from __future__ import annotations

"""
04_score_fusion_from_checkpoints.py

Score-level fusion из уже сохраненных checkpoints.

Главная идея:
    У нас уже обучены базовые модели:
        1. tabular baseline;
        2. code-only sequence model;
        3. numeric-aware sequence model.

    Этот скрипт НЕ переобучает базовые модели.
    Он только:
        1. загружает checkpoints;
        2. получает base scores на tuning и held_out;
        3. обучает маленькую LogisticRegression на tuning;
        4. оценивает fusion на held_out;
        5. сохраняет predictions в общей reproducibility-схеме.

Почему так:
    held_out нельзя использовать для обучения fusion, иначе будет leakage.
    Поэтому:
        tuning  -> обучение meta-model;
        held_out -> финальная оценка.
"""

import argparse
import importlib.util
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_ehrshot_eval import (
    DEFAULT_TOP_FRACS,
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    summarize_metrics_mean_std,
    topk_metrics,
)

# Колонки, которые не являются признаками в tabular feature cache.
META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}

# Дефолтный fusion config.
# Он используется только если отдельный configs/fusion_final_runs.json не найден.
DEFAULT_FUSION_CONFIG: dict[str, Any] = {
    "run_set": "final_repro_v1",
    "seeds": [42, 43, 44, 45, 46],
    "tasks": ["guo_readmission", "guo_icu"],
    "base_models": {
        "guo_readmission": {
            "tabular": {
                "model_family": "tabular",
                "model_name": "HistGradientBoosting",
                "representation": "tabular_all_features",
                "compression_version": "none",
                "numeric_on": True,
            },
            "sequence": {
                "model_family": "sequence",
                "model_name": "RETAIN_lite",
                "representation": "code_sequence_time2vec",
                "compression_version": "raw",
                "numeric_on": False,
            },
            "numeric_sequence": {
                "model_family": "numeric_sequence",
                "model_name": "RETAIN_lite_numeric",
                "representation": "numeric_sequence_time2vec",
                "compression_version": "raw",
                "numeric_on": True,
            },
        },
        "guo_icu": {
            "tabular": {
                "model_family": "tabular",
                "model_name": "RandomForest_balanced",
                "representation": "tabular_all_features",
                "compression_version": "none",
                "numeric_on": True,
            },
            "sequence": {
                "model_family": "sequence",
                "model_name": "GRU_2L",
                "representation": "code_sequence_time2vec",
                "compression_version": "raw",
                "numeric_on": False,
            },
            "numeric_sequence": {
                "model_family": "numeric_sequence",
                "model_name": "GRU_2L_numeric",
                "representation": "numeric_sequence_time2vec",
                "compression_version": "raw",
                "numeric_on": True,
            },
        },
    },
    "fusion_variants": {
        "score_tabular_only": ["p_tabular"],
        "score_sequence_only": ["p_sequence"],
        "score_numeric_sequence_only": ["p_numeric_sequence"],
        "score_tabular_sequence": ["p_tabular", "p_sequence"],
        "score_tabular_numeric_sequence": ["p_tabular", "p_numeric_sequence"],
        "score_all": ["p_tabular", "p_sequence", "p_numeric_sequence"],
    },
}


# -----------------------------------------------------------------------------
# Общие helper-функции
# -----------------------------------------------------------------------------

def read_json_if_exists(path: Path | None) -> dict[str, Any]:
    """
    Читает JSON config.

    Если путь пустой или файла нет, используем DEFAULT_FUSION_CONFIG.
    Это полезно для smoke-run, но для финального запуска лучше иметь явный config.
    """
    if path is None or str(path).strip() == "":
        return DEFAULT_FUSION_CONFIG
    path = Path(path)
    if not path.exists():
        print(f"WARNING: fusion config not found: {path}. Using built-in defaults.")
        return DEFAULT_FUSION_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_list(x: str) -> list[str]:
    """
    Превращает строку вида "a,b,c" в список ["a", "b", "c"].

    Используется для CLI-аргументов:
        --tasks guo_readmission,guo_icu
        --seeds 42,43,44
    """
    return [v.strip() for v in str(x).split(",") if v.strip()]


def safe_bool(x: Any) -> bool:
    """
    Безопасно приводит значение к bool.

    Нужно, потому что из JSON/ClearML значение может прийти как:
        True
        "true"
        "yes"
        "1"
    """
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def safe_path_part(value: object) -> str:
    """
    Делает строку безопасной для пути.

    Например:
        "a/b"    -> "a__b"
        "GRU 2L" -> "GRU_2L"
    """
    return str(value).replace("/", "__").replace(" ", "_")


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Обрезает вероятности, чтобы они не были ровно 0 или 1.

    Это нужно для logit-преобразования:
        log(p / (1 - p))

    Если p = 0 или p = 1, logit становится бесконечным.
    """
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    """
    Преобразует вероятность в logit.

    Для fusion это часто лучше, чем подавать вероятности напрямую,
    потому что LogisticRegression линейно работает именно в logit-space.
    """
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def transform_score_features(df: pd.DataFrame, cols: list[str], mode: str) -> np.ndarray:
    """
    Преобразует вероятность в logit.

    Для fusion это часто лучше, чем подавать вероятности напрямую,
    потому что LogisticRegression линейно работает именно в logit-space.
    """
    x = df[cols].to_numpy(dtype=float)
    if mode == "prob":
        return clip_prob(x)
    if mode == "logit":
        return prob_to_logit(x)
    raise ValueError(f"Unknown score transform: {mode}")


def get_device(device_arg: str) -> torch.device:
    """
    Выбирает устройство для sequence inference.

    auto:
        cuda, если доступна;
        иначе cpu.
    """
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """
    Загружает torch checkpoint.

    На разных версиях PyTorch есть параметр weights_only.
    Поэтому проверяем его наличие через inspect.
    """
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def import_sequence_module(path: Path):
    """
    Импортирует 02_train_sequence_multiseed.py как обычный Python-модуль.

    Почему так:
        файл начинается с цифры, поэтому нельзя сделать обычный import:
            import 02_train_sequence_multiseed

        Поэтому используем importlib.util.spec_from_file_location.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sequence module not found: {path}")
    spec = importlib.util.spec_from_file_location("sequence_repro_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import sequence module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_required_local_copy(remote_url: str) -> Path:
    """
    Скачивает файл из MinIO/S3 через ClearML StorageManager.

    Если файл не найден или StorageManager вернул битый путь — падаем явно.
    """
    from clearml import StorageManager

    local_copy = StorageManager.get_local_copy(remote_url=remote_url)
    if local_copy is None:
        raise FileNotFoundError(f"Could not download: {remote_url}")
    path = Path(local_copy)
    if not path.exists():
        raise FileNotFoundError(f"Downloaded path does not exist: {path}")
    return path


def maybe_download_file(remote_url: str, dst: Path) -> Path:
    """
    Если dst уже существует локально — используем его.
    Иначе скачиваем remote_url и копируем в dst.
    """
    dst = Path(dst)
    if dst.exists():
        return dst
    if not remote_url:
        raise FileNotFoundError(dst)
    local = get_required_local_copy(remote_url)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dst)
    return dst


# -----------------------------------------------------------------------------
# Поиск checkpoints
# -----------------------------------------------------------------------------

def find_tabular_checkpoint(
    checkpoint_dir: Path,
    run_set: str,
    task: str,
    model_name: str,
    seed: int,
    checkpoint_s3_prefix: str = "",
) -> Path:
    """
    Ищет tabular checkpoint.

    Ожидаемое имя:
        <model_name>_tabular_all_features_seed<seed>_model.joblib

    Ожидаемая структура:
        checkpoints/<run_set>/<task>/tabular/<filename>
    """
    filename = f"{safe_path_part(model_name)}_tabular_all_features_seed{seed}_model.joblib"

    candidates = [
        Path(checkpoint_dir) / run_set / task / "tabular" / filename,
        Path(checkpoint_dir) / task / "tabular" / filename,
        Path(checkpoint_dir) / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(Path(checkpoint_dir).glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Multiple tabular checkpoints found:\n" + "\n".join(map(str, matches)))

    if checkpoint_s3_prefix:
        remote_url = f"{checkpoint_s3_prefix.rstrip('/')}/{safe_path_part(run_set)}/{safe_path_part(task)}/tabular/{filename}"
        dst = Path(checkpoint_dir) / run_set / task / "tabular" / filename
        return maybe_download_file(remote_url, dst)

    raise FileNotFoundError(
        f"Tabular checkpoint not found: task={task}, model={model_name}, seed={seed}. "
        f"Searched in {checkpoint_dir}"
    )


def find_sequence_checkpoint(
    checkpoint_dir: Path,
    run_set: str,
    task: str,
    model_family: str,
    model_name: str,
    compression_version: str,
    seed: int,
    checkpoint_s3_prefix: str = "",
) -> Path:
    """
    Ищет sequence / numeric-sequence checkpoint.

    Ожидаемое имя:
        <model_name>_<compression_version>_seed<seed>_model.pt

    Ожидаемая структура:
        checkpoints/<run_set>/<task>/<model_family>/<filename>
    """
    filename = (
        f"{safe_path_part(model_name)}_"
        f"{safe_path_part(compression_version)}_"
        f"seed{seed}_model.pt"
    )

    candidates = [
        Path(checkpoint_dir) / run_set / task / model_family / filename,
        Path(checkpoint_dir) / task / model_family / filename,
        Path(checkpoint_dir) / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(Path(checkpoint_dir).glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Multiple sequence checkpoints found:\n" + "\n".join(map(str, matches)))

    if checkpoint_s3_prefix:
        remote_url = (
            f"{checkpoint_s3_prefix.rstrip('/')}/"
            f"{safe_path_part(run_set)}/{safe_path_part(task)}/{safe_path_part(model_family)}/{filename}"
        )
        dst = Path(checkpoint_dir) / run_set / task / model_family / filename
        return maybe_download_file(remote_url, dst)

    raise FileNotFoundError(
        f"Sequence checkpoint not found: task={task}, family={model_family}, "
        f"model={model_name}, version={compression_version}, seed={seed}. "
        f"Searched in {checkpoint_dir}"
    )


def load_tabular_features(
    cache_dir: Path,
    cache_s3_prefix: str,
    task: str,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    """
    Загружает tabular feature cache для задачи.

    Ожидаемое имя:
        guo_readmission_features_top500_num40.parquet
        guo_icu_features_top500_num40.parquet
    """
    filename = f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
    path = Path(cache_dir) / filename

    if not path.exists() and cache_s3_prefix:
        remote_url = f"{cache_s3_prefix.rstrip('/')}/{filename}"
        maybe_download_file(remote_url, path)

    if not path.exists():
        raise FileNotFoundError(f"Tabular feature cache not found: {path}")

    df = pd.read_parquet(path)
    if "task" not in df.columns:
        df["task"] = task
    return df


def ensure_sequence_dataset_available(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    task: str,
    compression_version: str,
) -> None:
    """
    Проверяет, что для sequence dataset есть:
        examples.parquet;
        vocab.json.

    Если локально нет, скачивает из MinIO/S3.
    """
    local_dir = Path(sequence_data_dir) / task / compression_version
    expected = [local_dir / "examples.parquet", local_dir / "vocab.json"]
    if all(p.exists() for p in expected):
        return

    if not sequence_data_s3_prefix:
        missing = [str(p) for p in expected if not p.exists()]
        raise FileNotFoundError(f"Missing sequence dataset files: {missing}")

    prefix = sequence_data_s3_prefix.rstrip("/")
    for filename in ["examples.parquet", "vocab.json"]:
        dst = local_dir / filename
        if dst.exists():
            continue
        remote_url = f"{prefix}/{task}/{compression_version}/{filename}"
        maybe_download_file(remote_url, dst)


# -----------------------------------------------------------------------------
# Inference базовых моделей
# -----------------------------------------------------------------------------

def apply_platt_calibrator(calibrator: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    """
    Применяет Platt calibration к вероятностям tabular model.

    В tabular checkpoint calibrator обучался на tuning.
    Здесь мы можем использовать его, если хотим брать calibrated base scores.
    """
    logits = prob_to_logit(p_raw).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def make_tabular_scores_from_checkpoint(
    cache_dir: Path,
    cache_s3_prefix: str,
    checkpoint_dir: Path,
    checkpoint_s3_prefix: str,
    run_set: str,
    task: str,
    model_name: str,
    seed: int,
    top_n_codes: int,
    top_n_numeric_codes: int,
    use_calibrated_base_scores: bool,
) -> pd.DataFrame:
    """
    Загружает tabular checkpoint и делает predict_proba на tuning + held_out.

    Возвращает DataFrame:
        task
        split
        seed
        example_id
        subject_id
        y_true
        p_tabular
        p_tabular_raw
    """
    set_global_seed(seed)

    ckpt_path = find_tabular_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=run_set,
        task=task,
        model_name=model_name,
        seed=seed,
    )

    print(f"Loading tabular checkpoint: {ckpt_path}")
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]
    feature_cols = ckpt["feature_cols"]
    calibrator = ckpt.get("platt_calibrator")

    df = load_tabular_features(
        cache_dir=cache_dir,
        cache_s3_prefix=cache_s3_prefix,
        task=task,
        top_n_codes=top_n_codes,
        top_n_numeric_codes=top_n_numeric_codes,
    )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Tabular feature cache missing checkpoint feature columns: {missing[:20]}")

    frames = []
    for split_name in ["tuning", "held_out"]:
        part = df[df["split"] == split_name].copy()
        if part.empty:
            raise ValueError(f"No rows for split={split_name}, task={task}")

        x = part[feature_cols].to_numpy()
        p_raw = model.predict_proba(x)[:, 1]
        p = p_raw
        if use_calibrated_base_scores and calibrator is not None:
            p = apply_platt_calibrator(calibrator, p_raw)

        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "split": split_name,
                    "seed": int(seed),
                    "example_id": part["row_id"].astype(int).to_numpy(),
                    "subject_id": part["subject_id"].astype(int).to_numpy(),
                    "y_true": part["label"].astype(int).to_numpy(),
                    "p_tabular": p.astype(float),
                    "p_tabular_raw": p_raw.astype(float),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def make_sequence_scores_from_checkpoint(
    seq_mod,
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    checkpoint_dir: Path,
    checkpoint_s3_prefix: str,
    run_set: str,
    task: str,
    model_family: str,
    model_name: str,
    representation: str,
    compression_version: str,
    numeric_on: bool,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    score_col: str,
) -> pd.DataFrame:
    """
    Загружает sequence checkpoint и делает inference на tuning + held_out.

    score_col:
        p_sequence
        или
        p_numeric_sequence
    """
    set_global_seed(seed)

    ensure_sequence_dataset_available(
        sequence_data_dir=sequence_data_dir,
        sequence_data_s3_prefix=sequence_data_s3_prefix,
        task=task,
        compression_version=compression_version,
    )

    ckpt_path = find_sequence_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=run_set,
        task=task,
        model_family=model_family,
        model_name=model_name,
        compression_version=compression_version,
        seed=seed,
    )

    print(f"Loading sequence checkpoint: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, device)

    df = seq_mod.load_sequence_examples(
        sequence_data_dir=sequence_data_dir,
        task=task,
        compression_version=compression_version,
    )
    vocab = seq_mod.load_vocab(
        sequence_data_dir=sequence_data_dir,
        task=task,
        compression_version=compression_version,
    )

    run_cfg = {
        "task": task,
        "model_family": model_family,
        "model_name": model_name,
        "base_model_name": ckpt.get("base_model_name", str(model_name).replace("_numeric", "")),
        "representation": representation,
        "compression_version": compression_version,
        "numeric_on": bool(numeric_on),
        "max_len": int(ckpt.get("max_len", 4096)),
        "batch_size": int(batch_size),
    }

    args_like = argparse.Namespace(
        num_workers=int(num_workers),
        emb_dim=int(ckpt.get("emb_dim", 64)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        dropout=float(ckpt.get("dropout", 0.20)),
    )

    token_mean = ckpt.get("token_numeric_mean")
    token_std = ckpt.get("token_numeric_std")

    if token_mean is None or token_std is None:
        train_df = df[df["split"] == "train"].copy()
        token_mean, token_std, _ = seq_mod.compute_token_numeric_stats(
            df_train=train_df,
            vocab_size=len(vocab),
            min_count=int(ckpt.get("numeric_min_count", 3)),
        )

    loaders = seq_mod.make_loaders(
        df=df,
        run_cfg=run_cfg,
        args=args_like,
        token_numeric_mean=token_mean,
        token_numeric_std=token_std,
        seed=seed,
    )

    model = seq_mod.make_model(run_cfg=run_cfg, vocab_size=len(vocab), args=args_like)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    frames = []
    for split_name in ["tuning", "held_out"]:
        pred = seq_mod.run_inference(model, loaders[split_name], device)
        p = seq_mod.sigmoid_np(pred["logits"])

        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "split": split_name,
                    "seed": int(seed),
                    "example_id": pred["example_id"].astype(int),
                    "subject_id": pred["subject_id"].astype(int),
                    "y_true": pred["y_true"].astype(int),
                    score_col: p.astype(float),
                    f"{score_col}_logit": pred["logits"].astype(float),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def merge_base_scores(tabular: pd.DataFrame, sequence: pd.DataFrame, numeric_sequence: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет tabular / sequence / numeric_sequence scores.

    Merge идет по одному и тому же prediction example:
        task
        split
        seed
        example_id
        subject_id
        y_true
    """
    key = ["task", "split", "seed", "example_id", "subject_id", "y_true"]
    out = tabular.merge(sequence, on=key, how="inner")
    out = out.merge(numeric_sequence, on=key, how="inner")
    if out.empty:
        raise ValueError("Base score merge produced 0 rows")
    return out


# -----------------------------------------------------------------------------
# Обучение score-level fusion и evaluation
# -----------------------------------------------------------------------------

def fit_meta_lr(train_df: pd.DataFrame, score_cols: list[str], score_transform: str, class_weight: str | None) -> Pipeline:
    """
    Обучает LogisticRegression поверх base scores.

    train_df — только tuning split.
    """
    x_train = transform_score_features(train_df, score_cols, mode=score_transform)
    y_train = train_df["y_true"].astype(int).to_numpy()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=3000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def predict_meta_lr(model: Pipeline, df: pd.DataFrame, score_cols: list[str], score_transform: str) -> np.ndarray:
    """
    Получает fusion probability на eval split.
    """
    x = transform_score_features(df, score_cols, mode=score_transform)
    return model.predict_proba(x)[:, 1]


def evaluate_prediction(y_true: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """
    Считает основные метрики для held_out predictions.
    """
    m = binary_ranking_metrics(y_true, risk)
    tk = topk_metrics(y_true, risk, top_fracs=DEFAULT_TOP_FRACS).set_index("top_frac")
    for frac in DEFAULT_TOP_FRACS:
        pct = int(round(frac * 100))
        m[f"top_{pct}pct_precision"] = float(tk.loc[frac, "top_k_event_rate"])
    return m


def fusion_prediction_frame(
    heldout: pd.DataFrame,
    risk: np.ndarray,
    *,
    task: str,
    seed: int,
    variant: str,
    score_cols: list[str],
    score_transform: str,
    class_weight: str | None,
) -> pd.DataFrame:
    """
    Формирует predictions в общей reproducibility-схеме.

    Эти колонки нужны, чтобы потом 03_compare_repro_results.py
    мог прочитать fusion как обычный prediction source.
    """
    return pd.DataFrame(
        {
            "task": task,
            "model_family": "fusion",
            "model_name": "score_lr",
            "representation": variant,
            "compression_version": "fusion",
            "numeric_on": any("numeric" in c for c in score_cols),
            "calibration": f"tuning_meta_lr_{score_transform}",
            "seed": int(seed),
            "split": "held_out",
            "example_id": heldout["example_id"].astype(int).to_numpy(),
            "subject_id": heldout["subject_id"].astype(int).to_numpy(),
            "y_true": heldout["y_true"].astype(int).to_numpy(),
            "pred_proba": risk.astype(float),
            "variant": variant,
            "score_cols": ",".join(score_cols),
            "score_transform": score_transform,
            "class_weight": class_weight or "none",
        }
    )


def run_single_seed_fusion(
    base_scores: pd.DataFrame,
    task: str,
    seed: int,
    fusion_variants: dict[str, list[str]],
    score_transform: str,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучает fusion отдельно для одного seed.

    Это нужно, чтобы честно сравнивать устойчивость по seed.
    """
    part = base_scores[(base_scores["task"] == task) & (base_scores["seed"] == seed)].copy()
    tuning = part[part["split"] == "tuning"].copy()
    heldout = part[part["split"] == "held_out"].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out for task={task}, seed={seed}")

    metric_rows = []
    pred_frames = []

    for variant, score_cols in fusion_variants.items():
        meta_model = fit_meta_lr(
            train_df=tuning,
            score_cols=score_cols,
            score_transform=score_transform,
            class_weight=class_weight,
        )
        risk = predict_meta_lr(meta_model, heldout, score_cols, score_transform)
        y = heldout["y_true"].astype(int).to_numpy()
        metrics = evaluate_prediction(y, risk)

        metric_rows.append(
            {
                "task": task,
                "model_family": "fusion",
                "model_name": "score_lr",
                "representation": variant,
                "compression_version": "fusion",
                "numeric_on": any("numeric" in c for c in score_cols),
                "calibration": f"tuning_meta_lr_{score_transform}",
                "seed": int(seed),
                "split": "held_out",
                "variant": variant,
                "score_cols": ",".join(score_cols),
                "score_transform": score_transform,
                "class_weight": class_weight or "none",
                **metrics,
            }
        )
        pred_frames.append(
            fusion_prediction_frame(
                heldout,
                risk,
                task=task,
                seed=seed,
                variant=variant,
                score_cols=score_cols,
                score_transform=score_transform,
                class_weight=class_weight,
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def make_ensemble_base_scores(base_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Делает seed ensemble для base scores.

    Для каждого example усредняем base scores по seed-ам.
    Потом обучаем fusion поверх уже усредненных scores.
    """
    id_cols = ["task", "split", "example_id", "subject_id", "y_true"]
    score_cols = ["p_tabular", "p_sequence", "p_numeric_sequence"]
    out = base_scores.groupby(id_cols, dropna=False)[score_cols].mean().reset_index()
    out["seed"] = -1
    return out


def run_ensemble_fusion(
    ensemble_scores: pd.DataFrame,
    task: str,
    fusion_variants: dict[str, list[str]],
    score_transform: str,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучает fusion на seed-ensemble scores.

    seed=-1 означает, что это не конкретный seed,
    а mean-over-seeds представление.
    """
    tuning = ensemble_scores[(ensemble_scores["task"] == task) & (ensemble_scores["split"] == "tuning")].copy()
    heldout = ensemble_scores[(ensemble_scores["task"] == task) & (ensemble_scores["split"] == "held_out")].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing ensemble tuning/held_out for task={task}")

    metric_rows = []
    pred_frames = []

    for variant, score_cols in fusion_variants.items():
        meta_model = fit_meta_lr(tuning, score_cols, score_transform, class_weight)
        risk = predict_meta_lr(meta_model, heldout, score_cols, score_transform)
        y = heldout["y_true"].astype(int).to_numpy()
        metrics = evaluate_prediction(y, risk)

        metric_rows.append(
            {
                "task": task,
                "model_family": "fusion",
                "model_name": "score_lr",
                "representation": variant,
                "compression_version": "fusion",
                "numeric_on": any("numeric" in c for c in score_cols),
                "calibration": f"tuning_meta_lr_{score_transform}",
                "seed": -1,
                "split": "held_out",
                "variant": variant,
                "score_cols": ",".join(score_cols),
                "score_transform": score_transform,
                "class_weight": class_weight or "none",
                "base_scores": "mean_over_seeds",
                **metrics,
            }
        )
        pred_frames.append(
            fusion_prediction_frame(
                heldout,
                risk,
                task=task,
                seed=-1,
                variant=variant,
                score_cols=score_cols,
                score_transform=score_transform,
                class_weight=class_weight,
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


# -----------------------------------------------------------------------------
# ClearML
# -----------------------------------------------------------------------------

def is_clearml_agent_run() -> bool:
    """
    Проверяет, запущен ли скрипт внутри ClearML agent.
    """
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args: argparse.Namespace, fusion_config: dict[str, Any]) -> dict[str, Any]:
    """
    Готовит параметры для ClearML.

    Важно:
        Path переводим в str, потому что ClearML иногда плохо сериализует Path.
    """
    cfg = vars(args).copy()
    for key in [
        "fusion_config",
        "cache_dir",
        "sequence_data_dir",
        "checkpoint_dir",
        "tabular_checkpoint_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
        "sequence_module_path",
    ]:
        if key in cfg:
            cfg[key] = str(cfg[key])
    cfg["fusion_config_json"] = fusion_config
    return cfg


def _to_bool(x: Any) -> bool:
    """
    Приводит значение из ClearML config к bool.
    """
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """
    На remote agent CLI-аргументы могут не прийти как sys.argv.

    Поэтому:
        task.connect(config) возвращает сохраненные параметры,
        а эта функция записывает их обратно в args.
    """
    path_keys = {
        "fusion_config",
        "cache_dir",
        "sequence_data_dir",
        "checkpoint_dir",
        "tabular_checkpoint_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
        "sequence_module_path",
    }
    int_keys = {"top_n_codes", "top_n_numeric_codes", "batch_size", "num_workers"}
    bool_keys = {"use_calibrated_base_scores"}
    skip_keys = {"enable_clearml", "execute_remotely", "fusion_config_json"}

    for key, value in dict(cfg).items():
        if key in skip_keys or not hasattr(args, key):
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in bool_keys:
            setattr(args, key, _to_bool(value))
        else:
            setattr(args, key, value)


def maybe_init_clearml(args: argparse.Namespace, fusion_config: dict[str, Any]):
    """
    Инициализирует ClearML и синхронизирует параметры.

    Важно:
        sync делаем до чтения fusion_config,
        чтобы remote agent сначала восстановил путь к config.
    """
    remote_agent_run = is_clearml_agent_run()
    if not args.enable_clearml and not remote_agent_run:
        return None

    from clearml import Task

    if not remote_agent_run:
        Task.force_requirements_env_freeze(False, "requirements.txt")

    task = Task.current_task() if remote_agent_run else None
    if task is None:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            output_uri=args.clearml_output_uri or None,
            auto_connect_arg_parser=False,
        )

    connected = dict(task.connect(build_clearml_config(args, fusion_config)))
    sync_args_from_clearml_config(args, connected)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task_id = {task.id}")
    print(f"  fusion_config = {args.fusion_config}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


def safe_upload_artifact(task, name: str, obj: Any) -> None:
    """
    Безопасно загружает artifact в ClearML.

    Если MinIO опять ругнется на storage / upload,
    training/evaluation не должен падать из-за artifact upload.
    """
    if task is None:
        return
    try:
        task.upload_artifact(name, artifact_object=obj)
    except Exception as e:
        print(f"WARNING: failed to upload ClearML artifact {name}: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--fusion-config", type=Path, default=Path("configs/fusion_final_runs.json"))
    parser.add_argument("--sequence-module-path", type=Path, default=Path(__file__).with_name("02_train_sequence_multiseed.py"))

    parser.add_argument("--cache-dir", type=Path, default=Path("ehrshot_baseline_cache"))
    parser.add_argument("--cache-s3-prefix", type=str, default="")
    parser.add_argument("--sequence-data-dir", type=Path, default=Path("ehrshot_sequence_datasets"))
    parser.add_argument("--sequence-data-s3-prefix", type=str, default="")

    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-s3-prefix", type=str, default="")
    parser.add_argument("--tabular-checkpoint-dir", type=Path, default=Path(""))
    parser.add_argument("--sequence-checkpoint-dir", type=Path, default=Path(""))
    parser.add_argument("--numeric-checkpoint-dir", type=Path, default=Path(""))

    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_score_fusion_from_checkpoints"))
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")

    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--score-transform", type=str, default="logit", choices=["logit", "prob"])
    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"])
    parser.add_argument("--use-calibrated-base-scores", action="store_true")

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="score_fusion_from_checkpoints")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")

    args = parser.parse_args()

    fusion_config = read_json_if_exists(args.fusion_config)
    clearml_task = maybe_init_clearml(args, fusion_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_mod = import_sequence_module(args.sequence_module_path)
    device = get_device(args.device)

    run_set = fusion_config.get("run_set", "final_repro_v1")
    seeds = parse_int_list(args.seeds) if args.seeds.strip() else [int(x) for x in fusion_config.get("seeds", [42])]
    tasks = parse_csv_list(args.tasks) if args.tasks.strip() else list(fusion_config.get("tasks", []))
    base_models = fusion_config["base_models"]
    fusion_variants = fusion_config.get("fusion_variants", DEFAULT_FUSION_CONFIG["fusion_variants"])
    class_weight = None if args.class_weight == "none" else "balanced"

    tab_ckpt_dir = args.tabular_checkpoint_dir if str(args.tabular_checkpoint_dir) else args.checkpoint_dir
    seq_ckpt_dir = args.sequence_checkpoint_dir if str(args.sequence_checkpoint_dir) else args.checkpoint_dir
    num_ckpt_dir = args.numeric_checkpoint_dir if str(args.numeric_checkpoint_dir) else args.checkpoint_dir

    print("DEVICE:", device)
    print("RUN_SET:", run_set)
    print("TASKS:", tasks)
    print("SEEDS:", seeds)
    print("FUSION VARIANTS:", fusion_variants)

    all_base_scores = []
    config_rows = []

    for task in tasks:
        if task not in base_models:
            raise ValueError(f"Task {task} not found in base_models config")

        task_cfg = base_models[task]
        for seed in seeds:
            print("=" * 100)
            print(f"Building base scores: task={task}, seed={seed}")

            tab_cfg = task_cfg["tabular"]
            seq_cfg = task_cfg["sequence"]
            num_cfg = task_cfg["numeric_sequence"]

            tab_scores = make_tabular_scores_from_checkpoint(
                cache_dir=args.cache_dir,
                cache_s3_prefix=args.cache_s3_prefix,
                checkpoint_dir=tab_ckpt_dir,
                checkpoint_s3_prefix=args.checkpoint_s3_prefix,
                run_set=run_set,
                task=task,
                model_name=tab_cfg["model_name"],
                seed=seed,
                top_n_codes=args.top_n_codes,
                top_n_numeric_codes=args.top_n_numeric_codes,
                use_calibrated_base_scores=bool(args.use_calibrated_base_scores),
            )

            seq_scores = make_sequence_scores_from_checkpoint(
                seq_mod=seq_mod,
                sequence_data_dir=args.sequence_data_dir,
                sequence_data_s3_prefix=args.sequence_data_s3_prefix,
                checkpoint_dir=seq_ckpt_dir,
                checkpoint_s3_prefix=args.checkpoint_s3_prefix,
                run_set=run_set,
                task=task,
                model_family=seq_cfg["model_family"],
                model_name=seq_cfg["model_name"],
                representation=seq_cfg["representation"],
                compression_version=seq_cfg["compression_version"],
                numeric_on=safe_bool(seq_cfg["numeric_on"]),
                seed=seed,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                score_col="p_sequence",
            )

            num_scores = make_sequence_scores_from_checkpoint(
                seq_mod=seq_mod,
                sequence_data_dir=args.sequence_data_dir,
                sequence_data_s3_prefix=args.sequence_data_s3_prefix,
                checkpoint_dir=num_ckpt_dir,
                checkpoint_s3_prefix=args.checkpoint_s3_prefix,
                run_set=run_set,
                task=task,
                model_family=num_cfg["model_family"],
                model_name=num_cfg["model_name"],
                representation=num_cfg["representation"],
                compression_version=num_cfg["compression_version"],
                numeric_on=safe_bool(num_cfg["numeric_on"]),
                seed=seed,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                score_col="p_numeric_sequence",
            )

            merged = merge_base_scores(tab_scores, seq_scores, num_scores)
            all_base_scores.append(merged)

            config_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "run_set": run_set,
                    "tabular_model": tab_cfg["model_name"],
                    "sequence_model": seq_cfg["model_name"],
                    "sequence_version": seq_cfg["compression_version"],
                    "numeric_sequence_model": num_cfg["model_name"],
                    "numeric_sequence_version": num_cfg["compression_version"],
                    "score_transform": args.score_transform,
                    "base_scores_calibrated": bool(args.use_calibrated_base_scores),
                }
            )

    base_scores = pd.concat(all_base_scores, ignore_index=True)
    base_scores.to_csv(args.output_dir / "fusion_score_base_scores_tuning_heldout.csv", index=False)
    pd.DataFrame(config_rows).to_csv(args.output_dir / "fusion_score_configs.csv", index=False)

    print("Base scores shape:", base_scores.shape)
    print(
        base_scores.groupby(["task", "split", "seed"])
        .agg(n=("example_id", "size"), n_positive=("y_true", "sum"), event_rate=("y_true", "mean"))
        .reset_index()
        .to_string(index=False)
    )

    single_metric_parts = []
    single_pred_parts = []

    for task in tasks:
        for seed in seeds:
            m, p = run_single_seed_fusion(
                base_scores=base_scores,
                task=task,
                seed=seed,
                fusion_variants=fusion_variants,
                score_transform=args.score_transform,
                class_weight=class_weight,
            )
            single_metric_parts.append(m)
            single_pred_parts.append(p)

    single_metrics = pd.concat(single_metric_parts, ignore_index=True)
    single_predictions = pd.concat(single_pred_parts, ignore_index=True)
    seed_summary = summarize_metrics_mean_std(single_metrics)

    single_metrics.to_csv(args.output_dir / "fusion_score_single_seed_metrics.csv", index=False)
    single_predictions.to_csv(args.output_dir / "fusion_score_single_seed_predictions.csv", index=False)
    seed_summary.to_csv(args.output_dir / "fusion_score_seed_summary.csv", index=False)

    ensemble_scores = make_ensemble_base_scores(base_scores)
    ensemble_scores.to_csv(args.output_dir / "fusion_score_ensemble_base_scores_tuning_heldout.csv", index=False)

    ens_metric_parts = []
    ens_pred_parts = []
    for task in tasks:
        m, p = run_ensemble_fusion(
            ensemble_scores=ensemble_scores,
            task=task,
            fusion_variants=fusion_variants,
            score_transform=args.score_transform,
            class_weight=class_weight,
        )
        ens_metric_parts.append(m)
        ens_pred_parts.append(p)

    ensemble_metrics = pd.concat(ens_metric_parts, ignore_index=True)
    ensemble_predictions = pd.concat(ens_pred_parts, ignore_index=True)

    ensemble_metrics.to_csv(args.output_dir / "fusion_score_ensemble_metrics.csv", index=False)
    ensemble_predictions.to_csv(args.output_dir / "fusion_score_ensemble_predictions.csv", index=False)

    print("\nSaved outputs to:", args.output_dir)
    print("\nSingle-seed score fusion sorted by AUPRC:")
    show_cols = ["task", "representation", "n_seeds", "auprc_mean", "auprc_std", "auroc_mean", "brier_mean", "logloss_mean", "top_10pct_precision_mean"]
    show_cols = [c for c in show_cols if c in seed_summary.columns]
    print(seed_summary[show_cols].sort_values(["task", "auprc_mean"], ascending=[True, False]).to_string(index=False))

    print("\nEnsemble score fusion sorted by AUPRC:")
    show_cols = ["task", "representation", "auprc", "auroc", "brier", "logloss", "top_10pct_precision", "n", "n_positive"]
    print(ensemble_metrics[show_cols].sort_values(["task", "auprc"], ascending=[True, False]).to_string(index=False))

    safe_upload_artifact(clearml_task, "fusion_score_base_scores_tuning_heldout", base_scores)
    safe_upload_artifact(clearml_task, "fusion_score_single_seed_metrics", single_metrics)
    safe_upload_artifact(clearml_task, "fusion_score_single_seed_predictions", single_predictions)
    safe_upload_artifact(clearml_task, "fusion_score_seed_summary", seed_summary)
    safe_upload_artifact(clearml_task, "fusion_score_ensemble_metrics", ensemble_metrics)
    safe_upload_artifact(clearml_task, "fusion_score_ensemble_predictions", ensemble_predictions)


if __name__ == "__main__":
    main()
