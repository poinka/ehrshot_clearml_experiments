from __future__ import annotations

"""
01_train_tabular_multiseed_baselines.py

Назначение:
    Обучить табличные baseline-модели для EHRSHOT-задач на нескольких random seed.

Зачем нужен файл:
    Это один из ключевых файлов reproducibility package.
    Он отвечает за финальные tabular baselines, с которыми потом сравниваются:
        1. code-only sequence models;
        2. numeric-aware sequence models;
        3. compressed / raw sequence variants;
        4. fusion/exploratory models.

Какие задачи покрывает:
    - guo_readmission:
        модель: HistGradientBoosting
        смысл: основной табличный baseline для readmission.

    - guo_icu:
        модель: RandomForest_balanced
        смысл: основной табличный baseline для ICU transfer.

Какие артефакты сохраняет:
    - per-seed metrics;
    - per-seed held-out predictions;
    - top-k метрики;
    - общий файл metrics по всем seed;
    - общий файл predictions по всем seed;
    - общий файл top-k по всем seed;
    - configs.csv для воспроизводимости;
    - feature_manifest.csv со списком использованных признаков.

Важное ограничение:
    Этот скрипт НЕ пересобирает признаки из raw EHRSHOT.
    Он ожидает уже готовый feature cache:
        ehrshot_baseline_cache/
            guo_readmission_features_top500_num40.parquet
            guo_icu_features_top500_num40.parquet
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Callable
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common_ehrshot_eval import (
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    topk_metrics,
)


# -----------------------------------------------------------------------------
# 1. Общие константы
# -----------------------------------------------------------------------------

# Колонки, которые НЕ являются признаками модели.
# Все остальные колонки в feature parquet считаются model features.
META_COLS = {
    "task",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
}

# Основные табличные baseline-конфиги.
# Именно эти модели сейчас используются как reference models в сравнении.
DEFAULT_CONFIGS = [
    {
        "task": "guo_readmission",
        "model": "HistGradientBoosting",
        "family": "tabular",
        "representation": "tabular_all_features",
        "numeric_on": True,
        "notes": "Main tabular baseline for readmission.",
    },
    {
        "task": "guo_icu",
        "model": "RandomForest_balanced",
        "family": "tabular",
        "representation": "tabular_all_features",
        "numeric_on": True,
        "notes": "Main tabular baseline for ICU transfer.",
    },
]


DEFAULT_CHECKPOINT_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "checkpoints"
)


def safe_path_part(value: object) -> str:
    """
    Делает строку безопасной для использования как часть MinIO/S3 path.

    Например:
        "GRU 2L" -> "GRU_2L"
        "a/b"    -> "a__b"

    Это нужно, чтобы model_name / representation случайно не ломали структуру папок.
    """
    return str(value).replace("/", "__").replace(" ", "_")


def upload_file_to_minio(local_path: Path, remote_url: str) -> str:
    """
    Загружает локальный файл в MinIO/S3 через ClearML StorageManager.

    Возвращает remote_url, чтобы потом сохранить его в configs.csv.

    Если remote_url пустой, upload пропускается.
    """
    if not remote_url:
        return ""

    from clearml import StorageManager

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    print(f"Uploading checkpoint: {local_path} -> {remote_url}")

    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )

    return remote_url

def build_tabular_checkpoint_remote_url(
    checkpoint_s3_prefix: str,
    run_set: str,
    task: str,
    model_name: str,
    seed: int,
) -> str:
    """
    Строит MinIO/S3 path для tabular checkpoint.

    Итоговая структура:
        <checkpoint_s3_prefix>/
            <run_set>/
              <task>/
                tabular/
                  <model_name>_tabular_all_features_seed<seed>_model.joblib
    """
    if not checkpoint_s3_prefix:
        return ""

    filename = "_".join(
        [
            safe_path_part(model_name),
            "tabular_all_features",
            f"seed{int(seed)}",
            "model.joblib",
        ]
    )

    parts = [
        checkpoint_s3_prefix.rstrip("/"),
        safe_path_part(run_set),
        safe_path_part(task),
        "tabular",
        filename,
    ]

    return "/".join(parts)

# -----------------------------------------------------------------------------
# 2. Загрузка tabular feature cache
# -----------------------------------------------------------------------------

def load_cached_features(
    cache_dir: Path,
    task: str,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    """
    Загружает готовые табличные признаки для одной задачи.

    Ожидаемый формат файла:
        <task>_features_top<top_n_codes>_num<top_n_numeric_codes>.parquet

    Пример:
        guo_readmission_features_top500_num40.parquet

    Почему это важно:
        Для reproducibility package надо явно знать, какой feature cache
        использовался при обучении. Поэтому top_n_codes и top_n_numeric_codes
        становятся частью config.
    """
    path = cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Cached tabular features not found: {path}. "
            "Сначала нужно запустить classical baseline notebook "
            "или скопировать ehrshot_baseline_cache в рабочую директорию ClearML."
        )

    df = pd.read_parquet(path)

    # Минимальная проверка схемы, чтобы не обучить модель на битом кеше.
    required_cols = {"row_id", "subject_id", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature cache for task={task} is missing required columns: {missing}"
        )

    return df


def maybe_download_tabular_cache_from_s3_prefix(
    cache_dir: Path,
    cache_s3_prefix: str,
    tasks: list[str],
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> Path:
    """
    Проверяет, есть ли feature cache локально.

    Если локально файлов нет, пытается скачать их из MinIO/S3 через ClearML StorageManager.

    Ожидаемая remote-структура:
        <cache_s3_prefix>/guo_readmission_features_top500_num40.parquet
        <cache_s3_prefix>/guo_icu_features_top500_num40.parquet

    Зачем это нужно:
        ClearML-agent запускается в чистой среде.
        Поэтому данные надо либо заранее положить в рабочую директорию,
        либо уметь подтянуть их из object storage.
    """
    expected_files = [
        cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
        for task in tasks
    ]

    # Если все нужные файлы уже есть, ничего не скачиваем.
    if all(path.exists() for path in expected_files):
        print(f"Using local tabular cache: {cache_dir}")
        return cache_dir

    # Если файлов нет и S3 prefix не передан — честно падаем.
    if not cache_s3_prefix:
        missing = [str(path) for path in expected_files if not path.exists()]
        raise FileNotFoundError(
            "Tabular cache files are missing locally and --cache-s3-prefix is not provided. "
            f"Missing files: {missing}"
        )

    from clearml import StorageManager

    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_s3_prefix.rstrip("/")

    for task in tasks:
        filename = f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
        dst = cache_dir / filename

        if dst.exists():
            print(f"Already exists locally: {dst}")
            continue

        remote_url = f"{prefix}/{filename}"
        print(f"Downloading tabular cache: {remote_url}")

        local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

        if not local_copy.exists():
            raise FileNotFoundError(
                f"ClearML StorageManager returned non-existing local copy: {local_copy}"
            )

        print(f"Copying {local_copy} -> {dst}")
        shutil.copy2(local_copy, dst)

    return cache_dir


# -----------------------------------------------------------------------------
# 3. Создание моделей
# -----------------------------------------------------------------------------

def make_model(model_name: str, seed: int):
    """
    Создает sklearn-модель по имени.

    Важно:
        Здесь должны быть только те модели, которые мы реально хотим
        воспроизводимо запускать и сравнивать.

    Сейчас основные:
        - HistGradientBoosting для readmission;
        - RandomForest_balanced для ICU.

    LogisticRegression_balanced оставлена как дополнительный sanity baseline,
    но она не входит в DEFAULT_CONFIGS.
    """
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=seed,
        )

    if model_name == "RandomForest_balanced":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    if model_name == "LogisticRegression_balanced":
        return Pipeline(
            steps=[
                # Заполняем пропуски нулями.
                # Для sparse/count-like признаков это приемлемый простой baseline.
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),

                # Логистическая регрессия чувствительна к масштабу признаков.
                ("scaler", StandardScaler()),

                # class_weight нужен из-за несбалансированных labels.
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown tabular model: {model_name}")


def positive_proba(model, x: np.ndarray) -> np.ndarray:
    """
    Возвращает вероятность положительного класса.

    Для бинарной классификации sklearn predict_proba возвращает:
        [:, 0] — P(class=0)
        [:, 1] — P(class=1)

    Нам нужна P(label=1), то есть риск события.
    """
    return model.predict_proba(x)[:, 1]


# -----------------------------------------------------------------------------
# 4. Platt calibration
# -----------------------------------------------------------------------------

def fit_platt(
    y_tuning: np.ndarray,
    p_tuning: np.ndarray,
    seed: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], LogisticRegression]:
    """
    Обучает Platt calibration на tuning split.

    Что такое Platt:
        Это логистическая регрессия поверх score/logit модели.
        Она не меняет ranking, но может улучшить качество вероятностей:
            - Brier score;
            - LogLoss;
            - calibration behavior.

    Почему используем tuning, а не held_out:
        held_out должен оставаться финальной честной оценкой.
        Нельзя подбирать calibration на held_out, иначе будет leakage.

    Вход:
        y_tuning  — истинные labels на tuning;
        p_tuning  — raw probabilities модели на tuning.

    Выход:
        transform — функция, которая применяет calibration к новым вероятностям;
        calibrator — обученная LogisticRegression.
    """
    p = np.clip(np.asarray(p_tuning), 1e-6, 1 - 1e-6)

    # Переводим вероятность в logit.
    # Это стандартный вход для Platt scaling.
    logits = np.log(p / (1 - p)).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    calibrator.fit(logits, y_tuning.astype(int))

    def transform(p_raw: np.ndarray) -> np.ndarray:
        """
        Применяет обученную calibration к raw probabilities.
        """
        p_raw = np.clip(np.asarray(p_raw), 1e-6, 1 - 1e-6)
        x = np.log(p_raw / (1 - p_raw)).reshape(-1, 1)
        return calibrator.predict_proba(x)[:, 1]

    return transform, calibrator


# -----------------------------------------------------------------------------
# 5. Сохранение configs и feature manifest
# -----------------------------------------------------------------------------

def build_run_config_row(
    task: str,
    model_name: str,
    seed: int,
    split: str,
    cache_dir: Path,
    top_n_codes: int,
    top_n_numeric_codes: int,
    output_dir: Path,
    checkpoint_path: str | None,
    checkpoint_s3_url: str | None,
    clearml_task_id: str | None,
) -> dict:
    """
    Формирует одну строку config для reproducibility package.

    Эта таблица нужна, чтобы потом можно было понять:
        - какая задача запускалась;
        - какая модель;
        - какой seed;
        - какие признаки;
        - откуда брались данные;
        - куда сохранялись результаты;
        - был ли ClearML run.
    """
    return {
        "task": task,
        "model_family": "tabular",
        "model_name": model_name,
        "representation": "tabular_all_features",
        "compression_version": "none",
        "numeric_on": True,

        "seed": int(seed),
        "split": split,

        "max_len": None,
        "top_n_codes": top_n_codes,
        "top_n_numeric_codes": top_n_numeric_codes,

        "dataset_path_or_version": str(cache_dir),
        "checkpoint_path": checkpoint_path or "",
        "checkpoint_s3_url": checkpoint_s3_url or "",
        "output_dir": str(output_dir),

        "code_path": "code/01_train_tabular_multiseed_baselines.py",
        "notebook_path": "1.classical_baselines_readmission_icu.ipynb",
        "clearml_task_id": clearml_task_id or "",
    }


def build_feature_manifest(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    """
    Создает feature manifest: список признаков, которые реально ушли в модель.

    Зачем:
        Если через месяц модель надо будет воспроизвести,
        одного metrics.csv недостаточно.
        Нужно знать точный набор feature columns.
    """
    feature_cols = [col for col in df.columns if col not in META_COLS]

    rows = []
    for idx, col in enumerate(feature_cols):
        rows.append(
            {
                "task": task,
                "model": model_name,
                "family": "tabular",
                "representation": "tabular_all_features",
                "feature_index": idx,
                "feature_name": col,
                "top_n_codes": top_n_codes,
                "top_n_numeric_codes": top_n_numeric_codes,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. Один запуск: task + model + seed
# -----------------------------------------------------------------------------

def run_one_seed_task_model(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None, str | None]:
    """
    Обучает одну модель для одной задачи и одного seed.

    Делает:
        1. Делит уже готовый dataframe на train/tuning/held_out.
        2. Обучает модель на train.
        3. Считает raw probabilities на tuning и held_out.
        4. Обучает Platt calibration на tuning.
        5. Считает calibrated probabilities на held_out.
        6. Сохраняет:
            - metrics;
            - predictions;
            - top-k metrics.

        Именно held_out используется как финальная оценка в отчетах.
    """
    set_global_seed(seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [col for col in df.columns if col not in META_COLS]

    if not feature_cols:
        raise ValueError(f"No feature columns found for task={task}")

    train_df = df[df["split"] == "train"].copy()
    tuning_df = df[df["split"] == "tuning"].copy()
    heldout_df = df[df["split"] == "held_out"].copy()

    if train_df.empty or tuning_df.empty or heldout_df.empty:
        raise ValueError(
            f"Empty split detected for task={task}: "
            f"train={len(train_df)}, tuning={len(tuning_df)}, held_out={len(heldout_df)}"
        )

    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()

    x_tuning = tuning_df[feature_cols].to_numpy()
    y_tuning = tuning_df["label"].astype(int).to_numpy()

    x_heldout = heldout_df[feature_cols].to_numpy()
    y_heldout = heldout_df["label"].astype(int).to_numpy()

    model = make_model(model_name, seed=seed)

    # HistGradientBoosting не принимает class_weight напрямую в этой форме,
    # поэтому передаем sample_weight.
    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)

    # Raw probabilities.
    p_tuning_raw = positive_proba(model, x_tuning)
    p_heldout_raw = positive_proba(model, x_heldout)

    # Platt calibration обучается только на tuning.
    platt_transform, platt_calibrator = fit_platt(
        y_tuning=y_tuning,
        p_tuning=p_tuning_raw,
        seed=seed,
    )

    p_heldout_platt = platt_transform(p_heldout_raw)

    result_rows = []
    pred_frames = []
    topk_frames = []

    # Сохраняем обе версии:
    #   raw   — некалиброванные вероятности;
    #   platt — вероятности после Platt calibration.
    for calibration, p_heldout in [
        ("raw", p_heldout_raw),
        ("platt", p_heldout_platt),
    ]:
        metrics = binary_ranking_metrics(y_heldout, p_heldout)

        result_rows.append(
            {
                "task": task,
                "model_family": "tabular",
                "model_name": model_name,
                "representation": "tabular_all_features",
                "compression_version": "none",
                "numeric_on": True,
                "calibration": calibration,
                "seed": int(seed),
                "split": "held_out",
                **{f"heldout_{metric_name}": value for metric_name, value in metrics.items()},
            }
        )

        
        # Сохраняем predictions сразу в нормализованном формате reproducibility package.
        #
        # Единая схема predictions:
        #   task                 — guo_readmission / guo_icu;
        #   model_family         — tabular / sequence / numeric_sequence / fusion;
        #   model_name           — название конкретной модели;
        #   representation       — тип входного представления;
        #   compression_version  — версия сжатия, для tabular = none;
        #   numeric_on           — использовались ли numeric-признаки;
        #   calibration          — raw / platt.
        #   seed                 — random seed;
        #   split                — train / tuning / held_out;
        #   example_id           — id prediction example;
        #   subject_id           — id пациента;
        #   y_true               — истинный label;
        #   pred_proba           — предсказанная вероятность события;
        
        pred_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "model_family": "tabular",
                    "model_name": model_name,
                    "representation": "tabular_all_features",
                    "compression_version": "none",
                    "numeric_on": True,
                    "calibration": calibration,
                    "seed": int(seed),
                    "split": "held_out",
                    "example_id": heldout_df["row_id"].astype(int).to_numpy(),
                    "subject_id": heldout_df["subject_id"].astype(int).to_numpy(),
                    "y_true": y_heldout.astype(int),
                    "pred_proba": p_heldout.astype(float), 
                }
            )
        )

        topk = topk_metrics(y_heldout, p_heldout)

        topk.insert(0, "split", "held_out")
        topk.insert(0, "seed", int(seed))
        topk.insert(0, "calibration", calibration)
        topk.insert(0, "numeric_on", True)
        topk.insert(0, "compression_version", "none")
        topk.insert(0, "representation", "tabular_all_features")
        topk.insert(0, "model_name", model_name)
        topk.insert(0, "model_family", "tabular")
        topk.insert(0, "task", task)

        topk_frames.append(topk)

    result_df = pd.DataFrame(result_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    topk_df = pd.concat(topk_frames, ignore_index=True)

    stem = f"{task}__tabular_all_features__{model_name}__seed{seed}"

    result_df.to_csv(args.output_dir / f"{stem}__results.csv", index=False)
    pred_df.to_csv(args.output_dir / f"{stem}__heldout_predictions.csv", index=False)
    topk_df.to_csv(args.output_dir / f"{stem}__topk.csv", index=False)

    checkpoint_path = None
    checkpoint_s3_url = None

    if not args.no_save_checkpoints:
        local_ckpt_dir = (
            args.output_dir
            / "checkpoints"
            / safe_path_part(args.run_set)
            / safe_path_part(task)
            / "tabular"
        )
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_filename = "_".join(
            [
                safe_path_part(model_name),
                "tabular_all_features",
                f"seed{int(seed)}",
                "model.joblib",
            ]
        )

        ckpt_path = local_ckpt_dir / ckpt_filename

        joblib.dump(
            {
                # Идентификация запуска.
                "task": task,
                "model_family": "tabular",
                "model_name": model_name,
                "representation": "tabular_all_features",
                "compression_version": "none",
                "numeric_on": True,
                "seed": int(seed),

                # Данные и признаки.
                "feature_cols": feature_cols,
                "meta_cols": sorted(META_COLS),

                # Модель и calibration.
                "model": model,
                "platt_calibrator": platt_calibrator,

                # Для восстановления вероятностей:
                # raw   = model.predict_proba(X)[:, 1]
                # platt = calibrator(logit(raw))
                "calibration": "raw_and_platt",
            },
            ckpt_path,
        )

        checkpoint_path = str(ckpt_path)

        remote_ckpt_url = build_tabular_checkpoint_remote_url(
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            run_set=args.run_set,
            task=task,
            model_name=model_name,
            seed=seed,
        )

        checkpoint_s3_url = upload_file_to_minio(
            local_path=ckpt_path,
            remote_url=remote_ckpt_url,
        )

    return result_df, pred_df, topk_df, checkpoint_path, checkpoint_s3_url


# -----------------------------------------------------------------------------
# 7. ClearML helpers
# -----------------------------------------------------------------------------

def build_clearml_config(args) -> dict:
    """
    Готовит config для ClearML.

    ClearML плохо сериализует Path/PosixPath,
    поэтому пути явно переводим в строки.
    """
    config = vars(args).copy()

    for key in ["cache_dir", "output_dir"]:
        if key in config:
            config[key] = str(config[key])

    config["default_configs"] = DEFAULT_CONFIGS

    return config


def sync_args_from_clearml_config(args, config: dict) -> None:
    """
    На remote ClearML agent скрипт часто стартует без исходных CLI-аргументов.

    task.connect(config) подтягивает параметры из ClearML,
    но не всегда автоматически обновляет argparse Namespace.

    Поэтому руками переносим значения из ClearML config обратно в args.
    """
    path_keys = {"cache_dir", "output_dir"}
    int_keys = {"top_n_codes", "top_n_numeric_codes"}
    bool_keys = {"no_save_checkpoints"}

    # Эти флаги нельзя синхронизировать на remote,
    # иначе remote job может попытаться заново поставить сам себя в очередь.
    skip_keys = {"enable_clearml", "execute_remotely"}

    for key, value in config.items():
        if key in skip_keys:
            continue

        if not hasattr(args, key):
            continue

        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in bool_keys:
            setattr(args, key, bool(value))
        else:
            setattr(args, key, value)


def is_clearml_agent_run() -> bool:
    """
    Проверяет, запущен ли код внутри ClearML agent.

    CLEARML_TASK_ID / TRAINS_TASK_ID появляются в окружении remote task.
    """
    return bool(
        os.environ.get("CLEARML_TASK_ID")
        or os.environ.get("TRAINS_TASK_ID")
    )


def maybe_init_clearml(args, config: dict):
    """
    Инициализирует ClearML, если он нужен.

    Логика:
        1. Если --enable-clearml не указан и мы не внутри agent — работаем локально.
        2. Если мы локально и указан --execute-remotely — создаем task и отправляем в очередь.
        3. Если мы уже внутри agent — берем текущий task и продолжаем выполнение.
    """
    remote_agent_run = is_clearml_agent_run()

    if not args.enable_clearml and not remote_agent_run:
        return None, config

    from clearml import Task

    # Не хотим, чтобы ClearML автоматически замораживал огромный env.
    # Берем requirements.txt как источник зависимостей.
    if not remote_agent_run:
        Task.force_requirements_env_freeze(False, "requirements.txt")

    should_execute_remotely = args.execute_remotely and not remote_agent_run

    if remote_agent_run:
        task = Task.current_task()

        if task is None:
            task = Task.init(
                project_name=args.clearml_project,
                task_name=args.clearml_task_name,
                output_uri=args.clearml_output_uri or None,
                auto_connect_arg_parser=False,
            )
    else:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            output_uri=args.clearml_output_uri or None,
            auto_connect_arg_parser=False,
        )

    connected_config = task.connect(config)
    connected_config = dict(connected_config)

    sync_args_from_clearml_config(args, connected_config)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task_id = {task.id}")
    print(f"  seeds = {args.seeds}")
    print(f"  cache_dir = {args.cache_dir}")
    print(f"  cache_s3_prefix = {args.cache_s3_prefix}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if should_execute_remotely:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")

        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task, connected_config


# -----------------------------------------------------------------------------
# 8. CLI и main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    CLI-аргументы.

    Пример локального запуска:
        python 01_train_tabular_multiseed_baselines.py \\
            --seeds 42,43,44,45,46

    Пример ClearML remote запуска:
        python 01_train_tabular_multiseed_baselines.py \\
            --seeds 42,43,44,45,46 \\
            --enable-clearml \\
            --execute-remotely \\
            --clearml-queue cpu \\
            --clearml-task-name tabular_multiseed_stability
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ehrshot_baseline_cache"),
        help="Local directory with cached tabular feature parquet files.",
    )

    parser.add_argument(
        "--run-set",
        type=str,
        default="tabular_v1",
        help="Название набора tabular-прогонов для структуры checkpoints в MinIO.",
    )

    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help=(
            "Не сохранять checkpoints. "
            "По умолчанию tabular checkpoints сохраняются локально и загружаются в MinIO."
        ),
    )

    parser.add_argument(
        "--checkpoint-s3-prefix",
        type=str,
        default=DEFAULT_CHECKPOINT_S3_PREFIX,
        help="MinIO/S3 prefix для checkpoints.",
    )

    parser.add_argument(
        "--cache-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with tabular cache parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_multiseed_tabular_results"),
        help="Directory where metrics, predictions, configs and manifests will be saved.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--top-n-codes",
        type=int,
        default=500,
        help="Number of top code features used in cached tabular features.",
    )

    parser.add_argument(
        "--top-n-numeric-codes",
        type=int,
        default=40,
        help="Number of top numeric code features used in cached tabular features.",
    )

    parser.add_argument(
        "--enable-clearml",
        action="store_true",
        help="Enable ClearML logging.",
    )

    parser.add_argument(
        "--execute-remotely",
        action="store_true",
        help="If set, enqueue this ClearML task to an agent queue and stop local execution.",
    )

    parser.add_argument(
        "--clearml-queue",
        type=str,
        default="cpu",
        help="ClearML queue name for remote execution.",
    )

    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
        help="ClearML project name.",
    )

    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="tabular_multiseed_stability",
        help="ClearML task name.",
    )

    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
        help="ClearML output URI for artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = build_clearml_config(args)
    task, _ = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)

    # Подтягиваем feature cache локально или из S3.
    args.cache_dir = maybe_download_tabular_cache_from_s3_prefix(
        cache_dir=args.cache_dir,
        cache_s3_prefix=args.cache_s3_prefix,
        tasks=[cfg["task"] for cfg in DEFAULT_CONFIGS],
        top_n_codes=args.top_n_codes,
        top_n_numeric_codes=args.top_n_numeric_codes,
    )

    all_results = []
    all_predictions = []
    all_topk = []
    all_configs = []
    all_feature_manifests = []

    clearml_task_id = task.id if task is not None else None

    for cfg in DEFAULT_CONFIGS:
        task_name = cfg["task"]
        model_name = cfg["model"]

        print("=" * 100)
        print(f"Loading cached features for task={task_name}")

        df = load_cached_features(
            cache_dir=args.cache_dir,
            task=task_name,
            top_n_codes=args.top_n_codes,
            top_n_numeric_codes=args.top_n_numeric_codes,
        )

        feature_manifest = build_feature_manifest(
            df=df,
            task=task_name,
            model_name=model_name,
            top_n_codes=args.top_n_codes,
            top_n_numeric_codes=args.top_n_numeric_codes,
        )
        all_feature_manifests.append(feature_manifest)

        for seed in seeds:
            print("=" * 100)
            print(
                f"Tabular experiment: "
                f"task={task_name}, model={model_name}, seed={seed}"
            )

            result_df, pred_df, topk_df, checkpoint_path, checkpoint_s3_url = run_one_seed_task_model(
                df=df,
                task=task_name,
                model_name=model_name,
                seed=seed,
                args=args,
            )

            all_results.append(result_df)
            all_predictions.append(pred_df)
            all_topk.append(topk_df)

            all_configs.append(
                build_run_config_row(
                    task=task_name,
                    model_name=model_name,
                    seed=seed,
                    split="held_out",
                    cache_dir=args.cache_dir,
                    top_n_codes=args.top_n_codes,
                    top_n_numeric_codes=args.top_n_numeric_codes,
                    output_dir=args.output_dir,
                    checkpoint_path=checkpoint_path,
                    checkpoint_s3_url=checkpoint_s3_url,
                    clearml_task_id=clearml_task_id,
                )
            )

    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    topk_df = pd.concat(all_topk, ignore_index=True)
    configs_df = pd.DataFrame(all_configs)
    feature_manifest_df = pd.concat(all_feature_manifests, ignore_index=True)

    # Главные итоговые файлы для reproducibility package.
    results_path = args.output_dir / "tabular_multiseed_results.csv"
    predictions_path = args.output_dir / "tabular_multiseed_heldout_predictions.csv"
    topk_path = args.output_dir / "tabular_multiseed_topk.csv"
    configs_path = args.output_dir / "tabular_multiseed_configs.csv"
    feature_manifest_path = args.output_dir / "tabular_feature_manifest.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    configs_df.to_csv(configs_path, index=False)
    feature_manifest_df.to_csv(feature_manifest_path, index=False)

    print("\nSaved tabular outputs:")
    print(f"  metrics: {results_path}")
    print(f"  predictions: {predictions_path}")
    print(f"  top-k: {topk_path}")
    print(f"  configs: {configs_path}")
    print(f"  feature manifest: {feature_manifest_path}")

    print("\nMetrics preview:")
    print(results_df.sort_values(["task", "seed", "calibration"]))

    # Загружаем артефакты в ClearML, если task активен.
    if task is not None:
        task.upload_artifact("tabular_multiseed_results", results_df)
        task.upload_artifact("tabular_multiseed_predictions", predictions_df)
        task.upload_artifact("tabular_multiseed_topk", topk_df)
        task.upload_artifact("tabular_multiseed_configs", configs_df)
        task.upload_artifact("tabular_feature_manifest", feature_manifest_df)


if __name__ == "__main__":
    main()