from __future__ import annotations

"""
05_embedding_fusion_from_checkpoints.py

Embedding-level fusion из уже сохраненных checkpoints.

Главная идея:
    В score fusion мы объединяли только вероятности:
        p_tabular
        p_sequence
        p_numeric_sequence

    Здесь мы делаем более богатую fusion-модель:
        1. берем tabular features;
        2. достаем embedding из sequence model;
        3. достаем embedding из numeric-sequence model;
        4. объединяем эти признаки;
        5. обучаем meta-model на tuning;
        6. оцениваем на held_out.

Важно:
    - базовые модели НЕ переобучаются;
    - sequence checkpoints только загружаются;
    - held_out не используется для обучения meta-model;
    - predictions сохраняются в общей reproducibility-схеме.
"""


import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence

from common_ehrshot_eval import (
    DEFAULT_TOP_FRACS,
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    summarize_metrics_mean_std,
    topk_metrics,
)


META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}

# Какие группы признаков можно комбинировать в embedding fusion.
DEFAULT_EMBEDDING_VARIANTS = {
    "embedding_tabular_only": ["tabular"],
    "embedding_sequence_only": ["sequence_embedding"],
    "embedding_numeric_sequence_only": ["numeric_sequence_embedding"],
    "embedding_tabular_sequence": ["tabular", "sequence_embedding"],
    "embedding_tabular_numeric_sequence": ["tabular", "numeric_sequence_embedding"],
    "embedding_all": ["tabular", "sequence_embedding", "numeric_sequence_embedding"],
}

# Meta-models поверх embedding/table features.
# Держим модели маленькими, чтобы не переобучаться на tuning.
META_MODEL_CONFIGS: list[dict[str, Any]] = [
    {"name": "lr_l2_C001", "kind": "logreg", "C": 0.01},
    {"name": "lr_l2_C003", "kind": "logreg", "C": 0.03},
    {"name": "lr_l2_C01", "kind": "logreg", "C": 0.10},
    {"name": "lr_l2_C1", "kind": "logreg", "C": 1.00},
    {
        "name": "histgb_small",
        "kind": "histgb",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "min_samples_leaf": 30,
    },
    {
        "name": "histgb_medium",
        "kind": "histgb",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.1,
        "min_samples_leaf": 20,
    },
    {
        "name": "rf_small",
        "kind": "random_forest",
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
]


# -----------------------------------------------------------------------------
# Импортируем helpers из score-fusion
# -----------------------------------------------------------------------------

def import_module_from_path(path: Path, module_name: str):
    """
    Импортирует модуль из произвольного пути.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_csv_list(x: str) -> list[str]:
    """
    Разбирает строку с разделителем ',' в список строк.
    """
    return [v.strip() for v in str(x).split(",") if v.strip()]


# -----------------------------------------------------------------------------
# Tabular feature frame из checkpoint feature_cols
# -----------------------------------------------------------------------------

def make_tabular_feature_frame_from_checkpoint(
    helper,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит tabular часть признаков для embedding fusion.

    Почему берем feature_cols из checkpoint:
        так мы гарантируем, что используем ровно те признаки,
        на которых обучался tabular baseline.

    Возвращает:
        1. DataFrame с tab__0000, tab__0001, ...;
        2. manifest, где видно соответствие source_feature -> fusion_feature.
    """
    ckpt_path = helper.find_tabular_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=run_set,
        task=task,
        model_name=model_name,
        seed=seed,
    )
    print(f"Loading tabular checkpoint for feature list: {ckpt_path}")
    ckpt = joblib.load(ckpt_path)
    feature_cols = list(ckpt["feature_cols"])

    df = helper.load_tabular_features(
        cache_dir=cache_dir,
        cache_s3_prefix=cache_s3_prefix,
        task=task,
        top_n_codes=top_n_codes,
        top_n_numeric_codes=top_n_numeric_codes,
    ).copy()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Tabular cache missing checkpoint feature columns: {missing[:20]}")

    out = df[df["split"].isin(["tuning", "held_out"])].copy()
    out["task"] = task
    out["seed"] = int(seed)
    out["y_true"] = out["label"].astype(int)

    rename_map = {c: f"tab__{i:04d}" for i, c in enumerate(feature_cols)}
    out = out.rename(columns=rename_map)
    tab_cols = list(rename_map.values())

    keep = ["task", "split", "seed", "row_id", "subject_id", "y_true"] + tab_cols
    out = out[keep].copy()
    out = out.rename(columns={"row_id": "example_id"})

    for c in ["example_id", "subject_id", "seed", "y_true"]:
        out[c] = out[c].astype(int)

    manifest = pd.DataFrame(
        {
            "task": task,
            "seed": int(seed),
            "feature_group": "tabular",
            "source_feature_name": feature_cols,
            "fusion_feature_name": tab_cols,
            "source_checkpoint": str(ckpt_path),
        }
    )

    return out, manifest


# -----------------------------------------------------------------------------
# Извлечение sequence embeddings
# -----------------------------------------------------------------------------

def get_sequence_embedding_and_logits(seq_mod, model, batch_dev: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Достает logits и embedding из sequence-модели.

    Поддерживает:
        RNNClassifier: GRU/LSTM;
        RetainLiteClassifier.

    Возвращает:
        logits;
        embedding последнего/агрегированного состояния.
    """
    tokens = batch_dev["tokens"]
    time_features = batch_dev["time_features"]
    numeric_features = batch_dev["numeric_features"]
    mask = batch_dev["mask"]
    lengths = batch_dev["lengths"]

    # RNNClassifier из 02_train_sequence_multiseed.py.
    if hasattr(model, "rnn"):
        x = model.event_emb(tokens, time_features, numeric_features)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = model.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        embedding = h[-1]
        logits = model.head(embedding).squeeze(-1)
        return logits, embedding

    # RetainLiteClassifier из 02_train_sequence_multiseed.py.
    if hasattr(model, "alpha_gru") and hasattr(model, "beta_gru"):
        x = model.event_emb(tokens, time_features, numeric_features)
        x_rev = seq_mod.reverse_by_lengths(x, lengths)
        mask_rev = seq_mod.reverse_by_lengths(mask.unsqueeze(-1).float(), lengths).squeeze(-1).bool()

        alpha_h, _ = model.alpha_gru(x_rev)
        beta_h, _ = model.beta_gru(x_rev)

        alpha_logits = model.alpha_fc(alpha_h).squeeze(-1).masked_fill(~mask_rev, -1e9)
        alpha = torch.softmax(alpha_logits, dim=1)
        beta = torch.tanh(model.beta_fc(beta_h))

        embedding = torch.sum(alpha.unsqueeze(-1) * beta * x_rev, dim=1)
        logits = model.head(embedding).squeeze(-1)
        return logits, embedding

    raise TypeError(f"Unsupported sequence model type: {type(model)}")


def run_embedding_inference(seq_mod, model, loader, device: torch.device) -> dict[str, np.ndarray]:
    """
    Прогоняет модель по DataLoader и собирает embeddings.

    Возвращает numpy-массивы:
        logits;
        embeddings;
        y_true;
        example_id;
        subject_id.
    """
    model.eval()

    logits_parts = []
    emb_parts = []
    y_parts = []
    example_parts = []
    subject_parts = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = seq_mod.move_batch(batch, device)
            logits, embeddings = get_sequence_embedding_and_logits(seq_mod, model, batch_dev)

            logits_parts.append(logits.detach().cpu().numpy())
            emb_parts.append(embeddings.detach().cpu().numpy())
            y_parts.append(batch["labels"].numpy())
            example_parts.append(batch["example_ids"].numpy())
            subject_parts.append(batch["subject_ids"].numpy())

    return {
        "logits": np.concatenate(logits_parts),
        "embeddings": np.concatenate(emb_parts, axis=0),
        "y_true": np.concatenate(y_parts).astype(int),
        "example_id": np.concatenate(example_parts).astype(int),
        "subject_id": np.concatenate(subject_parts).astype(int),
    }


def make_sequence_embeddings_from_checkpoint(
    helper,
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
    embedding_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает sequence checkpoint и достает embeddings на tuning + held_out.

    embedding_prefix:
        seq_emb    для code-only sequence;
        numseq_emb для numeric sequence.
    """
    set_global_seed(seed)

    helper.ensure_sequence_dataset_available(
        sequence_data_dir=sequence_data_dir,
        sequence_data_s3_prefix=sequence_data_s3_prefix,
        task=task,
        compression_version=compression_version,
    )

    ckpt_path = helper.find_sequence_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=run_set,
        task=task,
        model_family=model_family,
        model_name=model_name,
        compression_version=compression_version,
        seed=seed,
    )

    print(f"Loading checkpoint for embeddings: {ckpt_path}")
    ckpt = helper.safe_torch_load(ckpt_path, device)

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
    emb_cols = None

    for split_name in ["tuning", "held_out"]:
        pred = run_embedding_inference(seq_mod, model, loaders[split_name], device)
        embeddings = pred["embeddings"].astype(np.float32)

        if emb_cols is None:
            emb_cols = [f"{embedding_prefix}__{i:04d}" for i in range(embeddings.shape[1])]

        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        meta_df = pd.DataFrame(
            {
                "task": task,
                "split": split_name,
                "seed": int(seed),
                "example_id": pred["example_id"].astype(int),
                "subject_id": pred["subject_id"].astype(int),
                "y_true": pred["y_true"].astype(int),
            }
        )
        frames.append(pd.concat([meta_df, emb_df], axis=1))

    manifest = pd.DataFrame(
        {
            "task": task,
            "seed": int(seed),
            "feature_group": embedding_prefix,
            "source_feature_name": emb_cols or [],
            "fusion_feature_name": emb_cols or [],
            "source_checkpoint": str(ckpt_path),
            "model_family": model_family,
            "model_name": model_name,
            "compression_version": compression_version,
        }
    )

    return pd.concat(frames, ignore_index=True), manifest


def merge_feature_frames(tabular: pd.DataFrame, sequence_emb: pd.DataFrame, numeric_sequence_emb: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет tabular features, sequence embeddings и numeric sequence embeddings.
    """
    key = ["task", "split", "seed", "example_id", "subject_id", "y_true"]
    out = tabular.merge(sequence_emb, on=key, how="inner")
    out = out.merge(numeric_sequence_emb, on=key, how="inner")
    if out.empty:
        raise ValueError("Embedding feature merge produced 0 rows")
    return out


# -----------------------------------------------------------------------------
# Meta-models
# -----------------------------------------------------------------------------

def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Делит колонки feature_df на группы:
        tabular;
        sequence_embedding;
        numeric_sequence_embedding.
    """
    return {
        "tabular": [c for c in df.columns if c.startswith("tab__")],
        "sequence_embedding": [c for c in df.columns if c.startswith("seq_emb__")],
        "numeric_sequence_embedding": [c for c in df.columns if c.startswith("numseq_emb__")],
    }


def resolve_feature_cols(groups: dict[str, list[str]], group_names: list[str]) -> list[str]:
    """
    По списку групп возвращает реальные feature columns.
    """
    cols: list[str] = []
    for group_name in group_names:
        if group_name not in groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        cols.extend(groups[group_name])
    if not cols:
        raise ValueError(f"No columns resolved for groups={group_names}")
    return cols


def filter_usable_columns(train_df: pd.DataFrame, feature_cols: list[str], min_std: float = 1e-12) -> list[str]:
    """
    Выкидывает all-NaN и константные признаки.

    Это нужно, чтобы meta-model не ломалась на бесполезных колонках.
    """
    usable = []
    for c in feature_cols:
        values = train_df[c].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        if np.nanstd(finite) <= min_std:
            continue
        usable.append(c)

    if not usable:
        raise ValueError("No usable feature columns after dropping all-NaN/constant columns")

    dropped = len(feature_cols) - len(usable)
    if dropped > 0:
        print(f"Dropped {dropped} all-NaN or constant feature columns")
    return usable


def make_meta_model(model_cfg: dict[str, Any], class_weight: str | None, seed: int) -> Pipeline:
    """
    Создает meta-model по config.

    Поддерживает:
        LogisticRegression;
        HistGradientBoosting;
        RandomForest.
    """
    kind = model_cfg["kind"]

    if kind == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight=class_weight,
                        C=float(model_cfg["C"]),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if kind == "histgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=float(model_cfg["learning_rate"]),
                        max_iter=int(model_cfg["max_iter"]),
                        max_leaf_nodes=int(model_cfg["max_leaf_nodes"]),
                        l2_regularization=float(model_cfg["l2_regularization"]),
                        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if kind == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(model_cfg["n_estimators"]),
                        max_depth=int(model_cfg["max_depth"]),
                        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
                        max_features=model_cfg["max_features"],
                        class_weight="balanced_subsample" if class_weight == "balanced" else None,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown meta model kind: {kind}")


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


def fit_predict_meta_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_cfg: dict[str, Any],
    class_weight: str | None,
    seed: int,
) -> tuple[np.ndarray, int]:
    """
    Обучает meta-model на tuning и предсказывает на held_out.

    Возвращает:
        risk;
        число реально использованных признаков.
    """
    usable_cols = filter_usable_columns(train_df, feature_cols)
    x_train = train_df[usable_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y_true"].astype(int).to_numpy()
    x_eval = eval_df[usable_cols].to_numpy(dtype=np.float32)

    model = make_meta_model(model_cfg=model_cfg, class_weight=class_weight, seed=seed)
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1], len(usable_cols)


def embedding_prediction_frame(
    heldout: pd.DataFrame,
    risk: np.ndarray,
    *,
    task: str,
    seed: int,
    variant: str,
    group_names: list[str],
    model_cfg: dict[str, Any],
    n_features_raw: int,
    n_features_used: int,
    class_weight: str | None,
) -> pd.DataFrame:
    """
    Формирует prediction DataFrame в общей reproducibility-схеме.
    """
    return pd.DataFrame(
        {
            "task": task,
            "model_family": "fusion",
            "model_name": model_cfg["name"],
            "representation": variant,
            "compression_version": "fusion",
            "numeric_on": any("numeric" in g for g in group_names),
            "calibration": "tuning_meta_model",
            "seed": int(seed),
            "split": "held_out",
            "example_id": heldout["example_id"].astype(int).to_numpy(),
            "subject_id": heldout["subject_id"].astype(int).to_numpy(),
            "y_true": heldout["y_true"].astype(int).to_numpy(),
            "pred_proba": risk.astype(float),
            "variant": variant,
            "feature_groups": ",".join(group_names),
            "meta_model": model_cfg["name"],
            "meta_model_kind": model_cfg["kind"],
            "n_features_raw": int(n_features_raw),
            "n_features_used": int(n_features_used),
            "class_weight": class_weight or "none",
        }
    )


def run_single_seed_embedding_fusion(
    feature_df: pd.DataFrame,
    task: str,
    seed: int,
    embedding_variants: dict[str, list[str]],
    meta_model_configs: list[dict[str, Any]],
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучает embedding fusion отдельно для одного seed.
    """
    part = feature_df[(feature_df["task"] == task) & (feature_df["seed"] == seed)].copy()
    tuning = part[part["split"] == "tuning"].copy()
    heldout = part[part["split"] == "held_out"].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out for task={task}, seed={seed}")

    groups = get_feature_groups(part)
    metric_rows = []
    pred_frames = []

    for model_cfg in meta_model_configs:
        for variant, group_names in embedding_variants.items():
            feature_cols = resolve_feature_cols(groups, group_names)
            print(
                f"Fitting single-seed embedding fusion: task={task}, seed={seed}, "
                f"meta_model={model_cfg['name']}, variant={variant}, n_raw_features={len(feature_cols)}"
            )

            risk, n_used = fit_predict_meta_model(
                train_df=tuning,
                eval_df=heldout,
                feature_cols=feature_cols,
                model_cfg=model_cfg,
                class_weight=class_weight,
                seed=seed,
            )
            y = heldout["y_true"].astype(int).to_numpy()
            metrics = evaluate_prediction(y, risk)

            metric_rows.append(
                {
                    "task": task,
                    "model_family": "fusion",
                    "model_name": model_cfg["name"],
                    "representation": variant,
                    "compression_version": "fusion",
                    "numeric_on": any("numeric" in g for g in group_names),
                    "calibration": "tuning_meta_model",
                    "seed": int(seed),
                    "split": "held_out",
                    "variant": variant,
                    "feature_groups": ",".join(group_names),
                    "meta_model": model_cfg["name"],
                    "meta_model_kind": model_cfg["kind"],
                    "n_features_raw": len(feature_cols),
                    "n_features_used": n_used,
                    "class_weight": class_weight or "none",
                    **metrics,
                }
            )
            pred_frames.append(
                embedding_prediction_frame(
                    heldout,
                    risk,
                    task=task,
                    seed=seed,
                    variant=variant,
                    group_names=group_names,
                    model_cfg=model_cfg,
                    n_features_raw=len(feature_cols),
                    n_features_used=n_used,
                    class_weight=class_weight,
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def make_ensemble_feature_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает seed-ensemble feature frame.

    Tabular features одинаковые по seed-ам, поэтому берем drop_duplicates.
    Sequence embeddings усредняем по seed-ам.
    """
    id_cols = ["task", "split", "example_id", "subject_id", "y_true"]
    tab_cols = [c for c in feature_df.columns if c.startswith("tab__")]
    emb_cols = [c for c in feature_df.columns if c.startswith("seq_emb__") or c.startswith("numseq_emb__")]

    tabular_part = feature_df[id_cols + tab_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    emb_part = feature_df.groupby(id_cols, dropna=False)[emb_cols].mean().reset_index()
    out = tabular_part.merge(emb_part, on=id_cols, how="inner")
    out["seed"] = -1
    return out


def run_ensemble_embedding_fusion(
    ensemble_df: pd.DataFrame,
    task: str,
    embedding_variants: dict[str, list[str]],
    meta_model_configs: list[dict[str, Any]],
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучает embedding fusion на mean-over-seeds embeddings.
    """
    part = ensemble_df[ensemble_df["task"] == task].copy()
    tuning = part[part["split"] == "tuning"].copy()
    heldout = part[part["split"] == "held_out"].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing ensemble tuning/held_out for task={task}")

    groups = get_feature_groups(part)
    metric_rows = []
    pred_frames = []

    for model_cfg in meta_model_configs:
        for variant, group_names in embedding_variants.items():
            feature_cols = resolve_feature_cols(groups, group_names)
            print(
                f"Fitting ensemble embedding fusion: task={task}, "
                f"meta_model={model_cfg['name']}, variant={variant}, n_raw_features={len(feature_cols)}"
            )

            risk, n_used = fit_predict_meta_model(
                train_df=tuning,
                eval_df=heldout,
                feature_cols=feature_cols,
                model_cfg=model_cfg,
                class_weight=class_weight,
                seed=42,
            )
            y = heldout["y_true"].astype(int).to_numpy()
            metrics = evaluate_prediction(y, risk)

            metric_rows.append(
                {
                    "task": task,
                    "model_family": "fusion",
                    "model_name": model_cfg["name"],
                    "representation": variant,
                    "compression_version": "fusion",
                    "numeric_on": any("numeric" in g for g in group_names),
                    "calibration": "tuning_meta_model",
                    "seed": -1,
                    "split": "held_out",
                    "variant": variant,
                    "feature_groups": ",".join(group_names),
                    "meta_model": model_cfg["name"],
                    "meta_model_kind": model_cfg["kind"],
                    "n_features_raw": len(feature_cols),
                    "n_features_used": n_used,
                    "base_embeddings": "mean_over_seeds",
                    "class_weight": class_weight or "none",
                    **metrics,
                }
            )
            pred_frames.append(
                embedding_prediction_frame(
                    heldout,
                    risk,
                    task=task,
                    seed=-1,
                    variant=variant,
                    group_names=group_names,
                    model_cfg=model_cfg,
                    n_features_raw=len(feature_cols),
                    n_features_used=n_used,
                    class_weight=class_weight,
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def select_meta_model_configs(names: str) -> list[dict[str, Any]]:
    """
    Выбирает meta-model configs по CLI-аргументу.

    names:
        "all"
        или
        "histgb_small,rf_small"
    """
    requested = parse_csv_list(names)
    by_name = {cfg["name"]: cfg for cfg in META_MODEL_CONFIGS}

    if not requested or requested == ["all"]:
        return META_MODEL_CONFIGS

    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown meta model names: {missing}. Available: {list(by_name)}")

    return [by_name[name] for name in requested]


# -----------------------------------------------------------------------------
# ClearML
# -----------------------------------------------------------------------------

def build_clearml_config(args: argparse.Namespace, fusion_config: dict[str, Any]) -> dict[str, Any]:
    """
    Готовит параметры для ClearML.
    """
    cfg = vars(args).copy()
    for key in [
        "fusion_config",
        "score_helper_path",
        "sequence_module_path",
        "cache_dir",
        "sequence_data_dir",
        "checkpoint_dir",
        "tabular_checkpoint_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
        "features_path",
        "ensemble_features_path",
    ]:
        if key in cfg:
            cfg[key] = str(cfg[key])
    cfg["fusion_config_json"] = fusion_config
    cfg["meta_model_configs"] = META_MODEL_CONFIGS
    return cfg


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any], helper) -> None:
    """
    Восстанавливает параметры ClearML task обратно в args.
    """
    path_keys = {
        "fusion_config",
        "score_helper_path",
        "sequence_module_path",
        "cache_dir",
        "sequence_data_dir",
        "checkpoint_dir",
        "tabular_checkpoint_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
        "features_path",
        "ensemble_features_path",
    }
    int_keys = {"top_n_codes", "top_n_numeric_codes", "batch_size", "num_workers"}
    bool_keys = {"reuse_features", "save_feature_parquet"}
    skip_keys = {"enable_clearml", "execute_remotely", "fusion_config_json", "meta_model_configs"}

    for key, value in dict(cfg).items():
        if key in skip_keys or not hasattr(args, key):
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in bool_keys:
            setattr(args, key, helper._to_bool(value))
        else:
            setattr(args, key, value)


def maybe_init_clearml(args: argparse.Namespace, fusion_config: dict[str, Any], helper):
    """
    Инициализирует ClearML.

    sync делаем до чтения fusion_config, чтобы remote agent
    сначала восстановил путь к config.
    """
    remote_agent_run = helper.is_clearml_agent_run()
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
    sync_args_from_clearml_config(args, connected, helper)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task_id = {task.id}")
    print(f"  fusion_config = {args.fusion_config}")
    print(f"  meta_models = {args.meta_models}")
    print(f"  reuse_features = {args.reuse_features}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--fusion-config", type=Path, default=Path("configs/fusion_final_runs.json"))
    parser.add_argument("--score-helper-path", type=Path, default=Path(__file__).with_name("04_score_fusion_from_checkpoints.py"))
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

    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_embedding_fusion_from_checkpoints"))
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")

    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--meta-models", type=str, default="all")
    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"])
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--features-path", type=Path, default=Path(""))
    parser.add_argument("--ensemble-features-path", type=Path, default=Path(""))
    parser.add_argument("--save-feature-parquet", action="store_true")

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="embedding_fusion_from_checkpoints")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")

    args = parser.parse_args()

    helper = import_module_from_path(args.score_helper_path, "score_fusion_helper")
    fusion_config = helper.read_json_if_exists(args.fusion_config)
    clearml_task = maybe_init_clearml(args, fusion_config, helper)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_mod = helper.import_sequence_module(args.sequence_module_path)
    device = helper.get_device(args.device)

    run_set = fusion_config.get("run_set", "final_repro_v1")
    seeds = parse_int_list(args.seeds) if args.seeds.strip() else [int(x) for x in fusion_config.get("seeds", [42])]
    tasks = parse_csv_list(args.tasks) if args.tasks.strip() else list(fusion_config.get("tasks", []))
    base_models = fusion_config["base_models"]
    embedding_variants = fusion_config.get("embedding_variants", DEFAULT_EMBEDDING_VARIANTS)
    class_weight = None if args.class_weight == "none" else "balanced"
    meta_model_configs = select_meta_model_configs(args.meta_models)

    tab_ckpt_dir = args.tabular_checkpoint_dir if str(args.tabular_checkpoint_dir) else args.checkpoint_dir
    seq_ckpt_dir = args.sequence_checkpoint_dir if str(args.sequence_checkpoint_dir) else args.checkpoint_dir
    num_ckpt_dir = args.numeric_checkpoint_dir if str(args.numeric_checkpoint_dir) else args.checkpoint_dir

    print("DEVICE:", device)
    print("RUN_SET:", run_set)
    print("TASKS:", tasks)
    print("SEEDS:", seeds)
    print("META MODELS:", [cfg["name"] for cfg in meta_model_configs])
    print("EMBEDDING VARIANTS:", embedding_variants)

    manifest_df = pd.DataFrame()

    if args.reuse_features:
        if not str(args.features_path):
            raise ValueError("--reuse-features requires --features-path")
        print(f"Loading reused feature_df: {args.features_path}")
        feature_df = pd.read_parquet(args.features_path)

        if str(args.ensemble_features_path):
            print(f"Loading reused ensemble_df: {args.ensemble_features_path}")
            ensemble_df = pd.read_parquet(args.ensemble_features_path)
        else:
            ensemble_df = make_ensemble_feature_frame(feature_df)
    else:
        all_parts = []
        manifest_parts = []

        for task in tasks:
            if task not in base_models:
                raise ValueError(f"Task {task} not found in base_models config")

            task_cfg = base_models[task]
            tab_cfg = task_cfg["tabular"]
            seq_cfg = task_cfg["sequence"]
            num_cfg = task_cfg["numeric_sequence"]

            for seed in seeds:
                print("=" * 100)
                print(f"Building embedding features: task={task}, seed={seed}")

                tab_df, tab_manifest = make_tabular_feature_frame_from_checkpoint(
                    helper=helper,
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
                )

                seq_emb, seq_manifest = make_sequence_embeddings_from_checkpoint(
                    helper=helper,
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
                    numeric_on=helper.safe_bool(seq_cfg["numeric_on"]),
                    seed=seed,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    embedding_prefix="seq_emb",
                )

                num_emb, num_manifest = make_sequence_embeddings_from_checkpoint(
                    helper=helper,
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
                    numeric_on=helper.safe_bool(num_cfg["numeric_on"]),
                    seed=seed,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    embedding_prefix="numseq_emb",
                )

                merged = merge_feature_frames(tabular=tab_df, sequence_emb=seq_emb, numeric_sequence_emb=num_emb)
                all_parts.append(merged)
                manifest_parts.extend([tab_manifest, seq_manifest, num_manifest])

        feature_df = pd.concat(all_parts, ignore_index=True)
        manifest_df = pd.concat(manifest_parts, ignore_index=True)
        ensemble_df = make_ensemble_feature_frame(feature_df)

    feature_df = feature_df[feature_df["task"].isin(tasks)].copy()
    ensemble_df = ensemble_df[ensemble_df["task"].isin(tasks)].copy()

    if args.save_feature_parquet or not args.reuse_features:
        feature_df.to_parquet(args.output_dir / "embedding_features_tuning_heldout.parquet", index=False)
        ensemble_df.to_parquet(args.output_dir / "embedding_ensemble_features.parquet", index=False)

    if not manifest_df.empty:
        manifest_df.to_csv(args.output_dir / "embedding_feature_manifest.csv", index=False)

    print("Feature frame shape:", feature_df.shape)
    print(
        feature_df.groupby(["task", "split", "seed"])
        .agg(n=("example_id", "size"), n_positive=("y_true", "sum"), event_rate=("y_true", "mean"))
        .reset_index()
        .to_string(index=False)
    )

    single_metric_parts = []
    single_pred_parts = []

    for task in tasks:
        for seed in seeds:
            m, p = run_single_seed_embedding_fusion(
                feature_df=feature_df,
                task=task,
                seed=seed,
                embedding_variants=embedding_variants,
                meta_model_configs=meta_model_configs,
                class_weight=class_weight,
            )
            single_metric_parts.append(m)
            single_pred_parts.append(p)

    single_metrics = pd.concat(single_metric_parts, ignore_index=True)
    single_predictions = pd.concat(single_pred_parts, ignore_index=True)
    seed_summary = summarize_metrics_mean_std(single_metrics)

    single_metrics.to_csv(args.output_dir / "fusion_embedding_single_seed_metrics.csv", index=False)
    single_predictions.to_csv(args.output_dir / "fusion_embedding_single_seed_predictions.csv", index=False)
    seed_summary.to_csv(args.output_dir / "fusion_embedding_seed_summary.csv", index=False)

    ens_metric_parts = []
    ens_pred_parts = []

    for task in tasks:
        m, p = run_ensemble_embedding_fusion(
            ensemble_df=ensemble_df,
            task=task,
            embedding_variants=embedding_variants,
            meta_model_configs=meta_model_configs,
            class_weight=class_weight,
        )
        ens_metric_parts.append(m)
        ens_pred_parts.append(p)

    ensemble_metrics = pd.concat(ens_metric_parts, ignore_index=True)
    ensemble_predictions = pd.concat(ens_pred_parts, ignore_index=True)

    ensemble_metrics.to_csv(args.output_dir / "fusion_embedding_ensemble_metrics.csv", index=False)
    ensemble_predictions.to_csv(args.output_dir / "fusion_embedding_ensemble_predictions.csv", index=False)

    print("\nSaved outputs to:", args.output_dir)
    print("\nSingle-seed embedding fusion sorted by AUPRC:")
    show_cols = ["task", "model_name", "representation", "n_seeds", "auprc_mean", "auprc_std", "auroc_mean", "brier_mean", "logloss_mean", "top_10pct_precision_mean"]
    show_cols = [c for c in show_cols if c in seed_summary.columns]
    print(seed_summary[show_cols].sort_values(["task", "auprc_mean"], ascending=[True, False]).to_string(index=False))

    print("\nEnsemble embedding fusion sorted by AUPRC:")
    show_cols = ["task", "model_name", "representation", "auprc", "auroc", "brier", "logloss", "top_10pct_precision", "n", "n_positive"]
    print(ensemble_metrics[show_cols].sort_values(["task", "auprc"], ascending=[True, False]).to_string(index=False))

    helper.safe_upload_artifact(clearml_task, "embedding_feature_manifest", manifest_df)
    helper.safe_upload_artifact(clearml_task, "fusion_embedding_single_seed_metrics", single_metrics)
    helper.safe_upload_artifact(clearml_task, "fusion_embedding_single_seed_predictions", single_predictions)
    helper.safe_upload_artifact(clearml_task, "fusion_embedding_seed_summary", seed_summary)
    helper.safe_upload_artifact(clearml_task, "fusion_embedding_ensemble_metrics", ensemble_metrics)
    helper.safe_upload_artifact(clearml_task, "fusion_embedding_ensemble_predictions", ensemble_predictions)


if __name__ == "__main__":
    main()
