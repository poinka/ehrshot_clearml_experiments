from __future__ import annotations

"""
common_ehrshot_eval.py

Назначение:
    Общие функции для EHRSHOT reproducibility package.

Что здесь хранится:
    1. Метрики бинарной классификации.
    2. Top-k метрики.
    3. Нормализация prediction schema.
    4. Загрузка predictions из файлов/папок.
    5. Mean ± std по seed.
    6. Ensemble по seed.
    7. Patient-level bootstrap.
    8. Paired patient-level bootstrap.
    9. High-repeat subgroup из sequence examples.
    10. Sequence cost summary.
    11. MinIO upload helpers.

    Новая схема:
        task
        model_family
        model_name
        representation
        compression_version
        numeric_on
        calibration
        seed
        split
        example_id
        subject_id
        y_true
        pred_proba

    Старые имена тоже поддерживаются:
        family  -> model_family
        model   -> model_name
        version -> compression_version
        row_id  -> example_id
        risk    -> pred_proba
"""

import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------------

PREDICTION_ID_COLUMNS = [
    "task",
    "model_family",
    "model_name",
    "representation",
    "compression_version",
    "numeric_on",
    "calibration",
    "seed",
    "split",
]

PREDICTION_EXAMPLE_COLUMNS = [
    "example_id",
    "subject_id",
    "y_true",
    "pred_proba",
]

PREDICTION_REQUIRED_COLUMNS = PREDICTION_ID_COLUMNS + PREDICTION_EXAMPLE_COLUMNS

DEFAULT_METRICS = [
    "auroc",
    "auprc",
    "brier",
    "logloss",
    "top_5pct_precision",
    "top_10pct_precision",
    "top_20pct_precision",
]

DEFAULT_TOP_FRACS = [0.05, 0.10, 0.20]

LOWER_IS_BETTER = {"brier", "logloss"}


# -----------------------------------------------------------------------------
# 2. Small utils
# -----------------------------------------------------------------------------

def parse_int_list(x: str | Sequence[int]) -> list[int]:
    """
    Превращает строку вида "42,43,44" в список int.
    """
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]

    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def parse_csv_list(x: str | Sequence[str]) -> list[str]:
    """
    Превращает строку вида "a,b,c" в список строк.
    """
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for item in x:
            out.extend(parse_csv_list(item))
        return out

    return [v.strip() for v in str(x).split(",") if v.strip()]


def set_global_seed(seed: int) -> None:
    """
    Фиксирует random seed для python/numpy/torch.

    Полная детерминированность на GPU не гарантируется,
    но для seed-comparison это лучше, чем ничего.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    except Exception:
        pass


def read_json(path: Path) -> dict[str, Any]:
    """
    Читает JSON.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict[str, Any], path: Path) -> None:
    """
    Сохраняет JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def to_jsonable(x: Any) -> Any:
    """
    Приводит numpy/pandas types к JSON-friendly types.
    """
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, list):
        return [to_jsonable(v) for v in x]

    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        if np.isnan(x):
            return None
        return float(x)

    if isinstance(x, np.bool_):
        return bool(x)

    if pd.isna(x) and not isinstance(x, (list, dict, tuple)):
        return None

    return x


def safe_bool(x: Any) -> bool:
    """
    Безопасно приводит значение к bool.

    Нужно, потому что ClearML/CSV иногда возвращают bool как строки.
    """
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}

    return bool(x)


def safe_path_part(value: object) -> str:
    """
    Делает строку безопасной для части пути.
    """
    return str(value).replace("/", "__").replace(" ", "_")


def ensure_parent(path: Path) -> Path:
    """
    Создает родительскую папку для файла.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# 3. Metrics
# -----------------------------------------------------------------------------

def safe_auroc(y_true: Any, y_prob: Any) -> float:
    """
    AUROC с защитой от one-class subgroup.
    """
    y_true = np.asarray(y_true).astype(int)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: Any, y_prob: Any) -> float:
    """
    AUPRC с защитой от one-class subgroup.
    """
    y_true = np.asarray(y_true).astype(int)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_prob))


def binary_ranking_metrics(y_true: Any, y_prob: Any) -> dict[str, float]:
    """
    Основные метрики для binary risk prediction.

    AUPRC:
        Основная метрика для несбалансированных задач.

    AUROC:
        Ranking quality, но может выглядеть слишком оптимистично при редких events.

    Brier / LogLoss:
        Качество вероятностей.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_prob_clip = np.clip(y_prob, 1e-6, 1 - 1e-6)

    if len(y_true) == 0:
        return {
            "n": 0,
            "n_positive": 0,
            "event_rate": np.nan,
            "auroc": np.nan,
            "auprc": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
        }

    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob_clip)),
        "logloss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
    }


def topk_metrics(
    y_true: Any,
    y_prob: Any,
    top_fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """
    Считает top-k event rate / lift / capture.

    top_k_event_rate:
        Precision внутри top-k риска.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if n else np.nan

    order = np.argsort(-y_prob)

    rows = []

    for frac in top_fracs:
        k = max(1, int(np.ceil(n * float(frac)))) if n else 0

        if k == 0:
            rows.append(
                {
                    "top_frac": float(frac),
                    "top_k": 0,
                    "events_in_top_k": 0,
                    "top_k_event_rate": np.nan,
                    "top_k_lift": np.nan,
                    "event_capture": np.nan,
                    "base_event_rate": base_rate,
                    "n": int(n),
                    "n_positive": int(n_pos),
                }
            )
            continue

        idx = order[:k]
        events = int(y_true[idx].sum())
        event_rate = float(events / k)

        rows.append(
            {
                "top_frac": float(frac),
                "top_k": int(k),
                "events_in_top_k": events,
                "top_k_event_rate": event_rate,
                "top_k_lift": (
                    float(event_rate / base_rate)
                    if base_rate and base_rate > 0
                    else np.nan
                ),
                "event_capture": (
                    float(events / n_pos)
                    if n_pos > 0
                    else np.nan
                ),
                "base_event_rate": base_rate,
                "n": int(n),
                "n_positive": int(n_pos),
            }
        )

    return pd.DataFrame(rows)


def topk_metrics_wide(
    y_true: Any,
    y_prob: Any,
    top_fracs: Sequence[float] = DEFAULT_TOP_FRACS,
) -> dict[str, float]:
    """
    Возвращает top-k метрики в wide-формате.

    Пример колонок:
        top_5pct_precision
        top_5pct_lift
        top_5pct_event_capture
    """
    topk = topk_metrics(y_true, y_prob, top_fracs=top_fracs)

    out: dict[str, float] = {}

    for _, row in topk.iterrows():
        pct = int(round(float(row["top_frac"]) * 100))

        out[f"top_{pct}pct_precision"] = float(row["top_k_event_rate"])
        out[f"top_{pct}pct_lift"] = float(row["top_k_lift"])
        out[f"top_{pct}pct_event_capture"] = float(row["event_capture"])

    return out


def safe_metric(
    y_true: Any,
    y_prob: Any,
    metric_name: str,
) -> float:
    """
    Единая точка расчета одной метрики.

    Поддерживает:
        auroc
        auprc
        brier
        logloss
        top_5pct_precision
        top_10pct_precision
        top_20pct_precision
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_clip = np.clip(y_prob, 1e-6, 1 - 1e-6)

    if len(y_true) == 0:
        return np.nan

    if metric_name == "auroc":
        return safe_auroc(y_true, y_prob)

    if metric_name == "auprc":
        return safe_auprc(y_true, y_prob)

    if metric_name == "brier":
        return float(brier_score_loss(y_true, y_prob_clip))

    if metric_name == "logloss":
        return float(log_loss(y_true, y_prob_clip, labels=[0, 1]))

    match = re.match(r"^top_(\d+)pct_precision$", metric_name)
    if match:
        pct = int(match.group(1))
        k = max(1, int(np.ceil(len(y_true) * pct / 100)))
        order = np.argsort(-y_prob)
        return float(y_true[order[:k]].mean())

    match = re.match(r"^top_(\d+)pct_lift$", metric_name)
    if match:
        pct = int(match.group(1))
        k = max(1, int(np.ceil(len(y_true) * pct / 100)))
        order = np.argsort(-y_prob)
        precision = float(y_true[order[:k]].mean())
        base = float(y_true.mean())
        return float(precision / base) if base > 0 else np.nan

    match = re.match(r"^top_(\d+)pct_event_capture$", metric_name)
    if match:
        pct = int(match.group(1))
        k = max(1, int(np.ceil(len(y_true) * pct / 100)))
        order = np.argsort(-y_prob)
        events = int(y_true[order[:k]].sum())
        total_events = int(y_true.sum())
        return float(events / total_events) if total_events > 0 else np.nan

    raise ValueError(f"Unknown metric: {metric_name}")


def higher_is_better(metric_name: str) -> bool:
    """
    Для brier/logloss меньше лучше, для остальных больше лучше.
    """
    return metric_name not in LOWER_IS_BETTER


def summarize_values(values: Any) -> dict[str, float]:
    """
    Mean/std/95% CI по bootstrap values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid": 0,
        }

    return {
        "mean": float(np.mean(values)),
        "std": (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        ),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_valid": int(len(values)),
    }


# -----------------------------------------------------------------------------
# 4. Prediction schema normalization
# -----------------------------------------------------------------------------

def infer_representation(row: pd.Series) -> str:
    """
    Восстанавливает representation, если его нет в старых prediction files.
    """
    model_family = str(row.get("model_family", ""))

    if model_family == "tabular":
        return "tabular_all_features"

    if model_family == "numeric_sequence":
        return "numeric_sequence_time2vec"

    if model_family == "sequence":
        return "code_sequence_time2vec"

    return "unknown"


def normalize_prediction_schema(
    df: pd.DataFrame,
    default_model_family: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """
    Приводит старые и новые predictions к единой схеме.

    Старые колонки:
        family  -> model_family
        model   -> model_name
        version -> compression_version
        row_id  -> example_id
        risk    -> pred_proba

    Новые колонки:
        уже оставляются как есть.
    """
    out = df.copy()

    rename_map = {}

    if "model_family" not in out.columns and "family" in out.columns:
        rename_map["family"] = "model_family"

    if "model_name" not in out.columns and "model" in out.columns:
        rename_map["model"] = "model_name"

    if "compression_version" not in out.columns and "version" in out.columns:
        rename_map["version"] = "compression_version"

    if "example_id" not in out.columns and "row_id" in out.columns:
        rename_map["row_id"] = "example_id"

    if "pred_proba" not in out.columns and "risk" in out.columns:
        rename_map["risk"] = "pred_proba"

    out = out.rename(columns=rename_map)

    if "model_family" not in out.columns:
        if default_model_family is None:
            raise ValueError(
                "Prediction file has no model_family/family column "
                "and default_model_family was not provided."
            )
        out["model_family"] = default_model_family

    if "compression_version" not in out.columns:
        if (out["model_family"].astype(str) == "tabular").all():
            out["compression_version"] = "none"
        else:
            out["compression_version"] = "unknown"

    # Старый tabular мог иметь version=tabular_all_features.
    tabular_mask = out["model_family"].astype(str) == "tabular"
    if "compression_version" in out.columns:
        out.loc[
            tabular_mask & (out["compression_version"].astype(str) == "tabular_all_features"),
            "compression_version",
        ] = "none"

    if "representation" not in out.columns:
        out["representation"] = out.apply(infer_representation, axis=1)

    if "numeric_on" not in out.columns:
        out["numeric_on"] = out["model_family"].astype(str).isin(
            ["tabular", "numeric_sequence"]
        )

    if "calibration" not in out.columns:
        out["calibration"] = "raw"

    if "split" not in out.columns:
        out["split"] = "held_out"

    if "seed" not in out.columns:
        out["seed"] = -1

    if source_name is not None:
        out["prediction_source"] = source_name

    missing = set(PREDICTION_REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns after normalization: {missing}")

    out["task"] = out["task"].astype(str)
    out["model_family"] = out["model_family"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["representation"] = out["representation"].astype(str)
    out["compression_version"] = out["compression_version"].astype(str)
    out["calibration"] = out["calibration"].astype(str)
    out["split"] = out["split"].astype(str)

    out["numeric_on"] = out["numeric_on"].apply(safe_bool)
    out["seed"] = out["seed"].astype(int)
    out["example_id"] = out["example_id"].astype(int)
    out["subject_id"] = out["subject_id"].astype(int)
    out["y_true"] = out["y_true"].astype(int)
    out["pred_proba"] = out["pred_proba"].astype(float)

    first = PREDICTION_REQUIRED_COLUMNS
    rest = [c for c in out.columns if c not in first]

    return out[first + rest]


def load_predictions_from_path_or_dir(
    path: Path,
    default_model_family: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """
    Загружает predictions из файла или папки и нормализует схему.

    Если path — папка, ищет:
        *heldout_predictions.csv
        *__heldout_predictions.csv
        *multiseed*predictions.csv
    """
    path = Path(path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        patterns = [
            "*heldout_predictions.csv",
            "*__heldout_predictions.csv",
            "*multiseed*predictions.csv",
        ]

        files = []
        for pattern in patterns:
            files.extend(sorted(path.glob(pattern)))

        files = list(dict.fromkeys(files))

        if not files:
            raise FileNotFoundError(f"No prediction CSV files found in directory: {path}")
    else:
        raise FileNotFoundError(f"Prediction path not found: {path}")

    parts = []

    for file_path in files:
        df = pd.read_csv(file_path)
        df["source_file"] = str(file_path)

        normalized = normalize_prediction_schema(
            df,
            default_model_family=default_model_family,
            source_name=source_name,
        )

        parts.append(normalized)

    out = pd.concat(parts, ignore_index=True)

    dedup_key = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
        "seed",
        "split",
        "example_id",
        "subject_id",
    ]

    before = len(out)
    out = out.drop_duplicates(subset=dedup_key, keep="last").reset_index(drop=True)
    after = len(out)

    if before != after:
        print(f"Dropped duplicated prediction rows: {before - after}")

    return out


def filter_predictions_by_model_configs(
    pred: pd.DataFrame,
    model_configs: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """
    Оставляет только модели из config["primary_models"].
    """
    if not model_configs:
        return pred.copy()

    masks = []

    for cfg in model_configs:
        mask = pd.Series(True, index=pred.index)

        for col in [
            "task",
            "model_family",
            "model_name",
            "representation",
            "compression_version",
            "numeric_on",
        ]:
            if col in cfg and cfg[col] is not None:
                mask &= pred[col] == cfg[col]

        masks.append(mask)

    if not masks:
        return pred.iloc[0:0].copy()

    return pred[np.logical_or.reduce(masks)].copy()


def get_model_config_by_key(
    model_configs: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """
    Находит model config по key.
    """
    matches = [cfg for cfg in model_configs if cfg.get("key") == key]

    if not matches:
        raise ValueError(f"Unknown model config key: {key}")

    if len(matches) > 1:
        raise ValueError(f"Duplicate model config key: {key}")

    return matches[0]


def select_model_predictions(
    pred: pd.DataFrame,
    model_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Фильтрует predictions по model config.
    """
    out = pred.copy()

    for col in [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
    ]:
        if col in model_cfg and model_cfg[col] is not None:
            out = out[out[col] == model_cfg[col]]

    if out.empty:
        raise ValueError(f"No predictions for model config: {model_cfg}")

    return out.copy()


# -----------------------------------------------------------------------------
# 5. Metrics tables
# -----------------------------------------------------------------------------

def compute_metrics_for_group(
    pred: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """
    Считает metrics для произвольной группировки.
    """
    rows = []

    for key, part in pred.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_cols, key))

        y = part["y_true"].to_numpy().astype(int)
        p = part["pred_proba"].to_numpy().astype(float)

        base = binary_ranking_metrics(y, p)
        row.update(base)

        for metric in metrics:
            if metric in base:
                continue

            row[metric] = safe_metric(y, p, metric)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics_by_seed(
    pred: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Считает метрики по каждому seed.
    """
    group_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
        "seed",
    ]

    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    return compute_metrics_for_group(pred, group_cols=group_cols, metrics=metrics)


def summarize_metrics_mean_std(
    metrics_by_seed: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Делает mean/std/min/max по seed.
    """
    id_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
    ]

    if extra_group_cols:
        id_cols.extend(extra_group_cols)

    metric_cols = [
        col
        for col in ["n", "n_positive", "event_rate", *metrics]
        if col in metrics_by_seed.columns
    ]

    rows = []

    for key, group in metrics_by_seed.groupby(id_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(id_cols, key))
        row["n_seeds"] = int(group["seed"].nunique())

        for col in metric_cols:
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = (
                float(values.std(ddof=1))
                if len(values.dropna()) > 1
                else np.nan
            )
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())

        rows.append(row)

    return pd.DataFrame(rows)


def compute_topk_by_seed(
    pred: pd.DataFrame,
    top_fracs: Sequence[float] = DEFAULT_TOP_FRACS,
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Сохраняет top-k в long format по seed.
    """
    group_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
        "seed",
    ]

    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    rows = []

    for key, part in pred.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        meta = dict(zip(group_cols, key))
        topk = topk_metrics(
            part["y_true"].to_numpy(),
            part["pred_proba"].to_numpy(),
            top_fracs=top_fracs,
        )

        for _, r in topk.iterrows():
            rows.append({**meta, **r.to_dict()})

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. Ensemble and prediction stability
# -----------------------------------------------------------------------------

def ensemble_predictions(
    pred: pd.DataFrame,
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Делает seed ensemble: среднее pred_proba по seed для каждого example.
    """
    group_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
        "split",
        "example_id",
        "subject_id",
        "y_true",
    ]

    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    out = (
        pred
        .groupby(group_cols, dropna=False)
        .agg(
            pred_proba=("pred_proba", "mean"),
            pred_proba_std=("pred_proba", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    out["pred_proba_std"] = out["pred_proba_std"].fillna(0.0)

    return out


def prediction_stability_summary(
    pred: pd.DataFrame,
    extra_group_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Считает std предсказаний по seed на уровне example и summary по модели.
    """
    by_example = ensemble_predictions(
        pred,
        extra_group_cols=extra_group_cols,
    )

    group_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
    ]

    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    summary = (
        by_example
        .groupby(group_cols, dropna=False)
        .agg(
            n_examples=("example_id", "size"),
            n_patients=("subject_id", "nunique"),
            n_positive=("y_true", "sum"),
            mean_prediction_std=("pred_proba_std", "mean"),
            median_prediction_std=("pred_proba_std", "median"),
            p90_prediction_std=("pred_proba_std", lambda x: float(np.quantile(x, 0.90))),
            p95_prediction_std=("pred_proba_std", lambda x: float(np.quantile(x, 0.95))),
            max_prediction_std=("pred_proba_std", "max"),
        )
        .reset_index()
    )

    return by_example, summary


# -----------------------------------------------------------------------------
# 7. Patient bootstrap
# -----------------------------------------------------------------------------

def _cluster_index_groups(
    df: pd.DataFrame,
    cluster_col: str,
) -> list[np.ndarray]:
    """
    Индексы строк по patient/cluster.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col={cluster_col} not found in DataFrame")

    return [
        np.asarray(idx, dtype=np.int64)
        for idx in df.groupby(cluster_col, sort=False).indices.values()
    ]


def patient_bootstrap_metric_ci(
    part: pd.DataFrame,
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cluster_col: str = "subject_id",
) -> dict[str, float]:
    """
    Patient-level bootstrap CI для одной модели и одной метрики.
    """
    if part.empty:
        return {
            "observed": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap_valid": 0,
            "n_bootstrap_requested": int(n_bootstrap),
            "bootstrap_unit": cluster_col,
            "n_examples": 0,
            "n_patients": 0,
            "n_positive": 0,
        }

    y = part["y_true"].to_numpy().astype(int)
    p = part["pred_proba"].to_numpy().astype(float)

    observed = safe_metric(y, p, metric_name)

    groups = _cluster_index_groups(part, cluster_col=cluster_col)
    n_clusters = len(groups)

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[i] for i in sampled])

        values.append(
            safe_metric(
                y[idx],
                p[idx],
                metric_name,
            )
        )

    s = summarize_values(values)

    return {
        "observed": observed,
        "bootstrap_mean": s["mean"],
        "bootstrap_std": s["std"],
        "ci_low": s["ci_low"],
        "ci_high": s["ci_high"],
        "n_bootstrap_valid": s["n_valid"],
        "n_bootstrap_requested": int(n_bootstrap),
        "bootstrap_unit": cluster_col,
        "n_examples": int(len(part)),
        "n_patients": int(part[cluster_col].nunique()),
        "n_positive": int(part["y_true"].sum()),
    }


def make_patient_bootstrap_metric_ci_table(
    pred: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cluster_col: str = "subject_id",
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Patient-level bootstrap CI table для всех моделей.
    """
    group_cols = [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
        "calibration",
    ]

    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    rows = []

    for key, part in pred.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        meta = dict(zip(group_cols, key))

        for metric in metrics:
            ci = patient_bootstrap_metric_ci(
                part=part,
                metric_name=metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
                cluster_col=cluster_col,
            )

            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "higher_is_better": higher_is_better(metric),
                    **ci,
                }
            )

    return pd.DataFrame(rows)


def paired_patient_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cluster_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Paired patient-level bootstrap для сравнения двух моделей.

    Delta считается как:
        metric(model_a) - metric(model_b)
    """
    keep = [
        "task",
        "example_id",
        "subject_id",
        "y_true",
        "pred_proba",
    ]

    a = df_a[keep].rename(columns={"pred_proba": "pred_proba_a"})
    b = df_b[keep].rename(columns={"pred_proba": "pred_proba_b"})

    merged = a.merge(
        b,
        on=["task", "example_id", "subject_id", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("Paired merge produced 0 rows.")

    y = merged["y_true"].to_numpy().astype(int)
    pa = merged["pred_proba_a"].to_numpy().astype(float)
    pb = merged["pred_proba_b"].to_numpy().astype(float)

    groups = _cluster_index_groups(merged, cluster_col=cluster_col)
    n_clusters = len(groups)

    rng = np.random.default_rng(seed)

    rows = []

    for metric in metrics:
        point_a = safe_metric(y, pa, metric)
        point_b = safe_metric(y, pb, metric)
        point_delta = point_a - point_b

        values = []

        for _ in range(n_bootstrap):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([groups[i] for i in sampled])

            values.append(
                safe_metric(y[idx], pa[idx], metric)
                - safe_metric(y[idx], pb[idx], metric)
            )

        s = summarize_values(values)

        rows.append(
            {
                "metric": metric,
                "higher_is_better": higher_is_better(metric),
                "model_a_value": point_a,
                "model_b_value": point_b,
                "point_delta_a_minus_b": point_delta,
                "bootstrap_mean_delta": s["mean"],
                "bootstrap_std_delta": s["std"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "n_bootstrap_valid": s["n_valid"],
                "n_bootstrap_requested": int(n_bootstrap),
                "bootstrap_unit": cluster_col,
                "n_paired_examples": int(len(merged)),
                "n_paired_patients": int(merged[cluster_col].nunique()),
                "n_paired_positive": int(merged["y_true"].sum()),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8. High-repeat and sequence cost
# -----------------------------------------------------------------------------

def load_sequence_examples(
    sequence_dataset_dir: Path,
    task: str,
    compression_version: str,
) -> pd.DataFrame | None:
    """
    Загружает examples.parquet для task/version.
    """
    path = (
        Path(sequence_dataset_dir)
        / task
        / compression_version
        / "examples.parquet"
    )

    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["task"] = task
    df["compression_version"] = compression_version

    if "example_id" not in df.columns and "row_id" in df.columns:
        df = df.rename(columns={"row_id": "example_id"})

    if "y_true" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "y_true"})

    return df


def build_high_repeat_group_from_sequence_examples(
    sequence_dataset_dir: Path,
    tasks: Sequence[str],
    baseline_version: str = "raw",
    repeat_reference_version: str = "compressed_first_last",
    q: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит high-repeat subgroup из sequence examples.

    Определение:
        High-repeat subgroup = top q held-out строк по repeat_removed_first_last.

    repeat_removed_first_last:
        Количество повторяющихся событий диагноза, которые потенциально удаляются при first-last сжатии.

    Важно:
        Подгруппа определяется только по истории пациента до prediction_time.
        Метки НЕ используются для присвоения high_repeat_group.
    
    Выход high_repeat:
        одна строка = один held-out prediction example.
    
    Это технически на уровне примера, а не чисто на уровне пациента:
        один пациент может иметь несколько prediction examples.
    """
    rows = []
    threshold_rows = []

    for task in tasks:
        raw = load_sequence_examples(
            sequence_dataset_dir,
            task=task,
            compression_version=baseline_version,
        )

        first_last = load_sequence_examples(
            sequence_dataset_dir,
            task=task,
            compression_version=repeat_reference_version,
        )

        if raw is None:
            print(f"WARNING: raw sequence examples not found for task={task}")
            continue

        if first_last is None:
            print(
                f"WARNING: {repeat_reference_version} examples not found for task={task}; "
                "high-repeat group will be empty."
            )
            continue

        raw = raw[raw["split"] == "held_out"].copy()
        first_last = first_last[first_last["split"] == "held_out"].copy()

        raw_cols = [
            "task",
            "example_id",
            "subject_id",
            "split",
            "y_true",
            "seq_len",
        ]

        if "prediction_time" in raw.columns:
            raw_cols.append("prediction_time")

        raw_small = raw[raw_cols].copy()
        raw_small = raw_small.rename(columns={"seq_len": "raw_seq_len"})

        first_last_cols = [
            "task",
            "example_id",
            "subject_id",
            "split",
            "seq_len",
        ]

        if "n_events_removed_vs_raw" in first_last.columns:
            first_last_cols.append("n_events_removed_vs_raw")

        if "raw_seq_len" in first_last.columns:
            first_last_cols.append("raw_seq_len")

        first_last_small = first_last[first_last_cols].copy()
        first_last_small = first_last_small.rename(
            columns={
                "seq_len": "first_last_seq_len",
                "n_events_removed_vs_raw": "repeat_removed_first_last",
                "raw_seq_len": "raw_seq_len_from_first_last",
            }
        )

        tmp = raw_small.merge(
            first_last_small,
            on=["task", "example_id", "subject_id", "split"],
            how="inner",
            validate="one_to_one",
        )

        if tmp.empty:
            raise ValueError(
                f"High-repeat merge produced 0 rows for task={task}. "
                "Check example_id/subject_id alignment between raw and compressed_first_last."
            )

        # Main preferred source:
        #   n_events_removed_vs_raw from compressed_first_last examples.
        #
        # Fallback:
        #   raw_seq_len - first_last_seq_len.
        #
        # This score is computed from sequence history only.
        # y_true/label is not used to define the group.
        if "repeat_removed_first_last" not in tmp.columns:
            tmp["repeat_removed_first_last"] = (
                tmp["raw_seq_len"].astype(float)
                - tmp["first_last_seq_len"].astype(float)
            )
        else:
            tmp["repeat_removed_first_last"] = (
                tmp["repeat_removed_first_last"]
                .fillna(
                    tmp["raw_seq_len"].astype(float)
                    - tmp["first_last_seq_len"].astype(float)
                )
                .astype(float)
            )

        threshold = float(tmp["repeat_removed_first_last"].quantile(q))

        tmp["high_repeat_group"] = (
            tmp["repeat_removed_first_last"] >= threshold
        )

        tmp["group_name"] = np.where(
            tmp["high_repeat_group"],
            "high_repeat_top10",
            "other_90",
        )

        tmp["repeat_removed_threshold"] = threshold

        high_part = tmp[tmp["high_repeat_group"]].copy()
        other_part = tmp[~tmp["high_repeat_group"]].copy()

        threshold_rows.append(
            {
                "task": task,
                "baseline_version": baseline_version,
                "repeat_reference_version": repeat_reference_version,
                "q": float(q),
                "definition": (
                    "top 10% held-out prediction examples by "
                    "repeat_removed_first_last"
                ),
                "repeat_removed_threshold": threshold,
                "n_heldout_examples": int(len(tmp)),
                "n_heldout_patients": int(tmp["subject_id"].nunique()),
                "n_high_repeat_examples": int(len(high_part)),
                "n_high_repeat_patients": int(high_part["subject_id"].nunique()),
                "n_other_examples": int(len(other_part)),
                "n_other_patients": int(other_part["subject_id"].nunique()),

                # Эти поля можно использовать для описания subgroup.
                # Но они НЕ используются при выделении группы.
                "n_high_repeat_positive": int(high_part["y_true"].sum()),
                "high_repeat_event_rate": (
                    float(high_part["y_true"].mean())
                    if len(high_part)
                    else np.nan
                ),
                "other_event_rate": (
                    float(other_part["y_true"].mean())
                    if len(other_part)
                    else np.nan
                ),
                "median_repeat_removed_high_repeat": (
                    float(high_part["repeat_removed_first_last"].median())
                    if len(high_part)
                    else np.nan
                ),
                "median_repeat_removed_other": (
                    float(other_part["repeat_removed_first_last"].median())
                    if len(other_part)
                    else np.nan
                ),
            }
        )

        keep_cols = [
            "task",
            "example_id",
            "subject_id",
            "split",
            "y_true",
            "raw_seq_len",
            "first_last_seq_len",
            "repeat_removed_first_last",
            "repeat_removed_threshold",
            "high_repeat_group",
            "group_name",
        ]

        if "prediction_time" in tmp.columns:
            keep_cols.insert(4, "prediction_time")

        rows.append(tmp[keep_cols].copy())

    high_repeat = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame()
    )

    thresholds = pd.DataFrame(threshold_rows)

    return high_repeat, thresholds


def build_high_repeat_patient_list(
    high_repeat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Строит patient-level summary из example-level high-repeat subgroup.

    Важно:
        high_repeat_group присваивается на уровне prediction-example.
        Эта таблица отвечает на вопрос:
            какие пациенты имеют хотя бы один held-out example в high-repeat группе?
    """
    if high_repeat.empty:
        return pd.DataFrame()

    high = high_repeat[high_repeat["high_repeat_group"]].copy()

    if high.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "subject_id",
                "n_high_repeat_examples",
                "max_repeat_removed_first_last",
                "median_repeat_removed_first_last",
            ]
        )

    patient_list = (
        high
        .groupby(["task", "subject_id"])
        .agg(
            n_high_repeat_examples=("example_id", "nunique"),
            max_repeat_removed_first_last=("repeat_removed_first_last", "max"),
            median_repeat_removed_first_last=("repeat_removed_first_last", "median"),
        )
        .reset_index()
    )

    return patient_list


def add_high_repeat_group(
    pred: pd.DataFrame,
    high_repeat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет high_repeat_group к predictions.
    """
    if high_repeat.empty:
        out = pred.copy()
        out["high_repeat_group"] = False
        out["group_name"] = "all"
        return out

    cols = [
        "task",
        "example_id",
        "subject_id",
        "high_repeat_group",
        "group_name",
        "repeat_removed_first_last",
        "repeat_removed_threshold",
    ]

    out = pred.merge(
        high_repeat[cols],
        on=["task", "example_id", "subject_id"],
        how="inner",
    )

    if out.empty:
        raise ValueError("Prediction/high-repeat merge produced 0 rows.")

    return out


def build_sequence_cost_summary(
    sequence_dataset_dir: Path,
    tasks: Sequence[str],
    high_repeat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Собирает cost summary по sequence versions.

    Использует audit columns в examples.parquet:
        seq_len
        raw_seq_len
        n_events_removed_vs_raw
        n_synthetic_events
        n_compressible_chronic_events_raw
    """
    sequence_dataset_dir = Path(sequence_dataset_dir)

    rows = []

    if not sequence_dataset_dir.exists():
        return pd.DataFrame()

    for task in tasks:
        task_dir = sequence_dataset_dir / task

        if not task_dir.exists():
            continue

        for version_dir in sorted(task_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            compression_version = version_dir.name

            df = load_sequence_examples(
                sequence_dataset_dir,
                task=task,
                compression_version=compression_version,
            )

            if df is None or "seq_len" not in df.columns:
                continue

            df = df.copy()

            if "raw_seq_len" not in df.columns:
                if compression_version == "raw":
                    df["raw_seq_len"] = df["seq_len"]
                else:
                    raw = load_sequence_examples(
                        sequence_dataset_dir,
                        task=task,
                        compression_version="raw",
                    )

                    if raw is not None and "seq_len" in raw.columns:
                        raw_small = raw[
                            ["example_id", "subject_id", "split", "seq_len"]
                        ].rename(columns={"seq_len": "raw_seq_len"})

                        df = df.merge(
                            raw_small,
                            on=["example_id", "subject_id", "split"],
                            how="left",
                            validate="many_to_one",
                        )
                    else:
                        df["raw_seq_len"] = np.nan

            if "n_events_removed_vs_raw" not in df.columns:
                df["n_events_removed_vs_raw"] = (
                    df["raw_seq_len"].astype(float)
                    - df["seq_len"].astype(float)
                )

            if "n_synthetic_events" not in df.columns:
                if "n_compression_events" in df.columns:
                    df["n_synthetic_events"] = df["n_compression_events"]
                else:
                    df["n_synthetic_events"] = 0

            if "n_compressible_chronic_events_raw" not in df.columns:
                df["n_compressible_chronic_events_raw"] = np.nan

            parts = [
                ("all", df),
                ("held_out", df[df["split"] == "held_out"].copy()),
            ]

            if high_repeat is not None and len(high_repeat):
                h = high_repeat[
                    (high_repeat["task"] == task)
                    & (high_repeat["high_repeat_group"])
                ]

                if len(h):
                    high = df.merge(
                        h[["task", "example_id", "subject_id"]],
                        on=["task", "example_id", "subject_id"],
                        how="inner",
                    )
                    parts.append(("held_out_high_repeat", high))

            for split_name, part in parts:
                if part.empty:
                    continue

                rows.append(
                    {
                        "task": task,
                        "compression_version": compression_version,
                        "split": split_name,
                        "n_examples": int(len(part)),
                        "n_patients": int(part["subject_id"].nunique()),
                        "mean_seq_len": float(part["seq_len"].mean()),
                        "median_seq_len": float(part["seq_len"].median()),
                        "p90_seq_len": float(np.quantile(part["seq_len"], 0.90)),
                        "max_seq_len": int(part["seq_len"].max()),
                        "mean_events_removed_vs_raw": float(
                            part["n_events_removed_vs_raw"].mean()
                        ),
                        "median_events_removed_vs_raw": float(
                            part["n_events_removed_vs_raw"].median()
                        ),
                        "p90_events_removed_vs_raw": float(
                            np.quantile(part["n_events_removed_vs_raw"], 0.90)
                        ),
                        "mean_synthetic_events": float(
                            part["n_synthetic_events"].mean()
                        ),
                        "mean_compressible_chronic_events_raw": float(
                            part["n_compressible_chronic_events_raw"].mean()
                        ),
                    }
                )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 9. MinIO helpers
# -----------------------------------------------------------------------------

def upload_file_to_minio(local_path: Path, remote_url: str) -> str:
    """
    Загружает один файл в MinIO/S3 через ClearML StorageManager.
    """
    if not remote_url:
        return ""

    from clearml import StorageManager

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    print(f"Uploading: {local_path} -> {remote_url}")

    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )

    return remote_url


def upload_tree_to_minio(
    local_root: Path,
    s3_prefix: str,
) -> pd.DataFrame:
    """
    Загружает всю папку в MinIO/S3 с сохранением relative layout.
    """
    if not s3_prefix:
        return pd.DataFrame()

    local_root = Path(local_root)

    rows = []

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(local_root).as_posix()
        remote_url = f"{s3_prefix.rstrip('/')}/{rel}"

        upload_file_to_minio(path, remote_url)

        rows.append(
            {
                "local_path": str(path),
                "relative_path": rel,
                "s3_url": remote_url,
            }
        )

    return pd.DataFrame(rows)


def maybe_download_file_from_minio(
    local_path: Path,
    remote_url: str,
) -> Path:
    """
    Скачивает файл из MinIO/S3, если локально его нет.
    """
    local_path = Path(local_path)

    if local_path.exists():
        return local_path

    if not remote_url:
        raise FileNotFoundError(
            f"Local file missing and remote_url is empty: {local_path}"
        )

    from clearml import StorageManager

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {remote_url}")

    local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

    if local_copy.is_file():
        shutil.copy2(local_copy, local_path)
    elif local_copy.is_dir():
        matches = sorted(local_copy.rglob(local_path.name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Could not find exactly one {local_path.name} inside {local_copy}"
            )
        shutil.copy2(matches[0], local_path)
    else:
        raise FileNotFoundError(f"StorageManager returned missing path: {local_copy}")

    return local_path


# -----------------------------------------------------------------------------
# 10. Backward-compatible aliases
# -----------------------------------------------------------------------------

def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Поддерживает старую схему row_id/risk и новую example_id/pred_proba.
    """
    a = normalize_prediction_schema(df_a)
    b = normalize_prediction_schema(df_b)

    return paired_patient_bootstrap_delta(
        df_a=a,
        df_b=b,
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        seed=seed,
        cluster_col="subject_id",
    )