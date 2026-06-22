from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def parse_int_list(x: str | Sequence[int]) -> list[int]:
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode can slow down training, but helps seed comparison.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def safe_auroc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def binary_ranking_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_clip = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "event_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob_clip)) if len(y_true) else np.nan,
        "logloss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])) if len(y_true) else np.nan,
    }


def topk_metrics(y_true, y_prob, top_fracs=(0.01, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if n else np.nan
    order = np.argsort(-y_prob)
    rows = []
    for frac in top_fracs:
        k = max(1, int(np.ceil(n * frac)))
        idx = order[:k]
        events = int(y_true[idx].sum())
        event_rate = events / k if k else np.nan
        rows.append(
            {
                "top_frac": float(frac),
                "top_k": int(k),
                "events_in_top_k": events,
                "top_k_event_rate": float(event_rate),
                "top_k_lift": float(event_rate / base_rate) if base_rate and base_rate > 0 else np.nan,
                "event_capture": float(events / n_pos) if n_pos > 0 else np.nan,
                "base_event_rate": base_rate,
                "n": int(n),
                "n_positive": int(n_pos),
            }
        )
    return pd.DataFrame(rows)


def safe_metric(y_true, y_prob, metric_name: str) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    if len(y_true) == 0:
        return np.nan
    if metric_name == "auroc":
        return safe_auroc(y_true, y_prob)
    if metric_name == "auprc":
        return safe_auprc(y_true, y_prob)
    if metric_name == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if metric_name == "logloss":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    if metric_name.startswith("top_") and metric_name.endswith("_precision"):
        pct = int(metric_name.replace("top_", "").replace("pct_precision", ""))
        k = max(1, int(np.ceil(len(y_true) * pct / 100)))
        order = np.argsort(-y_prob)
        return float(y_true[order[:k]].mean())
    raise ValueError(f"Unknown metric: {metric_name}")


def summarize_values(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_valid": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_valid": int(len(values)),
    }


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    keep = ["task", "row_id", "subject_id", "y_true", "risk"]
    a = df_a[keep].rename(columns={"risk": "risk_a"})
    b = df_b[keep].rename(columns={"risk": "risk_b"})
    merged = a.merge(b, on=["task", "row_id", "subject_id", "y_true"], how="inner")
    if merged.empty:
        raise ValueError("Paired merge produced 0 rows")
    y = merged["y_true"].to_numpy().astype(int)
    pa = merged["risk_a"].to_numpy().astype(float)
    pb = merged["risk_b"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    n = len(merged)
    rows = []
    for metric in metrics:
        point = safe_metric(y, pa, metric) - safe_metric(y, pb, metric)
        vals = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            vals.append(safe_metric(y[idx], pa[idx], metric) - safe_metric(y[idx], pb[idx], metric))
        s = summarize_values(vals)
        rows.append(
            {
                "metric": metric,
                "point_delta_a_minus_b": point,
                "bootstrap_mean_delta": s["mean"],
                "bootstrap_std_delta": s["std"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "n_bootstrap_valid": s["n_valid"],
                "n_paired_examples": int(n),
                "n_paired_positive": int(y.sum()),
            }
        )
    return pd.DataFrame(rows)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
