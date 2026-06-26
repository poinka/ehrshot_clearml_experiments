from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

# code-only sequence helpers
import train_sequence_multiseed as seq_mod

# numeric-aware sequence helpers
import train_numeric_sequence_multiseed as numseq_mod


META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}


TASK_CONFIGS = {
    "guo_readmission": {
        "tabular_model": "HistGradientBoosting",
        "sequence": {
            "version": "compressed_dedup",
            "model": "RETAIN_lite",
        },
        "numeric_sequence": {
            "version": "compressed_dedup",
            "model": "RETAIN_lite_numeric",
        },
    },
    "guo_icu": {
        "tabular_model": "RandomForest_balanced",
        "sequence": {
            "version": "compressed_condition_era",
            "model": "LSTM_1L",
        },
        "numeric_sequence": {
            "version": "compressed_hybrid",
            "model": "GRU_2L_numeric",
        },
    },
}


FUSION_SPECS = {
    "tabular_only": ["p_tabular"],
    "sequence_only": ["p_sequence"],
    "numeric_sequence_only": ["p_numeric_sequence"],
    "fusion_tabular_sequence": ["p_tabular", "p_sequence"],
    "fusion_tabular_numeric_sequence": ["p_tabular", "p_numeric_sequence"],
    "fusion_tabular_sequence_numeric": [
        "p_tabular",
        "p_sequence",
        "p_numeric_sequence",
    ],
}


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict:
    """
    PyTorch >= 2.6 can use weights_only=True by default.
    Our numeric checkpoints may contain numpy arrays, so we explicitly allow full load.
    """
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def transform_score_features(x: pd.DataFrame, score_cols: list[str], mode: str) -> np.ndarray:
    arr = x[score_cols].to_numpy(dtype=float)

    if mode == "prob":
        return clip_prob(arr)

    if mode == "logit":
        arr = clip_prob(arr)
        return np.log(arr / (1.0 - arr))

    raise ValueError(f"Unknown score transform: {mode}")


def make_tabular_model(model_name: str, seed: int):
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

    raise ValueError(f"Unknown tabular model: {model_name}")


def load_tabular_features(cache_dir: Path, task: str, top_n_codes: int, top_n_numeric_codes: int) -> pd.DataFrame:
    path = cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Tabular feature cache not found: {path}")
    return pd.read_parquet(path)


def make_tabular_scores(
    cache_dir: Path,
    task: str,
    model_name: str,
    seed: int,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    """
    Trains tabular baseline on train split and returns raw scores for tuning + held_out.

    Important:
    - We do not Platt-calibrate here.
    - Fusion LogisticRegression itself learns calibration/fusion on tuning.
    """
    set_global_seed(seed)

    df = load_tabular_features(
        cache_dir=cache_dir,
        task=task,
        top_n_codes=top_n_codes,
        top_n_numeric_codes=top_n_numeric_codes,
    )

    feature_cols = [c for c in df.columns if c not in META_COLS]

    train_df = df[df["split"] == "train"].copy()
    tuning_df = df[df["split"] == "tuning"].copy()
    heldout_df = df[df["split"] == "held_out"].copy()

    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()

    model = make_tabular_model(model_name, seed=seed)

    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)

    frames = []

    for split_name, part in [("tuning", tuning_df), ("held_out", heldout_df)]:
        x = part[feature_cols].to_numpy()
        p = model.predict_proba(x)[:, 1]

        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "split": split_name,
                    "seed": seed,
                    "row_id": part["row_id"].astype(int).to_numpy(),
                    "subject_id": part["subject_id"].astype(int).to_numpy(),
                    "y_true": part["label"].astype(int).to_numpy(),
                    "p_tabular": p.astype(float),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def find_checkpoint(
    ckpt_dir: Path,
    task: str,
    version: str,
    model_name: str,
    seed: int | None = None,
) -> Path:
    """
    Flexible checkpoint lookup.

    Supports:
    1) task__version__model__seed42.pt
    2) task__version__model.pt
    3) recursive search inside ckpt_dir
    """
    candidates = []

    if seed is not None:
        candidates.append(
            ckpt_dir / f"{task}__{version}__{model_name}__seed{seed}.pt"
        )

    candidates.append(
        ckpt_dir / f"{task}__{version}__{model_name}.pt"
    )

    for path in candidates:
        if path.exists():
            return path

    recursive_patterns = []

    if seed is not None:
        recursive_patterns.append(
            f"**/{task}__{version}__{model_name}__seed{seed}.pt"
        )

    recursive_patterns.append(
        f"**/{task}__{version}__{model_name}.pt"
    )

    matches = []
    for pattern in recursive_patterns:
        matches.extend(sorted(ckpt_dir.glob(pattern)))

    matches = list(dict.fromkeys(matches))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            "Found multiple matching checkpoints:\n"
            + "\n".join(str(x) for x in matches)
            + "\nPlease pass a more specific checkpoint directory."
        )

    raise FileNotFoundError(
        f"Checkpoint not found for task={task}, version={version}, model={model_name}, seed={seed}. "
        f"Searched inside: {ckpt_dir}"
    )


def make_sequence_scores(
    sequence_data_dir: Path,
    checkpoint_dir: Path,
    task: str,
    version: str,
    model_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """
    Loads a code-only sequence checkpoint and returns raw sigmoid scores
    for tuning + held_out.
    """
    set_global_seed(seed)

    ckpt_file = find_checkpoint(
        ckpt_dir=checkpoint_dir,
        task=task,
        version=version,
        model_name=model_name,
        seed=seed,
    )
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Sequence checkpoint not found: {ckpt_file}")

    ckpt = safe_torch_load(ckpt_file, device)

    df = seq_mod.load_sequence_examples(sequence_data_dir, task, version)
    vocab = seq_mod.load_vocab(sequence_data_dir, task, version)

    max_len = int(ckpt.get("max_len", 512))
    loaders = seq_mod.make_loaders(
        df=df,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
        num_workers=num_workers,
    )

    model = seq_mod.make_model(model_name, vocab_size=len(vocab))
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
                    "seed": seed,
                    "row_id": pred["row_id"].astype(int),
                    "subject_id": pred["subject_id"].astype(int),
                    "y_true": pred["y_true"].astype(int),
                    "p_sequence": p.astype(float),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def make_numeric_sequence_scores(
    sequence_data_dir: Path,
    checkpoint_dir: Path,
    task: str,
    version: str,
    model_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """
    Loads a numeric-aware sequence checkpoint and returns raw sigmoid scores
    for tuning + held_out.
    """
    set_global_seed(seed)

    ckpt_file = find_checkpoint(
        ckpt_dir=checkpoint_dir,
        task=task,
        version=version,
        model_name=model_name,
        seed=seed,
    )
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Numeric sequence checkpoint not found: {ckpt_file}")

    ckpt = safe_torch_load(ckpt_file, device)

    df = numseq_mod.load_sequence_examples(sequence_data_dir, task, version)
    vocab = numseq_mod.load_vocab(sequence_data_dir, task, version)

    max_len = int(ckpt.get("max_len", 1024))

    if "token_numeric_mean" in ckpt and "token_numeric_std" in ckpt:
        token_mean = np.asarray(ckpt["token_numeric_mean"], dtype=np.float32)
        token_std = np.asarray(ckpt["token_numeric_std"], dtype=np.float32)
    else:
        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        token_mean, token_std, _ = numseq_mod.compute_token_numeric_stats(
            train_df,
            vocab_size=len(vocab),
        )

    loaders = numseq_mod.make_loaders(
        df=df,
        token_mean=token_mean,
        token_std=token_std,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
        num_workers=num_workers,
    )

    model = numseq_mod.make_model(model_name, vocab_size=len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    frames = []

    for split_name in ["tuning", "held_out"]:
        pred = numseq_mod.run_inference(model, loaders[split_name], device)
        p = numseq_mod.sigmoid_np(pred["logits"])

        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "split": split_name,
                    "seed": seed,
                    "row_id": pred["row_id"].astype(int),
                    "subject_id": pred["subject_id"].astype(int),
                    "y_true": pred["y_true"].astype(int),
                    "p_numeric_sequence": p.astype(float),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def merge_base_scores(
    tabular: pd.DataFrame,
    sequence: pd.DataFrame,
    numeric_sequence: pd.DataFrame,
) -> pd.DataFrame:
    key = ["task", "split", "seed", "row_id", "subject_id", "y_true"]

    out = tabular.merge(sequence, on=key, how="inner")
    out = out.merge(numeric_sequence, on=key, how="inner")

    if out.empty:
        raise ValueError("Base score merge produced 0 rows")

    return out


def fit_meta_lr(
    train_df: pd.DataFrame,
    score_cols: list[str],
    score_transform: str,
    class_weight: str | None,
) -> Pipeline:
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


def predict_meta_lr(
    model: Pipeline,
    df: pd.DataFrame,
    score_cols: list[str],
    score_transform: str,
) -> np.ndarray:
    x = transform_score_features(df, score_cols, mode=score_transform)
    return model.predict_proba(x)[:, 1]


def evaluate_prediction(y_true: np.ndarray, risk: np.ndarray) -> dict:
    m = binary_ranking_metrics(y_true, risk)
    tk = topk_metrics(y_true, risk).set_index("top_frac")

    m["top_5pct_precision"] = float(tk.loc[0.05, "top_k_event_rate"])
    m["top_10pct_precision"] = float(tk.loc[0.10, "top_k_event_rate"])
    m["top_20pct_precision"] = float(tk.loc[0.20, "top_k_event_rate"])

    return m


def run_single_seed_fusion(
    base_scores: pd.DataFrame,
    task: str,
    seed: int,
    score_transform: str,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning = base_scores[
        (base_scores["task"] == task)
        & (base_scores["seed"] == seed)
        & (base_scores["split"] == "tuning")
    ].copy()

    heldout = base_scores[
        (base_scores["task"] == task)
        & (base_scores["seed"] == seed)
        & (base_scores["split"] == "held_out")
    ].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out rows for task={task}, seed={seed}")

    metric_rows = []
    pred_rows = []

    for variant, score_cols in FUSION_SPECS.items():
        meta_model = fit_meta_lr(
            train_df=tuning,
            score_cols=score_cols,
            score_transform=score_transform,
            class_weight=class_weight,
        )

        risk = predict_meta_lr(
            model=meta_model,
            df=heldout,
            score_cols=score_cols,
            score_transform=score_transform,
        )

        y = heldout["y_true"].astype(int).to_numpy()
        metrics = evaluate_prediction(y, risk)

        metric_rows.append(
            {
                "task": task,
                "seed": seed,
                "variant": variant,
                "score_cols": ",".join(score_cols),
                "score_transform": score_transform,
                "meta_model": "LogisticRegression",
                **metrics,
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "task": task,
                    "seed": seed,
                    "variant": variant,
                    "split": "held_out",
                    "row_id": heldout["row_id"].astype(int).to_numpy(),
                    "subject_id": heldout["subject_id"].astype(int).to_numpy(),
                    "y_true": y,
                    "risk": risk.astype(float),
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def summarize_single_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "n",
        "n_positive",
        "event_rate",
        "auroc",
        "auprc",
        "brier",
        "logloss",
        "top_5pct_precision",
        "top_10pct_precision",
        "top_20pct_precision",
    ]

    id_cols = ["task", "variant", "score_cols", "score_transform", "meta_model"]

    agg = (
        metrics
        .groupby(id_cols)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    agg.columns = [
        "__".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    counts = (
        metrics
        .groupby(id_cols)
        .agg(n_seeds=("seed", "nunique"))
        .reset_index()
    )

    return counts.merge(agg, on=id_cols, how="left")


def make_ensemble_base_scores(base_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Average base model scores over seeds first.
    Then train one meta LogisticRegression on tuning and evaluate on held_out.
    """
    id_cols = ["task", "split", "row_id", "subject_id", "y_true"]
    score_cols = ["p_tabular", "p_sequence", "p_numeric_sequence"]

    ens = (
        base_scores
        .groupby(id_cols)[score_cols]
        .mean()
        .reset_index()
    )

    ens["seed"] = -1
    return ens


def run_ensemble_fusion(
    ensemble_scores: pd.DataFrame,
    task: str,
    score_transform: str,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning = ensemble_scores[
        (ensemble_scores["task"] == task)
        & (ensemble_scores["split"] == "tuning")
    ].copy()

    heldout = ensemble_scores[
        (ensemble_scores["task"] == task)
        & (ensemble_scores["split"] == "held_out")
    ].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out ensemble rows for task={task}")

    metric_rows = []
    pred_rows = []

    for variant, score_cols in FUSION_SPECS.items():
        meta_model = fit_meta_lr(
            train_df=tuning,
            score_cols=score_cols,
            score_transform=score_transform,
            class_weight=class_weight,
        )

        risk = predict_meta_lr(
            model=meta_model,
            df=heldout,
            score_cols=score_cols,
            score_transform=score_transform,
        )

        y = heldout["y_true"].astype(int).to_numpy()
        metrics = evaluate_prediction(y, risk)

        metric_rows.append(
            {
                "task": task,
                "variant": variant,
                "score_cols": ",".join(score_cols),
                "score_transform": score_transform,
                "meta_model": "LogisticRegression",
                "base_scores": "mean_over_seeds",
                **metrics,
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "task": task,
                    "variant": variant,
                    "split": "held_out",
                    "row_id": heldout["row_id"].astype(int).to_numpy(),
                    "subject_id": heldout["subject_id"].astype(int).to_numpy(),
                    "y_true": y,
                    "risk": risk.astype(float),
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-dir", type=Path, default=Path("ehrshot_baseline_cache"))
    parser.add_argument("--sequence-data-dir", type=Path, default=Path("ehrshot_sequence_datasets"))

    parser.add_argument(
        "--sequence-checkpoint-dir",
        type=Path,
        default=Path("ehrshot_multiseed_sequence_results/checkpoints"),
    )
    parser.add_argument(
        "--numeric-checkpoint-dir",
        type=Path,
        default=Path("ehrshot_multiseed_numeric_sequence_results/checkpoints"),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_score_fusion_minimal"))

    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--tasks", type=str, default="guo_readmission,guo_icu")

    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size-sequence", type=int, default=64)
    parser.add_argument("--batch-size-numeric", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--score-transform",
        type=str,
        default="logit",
        choices=["logit", "prob"],
        help="Use logits of base probabilities or raw probabilities as meta-features.",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Class weight for meta LogisticRegression.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    device = get_device(args.device)

    class_weight = None if args.class_weight == "none" else "balanced"

    print("DEVICE:", device)
    print("TASKS:", tasks)
    print("SEEDS:", seeds)
    print("score_transform:", args.score_transform)
    print("class_weight:", class_weight)

    all_base_scores = []

    for task in tasks:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task={task}. Available: {list(TASK_CONFIGS)}")

        cfg = TASK_CONFIGS[task]

        for seed in seeds:
            print("=" * 100)
            print(f"Building base scores: task={task}, seed={seed}")

            tab_scores = make_tabular_scores(
                cache_dir=args.cache_dir,
                task=task,
                model_name=cfg["tabular_model"],
                seed=seed,
                top_n_codes=args.top_n_codes,
                top_n_numeric_codes=args.top_n_numeric_codes,
            )

            seq_scores = make_sequence_scores(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.sequence_checkpoint_dir,
                task=task,
                version=cfg["sequence"]["version"],
                model_name=cfg["sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_sequence,
                num_workers=args.num_workers,
            )

            numseq_scores = make_numeric_sequence_scores(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.numeric_checkpoint_dir,
                task=task,
                version=cfg["numeric_sequence"]["version"],
                model_name=cfg["numeric_sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_numeric,
                num_workers=args.num_workers,
            )

            merged = merge_base_scores(
                tabular=tab_scores,
                sequence=seq_scores,
                numeric_sequence=numseq_scores,
            )

            all_base_scores.append(merged)

    base_scores = pd.concat(all_base_scores, ignore_index=True)
    base_scores.to_csv(args.output_dir / "base_scores_tuning_heldout.csv", index=False)

    print("Base scores shape:", base_scores.shape)
    print(
        base_scores
        .groupby(["task", "split", "seed"])
        .agg(
            n=("row_id", "size"),
            n_positive=("y_true", "sum"),
            event_rate=("y_true", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 1. Per-seed fusion
    # ------------------------------------------------------------
    single_metric_parts = []
    single_pred_parts = []

    for task in tasks:
        for seed in seeds:
            m, p = run_single_seed_fusion(
                base_scores=base_scores,
                task=task,
                seed=seed,
                score_transform=args.score_transform,
                class_weight=class_weight,
            )
            single_metric_parts.append(m)
            single_pred_parts.append(p)

    single_metrics = pd.concat(single_metric_parts, ignore_index=True)
    single_predictions = pd.concat(single_pred_parts, ignore_index=True)

    single_metrics.to_csv(args.output_dir / "fusion_single_seed_metrics.csv", index=False)
    single_predictions.to_csv(args.output_dir / "fusion_single_seed_predictions.csv", index=False)

    seed_summary = summarize_single_seed_metrics(single_metrics)
    seed_summary.to_csv(args.output_dir / "fusion_seed_summary.csv", index=False)

    # ------------------------------------------------------------
    # 2. Ensemble-over-seeds fusion
    # ------------------------------------------------------------
    ensemble_scores = make_ensemble_base_scores(base_scores)
    ensemble_scores.to_csv(args.output_dir / "ensemble_base_scores_tuning_heldout.csv", index=False)

    ens_metric_parts = []
    ens_pred_parts = []

    for task in tasks:
        m, p = run_ensemble_fusion(
            ensemble_scores=ensemble_scores,
            task=task,
            score_transform=args.score_transform,
            class_weight=class_weight,
        )
        ens_metric_parts.append(m)
        ens_pred_parts.append(p)

    ensemble_metrics = pd.concat(ens_metric_parts, ignore_index=True)
    ensemble_predictions = pd.concat(ens_pred_parts, ignore_index=True)

    ensemble_metrics.to_csv(args.output_dir / "fusion_ensemble_metrics.csv", index=False)
    ensemble_predictions.to_csv(args.output_dir / "fusion_ensemble_predictions.csv", index=False)

    print("\nSaved outputs to:", args.output_dir)

    print("\nSingle-seed summary sorted by AUPRC:")
    cols = [
        "task",
        "variant",
        "n_seeds",
        "auprc__mean",
        "auprc__std",
        "auroc__mean",
        "brier__mean",
        "logloss__mean",
        "top_10pct_precision__mean",
    ]
    print(
        seed_summary[cols]
        .sort_values(["task", "auprc__mean"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nEnsemble fusion metrics sorted by AUPRC:")
    cols = [
        "task",
        "variant",
        "auprc",
        "auroc",
        "brier",
        "logloss",
        "top_10pct_precision",
        "n",
        "n_positive",
    ]
    print(
        ensemble_metrics[cols]
        .sort_values(["task", "auprc"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()