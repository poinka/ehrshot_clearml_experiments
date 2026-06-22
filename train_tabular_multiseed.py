from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common_ehrshot_eval import (
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    topk_metrics,
)

META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}

DEFAULT_CONFIGS = [
    # Main tabular reference models from the notebooks.
    {"task": "guo_readmission", "model": "HistGradientBoosting"},
    {"task": "guo_icu", "model": "RandomForest_balanced"},
]


def load_cached_features(cache_dir: Path, task: str, top_n_codes: int, top_n_numeric_codes: int) -> pd.DataFrame:
    path = cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached tabular features not found: {path}. "
            "Run the classical baseline notebook once or copy ehrshot_baseline_cache to the ClearML working directory."
        )
    df = pd.read_parquet(path)
    return df

def maybe_download_tabular_cache_from_s3_prefix(
    cache_dir: Path,
    cache_s3_prefix: str,
    tasks: list[str],
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> Path:
    """
    If tabular cache exists locally, use it.
    Otherwise download required parquet files from MinIO/S3 prefix.

    Expected remote layout:
        <cache_s3_prefix>/guo_readmission_features_top500_num40.parquet
        <cache_s3_prefix>/guo_icu_features_top500_num40.parquet
    """
    expected_files = [
        cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
        for task in tasks
    ]

    if all(p.exists() for p in expected_files):
        print(f"Using local tabular cache: {cache_dir}")
        return cache_dir

    if not cache_s3_prefix:
        missing = [str(p) for p in expected_files if not p.exists()]
        raise FileNotFoundError(
            "Tabular cache files are missing locally and --cache-s3-prefix is not provided. "
            f"Missing files: {missing}"
        )

    from clearml import StorageManager

    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_s3_prefix.rstrip("/")

    for task in tasks:
        fname = f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"
        dst = cache_dir / fname

        if dst.exists():
            print(f"Already exists locally: {dst}")
            continue

        remote_url = f"{prefix}/{fname}"
        print(f"Downloading {remote_url}")
        local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

        print(f"Copying {local_copy} -> {dst}")
        shutil.copy2(local_copy, dst)

    return cache_dir


def make_model(model_name: str, seed: int):
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
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
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


def positive_proba(model, x):
    return model.predict_proba(x)[:, 1]


def fit_platt(y_tuning: np.ndarray, p_tuning: np.ndarray, seed: int):
    p = np.clip(np.asarray(p_tuning), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    calibrator.fit(logits, y_tuning.astype(int))

    def transform(p_raw):
        p_raw = np.clip(np.asarray(p_raw), 1e-6, 1 - 1e-6)
        x = np.log(p_raw / (1 - p_raw)).reshape(-1, 1)
        return calibrator.predict_proba(x)[:, 1]

    return transform, calibrator


def run_one_seed_task_model(df: pd.DataFrame, task: str, model_name: str, seed: int, out_dir: Path):
    set_global_seed(seed)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    train_df = df[df["split"] == "train"].copy()
    tuning_df = df[df["split"] == "tuning"].copy()
    heldout_df = df[df["split"] == "held_out"].copy()

    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()
    x_tuning = tuning_df[feature_cols].to_numpy()
    y_tuning = tuning_df["label"].astype(int).to_numpy()
    x_heldout = heldout_df[feature_cols].to_numpy()
    y_heldout = heldout_df["label"].astype(int).to_numpy()

    model = make_model(model_name, seed=seed)
    if model_name == "HistGradientBoosting":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, sample_weight=sw)
    else:
        model.fit(x_train, y_train)

    p_tuning_raw = positive_proba(model, x_tuning)
    p_heldout_raw = positive_proba(model, x_heldout)
    platt_transform, _ = fit_platt(y_tuning, p_tuning_raw, seed=seed)
    p_heldout_platt = platt_transform(p_heldout_raw)

    rows = []
    pred_frames = []
    topk_frames = []

    for calibration, p_heldout in [("raw", p_heldout_raw), ("platt", p_heldout_platt)]:
        metrics = binary_ranking_metrics(y_heldout, p_heldout)
        rows.append(
            {
                "task": task,
                "family": "tabular",
                "source": "tabular",
                "version": "tabular_all_features",
                "model": model_name,
                "seed": seed,
                "calibration": calibration,
                **{f"heldout_{k}": v for k, v in metrics.items()},
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "family": "tabular",
                    "source": "tabular",
                    "version": "tabular_all_features",
                    "model": model_name,
                    "seed": seed,
                    "calibration": calibration,
                    "row_id": heldout_df["row_id"].astype(int).to_numpy(),
                    "subject_id": heldout_df["subject_id"].astype(int).to_numpy(),
                    "y_true": y_heldout,
                    "risk": p_heldout,
                }
            )
        )
        tk = topk_metrics(y_heldout, p_heldout)
        tk.insert(0, "calibration", calibration)
        tk.insert(0, "seed", seed)
        tk.insert(0, "model", model_name)
        tk.insert(0, "version", "tabular_all_features")
        tk.insert(0, "family", "tabular")
        tk.insert(0, "task", task)
        topk_frames.append(tk)

    result_df = pd.DataFrame(rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    topk_df = pd.concat(topk_frames, ignore_index=True)

    stem = f"{task}__tabular_all_features__{model_name}__seed{seed}"
    result_df.to_csv(out_dir / f"{stem}__results.csv", index=False)
    pred_df.to_csv(out_dir / f"{stem}__heldout_predictions.csv", index=False)
    topk_df.to_csv(out_dir / f"{stem}__topk.csv", index=False)
    return result_df, pred_df, topk_df


def maybe_init_clearml(args, config: dict):
    if not args.enable_clearml:
        return None

    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        output_uri=args.clearml_output_uri or None,
    )

    task.connect(config)

    if args.execute_remotely:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("ehrshot_baseline_cache"))
    parser.add_argument(
        "--cache-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with tabular cache parquet files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_multiseed_tabular_results"))
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)
    parser.add_argument("--enable-clearml", action="store_true")
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
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="tabular_multiseed_stability")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    config = vars(args).copy()
    config["seeds"] = seeds
    task = maybe_init_clearml(args, config)

    args.cache_dir = maybe_download_tabular_cache_from_s3_prefix(
        cache_dir=args.cache_dir,
        cache_s3_prefix=args.cache_s3_prefix,
        tasks=[cfg["task"] for cfg in DEFAULT_CONFIGS],
        top_n_codes=args.top_n_codes,
        top_n_numeric_codes=args.top_n_numeric_codes,
    )
    all_results, all_predictions, all_topk = [], [], []
    for cfg in DEFAULT_CONFIGS:
        df = load_cached_features(args.cache_dir, cfg["task"], args.top_n_codes, args.top_n_numeric_codes)
        for seed in seeds:
            print("=" * 100)
            print(f"Tabular experiment: task={cfg['task']} model={cfg['model']} seed={seed}")
            r, p, t = run_one_seed_task_model(df, cfg["task"], cfg["model"], seed, args.output_dir)
            all_results.append(r)
            all_predictions.append(p)
            all_topk.append(t)

    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    topk_df = pd.concat(all_topk, ignore_index=True)
    results_df.to_csv(args.output_dir / "tabular_multiseed_results.csv", index=False)
    predictions_df.to_csv(args.output_dir / "tabular_multiseed_heldout_predictions.csv", index=False)
    topk_df.to_csv(args.output_dir / "tabular_multiseed_topk.csv", index=False)
    print(results_df.sort_values(["task", "seed", "calibration"]))

    if task is not None:
        task.upload_artifact("tabular_multiseed_results", results_df)
        task.upload_artifact("tabular_multiseed_predictions", predictions_df)
        task.upload_artifact("tabular_multiseed_topk", topk_df)


if __name__ == "__main__":
    main()
