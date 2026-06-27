#!/usr/bin/env python
"""Compare sequence compression experiments stored as ClearML artifacts.

This script is intended for the current EHRSHOT ClearML/MinIO workflow:

1. Training tasks are named like:
   cmp_readmission_<version>_RETAIN_lite_seed42
   cmp_icu_<version>_GRU_2L_numeric_seed42

2. Each training task uploads artifacts:
   heldout_predictions, metrics, topk, history, numeric_stats

3. Compression datasets are stored under S3/MinIO prefix:
   .../ehrshot_multiseed_inputs/ehrshot_sequence_datasets_compression_v2

The script downloads heldout_predictions from ClearML tasks, optionally downloads
examples.parquet files for sequence-cost analysis, then computes:
  - metrics_by_seed.csv
  - metrics_mean_std.csv
  - patient_bootstrap_metric_ci.csv
  - paired_bootstrap_vs_raw.csv
  - high_repeat_group.csv
  - high_repeat_metrics_by_seed.csv
  - non_high_repeat_metrics_by_seed.csv
  - paired_bootstrap_vs_raw_high_repeat.csv
  - sequence_cost_summary.csv
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

DEFAULT_PROJECT = "pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
DEFAULT_OUTPUT_URI = "s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"
DEFAULT_SEQUENCE_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "ehrshot_multiseed_inputs/ehrshot_sequence_datasets"
)

DEFAULT_GRID_RESULTS_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "ehrshot_sequence_compression_grid_results"
)

def parse_csv_list(x: str | Sequence[str]) -> list[str]:
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for item in x:
            out.extend(parse_csv_list(item))
        return out
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Where to save downloaded predictions and comparison outputs.
    parser.add_argument("--results-dir", type=Path, default=Path("ehrshot_sequence_compression_results_v2"))
    parser.add_argument(
        "--grid-results-s3-prefix",
        type=str,
        default=DEFAULT_GRID_RESULTS_S3_PREFIX,
        help="S3/MinIO prefix with grid_all_heldout_predictions.csv from one-task grid training.",
    )
    parser.add_argument("--sequence-dir", type=Path, default=Path("ehrshot_sequence_datasets_compression_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_sequence_compression_comparison_v2"))

    # What to compare.
    parser.add_argument("--tasks", nargs="+", default=["guo_readmission", "guo_icu"])
    parser.add_argument("--baseline-version", default="raw")
    parser.add_argument("--calibration", default="platt", choices=["raw", "platt", "all"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high-repeat-quantile", type=float, default=0.90)
    parser.add_argument("--top-fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])

    # ClearML source tasks. By default we look for training task names starting with cmp_readmission_ / cmp_icu_.
    parser.add_argument("--load-from-clearml", action="store_true")
    parser.add_argument(
        "--task-name-prefixes",
        type=str,
        default="cmp_readmission_,cmp_icu_",
        help="Comma-separated ClearML task name prefixes to scan for heldout_predictions artifacts.",
    )
    parser.add_argument(
        "--prediction-task-ids",
        type=str,
        default="",
        help="Optional comma-separated explicit ClearML task IDs. If provided, task-name search is skipped.",
    )
    parser.add_argument("--artifact-name", default="heldout_predictions")
    parser.add_argument("--skip-incomplete", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", help="Fail if an expected artifact cannot be loaded.")

    # Sequence examples for cost summary.
    parser.add_argument("--sequence-data-s3-prefix", type=str, default=DEFAULT_SEQUENCE_S3_PREFIX)
    parser.add_argument("--download-sequence-examples", action="store_true", default=True)

    # ClearML execution for this comparison task.
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", default="cpu")
    parser.add_argument("--clearml-project", default=DEFAULT_PROJECT)
    parser.add_argument("--clearml-task-name", default="sequence_compression_comparison_from_artifacts")
    parser.add_argument("--clearml-output-uri", default=DEFAULT_OUTPUT_URI)

    return parser.parse_args()


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args: argparse.Namespace) -> dict:
    cfg = vars(args).copy()
    for key in ["results_dir", "sequence_dir", "output_dir"]:
        cfg[key] = str(cfg[key])
    return cfg


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict) -> None:
    path_keys = {"results_dir", "sequence_dir", "output_dir"}
    int_keys = {"n_bootstrap", "seed"}
    float_keys = {"high_repeat_quantile"}
    bool_keys = {
        "load_from_clearml", "skip_incomplete", "strict", "download_sequence_examples",
    }
    skip_keys = {"enable_clearml", "execute_remotely"}
    for key, value in cfg.items():
        if key in skip_keys or not hasattr(args, key):
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in float_keys:
            setattr(args, key, float(value))
        elif key in bool_keys:
            if isinstance(value, str):
                setattr(args, key, value.lower() in {"1", "true", "yes", "y"})
            else:
                setattr(args, key, bool(value))
        else:
            setattr(args, key, value)


def maybe_init_clearml(args: argparse.Namespace):
    remote_agent_run = is_clearml_agent_run()
    if not args.enable_clearml and not remote_agent_run:
        return None

    from clearml import Task

    if not remote_agent_run:
        Task.force_requirements_env_freeze(False, "requirements.txt")

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

    cfg = dict(task.connect(build_clearml_config(args)))
    sync_args_from_clearml_config(args, cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  load_from_clearml = {args.load_from_clearml}")
    print(f"  task_name_prefixes = {args.task_name_prefixes}")
    print(f"  prediction_task_ids = {args.prediction_task_ids}")
    print(f"  results_dir = {args.results_dir}")
    print(f"  sequence_dir = {args.sequence_dir}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  clearml_queue = {args.clearml_queue}")
    print(f"  grid_results_s3_prefix = {args.grid_results_s3_prefix}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing comparison task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task

GRID_RESULT_FILES = [
    "grid_run_plan.csv",
    "grid_all_heldout_predictions.csv",
    "grid_all_metrics.csv",
    "grid_all_topk.csv",
    "grid_all_history.csv",
]


def maybe_download_grid_results_from_s3(args: argparse.Namespace) -> None:
    """
    Download outputs produced by train_sequence_compression_grid_clearml.py.

    This is the new workflow:
    one ClearML task trains all runs and uploads grid_all_*.csv files to MinIO.
    """
    if not args.grid_results_s3_prefix:
        print("No --grid-results-s3-prefix provided; using local results_dir only.")
        return

    from clearml import StorageManager

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.grid_results_s3_prefix.rstrip("/")

    for fname in GRID_RESULT_FILES:
        dst = args.results_dir / fname

        if dst.exists():
            print(f"Using local grid result: {dst}")
            continue

        remote_url = f"{prefix}/{fname}"
        print(f"Downloading grid result: {remote_url}")

        try:
            local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

            if local_copy.is_file():
                shutil.copy2(local_copy, dst)
            elif local_copy.is_dir():
                matches = sorted(local_copy.rglob(fname))
                if len(matches) == 1:
                    shutil.copy2(matches[0], dst)
                else:
                    raise FileNotFoundError(f"Could not find {fname} inside {local_copy}")
            else:
                raise FileNotFoundError(f"StorageManager returned missing path: {local_copy}")

        except Exception as e:
            if fname == "grid_all_heldout_predictions.csv":
                raise RuntimeError(
                    f"Cannot download required predictions file: {remote_url}"
                ) from e

            print(f"WARNING: optional grid result was not downloaded: {fname} | {e!r}")

# -----------------------------
# Metrics
# -----------------------------

def safe_auroc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def safe_brier(y, p):
    return float(brier_score_loss(np.asarray(y).astype(int), np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)))


def safe_logloss(y, p):
    return float(log_loss(np.asarray(y).astype(int), np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6), labels=[0, 1]))


def topk_stats(y, p, frac: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    if n == 0:
        return {"precision": np.nan, "lift": np.nan, "event_capture": np.nan}
    k = max(1, int(np.ceil(n * frac)))
    order = np.argsort(-p)
    top = order[:k]
    base = float(y.mean())
    events = int(y[top].sum())
    precision = events / k
    return {
        "precision": float(precision),
        "lift": float(precision / base) if base > 0 else np.nan,
        "event_capture": float(events / int(y.sum())) if int(y.sum()) > 0 else np.nan,
    }


def metric_functions(top_fracs: list[float]) -> dict[str, Callable]:
    fns: dict[str, Callable] = {
        "auroc": safe_auroc,
        "auprc": safe_auprc,
        "brier": safe_brier,
        "logloss": safe_logloss,
    }
    for frac in top_fracs:
        pct = int(round(frac * 100))
        fns[f"top{pct}_precision"] = lambda y, p, frac=frac: topk_stats(y, p, frac)["precision"]
        fns[f"top{pct}_lift"] = lambda y, p, frac=frac: topk_stats(y, p, frac)["lift"]
        fns[f"top{pct}_event_capture"] = lambda y, p, frac=frac: topk_stats(y, p, frac)["event_capture"]
    return fns


def higher_is_better(metric: str) -> bool:
    return metric not in {"brier", "logloss"}


# -----------------------------
# ClearML artifact loading
# -----------------------------

def task_name_matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(str(name).startswith(p) for p in prefixes)


def load_artifact_csv_from_task(task, artifact_name: str) -> pd.DataFrame:
    if artifact_name not in task.artifacts:
        raise KeyError(f"Artifact {artifact_name!r} not found in task {task.id} / {task.name}")
    artifact = task.artifacts[artifact_name]

    # artifact.get() may return DataFrame if it was uploaded as DataFrame.
    obj = artifact.get()
    if isinstance(obj, pd.DataFrame):
        return obj

    local = Path(artifact.get_local_copy())
    if local.is_file():
        return pd.read_csv(local)

    if local.is_dir():
        csv_files = sorted(local.rglob("*.csv"))
        if len(csv_files) == 1:
            return pd.read_csv(csv_files[0])
        if len(csv_files) > 1:
            # Prefer the concrete heldout prediction filename if present.
            pred_files = [p for p in csv_files if "heldout_predictions" in p.name]
            if len(pred_files) == 1:
                return pd.read_csv(pred_files[0])
            raise ValueError(f"Multiple csv files in artifact local copy {local}: {csv_files[:10]}")

    raise ValueError(f"Cannot read artifact {artifact_name!r} from task {task.id}; local={local}")


def discover_prediction_tasks(args: argparse.Namespace):
    from clearml import Task

    explicit_ids = parse_csv_list(args.prediction_task_ids)
    if explicit_ids:
        print(f"Loading explicit prediction task ids: {len(explicit_ids)}")
        return [Task.get_task(task_id=tid) for tid in explicit_ids]

    prefixes = parse_csv_list(args.task_name_prefixes)
    print(f"Searching ClearML project={args.clearml_project!r} for task prefixes={prefixes}")

    # ClearML API returns Task objects; filtering is done client-side to avoid relying on server-side regex syntax.
    tasks = Task.get_tasks(project_name=args.clearml_project)
    selected = []
    for t in tasks:
        name = getattr(t, "name", None) or getattr(getattr(t, "data", None), "name", "")
        if task_name_matches(str(name), prefixes):
            selected.append(t)
    selected = sorted(selected, key=lambda t: getattr(t, "name", ""))
    print(f"Found matching ClearML tasks: {len(selected)}")
    return selected


def download_predictions_from_clearml(args: argparse.Namespace) -> pd.DataFrame:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_prediction_tasks(args)
    parts = []
    errors = []

    for t in tasks:
        name = getattr(t, "name", None) or getattr(getattr(t, "data", None), "name", "")
        try:
            df = load_artifact_csv_from_task(t, args.artifact_name)
            df["source_task_id"] = t.id
            df["source_task_name"] = str(name)
            parts.append(df)
            print(f"Loaded {len(df):6d} rows from task={t.id} name={name}")
        except Exception as e:
            msg = f"Could not load artifact from task={getattr(t, 'id', '?')} name={name}: {e!r}"
            if args.strict:
                raise RuntimeError(msg) from e
            errors.append(msg)
            print("WARNING:", msg)

    if not parts:
        raise RuntimeError("No prediction artifacts loaded from ClearML tasks")

    pred = pd.concat(parts, ignore_index=True)

    # Save raw downloaded predictions for reproducibility.
    raw_path = args.results_dir / "clearml_downloaded_heldout_predictions_raw.csv"
    pred.to_csv(raw_path, index=False)
    print(f"Saved raw downloaded predictions: {raw_path}")

    required = {"task", "version", "model", "seed", "calibration", "row_id", "subject_id", "y_true", "risk"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Missing columns in downloaded predictions: {missing}")

    for col in ["seed", "row_id", "subject_id", "y_true"]:
        pred[col] = pred[col].astype(int)
    pred["risk"] = pred["risk"].astype(float)

    if args.calibration != "all":
        pred = pred[pred["calibration"] == args.calibration].copy()

    # Remove duplicate prediction rows if the same config was submitted twice.
    dedup_key = ["task", "version", "model", "seed", "calibration", "row_id", "subject_id"]
    n_before = len(pred)
    pred = pred.drop_duplicates(subset=dedup_key, keep="last").copy()
    n_after = len(pred)
    if n_before != n_after:
        print(f"Dropped duplicate prediction rows: {n_before - n_after}")

    out_path = args.results_dir / "all_heldout_predictions.csv"
    pred.to_csv(out_path, index=False)
    print(f"Saved filtered/deduped predictions: {out_path}")

    if errors:
        err_path = args.results_dir / "clearml_prediction_download_errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"Saved download warnings: {err_path}")

    return pred.reset_index(drop=True)


def load_predictions_from_local(results_dir: Path, calibration: str) -> pd.DataFrame:
    """
    Load predictions from local results_dir.

    New preferred input:
        grid_all_heldout_predictions.csv

    Backward-compatible inputs:
        *__heldout_predictions.csv
        all_heldout_predictions.csv
    """
    grid_path = results_dir / "grid_all_heldout_predictions.csv"

    if grid_path.exists():
        files = [grid_path]
    else:
        files = sorted(results_dir.glob("*__heldout_predictions.csv"))
        files += sorted(results_dir.glob("all_heldout_predictions.csv"))
        files = list(dict.fromkeys(files))

    if not files:
        raise FileNotFoundError(
            f"No prediction csv files found in {results_dir}. "
            "Expected grid_all_heldout_predictions.csv or *__heldout_predictions.csv"
        )

    parts = []

    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        parts.append(df)
        print(f"Loaded predictions: {path} shape={df.shape}")

    pred = pd.concat(parts, ignore_index=True)

    needed = {
        "task",
        "version",
        "model",
        "seed",
        "calibration",
        "row_id",
        "subject_id",
        "y_true",
        "risk",
    }
    missing = needed - set(pred.columns)

    if missing:
        raise ValueError(f"Missing columns in prediction files: {missing}")

    # If future version stores both tuning and held_out in one file, keep held_out here.
    if "split" in pred.columns:
        pred = pred[pred["split"] == "held_out"].copy()

    if calibration != "all":
        pred = pred[pred["calibration"] == calibration].copy()

    for col in ["seed", "row_id", "subject_id", "y_true"]:
        pred[col] = pred[col].astype(int)

    pred["risk"] = pred["risk"].astype(float)

    dedup_key = [
        "task",
        "version",
        "model",
        "seed",
        "calibration",
        "row_id",
        "subject_id",
    ]

    n_before = len(pred)
    pred = pred.drop_duplicates(subset=dedup_key, keep="last").copy()
    n_after = len(pred)

    if n_before != n_after:
        print(f"Dropped duplicate prediction rows: {n_before - n_after}")

    out_path = results_dir / "all_heldout_predictions.csv"
    pred.to_csv(out_path, index=False)
    print(f"Saved filtered/deduped predictions: {out_path}")

    return pred.reset_index(drop=True)


# -----------------------------
# Sequence examples / cost summary
# -----------------------------

def maybe_download_sequence_examples(args: argparse.Namespace, pred: pd.DataFrame) -> None:
    if not args.download_sequence_examples:
        return
    if not args.sequence_data_s3_prefix:
        return

    from clearml import StorageManager

    args.sequence_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.sequence_data_s3_prefix.rstrip("/")
    pairs = sorted({(str(r.task), str(r.version)) for r in pred[["task", "version"]].drop_duplicates().itertuples(index=False)})

    for task, version in pairs:
        local_dir = args.sequence_dir / task / version
        local_dir.mkdir(parents=True, exist_ok=True)
        dst = local_dir / "examples.parquet"
        if dst.exists():
            continue
        remote = f"{prefix}/{task}/{version}/examples.parquet"
        print(f"Downloading sequence examples: {remote}")
        try:
            local_copy = Path(StorageManager.get_local_copy(remote_url=remote))
            if local_copy.is_file():
                shutil.copy2(local_copy, dst)
            else:
                parquet_files = sorted(local_copy.rglob("examples.parquet")) if local_copy.exists() else []
                if len(parquet_files) == 1:
                    shutil.copy2(parquet_files[0], dst)
                else:
                    raise FileNotFoundError(f"StorageManager local copy is not a file: {local_copy}")
        except Exception as e:
            print(f"WARNING: could not download examples for {task}/{version}: {e!r}")


def load_examples(sequence_dir: Path, task: str, version: str) -> pd.DataFrame | None:
    path = sequence_dir / task / version / "examples.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["task"] = task
    df["version"] = version
    return df


def build_high_repeat_group(sequence_dir: Path, tasks: list[str], baseline_version: str, q: float) -> pd.DataFrame:
    """
    Define high-repeat held-out examples as top-q by amount of events removed
    by compressed_first_last.

    This matches our current dataset format:
        compressed_first_last/examples.parquet has n_events_removed_vs_raw.
    """
    rows = []

    for task in tasks:
        raw = load_examples(sequence_dir, task, baseline_version)
        first_last = load_examples(sequence_dir, task, "compressed_first_last")

        if raw is None:
            print(f"WARNING: raw examples not found for {task}; high-repeat group will be unavailable")
            continue

        raw_small = raw[
            ["row_id", "subject_id", "split", "label", "seq_len"]
        ].copy()

        raw_small = raw_small[raw_small["split"] == "held_out"].copy()
        raw_small = raw_small.rename(columns={"seq_len": "raw_seq_len"})

        if first_last is not None:
            fl_cols = ["row_id", "subject_id", "split", "seq_len"]

            if "n_events_removed_vs_raw" in first_last.columns:
                fl_cols.append("n_events_removed_vs_raw")

            fl = first_last[fl_cols].copy()
            fl = fl[fl["split"] == "held_out"].copy()
            fl = fl.rename(
                columns={
                    "seq_len": "first_last_seq_len",
                    "n_events_removed_vs_raw": "first_last_events_removed_vs_raw",
                }
            )

            tmp = raw_small.merge(
                fl,
                on=["row_id", "subject_id", "split"],
                how="left",
                validate="one_to_one",
            )

            if "first_last_events_removed_vs_raw" in tmp.columns:
                tmp["repeat_removed_proxy"] = tmp["first_last_events_removed_vs_raw"].fillna(0)
            else:
                tmp["repeat_removed_proxy"] = (
                    tmp["raw_seq_len"] - tmp["first_last_seq_len"]
                ).fillna(0)
        else:
            tmp = raw_small.copy()
            tmp["repeat_removed_proxy"] = 0.0

        threshold = float(np.quantile(tmp["repeat_removed_proxy"], q)) if len(tmp) else np.nan

        tmp["high_repeat_group"] = tmp["repeat_removed_proxy"] >= threshold
        tmp["high_repeat_threshold"] = threshold
        tmp["task"] = task

        rows.append(
            tmp[
                [
                    "task",
                    "row_id",
                    "subject_id",
                    "label",
                    "repeat_removed_proxy",
                    "high_repeat_group",
                    "high_repeat_threshold",
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_sequence_cost_summary(sequence_dir: Path, tasks: list[str], high_repeat_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build cost table for each compression version using audit columns already saved
    in examples.parquet.

    Expected new columns:
        seq_len
        raw_seq_len
        n_events_removed_vs_raw
        n_synthetic_events
        n_compressible_chronic_events_raw
    """
    rows = []

    if not sequence_dir.exists():
        return pd.DataFrame(rows)

    for task in tasks:
        task_dir = sequence_dir / task

        if not task_dir.exists():
            print(f"WARNING: sequence task dir not found: {task_dir}")
            continue

        for version_dir in sorted(task_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            version = version_dir.name
            df = load_examples(sequence_dir, task, version)

            if df is None or "seq_len" not in df.columns:
                print(f"WARNING: skip cost summary task={task} version={version}: missing examples or seq_len")
                continue

            df = df.copy()

            if "raw_seq_len" not in df.columns:
                if version == "raw":
                    df["raw_seq_len"] = df["seq_len"]
                else:
                    raw = load_examples(sequence_dir, task, "raw")
                    if raw is None:
                        print(f"WARNING: raw examples missing for cost summary task={task} version={version}")
                        continue

                    raw_small = raw[["row_id", "subject_id", "split", "seq_len"]].rename(
                        columns={"seq_len": "raw_seq_len"}
                    )

                    df = df.merge(
                        raw_small,
                        on=["row_id", "subject_id", "split"],
                        how="left",
                        validate="many_to_one",
                    )

            if "n_events_removed_vs_raw" not in df.columns:
                df["n_events_removed_vs_raw"] = df["raw_seq_len"] - df["seq_len"]

            if "n_synthetic_events" not in df.columns:
                if "n_compression_events" in df.columns:
                    df["n_synthetic_events"] = df["n_compression_events"]
                else:
                    df["n_synthetic_events"] = 0

            if "n_compressible_chronic_events_raw" not in df.columns:
                df["n_compressible_chronic_events_raw"] = np.nan

            split_parts = [
                ("all", df),
                ("held_out", df[df["split"] == "held_out"]),
            ]

            if high_repeat_df is not None and len(high_repeat_df):
                h = high_repeat_df[
                    (high_repeat_df["task"] == task)
                    & (high_repeat_df["high_repeat_group"])
                ]

                if len(h):
                    high_part = df.merge(
                        h[["task", "row_id", "subject_id", "high_repeat_group"]],
                        on=["task", "row_id", "subject_id"],
                        how="inner",
                    )
                    split_parts.append(("held_out_high_repeat", high_part))

            for split_name, part in split_parts:
                if len(part) == 0:
                    continue

                rows.append(
                    {
                        "task": task,
                        "version": version,
                        "split": split_name,
                        "n_examples": int(len(part)),
                        "n_patients": int(part["subject_id"].nunique()),
                        "mean_seq_len": float(part["seq_len"].mean()),
                        "median_seq_len": float(part["seq_len"].median()),
                        "p90_seq_len": float(np.quantile(part["seq_len"], 0.90)),
                        "max_seq_len": int(part["seq_len"].max()),
                        "mean_events_removed_vs_raw": float(part["n_events_removed_vs_raw"].mean()),
                        "median_events_removed_vs_raw": float(part["n_events_removed_vs_raw"].median()),
                        "p90_events_removed_vs_raw": float(np.quantile(part["n_events_removed_vs_raw"], 0.90)),
                        "mean_synthetic_events": float(part["n_synthetic_events"].mean()),
                        "mean_compressible_chronic_events_raw": float(
                            part["n_compressible_chronic_events_raw"].mean()
                        ),
                    }
                )

    return pd.DataFrame(rows)


# -----------------------------
# Aggregation / bootstrap
# -----------------------------

def compute_metrics_for_group(pred: pd.DataFrame, fns: dict[str, Callable], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in pred.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        y = part["y_true"].values
        p = part["risk"].values
        row.update({"n": int(len(part)), "n_positive": int(part["y_true"].sum()), "event_rate": float(part["y_true"].mean())})
        for name, fn in fns.items():
            try:
                row[name] = float(fn(y, p))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def patient_bootstrap_metric_ci(part: pd.DataFrame, fn: Callable, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = {sid: g.index.to_numpy() for sid, g in part.groupby("subject_id")}
    subjects = np.array(list(groups.keys()))
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[sid] for sid in sampled])
        sample = part.loc[idx]
        try:
            values.append(fn(sample["y_true"].values, sample["risk"].values))
        except Exception:
            values.append(np.nan)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"bootstrap_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_std": np.nan, "n_bootstrap_valid": 0}
    return {
        "bootstrap_mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "bootstrap_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "n_bootstrap_valid": int(len(vals)),
    }


def paired_patient_bootstrap_diff(aligned: pd.DataFrame, metric_fn: Callable, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = {sid: g.index.to_numpy() for sid, g in aligned.groupby("subject_id")}
    subjects = np.array(list(groups.keys()))
    diffs = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[sid] for sid in sampled])
        sample = aligned.loc[idx]
        y = sample["y_true"].values
        try:
            diff = metric_fn(y, sample["risk_compressed"].values) - metric_fn(y, sample["risk_raw"].values)
        except Exception:
            diff = np.nan
        diffs.append(diff)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {
            "bootstrap_mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_std_diff": np.nan,
            "p_two_sided_zero_diff": np.nan,
            "n_bootstrap_valid": 0,
        }
    prop_le_zero = float(np.mean(diffs <= 0))
    prop_ge_zero = float(np.mean(diffs >= 0))
    return {
        "bootstrap_mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "bootstrap_std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan,
        "p_two_sided_zero_diff": float(min(1.0, 2 * min(prop_le_zero, prop_ge_zero))),
        "n_bootstrap_valid": int(len(diffs)),
    }


def make_metric_ci_table(pred: pd.DataFrame, fns: dict[str, Callable], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    group_cols = ["task", "version", "model", "seed", "calibration"]
    for key, part in pred.groupby(group_cols):
        meta = dict(zip(group_cols, key))
        for metric, fn in fns.items():
            try:
                observed = fn(part["y_true"].values, part["risk"].values)
            except Exception:
                observed = np.nan
            ci = patient_bootstrap_metric_ci(part, fn, args.n_bootstrap, args.seed)
            rows.append({**meta, "metric": metric, "observed": observed, **ci})
    return pd.DataFrame(rows)


def make_paired_table(pred: pd.DataFrame, fns: dict[str, Callable], args: argparse.Namespace, suffix: str = "") -> pd.DataFrame:
    rows = []
    group_cols = ["task", "model", "seed", "calibration"]
    for key, group in pred.groupby(group_cols):
        meta = dict(zip(group_cols, key))
        raw = group[group["version"] == args.baseline_version].copy()
        if raw.empty:
            continue
        raw = raw[["row_id", "subject_id", "y_true", "risk"]].rename(columns={"risk": "risk_raw"})
        for version, comp in group.groupby("version"):
            if version == args.baseline_version:
                continue
            comp = comp[["row_id", "subject_id", "y_true", "risk"]].rename(columns={"risk": "risk_compressed", "y_true": "y_true_comp"})
            aligned = raw.merge(comp, on=["row_id", "subject_id"], how="inner")
            if aligned.empty:
                continue
            if not (aligned["y_true"] == aligned["y_true_comp"]).all():
                raise ValueError(f"Mismatched labels for {meta}, version={version}")
            for metric, fn in fns.items():
                y = aligned["y_true"].values
                try:
                    raw_value = fn(y, aligned["risk_raw"].values)
                    comp_value = fn(y, aligned["risk_compressed"].values)
                    diff = comp_value - raw_value
                except Exception:
                    raw_value = comp_value = diff = np.nan
                ci = paired_patient_bootstrap_diff(aligned, fn, args.n_bootstrap, args.seed)
                rows.append({
                    **meta,
                    "compressed_version": version,
                    "baseline_version": args.baseline_version,
                    "metric": metric,
                    "higher_is_better": higher_is_better(metric),
                    "baseline_value": raw_value,
                    "compressed_value": comp_value,
                    "observed_diff_compressed_minus_baseline": diff,
                    "n_examples_aligned": int(len(aligned)),
                    "n_patients_aligned": int(aligned["subject_id"].nunique()),
                    "n_positive_aligned": int(aligned["y_true"].sum()),
                    **ci,
                })
    out = pd.DataFrame(rows)
    if suffix and len(out):
        out["subgroup"] = suffix
    return out


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path, "shape=", df.shape)
    return path


def main() -> None:
    args = parse_args()
    clearml_task = maybe_init_clearml(args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.sequence_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fns = metric_functions(args.top_fracs)

    if args.load_from_clearml:
        pred = download_predictions_from_clearml(args)
    else:
        maybe_download_grid_results_from_s3(args)

    pred = load_predictions_from_local(args.results_dir, args.calibration)
    pred = pred[pred["task"].isin(args.tasks)].copy()
    if pred.empty:
        raise ValueError("No predictions left after task/calibration filtering")

    maybe_download_sequence_examples(args, pred)

    # Basic sanity summary.
    sanity = pred.groupby(["task", "version", "model", "calibration"]).agg(
        n_seeds=("seed", "nunique"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        n_rows=("row_id", "size"),
        n_examples=("row_id", "nunique"),
        n_patients=("subject_id", "nunique"),
        n_positive=("y_true", "sum"),
    ).reset_index()
    save_table(sanity, args.output_dir / "prediction_run_sanity.csv")
    print("\nPrediction sanity preview:")
    print(sanity.sort_values(["task", "model", "version"]).to_string(index=False))

    metrics_by_seed = compute_metrics_for_group(pred, fns, ["task", "version", "model", "seed", "calibration"])
    metrics_by_seed_path = save_table(metrics_by_seed, args.output_dir / "metrics_by_seed.csv")

    metric_cols = [c for c in metrics_by_seed.columns if c not in {"task", "version", "model", "seed", "calibration"}]
    rows = []
    for key, group in metrics_by_seed.groupby(["task", "version", "model", "calibration"]):
        row = dict(zip(["task", "version", "model", "calibration"], key))
        row["n_seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group[col].dropna()) > 1 else np.nan
        rows.append(row)
    metrics_mean_std = pd.DataFrame(rows)
    metrics_mean_std_path = save_table(metrics_mean_std, args.output_dir / "metrics_mean_std.csv")

    high_repeat = build_high_repeat_group(args.sequence_dir, args.tasks, args.baseline_version, args.high_repeat_quantile)
    high_repeat_path = save_table(high_repeat, args.output_dir / "high_repeat_group.csv")

    cost = build_sequence_cost_summary(args.sequence_dir, args.tasks, high_repeat)
    cost_path = save_table(cost, args.output_dir / "sequence_cost_summary.csv")

    metric_ci = make_metric_ci_table(pred, fns, args)
    metric_ci_path = save_table(metric_ci, args.output_dir / "patient_bootstrap_metric_ci.csv")

    paired = make_paired_table(pred, fns, args)
    paired_path = save_table(paired, args.output_dir / "paired_bootstrap_vs_raw.csv")

    high_metrics_path = low_metrics_path = paired_high_path = None
    if len(high_repeat):
        pred_h = pred.merge(high_repeat[["task", "row_id", "subject_id", "high_repeat_group"]], on=["task", "row_id", "subject_id"], how="inner")
        pred_high = pred_h[pred_h["high_repeat_group"]].copy()
        pred_low = pred_h[~pred_h["high_repeat_group"]].copy()
        high_metrics = compute_metrics_for_group(pred_high, fns, ["task", "version", "model", "seed", "calibration"])
        high_metrics_path = save_table(high_metrics, args.output_dir / "high_repeat_metrics_by_seed.csv")
        low_metrics = compute_metrics_for_group(pred_low, fns, ["task", "version", "model", "seed", "calibration"])
        low_metrics_path = save_table(low_metrics, args.output_dir / "non_high_repeat_metrics_by_seed.csv")
        paired_high = make_paired_table(pred_high, fns, args, suffix="high_repeat")
        paired_high_path = save_table(paired_high, args.output_dir / "paired_bootstrap_vs_raw_high_repeat.csv")

    print("\nMain mean±std preview:")
    preview_cols = ["task", "version", "model", "calibration", "n_seeds", "auprc_mean", "auprc_std", "auroc_mean", "brier_mean", "logloss_mean"]
    preview_cols = [c for c in preview_cols if c in metrics_mean_std.columns]
    print(metrics_mean_std[preview_cols].sort_values(["task", "auprc_mean"], ascending=[True, False]).head(50).to_string(index=False))

    if len(paired):
        print("\nPaired AUPRC vs raw preview:")
        view = paired[paired["metric"] == "auprc"].copy()
        cols = ["task", "model", "seed", "compressed_version", "observed_diff_compressed_minus_baseline", "ci_low", "ci_high", "n_positive_aligned"]
        print(view[cols].sort_values(["task", "model", "compressed_version", "seed"]).head(80).to_string(index=False))

    if clearml_task is not None:
        artifacts = {
            "prediction_run_sanity": args.output_dir / "prediction_run_sanity.csv",
            "metrics_by_seed": metrics_by_seed_path,
            "metrics_mean_std": metrics_mean_std_path,
            "patient_bootstrap_metric_ci": metric_ci_path,
            "paired_bootstrap_vs_raw": paired_path,
            "sequence_cost_summary": cost_path,
            "high_repeat_group": high_repeat_path,
        }
        if high_metrics_path is not None:
            artifacts["high_repeat_metrics_by_seed"] = high_metrics_path
        if low_metrics_path is not None:
            artifacts["non_high_repeat_metrics_by_seed"] = low_metrics_path
        if paired_high_path is not None:
            artifacts["paired_bootstrap_vs_raw_high_repeat"] = paired_high_path
        for name, path in artifacts.items():
            if path is not None and Path(path).exists():
                clearml_task.upload_artifact(name, artifact_object=path)


if __name__ == "__main__":
    main()
