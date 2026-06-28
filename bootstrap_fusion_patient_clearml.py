from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_COMPARISONS = [
    ("tabular_only", "fusion_tabular_sequence"),
    ("tabular_only", "fusion_tabular_numeric_sequence"),
    ("sequence_only", "fusion_tabular_sequence"),
    ("numeric_sequence_only", "fusion_tabular_numeric_sequence"),
]

METRIC_DIRECTIONS = {
    "auroc": "higher",
    "auprc": "higher",
    "brier": "lower",
    "logloss": "lower",
    "top_5pct_precision": "higher",
    "top_10pct_precision": "higher",
    "top_20pct_precision": "higher",
}


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def get_required_local_copy(remote_url: str) -> Path:
    from clearml import StorageManager

    local_copy = StorageManager.get_local_copy(remote_url=remote_url)

    if local_copy is None:
        raise FileNotFoundError(
            "ClearML StorageManager could not download object.\n"
            f"Remote URL not found or not accessible:\n{remote_url}"
        )

    path = Path(local_copy)

    if not path.exists():
        raise FileNotFoundError(
            "ClearML StorageManager returned a path, but it does not exist.\n"
            f"Remote URL: {remote_url}\n"
            f"Local path: {path}"
        )

    return path


def try_get_local_copy(remote_url: str) -> Path | None:
    from clearml import StorageManager

    try:
        local_copy = StorageManager.get_local_copy(remote_url=remote_url)
    except Exception as e:
        print(f"Could not download {remote_url}: {e}")
        return None

    if local_copy is None:
        return None

    path = Path(local_copy)
    if not path.exists():
        return None

    return path


def copy_to_workdir(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.resolve() == dst.resolve():
        return dst

    if src.is_dir():
        raise ValueError(f"Expected file, got directory: {src}")

    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)
    return dst


def find_local_prediction_file(input_dir: Path, stem: str) -> Path | None:
    candidates = [
        input_dir / f"{stem}.csv.gz",
        input_dir / f"{stem}.csv",
        input_dir / f"{stem}.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = []
    for suffix in ["csv.gz", "csv", "parquet"]:
        matches.extend(sorted(input_dir.rglob(f"{stem}.{suffix}")))

    matches = list(dict.fromkeys(matches))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Found multiple local files for {stem}:\n"
            + "\n".join(str(x) for x in matches)
        )

    return None


def get_artifact_from_source_task(source_task_id: str, artifact_name: str) -> Path | None:
    if not source_task_id:
        return None

    from clearml import Task

    source_task = Task.get_task(task_id=source_task_id)

    if artifact_name not in source_task.artifacts:
        print(
            f"Artifact {artifact_name!r} not found in source task {source_task_id}. "
            f"Available artifacts: {list(source_task.artifacts)}"
        )
        return None

    artifact = source_task.artifacts[artifact_name]
    local_copy = artifact.get_local_copy()

    if local_copy is None:
        return None

    path = Path(local_copy)

    if not path.exists():
        return None

    return path


def resolve_prediction_file(
    *,
    stem: str,
    artifact_name: str,
    local_path: str,
    direct_url: str,
    input_dir: Path,
    input_s3_prefix: str,
    source_task_id: str,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)

    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local prediction file does not exist: {path}")
        return path

    local_found = find_local_prediction_file(input_dir, stem)
    if local_found is not None:
        print(f"Using local prediction file: {local_found}")
        return local_found

    if direct_url:
        local_copy = get_required_local_copy(direct_url)
        suffix = ".csv.gz" if direct_url.endswith(".csv.gz") else Path(direct_url).suffix
        if suffix == "":
            suffix = ".csv.gz"
        return copy_to_workdir(local_copy, work_dir / f"{stem}{suffix}")

    artifact_path = get_artifact_from_source_task(source_task_id, artifact_name)
    if artifact_path is not None:
        print(f"Using prediction artifact from source task: {artifact_name} -> {artifact_path}")
        return artifact_path

    if input_s3_prefix:
        prefix = input_s3_prefix.rstrip("/")
        for suffix in ["csv.gz", "csv", "parquet"]:
            remote_url = f"{prefix}/{stem}.{suffix}"
            print(f"Trying prediction file: {remote_url}")
            local_copy = try_get_local_copy(remote_url)
            if local_copy is not None:
                return copy_to_workdir(local_copy, work_dir / f"{stem}.{suffix}")

    raise FileNotFoundError(
        f"Could not resolve prediction file {stem}. Provide one of:\n"
        f"  --{stem.replace('_', '-')}-path\n"
        f"  --{stem.replace('_', '-')}-url\n"
        f"  --source-task-id\n"
        f"  --input-dir\n"
        f"  --input-s3-prefix"
    )


def read_predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    print(f"Reading predictions: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"task", "variant", "row_id", "subject_id", "y_true", "risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"] == "held_out"].copy()

    df["row_id"] = df["row_id"].astype(int)
    df["subject_id"] = df["subject_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["risk"] = df["risk"].astype(float)

    if "seed" in df.columns:
        df["seed"] = df["seed"].astype(int)

    print("Predictions shape:", df.shape)
    print(
        df.groupby(["task", "variant"])
        .agg(n=("row_id", "size"), n_positive=("y_true", "sum"), event_rate=("y_true", "mean"))
        .reset_index()
        .to_string(index=False)
    )

    return df


def upload_file_to_s3(local_path: Path, root_dir: Path, s3_prefix: str) -> None:
    if not s3_prefix:
        return

    from clearml import StorageManager

    local_path = Path(local_path)
    root_dir = Path(root_dir)
    rel = local_path.relative_to(root_dir).as_posix()
    remote_url = f"{s3_prefix.rstrip('/')}/{rel}"

    print(f"Uploading {local_path} -> {remote_url}")
    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )


# --------------------------------------------------------------------------------------
# Metrics and bootstrap
# --------------------------------------------------------------------------------------


def clip_prob(y_prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)


def topk_precision(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    if n == 0:
        return np.nan

    k = max(1, int(np.ceil(n * frac)))
    order = np.argsort(-y_prob)
    idx = order[:k]
    return float(y_true[idx].mean())


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_prob(y_prob)

    has_two_classes = len(np.unique(y_true)) > 1

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_two_classes else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if has_two_classes else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "top_5pct_precision": topk_precision(y_true, y_prob, 0.05),
        "top_10pct_precision": topk_precision(y_true, y_prob, 0.10),
        "top_20pct_precision": topk_precision(y_true, y_prob, 0.20),
    }


def make_paired_prediction_frame(
    pred_df: pd.DataFrame,
    *,
    task: str,
    baseline_variant: str,
    candidate_variant: str,
    seed: int | None = None,
) -> pd.DataFrame:
    part = pred_df[pred_df["task"] == task].copy()

    if seed is not None:
        if "seed" not in part.columns:
            raise ValueError("seed was requested, but prediction dataframe has no seed column")
        part = part[part["seed"] == seed].copy()

    base = part[part["variant"] == baseline_variant].copy()
    cand = part[part["variant"] == candidate_variant].copy()

    if base.empty:
        raise ValueError(f"No rows for baseline variant={baseline_variant}, task={task}, seed={seed}")
    if cand.empty:
        raise ValueError(f"No rows for candidate variant={candidate_variant}, task={task}, seed={seed}")

    key = ["task", "row_id", "subject_id", "y_true"]
    if seed is not None:
        key = ["task", "seed", "row_id", "subject_id", "y_true"]

    base = base[key + ["risk"]].rename(columns={"risk": "risk_baseline"})
    cand = cand[key + ["risk"]].rename(columns={"risk": "risk_candidate"})

    paired = base.merge(cand, on=key, how="inner")

    if paired.empty:
        raise ValueError(
            f"Paired merge produced 0 rows for task={task}, "
            f"{baseline_variant} vs {candidate_variant}, seed={seed}"
        )

    return paired


def make_subject_index_groups(subject_ids: np.ndarray) -> dict[int, np.ndarray]:
    groups = {}
    subject_ids = np.asarray(subject_ids)

    for subject_id in np.unique(subject_ids):
        groups[int(subject_id)] = np.flatnonzero(subject_ids == subject_id)

    return groups


def bootstrap_one_comparison(
    paired: pd.DataFrame,
    *,
    task: str,
    baseline_variant: str,
    candidate_variant: str,
    n_boot: int,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = paired["y_true"].astype(int).to_numpy()
    risk_base = paired["risk_baseline"].astype(float).to_numpy()
    risk_cand = paired["risk_candidate"].astype(float).to_numpy()
    subject_ids = paired["subject_id"].astype(int).to_numpy()

    subjects = np.array(sorted(np.unique(subject_ids)), dtype=int)
    groups = make_subject_index_groups(subject_ids)

    obs_base = compute_metrics(y, risk_base)
    obs_cand = compute_metrics(y, risk_cand)

    sample_rows = []

    for boot_id in range(n_boot):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[int(s)] for s in sampled_subjects])

        y_b = y[idx]
        base_b = risk_base[idx]
        cand_b = risk_cand[idx]

        m_base = compute_metrics(y_b, base_b)
        m_cand = compute_metrics(y_b, cand_b)

        for metric, direction in METRIC_DIRECTIONS.items():
            baseline_value = m_base[metric]
            candidate_value = m_cand[metric]
            delta = candidate_value - baseline_value

            if direction == "higher":
                improvement = delta
            else:
                improvement = -delta

            sample_rows.append(
                {
                    "task": task,
                    "seed": seed if seed is not None else -1,
                    "bootstrap_id": boot_id,
                    "baseline_variant": baseline_variant,
                    "candidate_variant": candidate_variant,
                    "metric": metric,
                    "direction": direction,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "delta_candidate_minus_baseline": delta,
                    "improvement": improvement,
                    "n_rows": int(len(idx)),
                    "n_subjects_sampled": int(len(sampled_subjects)),
                    "n_unique_subjects_sampled": int(len(np.unique(sampled_subjects))),
                    "n_positive": int(y_b.sum()),
                    "event_rate": float(y_b.mean()),
                }
            )

    samples = pd.DataFrame(sample_rows)

    summary_rows = []

    for metric, direction in METRIC_DIRECTIONS.items():
        sub = samples[samples["metric"] == metric].copy()
        improvements = sub["improvement"].to_numpy(dtype=float)
        deltas = sub["delta_candidate_minus_baseline"].to_numpy(dtype=float)

        improvements = improvements[np.isfinite(improvements)]
        deltas = deltas[np.isfinite(deltas)]

        obs_delta = obs_cand[metric] - obs_base[metric]
        obs_improvement = obs_delta if direction == "higher" else -obs_delta

        if len(improvements) == 0:
            ci_low = ci_high = p_two_sided = prob_improvement = np.nan
            delta_ci_low = delta_ci_high = np.nan
        else:
            ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])
            delta_ci_low, delta_ci_high = np.percentile(deltas, [2.5, 97.5])
            prob_improvement = float(np.mean(improvements > 0.0))
            p_left = float(np.mean(improvements <= 0.0))
            p_right = float(np.mean(improvements >= 0.0))
            p_two_sided = min(1.0, 2.0 * min(p_left, p_right))

        summary_rows.append(
            {
                "task": task,
                "seed": seed if seed is not None else -1,
                "baseline_variant": baseline_variant,
                "candidate_variant": candidate_variant,
                "metric": metric,
                "direction": direction,
                "n_rows": int(len(paired)),
                "n_subjects": int(len(subjects)),
                "n_positive": int(y.sum()),
                "event_rate": float(y.mean()),
                "baseline_observed": obs_base[metric],
                "candidate_observed": obs_cand[metric],
                "delta_candidate_minus_baseline_observed": obs_delta,
                "delta_ci_low": delta_ci_low,
                "delta_ci_high": delta_ci_high,
                "improvement_observed": obs_improvement,
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "prob_improvement": prob_improvement,
                "p_two_sided_improvement": p_two_sided,
                "n_boot": int(n_boot),
            }
        )

    return pd.DataFrame(summary_rows), samples


def parse_comparisons(comparisons_arg: str) -> list[tuple[str, str]]:
    if not comparisons_arg.strip():
        return DEFAULT_COMPARISONS

    out = []
    for item in comparisons_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Custom comparisons must have format baseline:candidate,baseline:candidate"
            )
        baseline, candidate = item.split(":", 1)
        out.append((baseline.strip(), candidate.strip()))
    return out


# --------------------------------------------------------------------------------------
# ClearML
# --------------------------------------------------------------------------------------


def build_clearml_config(args) -> dict:
    cfg = vars(args).copy()
    for key in ["input_dir", "output_dir", "work_dir"]:
        if key in cfg:
            cfg[key] = str(cfg[key])
    return cfg


def _to_bool(x):
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def sync_args_from_clearml_config(args, cfg: dict) -> None:
    path_keys = {"input_dir", "output_dir", "work_dir"}
    int_keys = {"n_boot", "random_seed"}
    bool_keys = {"save_bootstrap_samples"}
    skip_keys = {"enable_clearml", "execute_remotely"}

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


def maybe_init_clearml(args, config: dict):
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

    connected_cfg = dict(task.connect(config))
    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  mode = {args.mode}")
    print(f"  tasks = {args.tasks}")
    print(f"  comparisons = {args.comparisons}")
    print(f"  n_boot = {args.n_boot}")
    print(f"  input_dir = {args.input_dir}")
    print(f"  input_s3_prefix = {args.input_s3_prefix}")
    print(f"  source_task_id = {args.source_task_id}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  results_s3_prefix = {args.results_s3_prefix}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def run_bootstrap_for_predictions(
    pred_df: pd.DataFrame,
    *,
    tasks: list[str],
    comparisons: list[tuple[str, str]],
    n_boot: int,
    random_seed: int,
    is_single_seed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)

    all_summary = []
    all_samples = []

    for task in tasks:
        task_df = pred_df[pred_df["task"] == task].copy()
        if task_df.empty:
            raise ValueError(f"No predictions for task={task}")

        seeds: list[int | None]
        if is_single_seed:
            if "seed" not in task_df.columns:
                raise ValueError("single_seed mode requires seed column")
            seeds = sorted(task_df["seed"].dropna().astype(int).unique().tolist())
        else:
            seeds = [None]

        for seed in seeds:
            for baseline_variant, candidate_variant in comparisons:
                print("=" * 100)
                print(
                    f"Bootstrap: task={task}, seed={seed}, "
                    f"{baseline_variant} vs {candidate_variant}"
                )

                paired = make_paired_prediction_frame(
                    task_df,
                    task=task,
                    baseline_variant=baseline_variant,
                    candidate_variant=candidate_variant,
                    seed=seed,
                )

                summary, samples = bootstrap_one_comparison(
                    paired,
                    task=task,
                    baseline_variant=baseline_variant,
                    candidate_variant=candidate_variant,
                    n_boot=n_boot,
                    rng=rng,
                    seed=seed,
                )

                all_summary.append(summary)
                all_samples.append(samples)

    return pd.concat(all_summary, ignore_index=True), pd.concat(all_samples, ignore_index=True)


def make_single_seed_across_seed_summary(single_seed_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["task", "baseline_variant", "candidate_variant", "metric", "direction"]
    value_cols = [
        "baseline_observed",
        "candidate_observed",
        "delta_candidate_minus_baseline_observed",
        "improvement_observed",
        "prob_improvement",
        "p_two_sided_improvement",
    ]

    out = (
        single_seed_summary
        .groupby(group_cols)[value_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    out.columns = [
        "__".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
        for col in out.columns
    ]

    counts = (
        single_seed_summary
        .groupby(group_cols)
        .agg(n_seeds=("seed", "nunique"))
        .reset_index()
    )

    return counts.merge(out, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level paired bootstrap for score-level fusion predictions."
    )

    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--input-s3-prefix", type=str, default="")
    parser.add_argument("--source-task-id", type=str, default="")

    parser.add_argument("--fusion-ensemble-predictions-path", type=str, default="")
    parser.add_argument("--fusion-single-seed-predictions-path", type=str, default="")
    parser.add_argument("--fusion-ensemble-predictions-url", type=str, default="")
    parser.add_argument("--fusion-single-seed-predictions-url", type=str, default="")

    parser.add_argument("--output-dir", type=Path, default=Path("fusion_patient_bootstrap"))
    parser.add_argument("--work-dir", type=Path, default=Path("bootstrap_inputs"))
    parser.add_argument("--results-s3-prefix", type=str, default="")

    parser.add_argument("--mode", type=str, default="both", choices=["ensemble", "single_seed", "both"])
    parser.add_argument("--tasks", type=str, default="guo_readmission,guo_icu")
    parser.add_argument("--comparisons", type=str, default="")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260628)
    parser.add_argument("--save-bootstrap-samples", action="store_true")

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="fusion_patient_bootstrap",
    )
    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )

    args = parser.parse_args()

    clearml_task = maybe_init_clearml(args, build_clearml_config(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    comparisons = parse_comparisons(args.comparisons)

    print("TASKS:", tasks)
    print("COMPARISONS:", comparisons)
    print("N_BOOT:", args.n_boot)

    output_paths = []

    if args.mode in {"ensemble", "both"}:
        ensemble_path = resolve_prediction_file(
            stem="fusion_ensemble_predictions",
            artifact_name="fusion_ensemble_predictions",
            local_path=args.fusion_ensemble_predictions_path,
            direct_url=args.fusion_ensemble_predictions_url,
            input_dir=args.input_dir,
            input_s3_prefix=args.input_s3_prefix,
            source_task_id=args.source_task_id,
            work_dir=args.work_dir,
        )

        ensemble_pred = read_predictions(ensemble_path)

        ens_summary, ens_samples = run_bootstrap_for_predictions(
            ensemble_pred,
            tasks=tasks,
            comparisons=comparisons,
            n_boot=args.n_boot,
            random_seed=args.random_seed,
            is_single_seed=False,
        )

        ens_summary_path = args.output_dir / "patient_bootstrap_ensemble_summary.csv"
        ens_samples_path = args.output_dir / "patient_bootstrap_ensemble_samples.csv.gz"

        ens_summary.to_csv(ens_summary_path, index=False)
        output_paths.append(ens_summary_path)

        if args.save_bootstrap_samples:
            ens_samples.to_csv(ens_samples_path, index=False, compression="gzip")
            output_paths.append(ens_samples_path)

        print("\nEnsemble bootstrap summary:")
        print(
            ens_summary.sort_values(
                ["task", "candidate_variant", "metric"]
            ).to_string(index=False)
        )

        if clearml_task is not None:
            clearml_task.upload_artifact("patient_bootstrap_ensemble_summary", ens_summary)
            if args.save_bootstrap_samples:
                clearml_task.upload_artifact("patient_bootstrap_ensemble_samples", ens_samples_path)

    if args.mode in {"single_seed", "both"}:
        single_path = resolve_prediction_file(
            stem="fusion_single_seed_predictions",
            artifact_name="fusion_single_seed_predictions",
            local_path=args.fusion_single_seed_predictions_path,
            direct_url=args.fusion_single_seed_predictions_url,
            input_dir=args.input_dir,
            input_s3_prefix=args.input_s3_prefix,
            source_task_id=args.source_task_id,
            work_dir=args.work_dir,
        )

        single_pred = read_predictions(single_path)

        single_summary, single_samples = run_bootstrap_for_predictions(
            single_pred,
            tasks=tasks,
            comparisons=comparisons,
            n_boot=args.n_boot,
            random_seed=args.random_seed + 17,
            is_single_seed=True,
        )

        single_summary_path = args.output_dir / "patient_bootstrap_single_seed_summary.csv"
        single_samples_path = args.output_dir / "patient_bootstrap_single_seed_samples.csv.gz"
        across_seed_summary_path = args.output_dir / "patient_bootstrap_single_seed_across_seed_summary.csv"

        across_seed_summary = make_single_seed_across_seed_summary(single_summary)

        single_summary.to_csv(single_summary_path, index=False)
        across_seed_summary.to_csv(across_seed_summary_path, index=False)

        output_paths.extend([single_summary_path, across_seed_summary_path])

        if args.save_bootstrap_samples:
            single_samples.to_csv(single_samples_path, index=False, compression="gzip")
            output_paths.append(single_samples_path)

        print("\nSingle-seed bootstrap summary:")
        print(
            single_summary.sort_values(
                ["task", "seed", "candidate_variant", "metric"]
            ).to_string(index=False)
        )

        print("\nSingle-seed across-seed summary:")
        print(
            across_seed_summary.sort_values(
                ["task", "candidate_variant", "metric"]
            ).to_string(index=False)
        )

        if clearml_task is not None:
            clearml_task.upload_artifact("patient_bootstrap_single_seed_summary", single_summary)
            clearml_task.upload_artifact("patient_bootstrap_single_seed_across_seed_summary", across_seed_summary)
            if args.save_bootstrap_samples:
                clearml_task.upload_artifact("patient_bootstrap_single_seed_samples", single_samples_path)

    for path in output_paths:
        upload_file_to_s3(path, args.output_dir, args.results_s3_prefix)

    print("\nSaved outputs to:", args.output_dir)
    for path in output_paths:
        print("  ", path)


if __name__ == "__main__":
    main()
