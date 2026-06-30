from __future__ import annotations

"""
04_compare_repro_results.py

Назначение:
    Единый финальный comparison script для reproducibility package.

Что делает:
    1. Читает configs/comparison_final_runs.json.
    2. Загружает tabular / sequence / numeric_sequence predictions.
    3. Нормализует schema в один формат.
    4. Фильтрует primary_models из JSON.
    5. Считает:
        - all_predictions_normalized.csv
        - input_manifest.csv
        - metrics_by_seed.csv
        - metrics_mean_std.csv
        - topk_by_seed.csv
        - high_repeat_group.csv
        - high_repeat_thresholds.csv
        - sequence_cost_summary.csv
        - subgroup_metrics_by_seed.csv
        - subgroup_metrics_mean_std.csv
        - ensemble_mean_predictions.csv
        - ensemble_metrics.csv
        - prediction_stability_by_example.csv
        - prediction_stability_summary.csv
        - patient_bootstrap_metric_ci.csv
        - paired_bootstrap_delta.csv
        - huly_compact_results.csv
        - huly_paired_auprc_delta_table.csv
    6. При необходимости загружает всю output папку в MinIO.

Пример:
    python code/04_compare_repro_results.py \\
      --comparison-config configs/comparison_final_runs.json \\
      --output-dir ehrshot_final_comparison \\
      --enable-clearml \\
      --execute-remotely \\
      --clearml-queue cpu
"""

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_ehrshot_eval import (
    DEFAULT_METRICS,
    DEFAULT_TOP_FRACS,
    add_high_repeat_group,
    build_high_repeat_group_from_sequence_examples,
    build_high_repeat_patient_list,
    build_sequence_cost_summary,
    compute_metrics_by_seed,
    compute_metrics_for_group,
    compute_topk_by_seed,
    ensemble_predictions,
    filter_predictions_by_model_configs,
    get_model_config_by_key,
    load_predictions_from_path_or_dir,
    make_patient_bootstrap_metric_ci_table,
    paired_patient_bootstrap_delta,
    prediction_stability_summary,
    read_json,
    safe_path_part,
    select_model_predictions,
    summarize_metrics_mean_std,
    upload_tree_to_minio,
    write_json,
)


DEFAULT_PROJECT = "pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
DEFAULT_OUTPUT_URI = "s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"

DEFAULT_OUTPUT_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "final_comparison"
)


# -----------------------------------------------------------------------------
# 1. CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/comparison_final_runs.json"),
        help="JSON config со списком prediction sources, primary models и pairwise comparisons.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_final_comparison"),
        help="Куда сохранить итоговые comparison CSV.",
    )

    parser.add_argument(
        "--output-s3-prefix",
        type=str,
        default="",
        help=(
            "MinIO/S3 prefix для загрузки итоговых comparison CSV. "
            "Если пусто — upload не выполняется."
        ),
    )

    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Не загружать output-dir в MinIO.",
    )

    parser.add_argument(
        "--enable-clearml",
        action="store_true",
        help="Инициализировать ClearML task.",
    )

    parser.add_argument(
        "--execute-remotely",
        action="store_true",
        help="Поставить задачу в ClearML очередь и завершить локальный процесс.",
    )

    parser.add_argument(
        "--clearml-queue",
        type=str,
        default="cpu",
        help="ClearML queue.",
    )

    parser.add_argument(
        "--clearml-project",
        type=str,
        default=DEFAULT_PROJECT,
        help="ClearML project.",
    )

    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="final_repro_comparison",
        help="ClearML task name.",
    )

    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default=DEFAULT_OUTPUT_URI,
        help="ClearML output URI.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 2. ClearML
# -----------------------------------------------------------------------------

def is_clearml_agent_run() -> bool:
    return bool(
        os.environ.get("CLEARML_TASK_ID")
        or os.environ.get("TRAINS_TASK_ID")
    )


def _to_bool(x: Any) -> bool:
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def build_clearml_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    out = vars(args).copy()

    for key in ["comparison_config", "output_dir"]:
        out[key] = str(out[key])

    out["comparison_config_json"] = config

    return out


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path_keys = {
        "comparison_config",
        "output_dir",
    }

    bool_keys = {
        "skip_upload",
    }

    skip_keys = {
        "enable_clearml",
        "execute_remotely",
        "comparison_config_json",
    }

    for key, value in dict(cfg).items():
        if key in skip_keys:
            continue

        if not hasattr(args, key):
            continue

        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in bool_keys:
            setattr(args, key, _to_bool(value))
        else:
            setattr(args, key, value)


def maybe_init_clearml(args: argparse.Namespace, config: dict[str, Any]):
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

    connected_cfg = dict(
        task.connect(
            build_clearml_config(args, config),
        )
    )

    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task_id = {task.id}")
    print(f"  comparison_config = {args.comparison_config}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  output_s3_prefix = {args.output_s3_prefix}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")

        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task


# -----------------------------------------------------------------------------
# 3. Loading predictions
# -----------------------------------------------------------------------------

def load_prediction_sources(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает prediction_sources из JSON config.
    """
    sources = config.get("prediction_sources", [])

    if not sources:
        raise ValueError("comparison config has empty prediction_sources")

    parts = []
    manifest_rows = []

    for source in sources:
        name = source["name"]
        path = Path(source["path"])

        default_model_family = source.get("model_family")

        print("=" * 100)
        print(f"Loading prediction source: {name}")
        print(f"path={path}")

        df = load_predictions_from_path_or_dir(
            path=path,
            default_model_family=default_model_family,
            source_name=name,
        )

        parts.append(df)

        manifest_rows.append(
            {
                "source_name": name,
                "path": str(path),
                "model_family_default": default_model_family or "",
                "n_rows": int(len(df)),
                "n_tasks": int(df["task"].nunique()),
                "n_models": int(
                    df[
                        [
                            "task",
                            "model_family",
                            "model_name",
                            "compression_version",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_seeds": int(df["seed"].nunique()),
                "calibrations": ",".join(sorted(df["calibration"].unique())),
            }
        )

    pred = pd.concat(parts, ignore_index=True)

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

    before = len(pred)
    pred = pred.drop_duplicates(subset=dedup_key, keep="last").reset_index(drop=True)
    after = len(pred)

    if before != after:
        print(f"Dropped duplicated rows after concatenating sources: {before - after}")

    manifest = pd.DataFrame(manifest_rows)

    return pred, manifest


def filter_predictions_from_config(
    pred: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Применяет calibration/split/tasks/primary_models фильтры из config.
    """
    out = pred.copy()

    calibration = config.get("calibration", "platt")

    if calibration != "all":
        out = out[out["calibration"] == calibration].copy()

    split = config.get("split", "held_out")

    if split != "all":
        out = out[out["split"] == split].copy()

    tasks = config.get("tasks", [])

    if tasks:
        out = out[out["task"].isin(tasks)].copy()

    primary_models = config.get("primary_models", [])

    if primary_models:
        out = filter_predictions_by_model_configs(out, primary_models)

    if out.empty:
        raise ValueError("No predictions left after config filtering.")

    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 4. Pairwise comparisons
# -----------------------------------------------------------------------------

def run_pairwise_comparisons(
    ensemble_pred: pd.DataFrame,
    config: dict[str, Any],
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
    cluster_col: str,
    subgroup_col: str | None = None,
) -> pd.DataFrame:
    """
    Считает pairwise patient-level bootstrap comparisons.

    Если subgroup_col задан, сравнения считаются отдельно по subgroup.
    """
    model_configs = config.get("primary_models", [])
    comparisons = config.get("pairwise_comparisons", [])

    if not comparisons:
        return pd.DataFrame()

    rows = []

    if subgroup_col and subgroup_col in ensemble_pred.columns:
        subgroup_values = list(ensemble_pred[subgroup_col].dropna().unique())
    else:
        subgroup_values = [None]

    for cmp_cfg in comparisons:
        model_a_cfg = get_model_config_by_key(model_configs, cmp_cfg["a"])
        model_b_cfg = get_model_config_by_key(model_configs, cmp_cfg["b"])

        a_all = select_model_predictions(ensemble_pred, model_a_cfg)
        b_all = select_model_predictions(ensemble_pred, model_b_cfg)

        for subgroup_value in subgroup_values:
            if subgroup_value is None:
                a = a_all.copy()
                b = b_all.copy()
                subgroup_name = "all"
            else:
                a = a_all[a_all[subgroup_col] == subgroup_value].copy()
                b = b_all[b_all[subgroup_col] == subgroup_value].copy()

                if "group_name" in a.columns and len(a):
                    subgroup_name = str(a["group_name"].iloc[0])
                else:
                    subgroup_name = str(subgroup_value)

            if a.empty or b.empty:
                print(
                    f"WARNING: empty pairwise subset for "
                    f"{cmp_cfg['comparison']} subgroup={subgroup_name}"
                )
                continue

            print("=" * 100)
            print(f"Pairwise comparison: {cmp_cfg['comparison']}")
            print(f"subgroup={subgroup_name}")
            print(f"a={cmp_cfg['a']}")
            print(f"b={cmp_cfg['b']}")

            delta = paired_patient_bootstrap_delta(
                df_a=a,
                df_b=b,
                metrics=metrics,
                n_bootstrap=n_bootstrap,
                seed=seed,
                cluster_col=cluster_col,
            )

            for _, r in delta.iterrows():
                rows.append(
                    {
                        "comparison": cmp_cfg["comparison"],
                        "description": cmp_cfg.get("description", ""),
                        "task": cmp_cfg.get("task", model_a_cfg.get("task", "")),
                        "subgroup": subgroup_name,
                        "model_a_key": cmp_cfg["a"],
                        "model_b_key": cmp_cfg["b"],
                        "model_a": format_model_label(model_a_cfg),
                        "model_b": format_model_label(model_b_cfg),
                        **r.to_dict(),
                    }
                )

    return pd.DataFrame(rows)


def format_model_label(model_cfg: dict[str, Any]) -> str:
    """
    Читабельное имя модели для Huly/report tables.
    """
    family = model_cfg.get("model_family", "")
    model = model_cfg.get("model_name", "")
    version = model_cfg.get("compression_version", "")

    return f"{family}: {version} + {model}"


# -----------------------------------------------------------------------------
# 5. Huly tables
# -----------------------------------------------------------------------------

def make_metric_mean_std_string(
    row: pd.Series,
    metric: str,
    digits: int = 3,
) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in row:
        return ""

    mean = row[mean_col]
    std = row.get(std_col, np.nan)

    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_huly_compact_results(
    metrics_mean_std: pd.DataFrame,
    primary_models: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Compact table для Huly:
        Task / Model / AUROC / AUPRC / Brier / LogLoss / top-10%.
    """
    rows = []

    for cfg in primary_models:
        part = select_model_predictions_like_metrics(metrics_mean_std, cfg)

        if part.empty:
            continue

        # Если вдруг несколько calibration, берем первую строку после сортировки.
        part = part.sort_values(["task", "model_family", "model_name"]).copy()
        row = part.iloc[0]

        rows.append(
            {
                "task": cfg.get("task", row["task"]),
                "model_key": cfg.get("key", ""),
                "model": format_model_label(cfg),
                "n_seeds": int(row.get("n_seeds", 0)),
                "n": int(row.get("n_mean", 0)) if "n_mean" in row else "",
                "n_positive": int(row.get("n_positive_mean", 0)) if "n_positive_mean" in row else "",
                "event_rate": (
                    float(row["event_rate_mean"])
                    if "event_rate_mean" in row and not pd.isna(row["event_rate_mean"])
                    else np.nan
                ),
                "AUROC": make_metric_mean_std_string(row, "auroc"),
                "AUPRC": make_metric_mean_std_string(row, "auprc"),
                "Brier ↓": make_metric_mean_std_string(row, "brier"),
                "LogLoss ↓": make_metric_mean_std_string(row, "logloss"),
                "Top-10% precision": make_metric_mean_std_string(row, "top_10pct_precision"),
            }
        )

    return pd.DataFrame(rows)


def select_model_predictions_like_metrics(
    df: pd.DataFrame,
    model_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Фильтрует metrics_mean_std по model config.
    """
    out = df.copy()

    for col in [
        "task",
        "model_family",
        "model_name",
        "representation",
        "compression_version",
        "numeric_on",
    ]:
        if col in model_cfg and model_cfg[col] is not None and col in out.columns:
            out = out[out[col] == model_cfg[col]]

    return out


def make_huly_paired_auprc_table(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Huly table только по AUPRC deltas.
    """
    if paired.empty:
        return pd.DataFrame()

    auprc = paired[paired["metric"] == "auprc"].copy()

    if auprc.empty:
        return pd.DataFrame()

    auprc["delta_auprc_with_ci"] = auprc.apply(
        lambda r: (
            f"{r['point_delta_a_minus_b']:+.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        ),
        axis=1,
    )

    cols = [
        "task",
        "subgroup",
        "comparison",
        "model_a",
        "model_b",
        "bootstrap_unit",
        "n_paired_patients",
        "n_paired_examples",
        "n_paired_positive",
        "delta_auprc_with_ci",
    ]

    cols = [c for c in cols if c in auprc.columns]

    return auprc[cols].copy()


# -----------------------------------------------------------------------------
# 6. Save helpers
# -----------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Saved: {path} shape={df.shape}")

    return path


def save_outputs_to_clearml(
    clearml_task,
    outputs: dict[str, Path],
) -> None:
    if clearml_task is None:
        return

    for name, path in outputs.items():
        path = Path(path)

        if path.exists():
            clearml_task.upload_artifact(name, artifact_object=path)


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config = read_json(args.comparison_config)

    clearml_task = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_set = config.get("run_set", "final_repro_comparison_v1")

    metrics = config.get("metrics", DEFAULT_METRICS)
    top_fracs = config.get("top_fracs", DEFAULT_TOP_FRACS)

    bootstrap_cfg = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap_cfg.get("n_bootstrap", 1000))
    bootstrap_seed = int(bootstrap_cfg.get("seed", 42))
    cluster_col = bootstrap_cfg.get("cluster_col", "subject_id")

    primary_models = config.get("primary_models", [])

    print("=" * 100)
    print("FINAL REPRO COMPARISON")
    print(f"run_set={run_set}")
    print(f"output_dir={args.output_dir}")
    print(f"metrics={metrics}")
    print(f"top_fracs={top_fracs}")
    print(f"n_bootstrap={n_bootstrap}")
    print(f"cluster_col={cluster_col}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 1. Load and normalize predictions
    # -------------------------------------------------------------------------

    raw_pred, input_manifest = load_prediction_sources(config)
    pred = filter_predictions_from_config(raw_pred, config)

    outputs: dict[str, Path] = {}

    outputs["input_manifest"] = save_table(
        input_manifest,
        args.output_dir / "input_manifest.csv",
    )

    outputs["all_predictions_normalized"] = save_table(
        pred,
        args.output_dir / "all_predictions_normalized.csv",
    )

    write_json(
        config,
        args.output_dir / "comparison_config_used.json",
    )

    # -------------------------------------------------------------------------
    # 2. Overall metrics by seed
    # -------------------------------------------------------------------------

    metrics_by_seed = compute_metrics_by_seed(
        pred,
        metrics=metrics,
    )

    outputs["metrics_by_seed"] = save_table(
        metrics_by_seed,
        args.output_dir / "metrics_by_seed.csv",
    )

    metrics_mean_std = summarize_metrics_mean_std(
        metrics_by_seed,
        metrics=metrics,
    )

    outputs["metrics_mean_std"] = save_table(
        metrics_mean_std,
        args.output_dir / "metrics_mean_std.csv",
    )

    topk_by_seed = compute_topk_by_seed(
        pred,
        top_fracs=top_fracs,
    )

    outputs["topk_by_seed"] = save_table(
        topk_by_seed,
        args.output_dir / "topk_by_seed.csv",
    )

    # -------------------------------------------------------------------------
    # 3. High-repeat and sequence cost
    # -------------------------------------------------------------------------

    high_repeat = pd.DataFrame()
    high_repeat_thresholds = pd.DataFrame()
    subgroup_pred = pred.copy()

    sequence_dataset_dir = config.get("sequence_dataset_dir", "")

    if sequence_dataset_dir:
        high_repeat_cfg = config.get("high_repeat", {})

        high_repeat, high_repeat_thresholds = build_high_repeat_group_from_sequence_examples(
            sequence_dataset_dir=Path(sequence_dataset_dir),
            tasks=config.get("tasks", sorted(pred["task"].unique())),
            baseline_version=high_repeat_cfg.get("baseline_version", "raw"),
            repeat_reference_version=high_repeat_cfg.get(
                "repeat_reference_version",
                "compressed_first_last",
            ),
            q=float(high_repeat_cfg.get("q", 0.90)),
        )

        outputs["high_repeat_group"] = save_table(
            high_repeat,
            args.output_dir / "high_repeat_group.csv",
        )

        outputs["high_repeat_thresholds"] = save_table(
            high_repeat_thresholds,
            args.output_dir / "high_repeat_thresholds.csv",
        )

        high_repeat_patients = build_high_repeat_patient_list(high_repeat)

        outputs["high_repeat_patients"] = save_table(
            high_repeat_patients,
            args.output_dir / "high_repeat_patients.csv",
        )

        if len(high_repeat):
            subgroup_pred = add_high_repeat_group(pred, high_repeat)

            outputs["all_predictions_with_high_repeat"] = save_table(
                subgroup_pred,
                args.output_dir / "all_predictions_with_high_repeat.csv",
            )

        sequence_cost = build_sequence_cost_summary(
            sequence_dataset_dir=Path(sequence_dataset_dir),
            tasks=config.get("tasks", sorted(pred["task"].unique())),
            high_repeat=high_repeat if len(high_repeat) else None,
        )

        outputs["sequence_cost_summary"] = save_table(
            sequence_cost,
            args.output_dir / "sequence_cost_summary.csv",
        )
    else:
        print("No sequence_dataset_dir in config; skip high-repeat and sequence cost.")

    # -------------------------------------------------------------------------
    # 4. Subgroup metrics
    # -------------------------------------------------------------------------

    if "group_name" in subgroup_pred.columns:
        subgroup_metrics_by_seed = compute_metrics_by_seed(
            subgroup_pred,
            metrics=metrics,
            extra_group_cols=["high_repeat_group", "group_name"],
        )

        outputs["subgroup_metrics_by_seed"] = save_table(
            subgroup_metrics_by_seed,
            args.output_dir / "subgroup_metrics_by_seed.csv",
        )

        subgroup_metrics_mean_std = summarize_metrics_mean_std(
            subgroup_metrics_by_seed,
            metrics=metrics,
            extra_group_cols=["high_repeat_group", "group_name"],
        )

        outputs["subgroup_metrics_mean_std"] = save_table(
            subgroup_metrics_mean_std,
            args.output_dir / "subgroup_metrics_mean_std.csv",
        )
    else:
        subgroup_metrics_by_seed = pd.DataFrame()
        subgroup_metrics_mean_std = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 5. Ensemble and stability
    # -------------------------------------------------------------------------

    ensemble_pred = ensemble_predictions(pred)

    outputs["ensemble_mean_predictions"] = save_table(
        ensemble_pred,
        args.output_dir / "ensemble_mean_predictions.csv",
    )

    ensemble_metrics = compute_metrics_for_group(
        ensemble_pred,
        group_cols=[
            "task",
            "model_family",
            "model_name",
            "representation",
            "compression_version",
            "numeric_on",
            "calibration",
        ],
        metrics=metrics,
    )

    outputs["ensemble_metrics"] = save_table(
        ensemble_metrics,
        args.output_dir / "ensemble_metrics.csv",
    )

    by_example, stability_summary = prediction_stability_summary(pred)

    outputs["prediction_stability_by_example"] = save_table(
        by_example,
        args.output_dir / "prediction_stability_by_example.csv",
    )

    outputs["prediction_stability_summary"] = save_table(
        stability_summary,
        args.output_dir / "prediction_stability_summary.csv",
    )

    # Ensemble with subgroup.
    if "group_name" in subgroup_pred.columns:
        ensemble_subgroup = ensemble_predictions(
            subgroup_pred,
            extra_group_cols=["high_repeat_group", "group_name"],
        )

        outputs["ensemble_mean_predictions_with_high_repeat"] = save_table(
            ensemble_subgroup,
            args.output_dir / "ensemble_mean_predictions_with_high_repeat.csv",
        )

        ensemble_subgroup_metrics = compute_metrics_for_group(
            ensemble_subgroup,
            group_cols=[
                "task",
                "model_family",
                "model_name",
                "representation",
                "compression_version",
                "numeric_on",
                "calibration",
                "high_repeat_group",
                "group_name",
            ],
            metrics=metrics,
        )

        outputs["ensemble_subgroup_metrics"] = save_table(
            ensemble_subgroup_metrics,
            args.output_dir / "ensemble_subgroup_metrics.csv",
        )
    else:
        ensemble_subgroup = pd.DataFrame()
        ensemble_subgroup_metrics = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 6. Patient bootstrap CI
    # -------------------------------------------------------------------------

    patient_ci = make_patient_bootstrap_metric_ci_table(
        pred=ensemble_pred,
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        cluster_col=cluster_col,
    )

    outputs["patient_bootstrap_metric_ci"] = save_table(
        patient_ci,
        args.output_dir / "patient_bootstrap_metric_ci.csv",
    )

    if len(ensemble_subgroup):
        patient_ci_subgroup = make_patient_bootstrap_metric_ci_table(
            pred=ensemble_subgroup,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            cluster_col=cluster_col,
            extra_group_cols=["high_repeat_group", "group_name"],
        )

        outputs["patient_bootstrap_metric_ci_subgroup"] = save_table(
            patient_ci_subgroup,
            args.output_dir / "patient_bootstrap_metric_ci_subgroup.csv",
        )

    # -------------------------------------------------------------------------
    # 7. Pairwise bootstrap
    # -------------------------------------------------------------------------

    paired_all = run_pairwise_comparisons(
        ensemble_pred=ensemble_pred,
        config=config,
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        cluster_col=cluster_col,
        subgroup_col=None,
    )

    outputs["paired_bootstrap_delta"] = save_table(
        paired_all,
        args.output_dir / "paired_bootstrap_delta.csv",
    )

    if len(ensemble_subgroup):
        paired_subgroup = run_pairwise_comparisons(
            ensemble_pred=ensemble_subgroup,
            config=config,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            cluster_col=cluster_col,
            subgroup_col="high_repeat_group",
        )

        outputs["paired_bootstrap_delta_subgroup"] = save_table(
            paired_subgroup,
            args.output_dir / "paired_bootstrap_delta_subgroup.csv",
        )

        paired_for_huly = pd.concat(
            [paired_all, paired_subgroup],
            ignore_index=True,
        )
    else:
        paired_subgroup = pd.DataFrame()
        paired_for_huly = paired_all

    # -------------------------------------------------------------------------
    # 8. Huly tables
    # -------------------------------------------------------------------------

    huly_compact = make_huly_compact_results(
        metrics_mean_std=metrics_mean_std,
        primary_models=primary_models,
    )

    outputs["huly_compact_results"] = save_table(
        huly_compact,
        args.output_dir / "huly_compact_results.csv",
    )

    huly_paired_auprc = make_huly_paired_auprc_table(paired_for_huly)

    outputs["huly_paired_auprc_delta_table"] = save_table(
        huly_paired_auprc,
        args.output_dir / "huly_paired_auprc_delta_table.csv",
    )

    # -------------------------------------------------------------------------
    # 9. Upload / ClearML artifacts
    # -------------------------------------------------------------------------

    upload_manifest = pd.DataFrame()

    output_s3_prefix = args.output_s3_prefix or config.get("output_s3_prefix", "")

    if output_s3_prefix and not args.skip_upload:
        run_output_s3_prefix = (
            f"{output_s3_prefix.rstrip('/')}/"
            f"{safe_path_part(run_set)}"
        )

        upload_manifest = upload_tree_to_minio(
            local_root=args.output_dir,
            s3_prefix=run_output_s3_prefix,
        )

        outputs["minio_upload_manifest"] = save_table(
            upload_manifest,
            args.output_dir / "minio_upload_manifest.csv",
        )

        print(f"Uploaded comparison outputs to: {run_output_s3_prefix}")
    else:
        print("Skip MinIO upload.")

    save_outputs_to_clearml(clearml_task, outputs)

    print("=" * 100)
    print("DONE")
    print(f"Local output: {args.output_dir}")

    print("\nMetrics mean±std preview:")
    preview_cols = [
        "task",
        "model_family",
        "model_name",
        "compression_version",
        "numeric_on",
        "n_seeds",
        "auprc_mean",
        "auprc_std",
        "auroc_mean",
        "brier_mean",
        "logloss_mean",
    ]
    preview_cols = [c for c in preview_cols if c in metrics_mean_std.columns]

    print(
        metrics_mean_std[preview_cols]
        .sort_values(["task", "auprc_mean"], ascending=[True, False])
        .to_string(index=False)
    )

    if len(huly_paired_auprc):
        print("\nPaired AUPRC deltas:")
        print(huly_paired_auprc.to_string(index=False))


if __name__ == "__main__":
    main()