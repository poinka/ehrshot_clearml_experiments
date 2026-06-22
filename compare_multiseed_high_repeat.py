from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd

from common_ehrshot_eval import binary_ranking_metrics, paired_bootstrap_delta, safe_metric, topk_metrics

TASKS = ["guo_readmission", "guo_icu"]

PRIMARY_CONFIGS = [
    # -------------------------
    # guo_readmission
    # -------------------------
    {"key": "readmission_tabular_hgb", "family": "tabular", "source": "tabular",
     "task": "guo_readmission", "version": "tabular_all_features", "model": "HistGradientBoosting"},

    {"key": "readmission_sequence_raw_retain", "family": "sequence", "source": "sequence",
     "task": "guo_readmission", "version": "raw", "model": "RETAIN_lite"},

    {"key": "readmission_sequence_dedup_retain", "family": "sequence", "source": "sequence",
     "task": "guo_readmission", "version": "compressed_dedup", "model": "RETAIN_lite"},

    {"key": "readmission_numeric_dedup_retain", "family": "numeric_sequence", "source": "numeric_sequence",
     "task": "guo_readmission", "version": "compressed_dedup", "model": "RETAIN_lite_numeric"},

    # -------------------------
    # guo_icu
    # -------------------------
    {"key": "icu_tabular_rf", "family": "tabular", "source": "tabular",
     "task": "guo_icu", "version": "tabular_all_features", "model": "RandomForest_balanced"},

    # code-only raw vs compressed, same architecture
    {"key": "icu_sequence_raw_gru2", "family": "sequence", "source": "sequence",
     "task": "guo_icu", "version": "raw", "model": "GRU_2L"},

    {"key": "icu_sequence_hybrid_gru2", "family": "sequence", "source": "sequence",
     "task": "guo_icu", "version": "compressed_hybrid", "model": "GRU_2L"},

    # old best code-only ICU reference
    {"key": "icu_sequence_condition_lstm1", "family": "sequence", "source": "sequence",
     "task": "guo_icu", "version": "compressed_condition_era", "model": "LSTM_1L"},

    # numeric raw baselines
    {"key": "icu_numeric_raw_gru1", "family": "numeric_sequence", "source": "numeric_sequence",
     "task": "guo_icu", "version": "raw", "model": "GRU_1L_numeric"},

    {"key": "icu_numeric_raw_lstm1", "family": "numeric_sequence", "source": "numeric_sequence",
     "task": "guo_icu", "version": "raw", "model": "LSTM_1L_numeric"},

    # numeric raw vs compressed, same architecture
    {"key": "icu_numeric_raw_gru2", "family": "numeric_sequence", "source": "numeric_sequence",
     "task": "guo_icu", "version": "raw", "model": "GRU_2L_numeric"},

    {"key": "icu_numeric_hybrid_gru2", "family": "numeric_sequence", "source": "numeric_sequence",
     "task": "guo_icu", "version": "compressed_hybrid", "model": "GRU_2L_numeric"},
]

METRICS = ["auroc", "auprc", "brier", "logloss", "top_5pct_precision", "top_10pct_precision", "top_20pct_precision"]

PAIRWISE_COMPARISONS = [
    # -------------------------------------------------
    # Tabular vs sequence / numeric sequence
    # -------------------------------------------------
    {
        "comparison": "readmission_sequence_dedup_minus_tabular",
        "task": "guo_readmission",
        "a": "readmission_sequence_dedup_retain",
        "b": "readmission_tabular_hgb",
        "description": "compressed_dedup RETAIN_lite vs tabular HGB",
    },
    {
        "comparison": "readmission_numeric_dedup_minus_tabular",
        "task": "guo_readmission",
        "a": "readmission_numeric_dedup_retain",
        "b": "readmission_tabular_hgb",
        "description": "compressed_dedup RETAIN_lite_numeric vs tabular HGB",
    },
    {
        "comparison": "icu_numeric_hybrid_minus_tabular",
        "task": "guo_icu",
        "a": "icu_numeric_hybrid_gru2",
        "b": "icu_tabular_rf",
        "description": "compressed_hybrid GRU_2L_numeric vs tabular RF",
    },
    {
        "comparison": "icu_sequence_hybrid_minus_tabular",
        "task": "guo_icu",
        "a": "icu_sequence_hybrid_gru2",
        "b": "icu_tabular_rf",
        "description": "compressed_hybrid GRU_2L code-only vs tabular RF",
    },

    # -------------------------------------------------
    # Raw vs compressed
    # -------------------------------------------------
    {
        "comparison": "readmission_sequence_dedup_minus_raw",
        "task": "guo_readmission",
        "a": "readmission_sequence_dedup_retain",
        "b": "readmission_sequence_raw_retain",
        "description": "readmission code-only compressed_dedup vs raw, same RETAIN_lite architecture",
    },
    {
        "comparison": "icu_sequence_hybrid_minus_raw",
        "task": "guo_icu",
        "a": "icu_sequence_hybrid_gru2",
        "b": "icu_sequence_raw_gru2",
        "description": "ICU code-only compressed_hybrid vs raw, same GRU_2L architecture",
    },
    {
        "comparison": "icu_numeric_hybrid_minus_raw",
        "task": "guo_icu",
        "a": "icu_numeric_hybrid_gru2",
        "b": "icu_numeric_raw_gru2",
        "description": "ICU numeric compressed_hybrid vs raw, same GRU_2L_numeric architecture",
    },

    # -------------------------------------------------
    # Code-only vs numeric
    # -------------------------------------------------
    {
        "comparison": "readmission_numeric_dedup_minus_code_only_dedup",
        "task": "guo_readmission",
        "a": "readmission_numeric_dedup_retain",
        "b": "readmission_sequence_dedup_retain",
        "description": "readmission numeric-aware vs code-only, same compressed_dedup version",
    },
    {
        "comparison": "icu_numeric_hybrid_minus_code_only_hybrid",
        "task": "guo_icu",
        "a": "icu_numeric_hybrid_gru2",
        "b": "icu_sequence_hybrid_gru2",
        "description": "ICU numeric-aware vs code-only, same compressed_hybrid version and GRU_2L architecture",
    },

    # -------------------------------------------------
    # Extra ICU numeric raw architecture sanity checks
    # -------------------------------------------------
    {
        "comparison": "icu_numeric_raw_gru1_minus_raw_lstm1",
        "task": "guo_icu",
        "a": "icu_numeric_raw_gru1",
        "b": "icu_numeric_raw_lstm1",
        "description": "ICU raw numeric GRU_1L vs raw numeric LSTM_1L",
    },
]

def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args) -> dict:
    config = vars(args).copy()

    for key in [
        "tabular_predictions",
        "sequence_predictions",
        "numeric_predictions",
        "audit_dir",
        "output_dir",
    ]:
        if key in config:
            config[key] = str(config[key])

    return config


def sync_args_from_clearml_config(args, config: dict) -> None:
    path_keys = {
        "tabular_predictions",
        "sequence_predictions",
        "numeric_predictions",
        "audit_dir",
        "output_dir",
    }
    int_keys = {"bootstrap", "seed"}
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
        else:
            setattr(args, key, value)


def maybe_init_clearml(args, config: dict):
    remote_agent_run = is_clearml_agent_run()

    if not args.enable_clearml and not remote_agent_run:
        return None, config

    from clearml import Task

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
    print(f"  tabular_task_id = {args.tabular_task_id}")
    print(f"  sequence_task_id = {args.sequence_task_id}")
    print(f"  numeric_task_id = {args.numeric_task_id}")
    print(f"  audit_s3_prefix = {args.audit_s3_prefix}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if should_execute_remotely:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task, connected_config

def load_dataframe_artifact(task_id: str, artifact_name: str) -> pd.DataFrame:
    from clearml import Task

    print(f"Loading artifact '{artifact_name}' from task_id={task_id}")
    src_task = Task.get_task(task_id=task_id)

    if artifact_name not in src_task.artifacts:
        available = list(src_task.artifacts.keys())
        raise KeyError(
            f"Artifact '{artifact_name}' not found in task {task_id}. "
            f"Available artifacts: {available}"
        )

    artifact = src_task.artifacts[artifact_name]

    # Если artifact был загружен как pandas DataFrame через upload_artifact(name, df),
    # artifact.get() обычно вернет сам DataFrame.
    obj = artifact.get()
    if isinstance(obj, pd.DataFrame):
        return obj

    # Fallback: если вернулся путь/файл.
    local_copy = Path(artifact.get_local_copy())
    if local_copy.is_file():
        return pd.read_csv(local_copy)

    csv_files = sorted(local_copy.glob("*.csv")) if local_copy.is_dir() else []
    if len(csv_files) == 1:
        return pd.read_csv(csv_files[0])

    raise ValueError(
        f"Could not load artifact '{artifact_name}' from task {task_id}. "
        f"get() returned type={type(obj)}, local_copy={local_copy}"
    )

def load_predictions(path: Path, family_name: str) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*multiseed*heldout_predictions.csv"))
        files += sorted(path.glob("*__heldout_predictions.csv"))
        files = list(dict.fromkeys(files))
        if not files:
            raise FileNotFoundError(f"No prediction csvs found in {path}")
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    else:
        df = pd.read_csv(path)
    if "family" not in df.columns:
        df["family"] = family_name
    if "source" not in df.columns:
        df["source"] = family_name
    if "version" not in df.columns and family_name == "tabular":
        df["version"] = "tabular_all_features"
    return df


def load_all_predictions(args) -> pd.DataFrame:
    parts = []

    if args.tabular_task_id:
        parts.append(
            load_dataframe_artifact(
                args.tabular_task_id,
                "tabular_multiseed_predictions",
            )
        )
    else:
        parts.append(load_predictions(args.tabular_predictions, "tabular"))

    if args.sequence_task_id:
        parts.append(
            load_dataframe_artifact(
                args.sequence_task_id,
                "sequence_multiseed_predictions",
            )
        )
    else:
        parts.append(load_predictions(args.sequence_predictions, "sequence"))

    if args.numeric_task_id:
        parts.append(
            load_dataframe_artifact(
                args.numeric_task_id,
                "numeric_sequence_multiseed_predictions",
            )
        )
    else:
        parts.append(load_predictions(args.numeric_predictions, "numeric_sequence"))

    df = pd.concat(parts, ignore_index=True)

    required = {
        "task", "family", "source", "version", "model", "seed", "calibration",
        "row_id", "subject_id", "y_true", "risk",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions: {missing}")

    df = df[df["calibration"] == args.calibration].copy()

    for c in ["row_id", "subject_id", "seed", "y_true"]:
        df[c] = df[c].astype(int)

    df["risk"] = df["risk"].astype(float)

    masks = []
    for cfg in PRIMARY_CONFIGS:
        m = (
            (df["family"] == cfg["family"])
            & (df["task"] == cfg["task"])
            & (df["version"] == cfg["version"])
            & (df["model"] == cfg["model"])
        )
        masks.append(m)

    df = df[np.logical_or.reduce(masks)].copy()
    return df

def maybe_download_audit_from_s3_prefix(audit_dir: Path, audit_s3_prefix: str) -> Path:
    expected = [
        audit_dir / f"task_history_copy_forwarding_audit__{task}.csv"
        for task in TASKS
    ]

    if all(p.exists() for p in expected):
        print(f"Using local audit dir: {audit_dir}")
        return audit_dir

    if not audit_s3_prefix:
        missing = [str(p) for p in expected if not p.exists()]
        raise FileNotFoundError(
            "Audit files are missing locally and --audit-s3-prefix is not provided. "
            f"Missing files: {missing}"
        )

    from clearml import StorageManager

    audit_dir.mkdir(parents=True, exist_ok=True)
    prefix = audit_s3_prefix.rstrip("/")

    for task in TASKS:
        fname = f"task_history_copy_forwarding_audit__{task}.csv"
        dst = audit_dir / fname

        if dst.exists():
            continue

        remote_url = f"{prefix}/{fname}"
        print(f"Downloading {remote_url}")
        local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

        print(f"Copying {local_copy} -> {dst}")
        shutil.copy2(local_copy, dst)

    return audit_dir

def load_high_repeat(audit_dir: Path, threshold_source: str = "held_out", q: float = 0.90):
    parts = []
    for task in TASKS:
        path = audit_dir / f"task_history_copy_forwarding_audit__{task}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Audit file not found: {path}")
        part = pd.read_csv(path)
        part["task"] = task
        parts.append(part)
    audit = pd.concat(parts, ignore_index=True)
    keep = ["task", "row_id", "subject_id", "split", "label", "repeat_removed_first_last", "raw_sequence_length_non_static", "n_diag_codes_persistent_365d"]
    audit = audit[keep].copy()
    rows = []
    for task in TASKS:
        t = audit[audit["task"] == task]
        if threshold_source == "held_out":
            src = t[t["split"] == "held_out"]
        elif threshold_source == "train":
            src = t[t["split"] == "train"]
        elif threshold_source == "all":
            src = t
        else:
            raise ValueError(threshold_source)
        thr = src["repeat_removed_first_last"].quantile(q)
        rows.append({"task": task, "threshold_source": threshold_source, "q": q, "repeat_removed_threshold": thr, "n_source_examples": len(src)})
    thr_df = pd.DataFrame(rows)
    audit = audit.merge(thr_df[["task", "repeat_removed_threshold"]], on="task", how="left")
    audit["high_repeat_group"] = audit["repeat_removed_first_last"] >= audit["repeat_removed_threshold"]
    audit_heldout = audit[audit["split"] == "held_out"].copy()
    return audit_heldout, thr_df


def add_high_repeat(pred: pd.DataFrame, audit_heldout: pd.DataFrame) -> pd.DataFrame:
    audit_small = audit_heldout[["task", "row_id", "subject_id", "high_repeat_group", "repeat_removed_first_last", "raw_sequence_length_non_static", "n_diag_codes_persistent_365d"]].copy()
    for c in ["row_id", "subject_id"]:
        audit_small[c] = audit_small[c].astype(int)
    merged = pred.merge(audit_small, on=["task", "row_id", "subject_id"], how="inner")
    if merged.empty:
        raise ValueError("Prediction/audit merge produced 0 rows")
    return merged


def group_name(high_repeat_group):
    if pd.isna(high_repeat_group):
        return "all"
    return "high_repeat_top10" if bool(high_repeat_group) else "other_90"


def compute_single_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["task", "family", "source", "version", "model", "seed", "calibration", "high_repeat_group"]
    for key, g in df.groupby(group_cols, dropna=False):
        task, family, source, version, model, seed, calibration, hrg = key
        y = g["y_true"].to_numpy()
        p = g["risk"].to_numpy()
        m = binary_ranking_metrics(y, p)
        topk = topk_metrics(y, p).set_index("top_frac")
        rows.append({
            "task": task, "family": family, "source": source, "version": version, "model": model, "seed": int(seed), "calibration": calibration,
            "high_repeat_group": bool(hrg), "group_name": group_name(hrg),
            **m,
            "top_5pct_precision": float(topk.loc[0.05, "top_k_event_rate"]),
            "top_10pct_precision": float(topk.loc[0.10, "top_k_event_rate"]),
            "top_20pct_precision": float(topk.loc[0.20, "top_k_event_rate"]),
        })
    return pd.DataFrame(rows)


def summarize_seed_metrics(single_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auroc", "auprc", "brier", "logloss", "top_5pct_precision", "top_10pct_precision", "top_20pct_precision"]
    id_cols = ["task", "family", "source", "version", "model", "calibration", "high_repeat_group", "group_name"]
    agg = single_seed_metrics.groupby(id_cols)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["__".join([x for x in col if x]) if isinstance(col, tuple) else col for col in agg.columns]
    counts = single_seed_metrics.groupby(id_cols).agg(n_seeds=("seed", "nunique"), n=("n", "first"), n_positive=("n_positive", "first"), event_rate=("event_rate", "first")).reset_index()
    return counts.merge(agg, on=id_cols, how="left")


def prediction_std_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = ["task", "family", "source", "version", "model", "calibration", "high_repeat_group", "row_id", "subject_id", "y_true"]
    by_example = df.groupby(id_cols).agg(risk_mean=("risk", "mean"), risk_std=("risk", "std"), n_seeds=("seed", "nunique")).reset_index()
    by_example["risk_std"] = by_example["risk_std"].fillna(0.0)
    by_example["group_name"] = by_example["high_repeat_group"].map(group_name)
    summary = by_example.groupby(["task", "family", "source", "version", "model", "calibration", "high_repeat_group", "group_name"]).agg(
        n_examples=("row_id", "size"),
        n_positive=("y_true", "sum"),
        mean_prediction_std=("risk_std", "mean"),
        median_prediction_std=("risk_std", "median"),
        p90_prediction_std=("risk_std", lambda x: float(np.quantile(x, 0.90))),
        p95_prediction_std=("risk_std", lambda x: float(np.quantile(x, 0.95))),
        max_prediction_std=("risk_std", "max"),
    ).reset_index()
    return by_example, summary


def ensemble_predictions(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["task", "family", "source", "version", "model", "calibration", "high_repeat_group", "row_id", "subject_id", "y_true"]
    out = df.groupby(id_cols).agg(risk=("risk", "mean"), risk_std=("risk", "std"), n_seeds=("seed", "nunique")).reset_index()
    out["risk_std"] = out["risk_std"].fillna(0.0)
    out["group_name"] = out["high_repeat_group"].map(group_name)
    return out


def ensemble_performance(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in ens.groupby(["task", "family", "source", "version", "model", "calibration", "high_repeat_group", "group_name"]):
        task, family, source, version, model, calibration, hrg, gname = key
        y = g["y_true"].to_numpy()
        p = g["risk"].to_numpy()
        m = binary_ranking_metrics(y, p)
        topk = topk_metrics(y, p).set_index("top_frac")
        rows.append({"task": task, "family": family, "source": source, "version": version, "model": model, "calibration": calibration, "high_repeat_group": bool(hrg), "group_name": gname, **m, "top_5pct_precision": topk.loc[0.05, "top_k_event_rate"], "top_10pct_precision": topk.loc[0.10, "top_k_event_rate"], "top_20pct_precision": topk.loc[0.20, "top_k_event_rate"]})
    return pd.DataFrame(rows)


def get_config_by_key(key: str) -> dict:
    matches = [c for c in PRIMARY_CONFIGS if c["key"] == key]
    if not matches:
        raise ValueError(f"Unknown config key: {key}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate config key: {key}")
    return matches[0]


def get_cfg_df(df: pd.DataFrame, key: str, high_repeat_group: bool) -> pd.DataFrame:
    cfg = get_config_by_key(key)
    out = df[
        (df["task"] == cfg["task"])
        & (df["family"] == cfg["family"])
        & (df["version"] == cfg["version"])
        & (df["model"] == cfg["model"])
        & (df["high_repeat_group"] == high_repeat_group)
    ].copy()

    if out.empty:
        raise ValueError(
            f"No rows for key={key}, "
            f"task={cfg['task']}, family={cfg['family']}, "
            f"version={cfg['version']}, model={cfg['model']}, "
            f"high_repeat_group={high_repeat_group}"
        )

    return out


def run_pairwise_bootstrap(ens: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []

    for cmp_cfg in PAIRWISE_COMPARISONS:
        task = cmp_cfg["task"]

        for hrg in [True, False]:
            a = get_cfg_df(ens, cmp_cfg["a"], hrg)
            b = get_cfg_df(ens, cmp_cfg["b"], hrg)

            delta = paired_bootstrap_delta(
                a,
                b,
                METRICS,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            for _, r in delta.iterrows():
                rows.append(
                    {
                        "task": task,
                        "group_name": group_name(hrg),
                        "high_repeat_group": hrg,
                        "comparison": cmp_cfg["comparison"],
                        "description": cmp_cfg["description"],
                        "model_a": f"{a['family'].iloc[0]}: {a['version'].iloc[0]} + {a['model'].iloc[0]}",
                        "model_b": f"{b['family'].iloc[0]}: {b['version'].iloc[0]} + {b['model'].iloc[0]}",
                        **r.to_dict(),
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular-predictions", type=Path, default=Path("ehrshot_multiseed_tabular_results/tabular_multiseed_heldout_predictions.csv"))
    parser.add_argument("--sequence-predictions", type=Path, default=Path("ehrshot_multiseed_sequence_results/sequence_multiseed_heldout_predictions.csv"))
    parser.add_argument("--numeric-predictions", type=Path, default=Path("ehrshot_multiseed_numeric_sequence_results/numeric_sequence_multiseed_heldout_predictions.csv"))
    parser.add_argument("--tabular-task-id", type=str, default="")
    parser.add_argument("--sequence-task-id", type=str, default="")
    parser.add_argument("--numeric-task-id", type=str, default="")
    parser.add_argument("--audit-s3-prefix", type=str, default="")
    parser.add_argument("--audit-dir", type=Path, default=Path("ehrshot_copy_forwarding_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_multiseed_all_model_comparison"))
    parser.add_argument("--calibration", type=str, default="platt")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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

    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )

    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="multiseed_high_repeat_comparison",
    )

    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    args = parser.parse_args()
    config = build_clearml_config(args)
    clearml_task, config = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_all_predictions(args)
    args.audit_dir = maybe_download_audit_from_s3_prefix(
        audit_dir=args.audit_dir,
        audit_s3_prefix=args.audit_s3_prefix,
    )
    audit, thresholds = load_high_repeat(args.audit_dir)
    merged = add_high_repeat(pred, audit)

    thresholds.to_csv(args.output_dir / "high_repeat_thresholds.csv", index=False)
    merged.to_csv(args.output_dir / "all_model_multiseed_predictions_with_high_repeat.csv", index=False)

    single = compute_single_seed_metrics(merged)
    single.to_csv(args.output_dir / "single_seed_subgroup_metrics.csv", index=False)
    seed_summary = summarize_seed_metrics(single)
    seed_summary.to_csv(args.output_dir / "seed_metric_summary.csv", index=False)

    by_example, std_summary = prediction_std_summary(merged)
    by_example.to_csv(args.output_dir / "prediction_stability_by_example.csv", index=False)
    std_summary.to_csv(args.output_dir / "prediction_stability_summary.csv", index=False)

    ens = ensemble_predictions(merged)
    ens.to_csv(args.output_dir / "ensemble_mean_predictions.csv", index=False)
    ens_perf = ensemble_performance(ens)
    ens_perf.to_csv(args.output_dir / "ensemble_subgroup_performance.csv", index=False)

    deltas = run_pairwise_bootstrap(ens, n_bootstrap=args.bootstrap, seed=args.seed)
    deltas.to_csv(args.output_dir / "ensemble_paired_bootstrap_delta.csv", index=False)

    auprc = deltas[deltas["metric"] == "auprc"].copy()
    auprc["delta_auprc_with_ci"] = auprc.apply(lambda r: f"{r['point_delta_a_minus_b']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1)
    auprc[["task", "group_name", "comparison", "model_a", "model_b", "n_paired_examples", "n_paired_positive", "delta_auprc_with_ci"]].to_csv(args.output_dir / "huly_paired_auprc_delta_table.csv", index=False)

    if clearml_task is not None:
        clearml_task.upload_artifact(
            "single_seed_subgroup_metrics",
            single,
        )
        clearml_task.upload_artifact(
            "seed_metric_summary",
            seed_summary,
        )
        clearml_task.upload_artifact(
            "prediction_stability_by_example",
            by_example,
        )
        clearml_task.upload_artifact(
            "prediction_stability_summary",
            std_summary,
        )
        clearml_task.upload_artifact(
            "ensemble_mean_predictions",
            ens,
        )
        clearml_task.upload_artifact(
            "ensemble_subgroup_performance",
            ens_perf,
        )
        clearml_task.upload_artifact(
            "ensemble_paired_bootstrap_delta",
            deltas,
        )
        clearml_task.upload_artifact(
            "huly_paired_auprc_delta_table",
            auprc,
        )

    print("Saved outputs to:", args.output_dir)
    print("\nSeed metric summary:")
    print(seed_summary)
    print("\nPrediction STD summary:")
    print(std_summary)
    print("\nEnsemble performance:")
    print(ens_perf)
    print("\nAUPRC paired bootstrap deltas:")
    print(auprc[["task", "group_name", "comparison", "delta_auprc_with_ci", "n_paired_positive"]])


if __name__ == "__main__":
    main()
