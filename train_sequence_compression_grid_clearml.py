#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import train_sequence_compression_model_clearml as base


DEFAULT_VERSIONS = (
    "raw,"
    "compressed_dedup,"
    "compressed_first_last,"
    "condition_era_90,"
    "condition_era_180,"
    "state_duration_90,"
    "state_duration_180"
)

DEFAULT_SEEDS = "42,43,44,45,46"

DEFAULT_SEQUENCE_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "ehrshot_multiseed_inputs/ehrshot_sequence_datasets"
)

DEFAULT_RESULTS_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "ehrshot_sequence_compression_grid_results"
)


MODEL_CONFIGS = [
    {
        "run_group": "readmission_RETAIN_sequence",
        "task": "guo_readmission",
        "model": "RETAIN_lite",
        "use_numeric": False,
        "max_len": 4096,
        "batch_size": 16,
    },
    {
        "run_group": "readmission_RETAIN_numeric",
        "task": "guo_readmission",
        "model": "RETAIN_lite",
        "use_numeric": True,
        "max_len": 4096,
        "batch_size": 16,
    },
    {
        "run_group": "icu_GRU2_numeric",
        "task": "guo_icu",
        "model": "GRU_2L",
        "use_numeric": True,
        "max_len": 4096,
        "batch_size": 16,
    },
    {
        "run_group": "icu_LSTM_sequence",
        "task": "guo_icu",
        "model": "LSTM_1L",
        "use_numeric": False,
        "max_len": 4096,
        "batch_size": 16,
    },
]


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_int_csv(x: str) -> list[int]:
    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all important sequence/numeric compression experiments inside one ClearML task."
    )

    parser.add_argument("--sequence-dir", type=Path, default=Path("ehrshot_sequence_datasets_compression_v3"))
    parser.add_argument("--sequence-data-s3-prefix", type=str, default=DEFAULT_SEQUENCE_S3_PREFIX)

    parser.add_argument("--results-dir", type=Path, default=Path("ehrshot_sequence_compression_grid_results_v3"))
    parser.add_argument("--results-s3-prefix", type=str, default=DEFAULT_RESULTS_S3_PREFIX)

    parser.add_argument("--versions", type=str, default=DEFAULT_VERSIONS)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--numeric-min-count", type=int, default=3)
    parser.add_argument("--top-fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--save-checkpoint", action="store_true")

    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="train_4models_all_compressions_v3")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")

    return parser.parse_args()


def build_clearml_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    for key in ["sequence_dir", "results_dir"]:
        cfg[key] = str(cfg[key])
    cfg["model_configs"] = MODEL_CONFIGS
    return cfg


def _to_bool(x: Any) -> bool:
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path_keys = {"sequence_dir", "results_dir"}
    int_keys = {
        "num_workers",
        "epochs",
        "patience",
        "emb_dim",
        "hidden_dim",
        "max_train_examples",
        "max_eval_examples",
        "numeric_min_count",
    }
    float_keys = {"lr", "weight_decay", "grad_clip", "dropout"}
    bool_keys = {"save_checkpoint", "skip_existing"}
    skip_keys = {"clearml", "execute_remotely"}

    for key, value in dict(cfg).items():
        if key in skip_keys or not hasattr(args, key):
            continue

        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in float_keys:
            setattr(args, key, float(value))
        elif key in bool_keys:
            setattr(args, key, _to_bool(value))
        else:
            setattr(args, key, value)


def maybe_init_clearml(args: argparse.Namespace):
    remote_agent_run = is_clearml_agent_run()

    if not args.clearml and not remote_agent_run:
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

    connected_cfg = dict(task.connect(build_clearml_config(args)))
    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  sequence_dir = {args.sequence_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
    print(f"  results_dir = {args.results_dir}")
    print(f"  results_s3_prefix = {args.results_s3_prefix}")
    print(f"  versions = {args.versions}")
    print(f"  seeds = {args.seeds}")
    print(f"  device = {args.device}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing one ClearML grid task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


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


def make_single_run_args(args: argparse.Namespace, cfg: dict[str, Any], version: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        sequence_dir=args.sequence_dir,
        sequence_data_s3_prefix=args.sequence_data_s3_prefix,
        sequence_data_artifact_name="",
        results_dir=args.results_dir,

        task=cfg["task"],
        version=version,
        model=cfg["model"],
        seed=seed,
        device=args.device,
        use_numeric=bool(cfg["use_numeric"]),

        max_len=int(cfg["max_len"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=args.num_workers,

        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,

        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        numeric_min_count=args.numeric_min_count,
        top_fracs=args.top_fracs,
        save_checkpoint=args.save_checkpoint,

        clearml=False,
        execute_remotely=False,
        clearml_queue=args.clearml_queue,
        clearml_task_name="",
        clearml_project=args.clearml_project,
        clearml_output_uri=args.clearml_output_uri,
    )


def expected_prediction_path(results_dir: Path, run_args: argparse.Namespace) -> Path:
    model_name = run_args.model + ("_numeric" if run_args.use_numeric else "")
    exp_name = f"{run_args.task}__{run_args.version}__{model_name}__seed{run_args.seed}"
    return results_dir / f"{exp_name}__heldout_predictions.csv"


def run_one_experiment(
    global_args: argparse.Namespace,
    run_args: argparse.Namespace,
    run_group: str,
    run_index: int,
    n_runs_total: int,
    clearml_task,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_name = run_args.model + ("_numeric" if run_args.use_numeric else "")
    exp_name = f"{run_args.task}__{run_args.version}__{model_name}__seed{run_args.seed}"

    pred_path = expected_prediction_path(global_args.results_dir, run_args)

    if global_args.skip_existing and pred_path.exists():
        print(f"[{run_index}/{n_runs_total}] SKIP existing: {exp_name}")
        metrics_path = global_args.results_dir / f"{exp_name}__metrics.csv"
        topk_path = global_args.results_dir / f"{exp_name}__topk.csv"
        hist_path = global_args.results_dir / f"{exp_name}__history.csv"

        return (
            pd.read_csv(metrics_path),
            pd.read_csv(pred_path),
            pd.read_csv(topk_path),
            pd.read_csv(hist_path),
        )

    print("=" * 120)
    print(f"[{run_index}/{n_runs_total}] RUN {exp_name}")
    print(f"  run_group={run_group}")
    print(f"  task={run_args.task}")
    print(f"  version={run_args.version}")
    print(f"  model={run_args.model}")
    print(f"  use_numeric={run_args.use_numeric}")
    print(f"  seed={run_args.seed}")
    print(f"  max_len={run_args.max_len}")
    print(f"  batch_size={run_args.batch_size}")

    base.set_seed(run_args.seed)
    device = base.get_device(run_args.device)

    global_args.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = global_args.results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_args.sequence_dir = base.prepare_sequence_dataset(run_args)

    print("DEVICE:", device)

    df = base.load_sequence_examples(run_args.sequence_dir, run_args.task, run_args.version)
    vocab = base.load_vocab(run_args.sequence_dir, run_args.task, run_args.version)
    vocab_size = len(vocab)

    df_train = df[df["split"] == "train"].copy()
    mean, std, numeric_stats_df = base.compute_token_numeric_stats(
        df_train,
        vocab_size,
        run_args.numeric_min_count,
    )

    loaders, split_dfs = base.make_loaders(df, run_args, mean, std)
    model = base.make_model(vocab_size, run_args)

    started = time.perf_counter()
    model, history_df, train_seconds, best_epoch = base.train_model(model, loaders, run_args, device)
    result_df, pred_df, topk_df = base.evaluate(model, loaders, run_args, device)
    elapsed = time.perf_counter() - started

    result_df["run_group"] = run_group
    result_df["train_seconds_total"] = train_seconds
    result_df["best_epoch"] = best_epoch
    result_df["grid_run_index"] = run_index
    result_df["grid_elapsed_seconds"] = elapsed

    pred_df["run_group"] = run_group
    pred_df["grid_run_index"] = run_index

    topk_df["run_group"] = run_group
    topk_df["grid_run_index"] = run_index

    history_df.insert(0, "run_group", run_group)
    history_df.insert(1, "task", run_args.task)
    history_df.insert(2, "version", run_args.version)
    history_df.insert(3, "model", model_name)
    history_df.insert(4, "seed", run_args.seed)
    history_df.insert(5, "grid_run_index", run_index)

    result_path = global_args.results_dir / f"{exp_name}__metrics.csv"
    pred_path = global_args.results_dir / f"{exp_name}__heldout_predictions.csv"
    topk_path = global_args.results_dir / f"{exp_name}__topk.csv"
    hist_path = global_args.results_dir / f"{exp_name}__history.csv"
    num_path = global_args.results_dir / f"{exp_name}__numeric_stats.csv"

    result_df.to_csv(result_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    history_df.to_csv(hist_path, index=False)
    numeric_stats_df.to_csv(num_path, index=False)

    paths_to_upload = [result_path, pred_path, topk_path, hist_path, num_path]

    if run_args.save_checkpoint:
        ckpt_path = ckpt_dir / f"{exp_name}.pt"
        torch.save(
            {
                "task": run_args.task,
                "version": run_args.version,
                "model": run_args.model,
                "use_numeric": run_args.use_numeric,
                "seed": run_args.seed,
                "vocab_size": vocab_size,
                "max_len": run_args.max_len,
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        paths_to_upload.append(ckpt_path)

    for path in paths_to_upload:
        upload_file_to_s3(path, global_args.results_dir, global_args.results_s3_prefix)

    print("Saved metrics:", result_path)
    print("Saved predictions:", pred_path)

    preview_cols = [
        "task",
        "version",
        "model",
        "seed",
        "calibration",
        "heldout_auroc",
        "heldout_auprc",
        "heldout_brier",
        "heldout_logloss",
    ]
    print(result_df[preview_cols].to_string(index=False))

    if clearml_task is not None:
        logger = clearml_task.get_logger()
        platt = result_df[result_df["calibration"] == "platt"]
        if len(platt):
            row = platt.iloc[0]
            title = f"{run_group}/{run_args.version}"
            logger.report_scalar(title, "heldout_auprc", float(row["heldout_auprc"]), iteration=run_index)
            logger.report_scalar(title, "heldout_auroc", float(row["heldout_auroc"]), iteration=run_index)
            logger.report_scalar(title, "heldout_brier", float(row["heldout_brier"]), iteration=run_index)

    del model, loaders, split_dfs, df, df_train, vocab, numeric_stats_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_df, pred_df, topk_df, history_df


def main() -> None:
    args = parse_args()
    clearml_task = maybe_init_clearml(args)

    versions = parse_csv_list(args.versions)
    seeds = parse_int_csv(args.seeds)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    for cfg in MODEL_CONFIGS:
        for version in versions:
            for seed in seeds:
                plan_rows.append(
                    {
                        "run_group": cfg["run_group"],
                        "task": cfg["task"],
                        "version": version,
                        "model": cfg["model"] + ("_numeric" if cfg["use_numeric"] else ""),
                        "base_model": cfg["model"],
                        "use_numeric": cfg["use_numeric"],
                        "max_len": cfg["max_len"],
                        "batch_size": cfg["batch_size"],
                        "seed": seed,
                    }
                )

    plan_df = pd.DataFrame(plan_rows)
    plan_path = args.results_dir / "grid_run_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    upload_file_to_s3(plan_path, args.results_dir, args.results_s3_prefix)

    print("=" * 120)
    print("GRID PLAN")
    print(plan_df.to_string(index=False))
    print(f"Total runs: {len(plan_df)}")
    print("=" * 120)

    all_metrics = []
    all_predictions = []
    all_topk = []
    all_history = []

    n_runs_total = len(plan_rows)
    run_index = 0

    for cfg in MODEL_CONFIGS:
        for version in versions:
            for seed in seeds:
                run_index += 1
                run_args = make_single_run_args(args, cfg, version, seed)

                result_df, pred_df, topk_df, history_df = run_one_experiment(
                    global_args=args,
                    run_args=run_args,
                    run_group=cfg["run_group"],
                    run_index=run_index,
                    n_runs_total=n_runs_total,
                    clearml_task=clearml_task,
                )

                all_metrics.append(result_df)
                all_predictions.append(pred_df)
                all_topk.append(topk_df)
                all_history.append(history_df)

                # Save aggregate incrementally, so partial results survive if the task fails later.
                pd.concat(all_metrics, ignore_index=True).to_csv(args.results_dir / "grid_all_metrics.csv", index=False)
                pd.concat(all_predictions, ignore_index=True).to_csv(args.results_dir / "grid_all_heldout_predictions.csv", index=False)
                pd.concat(all_topk, ignore_index=True).to_csv(args.results_dir / "grid_all_topk.csv", index=False)
                pd.concat(all_history, ignore_index=True).to_csv(args.results_dir / "grid_all_history.csv", index=False)

                for name in [
                    "grid_all_metrics.csv",
                    "grid_all_heldout_predictions.csv",
                    "grid_all_topk.csv",
                    "grid_all_history.csv",
                ]:
                    upload_file_to_s3(args.results_dir / name, args.results_dir, args.results_s3_prefix)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    topk_all = pd.concat(all_topk, ignore_index=True)
    history_all = pd.concat(all_history, ignore_index=True)

    metrics_path = args.results_dir / "grid_all_metrics.csv"
    predictions_path = args.results_dir / "grid_all_heldout_predictions.csv"
    topk_path = args.results_dir / "grid_all_topk.csv"
    history_path = args.results_dir / "grid_all_history.csv"

    metrics_all.to_csv(metrics_path, index=False)
    predictions_all.to_csv(predictions_path, index=False)
    topk_all.to_csv(topk_path, index=False)
    history_all.to_csv(history_path, index=False)

    for path in [metrics_path, predictions_path, topk_path, history_path]:
        upload_file_to_s3(path, args.results_dir, args.results_s3_prefix)

    print("=" * 120)
    print("DONE")
    print(f"Local results: {args.results_dir.resolve()}")
    print(f"S3/MinIO results: {args.results_s3_prefix.rstrip('/')}")
    print("Main metrics preview:")
    preview_cols = [
        "run_group",
        "task",
        "version",
        "model",
        "seed",
        "calibration",
        "heldout_auroc",
        "heldout_auprc",
        "heldout_brier",
        "heldout_logloss",
    ]
    print(metrics_all[preview_cols].sort_values(["task", "model", "version", "seed", "calibration"]).to_string(index=False))

    if clearml_task is not None:
        clearml_task.get_logger().report_text(
            f"Grid finished. Results saved as files/folders in MinIO/S3: {args.results_s3_prefix.rstrip('/')}"
        )


if __name__ == "__main__":
    main()