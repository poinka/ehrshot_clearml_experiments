from __future__ import annotations

import argparse
import inspect
import os
import shutil
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

import train_sequence_compression_model_clearml as seq_base


META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}


BEST_TASK_CONFIGS = {
    "guo_readmission": {
        "tabular_model": "HistGradientBoosting",

        # code-only companion model
        "sequence": {
            "version": "condition_era_180",
            "model": "RETAIN_lite",
        },

        # новая лучшая модель:
        # RETAIN_lite_numeric + raw
        # AUPRC = 0.397 ± 0.015
        # AUROC = 0.789 ± 0.008
        "numeric_sequence": {
            "version": "raw",
            "model": "RETAIN_lite_numeric",
        },
    },

    "guo_icu": {
        "tabular_model": "RandomForest_balanced",

        # code-only companion model 
        "sequence": {
            "version": "compressed_dedup",
            "model": "LSTM_1L",
        },

        # новая лучшая модель:
        # GRU_2L_numeric + condition_era_90
        # AUPRC = 0.204 ± 0.044
        # AUROC = 0.759 ± 0.041
        "numeric_sequence": {
            "version": "condition_era_90",
            "model": "GRU_2L_numeric",
        },
    },
}


RAW_TASK_CONFIGS = {
    "guo_readmission": {
        "tabular_model": "HistGradientBoosting",
        "sequence": {
            "version": "raw",
            "model": "RETAIN_lite",
        },
        "numeric_sequence": {
            "version": "raw",
            "model": "RETAIN_lite_numeric",
        },
    },
    "guo_icu": {
        "tabular_model": "RandomForest_balanced",
        "sequence": {
            "version": "raw",
            "model": "GRU_2L",
        },
        "numeric_sequence": {
            "version": "raw",
            "model": "GRU_2L_numeric",
        },
    },
}


TASK_CONFIG_PRESETS = {
    "best": BEST_TASK_CONFIGS,
    "raw": RAW_TASK_CONFIGS,
}


FUSION_SPECS = {
    "tabular_only": ["p_tabular"],
    "sequence_only": ["p_sequence"],
    "numeric_sequence_only": ["p_numeric_sequence"],
    "fusion_tabular_sequence": ["p_tabular", "p_sequence"],
    "fusion_tabular_numeric_sequence": ["p_tabular", "p_numeric_sequence"],
}

def get_required_local_copy(remote_url: str) -> Path:
    from clearml import StorageManager

    local_copy = StorageManager.get_local_copy(remote_url=remote_url)

    if local_copy is None:
        raise FileNotFoundError(
            "ClearML StorageManager could not download object.\n"
            f"Remote URL not found or not accessible:\n{remote_url}\n\n"
            "Check that the MinIO prefix is correct and that the file exists."
        )

    path = Path(local_copy)

    if not path.exists():
        raise FileNotFoundError(
            "ClearML StorageManager returned a path, but it does not exist.\n"
            f"Remote URL: {remote_url}\n"
            f"Local path: {path}"
        )

    return path

def split_model_name_for_seq_base(model_name: str) -> tuple[str, bool]:
    """
    File/checkpoint name can be:
      RETAIN_lite
      RETAIN_lite_numeric
      GRU_2L
      GRU_2L_numeric

    But train_sequence_compression_model_clearml.make_model expects:
      model = RETAIN_lite / GRU_2L / LSTM_1L ...
      use_numeric = True/False
    """
    if model_name.endswith("_numeric"):
        return model_name.removesuffix("_numeric"), True

    return model_name, False

def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict:
    """
    PyTorch >= 2.6 может использовать weights_only=True по умолчанию.
    В numeric checkpoints могут лежать numpy arrays, поэтому явно разрешаем full load.
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


def transform_score_features(
    x: pd.DataFrame,
    score_cols: list[str],
    mode: str,
) -> np.ndarray:
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


def load_tabular_features(
    cache_dir: Path,
    task: str,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    path = cache_dir / f"{task}_features_top{top_n_codes}_num{top_n_numeric_codes}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Tabular feature cache not found: {path}")

    return pd.read_parquet(path)


def maybe_download_tabular_cache_from_s3_prefix(
    cache_dir: Path,
    cache_s3_prefix: str,
    tasks: list[str],
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> Path:
    """
    Если tabular cache есть локально, используем его.
    Иначе скачиваем нужные parquet из MinIO/S3.

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

        local_copy = get_required_local_copy(remote_url)

        if not local_copy.exists():
            raise FileNotFoundError(f"StorageManager returned missing local file: {local_copy}")

        print(f"Copying {local_copy} -> {dst}")
        shutil.copy2(local_copy, dst)

    return cache_dir


def required_sequence_dataset_configs(
    task_configs: dict,
    tasks: list[str],
) -> list[dict]:
    configs = []
    seen = set()

    for task in tasks:
        cfg = task_configs[task]

        for role in ["sequence", "numeric_sequence"]:
            version = cfg[role]["version"]
            key = (task, version)

            if key in seen:
                continue

            seen.add(key)
            configs.append(
                {
                    "task": task,
                    "version": version,
                    "model": cfg[role]["model"],
                }
            )

    return configs


def required_checkpoint_configs(
    task_configs: dict,
    tasks: list[str],
    role: str,
) -> list[dict]:
    configs = []

    for task in tasks:
        cfg = task_configs[task][role]
        configs.append(
            {
                "task": task,
                "version": cfg["version"],
                "model": cfg["model"],
            }
        )

    return configs


def maybe_download_sequence_datasets_from_s3_prefix(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    configs: list[dict],
) -> Path:
    """
    Скачивает sequence datasets из MinIO/S3, если их нет локально.

    Expected remote layout:
        <prefix>/<task>/<version>/examples.parquet
        <prefix>/<task>/<version>/vocab.json
    """
    required_pairs = sorted({(cfg["task"], cfg["version"]) for cfg in configs})

    expected_files = []
    for task, version in required_pairs:
        expected_files.append(sequence_data_dir / task / version / "examples.parquet")
        expected_files.append(sequence_data_dir / task / version / "vocab.json")

    if all(p.exists() for p in expected_files):
        print(f"Using local sequence datasets: {sequence_data_dir}")
        return sequence_data_dir

    if not sequence_data_s3_prefix:
        missing = [str(p) for p in expected_files if not p.exists()]
        raise FileNotFoundError(
            "Sequence dataset files are missing locally and --sequence-data-s3-prefix is not provided. "
            f"Missing files: {missing[:20]}"
        )

    from clearml import StorageManager

    prefix = sequence_data_s3_prefix.rstrip("/")

    for task, version in required_pairs:
        local_dir = sequence_data_dir / task / version
        local_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["examples.parquet", "vocab.json"]:
            dst = local_dir / fname

            if dst.exists():
                print(f"Already exists locally: {dst}")
                continue

            remote_url = f"{prefix}/{task}/{version}/{fname}"
            print(f"Downloading {remote_url}")

            local_copy = get_required_local_copy(remote_url)

            if not local_copy.exists():
                raise FileNotFoundError(f"StorageManager returned missing local file: {local_copy}")

            print(f"Copying {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)

    return sequence_data_dir


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


def maybe_download_checkpoints_from_s3_prefix(
    checkpoint_dir: Path,
    checkpoint_s3_prefix: str,
    configs: list[dict],
    seeds: list[int],
) -> Path:
    """
    Скачивает выбранные checkpoints из MinIO/S3, если их нет локально.

    Expected remote layout:
        <checkpoint_s3_prefix>/<task>__<version>__<model>__seed<seed>.pt

    Example:
        guo_icu__raw__GRU_2L_numeric__seed42.pt
    """
    expected_files = []

    for cfg in configs:
        for seed in seeds:
            expected_files.append(
                checkpoint_dir / f"{cfg['task']}__{cfg['version']}__{cfg['model']}__seed{seed}.pt"
            )

    locally_ok = True

    for cfg in configs:
        for seed in seeds:
            try:
                find_checkpoint(
                    ckpt_dir=checkpoint_dir,
                    task=cfg["task"],
                    version=cfg["version"],
                    model_name=cfg["model"],
                    seed=seed,
                )
            except FileNotFoundError:
                locally_ok = False
                break

        if not locally_ok:
            break

    if locally_ok:
        print(f"Using local checkpoints: {checkpoint_dir}")
        return checkpoint_dir

    if not checkpoint_s3_prefix:
        missing = [str(p) for p in expected_files if not p.exists()]
        raise FileNotFoundError(
            "Checkpoint files are missing locally and checkpoint S3 prefix is not provided. "
            f"Missing files: {missing[:20]}"
        )

    from clearml import StorageManager

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prefix = checkpoint_s3_prefix.rstrip("/")

    seen_remote = set()

    for cfg in configs:
        for seed in seeds:
            fname = f"{cfg['task']}__{cfg['version']}__{cfg['model']}__seed{seed}.pt"
            dst = checkpoint_dir / fname

            if dst.exists():
                print(f"Already exists locally: {dst}")
                continue

            remote_url = f"{prefix}/{fname}"

            if remote_url in seen_remote:
                continue

            seen_remote.add(remote_url)

            print(f"Downloading {remote_url}")

            local_copy = get_required_local_copy(remote_url)

            if not local_copy.exists():
                raise FileNotFoundError(f"StorageManager returned missing local file: {local_copy}")

            print(f"Copying {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)

    return checkpoint_dir


def make_tabular_scores(
    cache_dir: Path,
    task: str,
    model_name: str,
    seed: int,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> pd.DataFrame:
    """
    Обучает tabular baseline на train split и возвращает raw scores для tuning + held_out.

    Важно:
    - здесь не делаем Platt calibration;
    - fusion LogisticRegression сама учится на tuning.
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


def make_compression_sequence_scores(
    sequence_data_dir: Path,
    checkpoint_dir: Path,
    task: str,
    version: str,
    model_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    score_col: str,
    numeric_min_count: int,
    emb_dim: int,
    hidden_dim: int,
    dropout: float,
) -> pd.DataFrame:
    """
    Inference for checkpoints trained by train_sequence_compression_model_clearml.py.

    Works for:
      - code-only sequence models, e.g. RETAIN_lite
      - numeric sequence models, e.g. RETAIN_lite_numeric / GRU_2L_numeric

    Important:
      model_name is the checkpoint filename model name.
      seq_base.make_model needs base_model_name + use_numeric.
    """
    set_global_seed(seed)

    base_model_name, use_numeric = split_model_name_for_seq_base(model_name)

    ckpt_file = find_checkpoint(
        ckpt_dir=checkpoint_dir,
        task=task,
        version=version,
        model_name=model_name,
        seed=seed,
    )

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Sequence checkpoint not found: {ckpt_file}")

    print(f"Loading compression checkpoint: {ckpt_file}")
    print(f"  file model_name={model_name}")
    print(f"  seq_base model={base_model_name}")
    print(f"  use_numeric={use_numeric}")

    ckpt = safe_torch_load(ckpt_file, device)

    df = seq_base.load_sequence_examples(sequence_data_dir, task, version)
    vocab = seq_base.load_vocab(sequence_data_dir, task, version)
    vocab_size = len(vocab)

    max_len = int(ckpt.get("max_len", 4096))

    run_args = argparse.Namespace(
        sequence_dir=sequence_data_dir,
        sequence_data_s3_prefix="",
        sequence_data_artifact_name="",
        results_dir=Path("."),

        task=task,
        version=version,
        model=base_model_name,
        seed=seed,
        device=str(device),
        use_numeric=use_numeric,

        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,

        epochs=0,
        patience=0,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,

        max_train_examples=0,
        max_eval_examples=0,
        numeric_min_count=numeric_min_count,
        top_fracs=[0.05, 0.10, 0.20],
        save_checkpoint=False,

        clearml=False,
        execute_remotely=False,
        clearml_queue="",
        clearml_task_name="",
        clearml_project="",
        clearml_output_uri="",
    )

    df_train = df[df["split"] == "train"].copy()

    token_mean, token_std, _ = seq_base.compute_token_numeric_stats(
        df_train,
        vocab_size=vocab_size,
        min_count=numeric_min_count,
    )

    loaders, _ = seq_base.make_loaders(
        df=df,
        args=run_args,
        mean=token_mean,
        std=token_std,
    )

    model = seq_base.make_model(vocab_size, run_args)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    frames = []

    for split_name in ["tuning", "held_out"]:
        pred = seq_base.run_inference(model, loaders[split_name], device)
        p = seq_base.sigmoid_np(pred["logits"])

        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "split": split_name,
                    "seed": seed,
                    "row_id": pred["row_id"].astype(int),
                    "subject_id": pred["subject_id"].astype(int),
                    "y_true": pred["y_true"].astype(int),
                    score_col: p.astype(float),
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


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args) -> dict:
    config = vars(args).copy()

    for key in [
        "cache_dir",
        "sequence_data_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
    ]:
        if key in config:
            config[key] = str(config[key])

    return config


def sync_args_from_clearml_config(args, config: dict) -> None:
    path_keys = {
        "cache_dir",
        "sequence_data_dir",
        "sequence_checkpoint_dir",
        "numeric_checkpoint_dir",
        "output_dir",
    }

    int_keys = {
        "top_n_codes",
        "top_n_numeric_codes",
        "batch_size_sequence",
        "batch_size_numeric",
        "num_workers",
        "emb_dim",
        "hidden_dim",
        "numeric_min_count",
    }

    float_keys = {
        "dropout",
    }

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
        elif key in float_keys:
            setattr(args, key, float(value))
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
    print(f"  sequence_config = {args.sequence_config}")
    print(f"  seeds = {args.seeds}")
    print(f"  tasks = {args.tasks}")
    print(f"  cache_dir = {args.cache_dir}")
    print(f"  cache_s3_prefix = {args.cache_s3_prefix}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
    print(f"  checkpoint_s3_prefix = {args.checkpoint_s3_prefix}")
    print(f"  sequence_checkpoint_s3_prefix = {args.sequence_checkpoint_s3_prefix}")
    print(f"  numeric_checkpoint_s3_prefix = {args.numeric_checkpoint_s3_prefix}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  device = {args.device}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if should_execute_remotely:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task, connected_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ehrshot_baseline_cache"),
    )
    parser.add_argument(
        "--cache-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with tabular cache parquet files.",
    )

    parser.add_argument(
        "--sequence-data-dir",
        type=Path,
        default=Path("ehrshot_sequence_datasets"),
    )
    parser.add_argument(
        "--sequence-data-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with sequence datasets.",
    )

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

    parser.add_argument(
        "--checkpoint-s3-prefix",
        type=str,
        default="",
        help="Common MinIO/S3 prefix with all sequence and numeric checkpoints.",
    )
    parser.add_argument(
        "--sequence-checkpoint-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with code-only sequence checkpoints. If empty, --checkpoint-s3-prefix is used.",
    )
    parser.add_argument(
        "--numeric-checkpoint-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with numeric sequence checkpoints. If empty, --checkpoint-s3-prefix is used.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_score_fusion_minimal"),
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="guo_readmission,guo_icu",
    )
    parser.add_argument(
        "--sequence-config",
        type=str,
        default="raw",
        choices=["raw", "best"],
        help="Which sequence checkpoints to fuse: raw or previously selected best compressed configs.",
    )

    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size-sequence", type=int, default=64)
    parser.add_argument("--batch-size-numeric", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--numeric-min-count", type=int, default=3)
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

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument(
        "--execute-remotely",
        action="store_true",
        help="If set, enqueue this ClearML task to an agent queue and stop local execution.",
    )
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="score_fusion_minimal_raw",
    )
    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )

    args = parser.parse_args()

    config = build_clearml_config(args)
    config["task_configs"] = TASK_CONFIG_PRESETS[args.sequence_config]

    clearml_task, config = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    device = get_device(args.device)

    class_weight = None if args.class_weight == "none" else "balanced"
    task_configs = TASK_CONFIG_PRESETS[args.sequence_config]

    print("DEVICE:", device)
    print("TASKS:", tasks)
    print("SEEDS:", seeds)
    print("score_transform:", args.score_transform)
    print("class_weight:", class_weight)
    print("sequence_config:", args.sequence_config)

    for task in tasks:
        if task not in task_configs:
            raise ValueError(f"Unknown task={task}. Available: {list(task_configs)}")

    args.cache_dir = maybe_download_tabular_cache_from_s3_prefix(
        cache_dir=args.cache_dir,
        cache_s3_prefix=args.cache_s3_prefix,
        tasks=tasks,
        top_n_codes=args.top_n_codes,
        top_n_numeric_codes=args.top_n_numeric_codes,
    )

    dataset_configs = required_sequence_dataset_configs(task_configs, tasks)

    args.sequence_data_dir = maybe_download_sequence_datasets_from_s3_prefix(
        sequence_data_dir=args.sequence_data_dir,
        sequence_data_s3_prefix=args.sequence_data_s3_prefix,
        configs=dataset_configs,
    )

    sequence_ckpt_prefix = args.sequence_checkpoint_s3_prefix or args.checkpoint_s3_prefix
    numeric_ckpt_prefix = args.numeric_checkpoint_s3_prefix or args.checkpoint_s3_prefix

    args.sequence_checkpoint_dir = maybe_download_checkpoints_from_s3_prefix(
        checkpoint_dir=args.sequence_checkpoint_dir,
        checkpoint_s3_prefix=sequence_ckpt_prefix,
        configs=required_checkpoint_configs(task_configs, tasks, role="sequence"),
        seeds=seeds,
    )

    args.numeric_checkpoint_dir = maybe_download_checkpoints_from_s3_prefix(
        checkpoint_dir=args.numeric_checkpoint_dir,
        checkpoint_s3_prefix=numeric_ckpt_prefix,
        configs=required_checkpoint_configs(task_configs, tasks, role="numeric_sequence"),
        seeds=seeds,
    )

    all_base_scores = []

    for task in tasks:
        cfg = task_configs[task]

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

            seq_scores = make_compression_sequence_scores(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.sequence_checkpoint_dir,
                task=task,
                version=cfg["sequence"]["version"],
                model_name=cfg["sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_sequence,
                num_workers=args.num_workers,
                score_col="p_sequence",
                numeric_min_count=args.numeric_min_count,
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )

            numseq_scores = make_compression_sequence_scores(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.numeric_checkpoint_dir,
                task=task,
                version=cfg["numeric_sequence"]["version"],
                model_name=cfg["numeric_sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_numeric,
                num_workers=args.num_workers,
                score_col="p_numeric_sequence",
                numeric_min_count=args.numeric_min_count,
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
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

    if clearml_task is not None:
        clearml_task.upload_artifact("base_scores_tuning_heldout", base_scores)
        clearml_task.upload_artifact("fusion_single_seed_metrics", single_metrics)
        clearml_task.upload_artifact("fusion_single_seed_predictions", single_predictions)
        clearml_task.upload_artifact("fusion_seed_summary", seed_summary)
        clearml_task.upload_artifact("ensemble_base_scores_tuning_heldout", ensemble_scores)
        clearml_task.upload_artifact("fusion_ensemble_metrics", ensemble_metrics)
        clearml_task.upload_artifact("fusion_ensemble_predictions", ensemble_predictions)


if __name__ == "__main__":
    main()