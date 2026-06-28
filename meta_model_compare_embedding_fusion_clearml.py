from __future__ import annotations

import argparse
import inspect
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch.nn.utils.rnn import pack_padded_sequence

from common_ehrshot_eval import (
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    topk_metrics,
)

import train_sequence_compression_model_clearml as seq_base


META_COLS = {"task", "row_id", "subject_id", "prediction_time", "label", "split"}


# Same configs as in score_fusion_minimal / embedding fusion experiments.
BEST_TASK_CONFIGS = {
    "guo_readmission": {
        "sequence": {
            "version": "condition_era_180",
            "model": "RETAIN_lite",
        },
        "numeric_sequence": {
            "version": "raw",
            "model": "RETAIN_lite_numeric",
        },
    },
    "guo_icu": {
        "sequence": {
            "version": "compressed_dedup",
            "model": "LSTM_1L",
        },
        "numeric_sequence": {
            "version": "condition_era_90",
            "model": "GRU_2L_numeric",
        },
    },
}


RAW_TASK_CONFIGS = {
    "guo_readmission": {
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


FEATURE_GROUP_SPECS = {
    "tabular_features_only": ["tabular"],
    "sequence_embedding_only": ["sequence_embedding"],
    "numeric_sequence_embedding_only": ["numeric_sequence_embedding"],
    "tabular_plus_sequence_embedding": ["tabular", "sequence_embedding"],
    "tabular_plus_numeric_sequence_embedding": ["tabular", "numeric_sequence_embedding"],
    "tabular_plus_sequence_numeric_embeddings": [
        "tabular",
        "sequence_embedding",
        "numeric_sequence_embedding",
    ],
}


META_MODEL_CONFIGS = [
    {
        "name": "lr_l2_C001",
        "kind": "logreg",
        "C": 0.01,
        "scale": True,
    },
    {
        "name": "lr_l2_C003",
        "kind": "logreg",
        "C": 0.03,
        "scale": True,
    },
    {
        "name": "lr_l2_C01",
        "kind": "logreg",
        "C": 0.1,
        "scale": True,
    },
    {
        "name": "lr_l2_C1",
        "kind": "logreg",
        "C": 1.0,
        "scale": True,
    },
    {
        "name": "histgb_small",
        "kind": "histgb",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "min_samples_leaf": 30,
    },
    {
        "name": "histgb_medium",
        "kind": "histgb",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.1,
        "min_samples_leaf": 20,
    },
    {
        "name": "rf_small",
        "kind": "random_forest",
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
]


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


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


def resolve_input_file(path_or_url: str | Path | None, output_dir: Path) -> Path | None:
    if path_or_url is None or str(path_or_url).strip() == "":
        return None

    value = str(path_or_url)

    if value.startswith("s3://") or value.startswith("http://") or value.startswith("https://"):
        local_copy = get_required_local_copy(value)
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / Path(value.rstrip("/")).name
        if not dst.exists():
            print(f"Copying downloaded input {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)
        return dst

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def split_model_name_for_seq_base(model_name: str) -> tuple[str, bool]:
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
    kwargs = {"map_location": device}

    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    return torch.load(path, **kwargs)


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
        raise FileNotFoundError(
            "Checkpoint files are missing locally and checkpoint S3 prefix is not provided."
        )

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

            print(f"Copying {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)

    return checkpoint_dir


def make_tabular_feature_frame(
    cache_dir: Path,
    task: str,
    seed: int,
    top_n_codes: int,
    top_n_numeric_codes: int,
) -> tuple[pd.DataFrame, list[str]]:
    df = load_tabular_features(
        cache_dir=cache_dir,
        task=task,
        top_n_codes=top_n_codes,
        top_n_numeric_codes=top_n_numeric_codes,
    ).copy()

    if "task" not in df.columns:
        df["task"] = task

    feature_cols = [c for c in df.columns if c not in META_COLS]

    out = df[df["split"].isin(["tuning", "held_out"])].copy()
    out["seed"] = seed
    out["y_true"] = out["label"].astype(int)

    rename_map = {c: f"tab__{i:04d}" for i, c in enumerate(feature_cols)}
    out = out.rename(columns=rename_map)

    tab_cols = list(rename_map.values())
    keep_cols = ["task", "split", "seed", "row_id", "subject_id", "y_true"] + tab_cols
    out = out[keep_cols].copy()

    for c in ["row_id", "subject_id", "seed", "y_true"]:
        out[c] = out[c].astype(int)

    return out, tab_cols


def get_sequence_embedding_and_logits(model, batch_dev: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = batch_dev["tokens"]
    time_features = batch_dev["time_features"]
    numeric_features = batch_dev["numeric_features"]
    mask = batch_dev["mask"]
    lengths = batch_dev["lengths"]

    if hasattr(model, "rnn"):
        x = model.event_emb(tokens, time_features, numeric_features)
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h = model.rnn(packed)

        if isinstance(h, tuple):
            h = h[0]

        embedding = h[-1]
        logits = model.head(embedding).squeeze(-1)
        return logits, embedding

    if hasattr(model, "alpha_gru") and hasattr(model, "beta_gru"):
        x = model.event_emb(tokens, time_features, numeric_features)

        x_rev = seq_base.reverse_by_lengths(x, lengths)
        mask_rev = seq_base.reverse_by_lengths(
            mask.unsqueeze(-1).float(),
            lengths,
        ).squeeze(-1).bool()

        alpha_h, _ = model.alpha_gru(x_rev)
        beta_h, _ = model.beta_gru(x_rev)

        alpha_logits = model.alpha_fc(alpha_h).squeeze(-1).masked_fill(~mask_rev, -1e9)
        alpha = torch.softmax(alpha_logits, dim=1)
        beta = torch.tanh(model.beta_fc(beta_h))

        embedding = torch.sum(alpha.unsqueeze(-1) * beta * x_rev, dim=1)
        logits = model.head(embedding).squeeze(-1)
        return logits, embedding

    raise TypeError(
        f"Unsupported model type for embedding extraction: {type(model)}. "
        "Expected RNNClassifier or RetainLiteClassifier-like object."
    )


def run_embedding_inference(model, loader, device: torch.device) -> dict:
    model.eval()

    all_logits = []
    all_embeddings = []
    all_y = []
    all_row_ids = []
    all_subject_ids = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = seq_base.move_batch(batch, device)
            logits, embeddings = get_sequence_embedding_and_logits(model, batch_dev)

            all_logits.append(logits.detach().cpu().numpy())
            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_y.append(batch["labels"].numpy())
            all_row_ids.append(batch["row_ids"].numpy())
            all_subject_ids.append(batch["subject_ids"].numpy())

    return {
        "logits": np.concatenate(all_logits),
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "y_true": np.concatenate(all_y).astype(int),
        "row_id": np.concatenate(all_row_ids).astype(int),
        "subject_id": np.concatenate(all_subject_ids).astype(int),
    }


def make_compression_sequence_embeddings(
    sequence_data_dir: Path,
    checkpoint_dir: Path,
    task: str,
    version: str,
    model_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_prefix: str,
    numeric_min_count: int,
    emb_dim: int,
    hidden_dim: int,
    dropout: float,
) -> pd.DataFrame:
    set_global_seed(seed)

    base_model_name, use_numeric = split_model_name_for_seq_base(model_name)

    ckpt_file = find_checkpoint(
        ckpt_dir=checkpoint_dir,
        task=task,
        version=version,
        model_name=model_name,
        seed=seed,
    )

    print(f"Loading checkpoint for embeddings: {ckpt_file}")
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
        pred = run_embedding_inference(model, loaders[split_name], device)
        embeddings = pred["embeddings"].astype(np.float32)
        emb_cols = [f"{embedding_prefix}__{i:04d}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)

        meta_df = pd.DataFrame(
            {
                "task": task,
                "split": split_name,
                "seed": seed,
                "row_id": pred["row_id"].astype(int),
                "subject_id": pred["subject_id"].astype(int),
                "y_true": pred["y_true"].astype(int),
            }
        )

        frames.append(pd.concat([meta_df, emb_df], axis=1))

    return pd.concat(frames, ignore_index=True)


def merge_feature_frames(
    tabular: pd.DataFrame,
    sequence_emb: pd.DataFrame,
    numeric_sequence_emb: pd.DataFrame,
) -> pd.DataFrame:
    key = ["task", "split", "seed", "row_id", "subject_id", "y_true"]

    out = tabular.merge(sequence_emb, on=key, how="inner")
    out = out.merge(numeric_sequence_emb, on=key, how="inner")

    if out.empty:
        raise ValueError("Feature merge produced 0 rows")

    return out


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "tabular": [c for c in df.columns if c.startswith("tab__")],
        "sequence_embedding": [c for c in df.columns if c.startswith("seq_emb__")],
        "numeric_sequence_embedding": [c for c in df.columns if c.startswith("numseq_emb__")],
    }


def resolve_feature_cols(groups: dict[str, list[str]], group_names: list[str]) -> list[str]:
    cols = []
    for group_name in group_names:
        cols.extend(groups[group_name])
    if not cols:
        raise ValueError(f"No columns resolved for groups={group_names}")
    return cols


def filter_usable_columns(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    min_std: float = 1e-12,
) -> list[str]:
    usable = []

    for c in feature_cols:
        values = train_df[c].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        if np.nanstd(finite) <= min_std:
            continue
        usable.append(c)

    if not usable:
        raise ValueError("No usable feature columns after dropping all-NaN/constant columns")

    dropped = len(feature_cols) - len(usable)
    if dropped > 0:
        print(f"Dropped {dropped} all-NaN or constant feature columns")

    return usable


def make_meta_model(model_cfg: dict[str, Any], class_weight: str | None, seed: int) -> Pipeline:
    kind = model_cfg["kind"]

    if kind == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight=class_weight,
                        C=float(model_cfg["C"]),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if kind == "histgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=float(model_cfg["learning_rate"]),
                        max_iter=int(model_cfg["max_iter"]),
                        max_leaf_nodes=int(model_cfg["max_leaf_nodes"]),
                        l2_regularization=float(model_cfg["l2_regularization"]),
                        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if kind == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(model_cfg["n_estimators"]),
                        max_depth=int(model_cfg["max_depth"]),
                        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
                        max_features=model_cfg["max_features"],
                        class_weight="balanced_subsample" if class_weight == "balanced" else None,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown meta model kind: {kind}")


def evaluate_prediction(y_true: np.ndarray, risk: np.ndarray) -> dict:
    m = binary_ranking_metrics(y_true, risk)
    tk = topk_metrics(y_true, risk).set_index("top_frac")

    m["top_5pct_precision"] = float(tk.loc[0.05, "top_k_event_rate"])
    m["top_10pct_precision"] = float(tk.loc[0.10, "top_k_event_rate"])
    m["top_20pct_precision"] = float(tk.loc[0.20, "top_k_event_rate"])

    return m


def fit_predict_meta_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_cfg: dict[str, Any],
    class_weight: str | None,
    seed: int,
) -> tuple[np.ndarray, int]:
    usable_cols = filter_usable_columns(train_df, feature_cols)

    x_train = train_df[usable_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y_true"].astype(int).to_numpy()
    x_eval = eval_df[usable_cols].to_numpy(dtype=np.float32)

    model = make_meta_model(model_cfg=model_cfg, class_weight=class_weight, seed=seed)
    model.fit(x_train, y_train)

    risk = model.predict_proba(x_eval)[:, 1]
    return risk, len(usable_cols)


def run_single_seed_model_comparison(
    feature_df: pd.DataFrame,
    task: str,
    seed: int,
    class_weight: str | None,
    meta_model_configs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_seed = feature_df[(feature_df["task"] == task) & (feature_df["seed"] == seed)].copy()

    tuning = task_seed[task_seed["split"] == "tuning"].copy()
    heldout = task_seed[task_seed["split"] == "held_out"].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out rows for task={task}, seed={seed}")

    groups = get_feature_groups(task_seed)

    metric_rows = []
    pred_rows = []

    for model_cfg in meta_model_configs:
        for variant, group_names in FEATURE_GROUP_SPECS.items():
            feature_cols = resolve_feature_cols(groups, group_names)

            print(
                f"Fitting single-seed meta model: task={task}, seed={seed}, "
                f"meta_model={model_cfg['name']}, variant={variant}, n_raw_features={len(feature_cols)}"
            )

            risk, n_features_used = fit_predict_meta_model(
                train_df=tuning,
                eval_df=heldout,
                feature_cols=feature_cols,
                model_cfg=model_cfg,
                class_weight=class_weight,
                seed=seed,
            )

            y = heldout["y_true"].astype(int).to_numpy()
            metrics = evaluate_prediction(y, risk)

            metric_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "variant": variant,
                    "feature_groups": ",".join(group_names),
                    "meta_model": model_cfg["name"],
                    "meta_model_kind": model_cfg["kind"],
                    "n_features_raw": len(feature_cols),
                    "n_features_used": n_features_used,
                    **metrics,
                }
            )

            pred_rows.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "seed": seed,
                        "variant": variant,
                        "meta_model": model_cfg["name"],
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
        "n_features_used",
    ]

    id_cols = ["task", "variant", "feature_groups", "meta_model", "meta_model_kind"]

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

    counts = metrics.groupby(id_cols).agg(n_seeds=("seed", "nunique")).reset_index()
    return counts.merge(agg, on=id_cols, how="left")


def make_ensemble_feature_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["task", "split", "row_id", "subject_id", "y_true"]

    tab_cols = [c for c in feature_df.columns if c.startswith("tab__")]
    emb_cols = [
        c for c in feature_df.columns
        if c.startswith("seq_emb__") or c.startswith("numseq_emb__")
    ]

    tabular_part = (
        feature_df[id_cols + tab_cols]
        .drop_duplicates(subset=id_cols)
        .reset_index(drop=True)
    )

    emb_part = feature_df.groupby(id_cols)[emb_cols].mean().reset_index()
    out = tabular_part.merge(emb_part, on=id_cols, how="inner")
    out["seed"] = -1
    return out


def run_ensemble_model_comparison(
    ensemble_df: pd.DataFrame,
    task: str,
    class_weight: str | None,
    meta_model_configs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_df = ensemble_df[ensemble_df["task"] == task].copy()

    tuning = task_df[task_df["split"] == "tuning"].copy()
    heldout = task_df[task_df["split"] == "held_out"].copy()

    if tuning.empty or heldout.empty:
        raise ValueError(f"Missing tuning/held_out rows for task={task}")

    groups = get_feature_groups(task_df)

    metric_rows = []
    pred_rows = []

    for model_cfg in meta_model_configs:
        for variant, group_names in FEATURE_GROUP_SPECS.items():
            feature_cols = resolve_feature_cols(groups, group_names)

            print(
                f"Fitting ensemble meta model: task={task}, meta_model={model_cfg['name']}, "
                f"variant={variant}, n_raw_features={len(feature_cols)}"
            )

            risk, n_features_used = fit_predict_meta_model(
                train_df=tuning,
                eval_df=heldout,
                feature_cols=feature_cols,
                model_cfg=model_cfg,
                class_weight=class_weight,
                seed=42,
            )

            y = heldout["y_true"].astype(int).to_numpy()
            metrics = evaluate_prediction(y, risk)

            metric_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "feature_groups": ",".join(group_names),
                    "meta_model": model_cfg["name"],
                    "meta_model_kind": model_cfg["kind"],
                    "n_features_raw": len(feature_cols),
                    "n_features_used": n_features_used,
                    "base_embeddings": "mean_over_seeds",
                    **metrics,
                }
            )

            pred_rows.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "variant": variant,
                        "meta_model": model_cfg["name"],
                        "split": "held_out",
                        "row_id": heldout["row_id"].astype(int).to_numpy(),
                        "subject_id": heldout["subject_id"].astype(int).to_numpy(),
                        "y_true": y,
                        "risk": risk.astype(float),
                    }
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def build_embedding_features(
    args: argparse.Namespace,
    tasks: list[str],
    seeds: list[int],
    task_configs: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    all_feature_parts = []
    manifest_rows = []

    for task in tasks:
        cfg = task_configs[task]

        for seed in seeds:
            print("=" * 100)
            print(f"Building embedding fusion features: task={task}, seed={seed}")

            tab_df, _ = make_tabular_feature_frame(
                cache_dir=args.cache_dir,
                task=task,
                seed=seed,
                top_n_codes=args.top_n_codes,
                top_n_numeric_codes=args.top_n_numeric_codes,
            )

            seq_emb = make_compression_sequence_embeddings(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.sequence_checkpoint_dir,
                task=task,
                version=cfg["sequence"]["version"],
                model_name=cfg["sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_sequence,
                num_workers=args.num_workers,
                embedding_prefix="seq_emb",
                numeric_min_count=args.numeric_min_count,
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )

            numseq_emb = make_compression_sequence_embeddings(
                sequence_data_dir=args.sequence_data_dir,
                checkpoint_dir=args.numeric_checkpoint_dir,
                task=task,
                version=cfg["numeric_sequence"]["version"],
                model_name=cfg["numeric_sequence"]["model"],
                seed=seed,
                device=device,
                batch_size=args.batch_size_numeric,
                num_workers=args.num_workers,
                embedding_prefix="numseq_emb",
                numeric_min_count=args.numeric_min_count,
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )

            merged = merge_feature_frames(
                tabular=tab_df,
                sequence_emb=seq_emb,
                numeric_sequence_emb=numseq_emb,
            )

            all_feature_parts.append(merged)

            manifest_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "n_rows": len(merged),
                    "n_tabular_features": len([c for c in merged.columns if c.startswith("tab__")]),
                    "n_sequence_embedding_features": len([c for c in merged.columns if c.startswith("seq_emb__")]),
                    "n_numeric_sequence_embedding_features": len([c for c in merged.columns if c.startswith("numseq_emb__")]),
                    "sequence_version": cfg["sequence"]["version"],
                    "sequence_model": cfg["sequence"]["model"],
                    "numeric_sequence_version": cfg["numeric_sequence"]["version"],
                    "numeric_sequence_model": cfg["numeric_sequence"]["model"],
                }
            )

    return pd.concat(all_feature_parts, ignore_index=True), pd.DataFrame(manifest_rows)


def select_meta_model_configs(names: str) -> list[dict[str, Any]]:
    requested = parse_csv_list(names)
    by_name = {cfg["name"]: cfg for cfg in META_MODEL_CONFIGS}

    if not requested or requested == ["all"]:
        return META_MODEL_CONFIGS

    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown meta model names: {missing}. Available: {list(by_name)}")

    return [by_name[name] for name in requested]


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


def _to_bool(x: Any) -> bool:
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


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

    float_keys = {"dropout"}
    bool_keys = {"reuse_features", "save_feature_parquet"}
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
        elif key in bool_keys:
            setattr(args, key, _to_bool(value))
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
    print(f"  meta_models = {args.meta_models}")
    print(f"  reuse_features = {args.reuse_features}")
    print(f"  features_path = {args.features_path}")
    print(f"  ensemble_features_path = {args.ensemble_features_path}")
    print(f"  cache_dir = {args.cache_dir}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  device = {args.device}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if should_execute_remotely:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task, connected_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-dir", type=Path, default=Path("ehrshot_baseline_cache"))
    parser.add_argument("--cache-s3-prefix", type=str, default="")

    parser.add_argument("--sequence-data-dir", type=Path, default=Path("ehrshot_sequence_datasets"))
    parser.add_argument("--sequence-data-s3-prefix", type=str, default="")

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

    parser.add_argument("--checkpoint-s3-prefix", type=str, default="")
    parser.add_argument("--sequence-checkpoint-s3-prefix", type=str, default="")
    parser.add_argument("--numeric-checkpoint-s3-prefix", type=str, default="")

    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_embedding_meta_model_comparison"))

    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--tasks", type=str, default="guo_readmission,guo_icu")
    parser.add_argument("--sequence-config", type=str, default="best", choices=["raw", "best"])

    parser.add_argument("--top-n-codes", type=int, default=500)
    parser.add_argument("--top-n-numeric-codes", type=int, default=40)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size-sequence", type=int, default=16)
    parser.add_argument("--batch-size-numeric", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--numeric-min-count", type=int, default=3)

    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"])

    parser.add_argument(
        "--meta-models",
        type=str,
        default="all",
        help="Comma-separated meta model names or 'all'. Available: "
        + ",".join([cfg["name"] for cfg in META_MODEL_CONFIGS]),
    )

    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Load existing embedding feature parquet instead of recomputing embeddings.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="",
        help="Local or S3 path to embedding_fusion_features_tuning_heldout.parquet.",
    )
    parser.add_argument(
        "--ensemble-features-path",
        type=str,
        default="",
        help="Optional local or S3 path to embedding_fusion_ensemble_features.parquet.",
    )
    parser.add_argument(
        "--save-feature-parquet",
        action="store_true",
        help="Save built/reused feature parquet artifacts in output-dir.",
    )

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="embedding_meta_model_comparison")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")

    args = parser.parse_args()

    config = build_clearml_config(args)
    config["task_configs"] = TASK_CONFIG_PRESETS[args.sequence_config]
    config["meta_model_configs"] = META_MODEL_CONFIGS

    clearml_task, _ = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    tasks = parse_csv_list(args.tasks)
    device = get_device(args.device)
    class_weight = None if args.class_weight == "none" else "balanced"
    task_configs = TASK_CONFIG_PRESETS[args.sequence_config]
    meta_model_configs = select_meta_model_configs(args.meta_models)

    print("DEVICE:", device)
    print("TASKS:", tasks)
    print("SEEDS:", seeds)
    print("sequence_config:", args.sequence_config)
    print("class_weight:", class_weight)
    print("meta models:", [cfg["name"] for cfg in meta_model_configs])

    for task in tasks:
        if task not in task_configs:
            raise ValueError(f"Unknown task={task}. Available: {list(task_configs)}")

    manifest_df = pd.DataFrame()

    if args.reuse_features:
        features_path = resolve_input_file(args.features_path, args.output_dir)
        if features_path is None:
            raise ValueError("--reuse-features requires --features-path")
        print(f"Loading existing feature_df: {features_path}")
        feature_df = pd.read_parquet(features_path)

        ensemble_features_path = resolve_input_file(args.ensemble_features_path, args.output_dir)
        if ensemble_features_path is not None:
            print(f"Loading existing ensemble_df: {ensemble_features_path}")
            ensemble_df = pd.read_parquet(ensemble_features_path)
        else:
            ensemble_df = make_ensemble_feature_frame(feature_df)
    else:
        feature_df, manifest_df = build_embedding_features(
            args=args,
            tasks=tasks,
            seeds=seeds,
            task_configs=task_configs,
            device=device,
        )
        ensemble_df = make_ensemble_feature_frame(feature_df)

    # Keep only requested tasks, in case reused features contain more.
    feature_df = feature_df[feature_df["task"].isin(tasks)].copy()
    ensemble_df = ensemble_df[ensemble_df["task"].isin(tasks)].copy()

    if args.save_feature_parquet or not args.reuse_features:
        feature_df.to_parquet(args.output_dir / "embedding_features_tuning_heldout.parquet", index=False)
        ensemble_df.to_parquet(args.output_dir / "embedding_ensemble_features.parquet", index=False)

    if not manifest_df.empty:
        manifest_df.to_csv(args.output_dir / "embedding_feature_manifest.csv", index=False)

    print("Feature frame shape:", feature_df.shape)
    print(
        feature_df
        .groupby(["task", "split", "seed"])
        .agg(n=("row_id", "size"), n_positive=("y_true", "sum"), event_rate=("y_true", "mean"))
        .reset_index()
        .to_string(index=False)
    )

    single_metric_parts = []
    single_pred_parts = []

    for task in tasks:
        for seed in seeds:
            m, p = run_single_seed_model_comparison(
                feature_df=feature_df,
                task=task,
                seed=seed,
                class_weight=class_weight,
                meta_model_configs=meta_model_configs,
            )
            single_metric_parts.append(m)
            single_pred_parts.append(p)

    single_metrics = pd.concat(single_metric_parts, ignore_index=True)
    single_predictions = pd.concat(single_pred_parts, ignore_index=True)
    seed_summary = summarize_single_seed_metrics(single_metrics)

    single_metrics.to_csv(args.output_dir / "meta_model_single_seed_metrics.csv", index=False)
    single_predictions.to_csv(args.output_dir / "meta_model_single_seed_predictions.csv", index=False)
    seed_summary.to_csv(args.output_dir / "meta_model_seed_summary.csv", index=False)

    ens_metric_parts = []
    ens_pred_parts = []

    for task in tasks:
        m, p = run_ensemble_model_comparison(
            ensemble_df=ensemble_df,
            task=task,
            class_weight=class_weight,
            meta_model_configs=meta_model_configs,
        )
        ens_metric_parts.append(m)
        ens_pred_parts.append(p)

    ensemble_metrics = pd.concat(ens_metric_parts, ignore_index=True)
    ensemble_predictions = pd.concat(ens_pred_parts, ignore_index=True)

    ensemble_metrics.to_csv(args.output_dir / "meta_model_ensemble_metrics.csv", index=False)
    ensemble_predictions.to_csv(args.output_dir / "meta_model_ensemble_predictions.csv", index=False)

    print("\nSaved outputs to:", args.output_dir)

    print("\nSingle-seed meta-model comparison sorted by AUPRC:")
    show_cols = [
        "task",
        "variant",
        "meta_model",
        "n_seeds",
        "n_features_used__mean",
        "auprc__mean",
        "auprc__std",
        "auroc__mean",
        "brier__mean",
        "logloss__mean",
        "top_10pct_precision__mean",
    ]
    print(
        seed_summary[show_cols]
        .sort_values(["task", "auprc__mean"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nEnsemble meta-model comparison sorted by AUPRC:")
    show_cols = [
        "task",
        "variant",
        "meta_model",
        "n_features_used",
        "auprc",
        "auroc",
        "brier",
        "logloss",
        "top_10pct_precision",
        "n",
        "n_positive",
    ]
    print(
        ensemble_metrics[show_cols]
        .sort_values(["task", "auprc"], ascending=[True, False])
        .to_string(index=False)
    )

    if clearml_task is not None:
        clearml_task.upload_artifact("meta_model_single_seed_metrics", single_metrics)
        clearml_task.upload_artifact("meta_model_single_seed_predictions", single_predictions)
        clearml_task.upload_artifact("meta_model_seed_summary", seed_summary)
        clearml_task.upload_artifact("meta_model_ensemble_metrics", ensemble_metrics)
        clearml_task.upload_artifact("meta_model_ensemble_predictions", ensemble_predictions)
        if not manifest_df.empty:
            clearml_task.upload_artifact("embedding_feature_manifest", manifest_df)


if __name__ == "__main__":
    main()
