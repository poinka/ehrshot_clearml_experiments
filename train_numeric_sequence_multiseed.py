from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from common_ehrshot_eval import (
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    topk_metrics,
)

PAD_ID = 0
UNK_ID = 1

DEFAULT_CONFIGS = [
    # readmission numeric
    {"task": "guo_readmission", "version": "raw", "model": "RETAIN_lite_numeric"},
    {"task": "guo_readmission", "version": "compressed_dedup", "model": "RETAIN_lite_numeric"},

    # ICU numeric: raw baselines
    {"task": "guo_icu", "version": "raw", "model": "GRU_1L_numeric"},
    {"task": "guo_icu", "version": "raw", "model": "LSTM_1L_numeric"},

    # ICU numeric: clean raw vs compressed for same architecture
    {"task": "guo_icu", "version": "raw", "model": "GRU_2L_numeric"},
    {"task": "guo_icu", "version": "compressed_hybrid", "model": "GRU_2L_numeric"},
]


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_plain_list(x):
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []


def load_sequence_examples(data_dir: Path, task: str, version: str) -> pd.DataFrame:
    path = data_dir / task / version / "examples.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Sequence examples not found: {path}")
    return pd.read_parquet(path)


def load_vocab(data_dir: Path, task: str, version: str) -> dict:
    with open(data_dir / task / version / "vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)

def maybe_download_sequence_datasets_from_s3_prefix(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    configs: list[dict],
) -> Path:
    """
    Download required sequence dataset files from MinIO/S3 if they are not available locally.

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
            local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

            if not local_copy.exists():
                raise FileNotFoundError(f"StorageManager returned missing local file: {local_copy}")

            print(f"Copying {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)

    return sequence_data_dir

def compute_token_numeric_stats(df_train: pd.DataFrame, vocab_size: int, min_count: int = 3):
    count = np.zeros(vocab_size, dtype=np.float64)
    sum_x = np.zeros(vocab_size, dtype=np.float64)
    sum_x2 = np.zeros(vocab_size, dtype=np.float64)
    for _, row in df_train.iterrows():
        token_ids = to_plain_list(row["token_ids"])
        nums = to_plain_list(row.get("numeric_values", []))
        if len(nums) < len(token_ids):
            nums += [np.nan] * (len(token_ids) - len(nums))
        for token_id, value in zip(token_ids, nums):
            try:
                x = float(value)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            token_id = int(token_id)
            if 0 <= token_id < vocab_size:
                count[token_id] += 1
                sum_x[token_id] += x
                sum_x2[token_id] += x * x
    mean = np.zeros(vocab_size, dtype=np.float32)
    std = np.ones(vocab_size, dtype=np.float32)
    ok = count >= min_count
    mean[ok] = (sum_x[ok] / count[ok]).astype(np.float32)
    var = np.zeros(vocab_size, dtype=np.float64)
    var[ok] = np.maximum(sum_x2[ok] / count[ok] - mean[ok].astype(np.float64) ** 2, 1e-6)
    std[ok] = np.sqrt(var[ok]).astype(np.float32)
    stats = pd.DataFrame({"token_id": np.arange(vocab_size), "numeric_count": count, "numeric_mean": mean, "numeric_std": std})
    return mean, std, stats


class EHRSequenceNumericDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        token_ids = to_plain_list(row["token_ids"])
        days_before = to_plain_list(row["days_before_prediction"])
        delta_days = to_plain_list(row["delta_days"])
        nums = to_plain_list(row.get("numeric_values", []))
        if len(token_ids) == 0:
            token_ids = [UNK_ID]
            days_before = [0.0]
            delta_days = [0.0]
            nums = [np.nan]
        n = len(token_ids)
        if len(days_before) < n:
            days_before += [0.0] * (n - len(days_before))
        if len(delta_days) < n:
            delta_days += [0.0] * (n - len(delta_days))
        if len(nums) < n:
            nums += [np.nan] * (n - len(nums))
        return {
            "row_id": int(row["row_id"]),
            "subject_id": int(row["subject_id"]),
            "tokens": token_ids[-self.max_len:],
            "days_before": days_before[-self.max_len:],
            "delta_days": delta_days[-self.max_len:],
            "numeric_values": nums[-self.max_len:],
            "label": int(row["label"]),
        }


def make_collate_fn(token_mean: np.ndarray, token_std: np.ndarray):
    mean_t = torch.tensor(token_mean, dtype=torch.float32)
    std_t = torch.tensor(token_std, dtype=torch.float32)

    def collate(batch):
        bs = len(batch)
        lengths = torch.tensor([len(x["tokens"]) for x in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        tokens = torch.full((bs, max_len), PAD_ID, dtype=torch.long)
        time_features = torch.zeros((bs, max_len, 2), dtype=torch.float32)
        numeric_features = torch.zeros((bs, max_len, 2), dtype=torch.float32)
        mask = torch.zeros((bs, max_len), dtype=torch.bool)
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
        row_ids = torch.tensor([x["row_id"] for x in batch], dtype=torch.long)
        subject_ids = torch.tensor([x["subject_id"] for x in batch], dtype=torch.long)
        for i, item in enumerate(batch):
            l = len(item["tokens"])
            tok = torch.tensor(item["tokens"], dtype=torch.long)
            tokens[i, :l] = tok
            mask[i, :l] = True
            days = torch.tensor(item["days_before"], dtype=torch.float32).clamp(min=0)
            delta = torch.tensor(item["delta_days"], dtype=torch.float32).clamp(min=0)
            time_features[i, :l, 0] = torch.log1p(days)
            time_features[i, :l, 1] = torch.log1p(delta)
            raw_nums = []
            present = []
            for v in item["numeric_values"]:
                try:
                    x = float(v)
                    ok = np.isfinite(x)
                except Exception:
                    x, ok = 0.0, False
                raw_nums.append(x if ok else 0.0)
                present.append(1.0 if ok else 0.0)
            raw_nums = torch.tensor(raw_nums, dtype=torch.float32)
            present = torch.tensor(present, dtype=torch.float32)
            z = ((raw_nums - mean_t[tok]) / std_t[tok].clamp(min=1e-6)).clamp(-10.0, 10.0)
            z = torch.where(present > 0, z, torch.zeros_like(z))
            numeric_features[i, :l, 0] = z
            numeric_features[i, :l, 1] = present
        return {"tokens": tokens, "time_features": time_features, "numeric_features": numeric_features, "mask": mask, "lengths": lengths, "labels": labels, "row_ids": row_ids, "subject_ids": subject_ids}

    return collate


class Time2VecEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, out_dim: int = 64):
        super().__init__()
        linear_dim = max(1, out_dim // 2)
        periodic_dim = out_dim - linear_dim
        self.linear = nn.Linear(input_dim, linear_dim)
        self.periodic = nn.Linear(input_dim, periodic_dim) if periodic_dim > 0 else None

    def forward(self, x):
        parts = [self.linear(x)]
        if self.periodic is not None:
            parts.append(torch.sin(self.periodic(x)))
        return torch.cat(parts, dim=-1)


class EventEmbeddingNumeric(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.time_proj = Time2VecEncoder(2, emb_dim)
        self.numeric_proj = nn.Sequential(nn.Linear(2, emb_dim), nn.GELU(), nn.LayerNorm(emb_dim))
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, time_features, numeric_features):
        x = self.token_emb(tokens) + self.time_proj(time_features) + self.numeric_proj(numeric_features)
        return self.dropout(self.norm(x))


def reverse_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for i, l in enumerate(lengths.tolist()):
        out[i, :l] = x[i, :l].flip(0)
    return out


class RNNNumericClassifier(nn.Module):
    def __init__(self, vocab_size: int, rnn_type: str = "GRU", emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.20):
        super().__init__()
        self.event_emb = EventEmbeddingNumeric(vocab_size, emb_dim, dropout)
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, tokens, time_features, numeric_features, mask, lengths):
        x = self.event_emb(tokens, time_features, numeric_features)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        return self.head(h[-1]).squeeze(-1)


class RetainLiteNumericClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.event_emb = EventEmbeddingNumeric(vocab_size, emb_dim, dropout)
        self.alpha_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, emb_dim)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(dropout), nn.Linear(emb_dim, 1))

    def forward(self, tokens, time_features, numeric_features, mask, lengths):
        x = self.event_emb(tokens, time_features, numeric_features)
        x_rev = reverse_by_lengths(x, lengths)
        mask_rev = reverse_by_lengths(mask.unsqueeze(-1).float(), lengths).squeeze(-1).bool()
        alpha_h, _ = self.alpha_gru(x_rev)
        beta_h, _ = self.beta_gru(x_rev)
        alpha_logits = self.alpha_fc(alpha_h).squeeze(-1).masked_fill(~mask_rev, -1e9)
        alpha = torch.softmax(alpha_logits, dim=1)
        beta = torch.tanh(self.beta_fc(beta_h))
        context = torch.sum(alpha.unsqueeze(-1) * beta * x_rev, dim=1)
        return self.head(context).squeeze(-1)


def make_model(model_name: str, vocab_size: int) -> nn.Module:
    if model_name == "GRU_1L_numeric":
        return RNNNumericClassifier(vocab_size, "GRU", num_layers=1)
    if model_name == "GRU_2L_numeric":
        return RNNNumericClassifier(vocab_size, "GRU", num_layers=2)
    if model_name == "LSTM_1L_numeric":
        return RNNNumericClassifier(vocab_size, "LSTM", num_layers=1)
    if model_name == "LSTM_2L_numeric":
        return RNNNumericClassifier(vocab_size, "LSTM", num_layers=2)
    if model_name == "RETAIN_lite_numeric":
        return RetainLiteNumericClassifier(vocab_size)
    raise ValueError(f"Unknown numeric sequence model: {model_name}")


def make_loaders(df, token_mean, token_std, batch_size, max_len, seed, num_workers):
    splits = {s: df[df["split"] == s].copy().reset_index(drop=True) for s in ["train", "tuning", "held_out"]}
    collate = make_collate_fn(token_mean, token_std)
    generator = torch.Generator(); generator.manual_seed(seed)
    loaders = {
        "train": DataLoader(EHRSequenceNumericDataset(splits["train"], max_len), batch_size=batch_size, shuffle=True, generator=generator, num_workers=num_workers, collate_fn=collate),
        "tuning": DataLoader(EHRSequenceNumericDataset(splits["tuning"], max_len), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate),
        "held_out": DataLoader(EHRSequenceNumericDataset(splits["held_out"], max_len), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate),
    }
    return loaders


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -30, 30)))


def run_inference(model, loader, device):
    model.eval()
    out = {"logits": [], "y_true": [], "row_id": [], "subject_id": []}
    with torch.no_grad():
        for batch in loader:
            b = move_batch(batch, device)
            logits = model(b["tokens"], b["time_features"], b["numeric_features"], b["mask"], b["lengths"])
            out["logits"].append(logits.detach().cpu().numpy())
            out["y_true"].append(batch["labels"].numpy())
            out["row_id"].append(batch["row_ids"].numpy())
            out["subject_id"].append(batch["subject_ids"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def train_model(model, loaders, args, device):
    model.to(device)
    y_train = np.concatenate([batch["labels"].numpy() for batch in loaders["train"]]).astype(int)
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - y_train.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_auprc = -np.inf
    best_epoch = -1
    bad = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            b = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(b["tokens"], b["time_features"], b["numeric_features"], b["mask"], b["lengths"])
            loss = criterion(logits, b["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        tuning = run_inference(model, loaders["tuning"], device)
        risk = sigmoid_np(tuning["logits"])
        m = binary_ranking_metrics(tuning["y_true"], risk)
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "tuning_auroc": m["auroc"], "tuning_auprc": m["auprc"], "tuning_brier": m["brier"], "best_epoch": best_epoch})
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} tuning_auprc={m['auprc']:.4f} tuning_auroc={m['auroc']:.4f}")
        if m["auprc"] > best_auprc + 1e-5:
            best_auprc = m["auprc"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows); history["best_epoch"] = best_epoch
    return model, history


def evaluate_with_platt(task, version, model_name, seed, model, loaders, device):
    tuning = run_inference(model, loaders["tuning"], device)
    heldout = run_inference(model, loaders["held_out"], device)
    probs = {"raw": sigmoid_np(heldout["logits"])}
    try:
        cal = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        cal.fit(tuning["logits"].reshape(-1, 1), tuning["y_true"].astype(int))
        probs["platt"] = cal.predict_proba(heldout["logits"].reshape(-1, 1))[:, 1]
    except Exception as e:
        print("Platt calibration failed:", repr(e))
    result_rows, pred_frames, topk_frames = [], [], []
    for calibration, risk in probs.items():
        m = binary_ranking_metrics(heldout["y_true"], risk)
        result_rows.append({"task": task, "family": "numeric_sequence", "source": "numeric_sequence", "version": version, "model": model_name, "seed": seed, "calibration": calibration, **{f"heldout_{k}": v for k, v in m.items()}})
        pred_frames.append(pd.DataFrame({"task": task, "family": "numeric_sequence", "source": "numeric_sequence", "version": version, "model": model_name, "seed": seed, "calibration": calibration, "row_id": heldout["row_id"].astype(int), "subject_id": heldout["subject_id"].astype(int), "y_true": heldout["y_true"].astype(int), "logit": heldout["logits"], "risk": risk}))
        tk = topk_metrics(heldout["y_true"], risk)
        tk.insert(0, "calibration", calibration)
        tk.insert(0, "seed", seed)
        tk.insert(0, "model", model_name)
        tk.insert(0, "version", version)
        tk.insert(0, "family", "numeric_sequence")
        tk.insert(0, "task", task)
        topk_frames.append(tk)
    return pd.DataFrame(result_rows), pd.concat(pred_frames, ignore_index=True), pd.concat(topk_frames, ignore_index=True)


def run_one(args, cfg, seed, device):
    set_global_seed(seed)
    df = load_sequence_examples(args.sequence_data_dir, cfg["task"], cfg["version"])
    vocab = load_vocab(args.sequence_data_dir, cfg["task"], cfg["version"])
    vocab_size = len(vocab)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    token_mean, token_std, token_stats = compute_token_numeric_stats(train_df, vocab_size=vocab_size)
    loaders = make_loaders(df, token_mean, token_std, args.batch_size, args.max_len, seed, args.num_workers)
    model = make_model(cfg["model"], vocab_size)
    model, history = train_model(model, loaders, args, device)
    result_df, pred_df, topk_df = evaluate_with_platt(cfg["task"], cfg["version"], cfg["model"], seed, model, loaders, device)
    stem = f"{cfg['task']}__{cfg['version']}__{cfg['model']}__seed{seed}"
    history.to_csv(args.output_dir / f"{stem}__history.csv", index=False)
    result_df.to_csv(args.output_dir / f"{stem}__results.csv", index=False)
    pred_df.to_csv(args.output_dir / f"{stem}__heldout_predictions.csv", index=False)
    topk_df.to_csv(args.output_dir / f"{stem}__topk.csv", index=False)
    token_stats.to_csv(args.output_dir / f"{stem}__token_numeric_stats.csv", index=False)
    if args.save_checkpoints:
        ckpt_dir = args.output_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"task": cfg["task"], "version": cfg["version"], "model": cfg["model"], "seed": seed, "vocab_size": vocab_size, "max_len": args.max_len, "token_numeric_mean": token_mean, "token_numeric_std": token_std, "state_dict": model.state_dict()}, ckpt_dir / f"{stem}.pt")
    return result_df, pred_df, topk_df, history

def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args) -> dict:
    config = vars(args).copy()

    for key in ["sequence_data_dir", "output_dir"]:
        if key in config:
            config[key] = str(config[key])

    return config


def sync_args_from_clearml_config(args, config: dict) -> None:
    path_keys = {"sequence_data_dir", "output_dir"}
    int_keys = {"max_len", "batch_size", "num_workers", "epochs", "patience"}
    float_keys = {"learning_rate", "weight_decay", "grad_clip"}
    bool_keys = {"save_checkpoints"}

    # Не синхронизируем эти флаги на remote, чтобы задача не пыталась enqueue-нуть сама себя.
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
            if isinstance(value, str):
                setattr(args, key, value.lower() in {"1", "true", "yes", "y"})
            else:
                setattr(args, key, bool(value))
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
    print(f"  seeds = {args.seeds}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
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
    parser.add_argument("--sequence-data-dir", type=Path, default=Path("ehrshot_sequence_datasets"))
    parser.add_argument(
        "--sequence-data-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix with sequence datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_multiseed_numeric_sequence_results"))
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument(
        "--execute-remotely",
        action="store_true",
        help="If set, enqueue this ClearML task to an agent queue and stop local execution.",
    )

    parser.add_argument(
        "--clearml-queue",
        type=str,
        default="gpu_40",
        help="ClearML queue name for remote execution.",
    )
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="numeric_sequence_multiseed_stability")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")
    args = parser.parse_args()
    config = build_clearml_config(args)
    config["configs"] = DEFAULT_CONFIGS

    clearml_task, config = maybe_init_clearml(args, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)

    args.sequence_data_dir = maybe_download_sequence_datasets_from_s3_prefix(
        sequence_data_dir=args.sequence_data_dir,
        sequence_data_s3_prefix=args.sequence_data_s3_prefix,
        configs=DEFAULT_CONFIGS,
    )

    device = get_device(args.device)
    print("DEVICE:", device)
    all_results, all_predictions, all_topk, all_history = [], [], [], []
    for cfg in DEFAULT_CONFIGS:
        for seed in seeds:
            print("=" * 100)
            print(f"Numeric sequence experiment: {cfg} seed={seed}")
            r, p, t, h = run_one(args, cfg, seed, device)
            all_results.append(r); all_predictions.append(p); all_topk.append(t)
            h.insert(0, "seed", seed); h.insert(0, "model", cfg["model"]); h.insert(0, "version", cfg["version"]); h.insert(0, "task", cfg["task"])
            all_history.append(h)
    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    topk_df = pd.concat(all_topk, ignore_index=True)
    history_df = pd.concat(all_history, ignore_index=True)
    results_df.to_csv(args.output_dir / "numeric_sequence_multiseed_results.csv", index=False)
    predictions_df.to_csv(args.output_dir / "numeric_sequence_multiseed_heldout_predictions.csv", index=False)
    topk_df.to_csv(args.output_dir / "numeric_sequence_multiseed_topk.csv", index=False)
    history_df.to_csv(args.output_dir / "numeric_sequence_multiseed_history.csv", index=False)
    print(results_df.sort_values(["task", "seed", "calibration"]))
    if clearml_task is not None:
        clearml_task.upload_artifact("numeric_sequence_multiseed_results", results_df)
        clearml_task.upload_artifact("numeric_sequence_multiseed_predictions", predictions_df)
        clearml_task.upload_artifact("numeric_sequence_multiseed_topk", topk_df)
        clearml_task.upload_artifact("numeric_sequence_multiseed_history", history_df)


if __name__ == "__main__":
    main()
