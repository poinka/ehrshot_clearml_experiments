#!/usr/bin/env python
"""Train fixed sequence models on one or more compression variants.

The script is deliberately controlled by task/version/model/seed so that ClearML runs differ
only by the requested compression version unless you explicitly change model args.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
UNK_ID = SPECIAL_TOKENS["<UNK>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, default=Path("ehrshot_sequence_datasets_compression_v2"))
    parser.add_argument("--sequence-data-s3-prefix", type=str, default="", help="S3/MinIO prefix with <task>/<version>/examples.parquet and vocab.json")
    parser.add_argument("--sequence-data-artifact-name", type=str, default="sequence_compression_datasets_zip")
    parser.add_argument("--results-dir", type=Path, default=Path("ehrshot_sequence_compression_results_v2"))
    parser.add_argument("--task", required=True, choices=["guo_readmission", "guo_icu"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", required=True, choices=["GRU_1L", "GRU_2L", "LSTM_1L", "LSTM_2L", "RETAIN_lite"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--use-numeric", action="store_true")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--max-train-examples", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-eval-examples", type=int, default=0, help="0 means no cap")
    parser.add_argument("--numeric-min-count", type=int, default=3)
    parser.add_argument("--top-fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true", help="Enqueue this task to a ClearML agent queue and stop local execution")
    parser.add_argument("--clearml-queue", type=str, default="gpu_40")
    parser.add_argument("--clearml-task-name", type=str, default="")
    parser.add_argument("--clearml-project", default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-output-uri", default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Do not prefer MPS on remote agents; use it only if explicitly requested.
    return torch.device("cpu")


def load_sequence_examples(sequence_dir: Path, task: str, version: str) -> pd.DataFrame:
    path = sequence_dir / task / version / "examples.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_vocab(sequence_dir: Path, task: str, version: str) -> dict[str, int]:
    path = sequence_dir / task / version / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def to_plain_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    return list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []


class EHRSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        token_ids = to_plain_list(row["token_ids"])
        days_before = to_plain_list(row["days_before_prediction"])
        delta_days = to_plain_list(row["delta_days"])
        numeric_values = to_plain_list(row.get("numeric_values", []))
        if len(token_ids) == 0:
            token_ids = [UNK_ID]
            days_before = [0.0]
            delta_days = [0.0]
            numeric_values = [np.nan]
        n = len(token_ids)
        if len(days_before) < n:
            days_before += [0.0] * (n - len(days_before))
        if len(delta_days) < n:
            delta_days += [0.0] * (n - len(delta_days))
        if len(numeric_values) < n:
            numeric_values += [np.nan] * (n - len(numeric_values))
        return {
            "row_id": int(row["row_id"]),
            "subject_id": int(row["subject_id"]),
            "tokens": token_ids[-self.max_len :],
            "days_before": days_before[-self.max_len :],
            "delta_days": delta_days[-self.max_len :],
            "numeric_values": numeric_values[-self.max_len :],
            "label": int(row["label"]),
        }


def compute_token_numeric_stats(df_train: pd.DataFrame, vocab_size: int, min_count: int = 3):
    count = np.zeros(vocab_size, dtype=np.float64)
    sum_x = np.zeros(vocab_size, dtype=np.float64)
    sum_x2 = np.zeros(vocab_size, dtype=np.float64)
    for _, row in df_train.iterrows():
        tokens = to_plain_list(row["token_ids"])
        nums = to_plain_list(row.get("numeric_values", []))
        if len(nums) < len(tokens):
            nums += [np.nan] * (len(tokens) - len(nums))
        for token_id, value in zip(tokens, nums):
            token_id = int(token_id)
            if token_id < 0 or token_id >= vocab_size:
                continue
            try:
                x = float(value)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            count[token_id] += 1
            sum_x[token_id] += x
            sum_x2[token_id] += x * x
    mean = np.zeros(vocab_size, dtype=np.float32)
    std = np.ones(vocab_size, dtype=np.float32)
    valid = count >= min_count
    mean[valid] = (sum_x[valid] / count[valid]).astype(np.float32)
    var = (sum_x2[valid] / np.maximum(count[valid], 1)) - mean[valid] ** 2
    std[valid] = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    stats_df = pd.DataFrame({
        "token_id": np.arange(vocab_size),
        "numeric_count_train": count.astype(int),
        "numeric_mean_train": mean,
        "numeric_std_train": std,
        "has_enough_numeric_stats": valid,
    })
    return mean, std, stats_df


def make_collate_fn(token_numeric_mean: np.ndarray, token_numeric_std: np.ndarray, use_numeric: bool):
    mean_t = torch.tensor(token_numeric_mean, dtype=torch.float32)
    std_t = torch.tensor(token_numeric_std, dtype=torch.float32)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        lengths = torch.tensor([len(x["tokens"]) for x in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        tokens = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        time_features = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
        numeric_features = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
        row_ids = torch.tensor([x["row_id"] for x in batch], dtype=torch.long)
        subject_ids = torch.tensor([x["subject_id"] for x in batch], dtype=torch.long)
        for i, item in enumerate(batch):
            l = len(item["tokens"])
            tok = torch.tensor(item["tokens"], dtype=torch.long)
            tokens[i, :l] = tok
            mask[i, :l] = True
            days_before = torch.tensor(item["days_before"], dtype=torch.float32).clamp(min=0)
            delta_days = torch.tensor(item["delta_days"], dtype=torch.float32).clamp(min=0)
            time_features[i, :l, 0] = torch.log1p(days_before)
            time_features[i, :l, 1] = torch.log1p(delta_days)
            if use_numeric:
                raw_nums, present = [], []
                for v in item["numeric_values"]:
                    try:
                        x = float(v)
                        ok = np.isfinite(x)
                    except Exception:
                        x, ok = 0.0, False
                    raw_nums.append(x if ok else 0.0)
                    present.append(1.0 if ok else 0.0)
                raw_nums_t = torch.tensor(raw_nums, dtype=torch.float32)
                present_t = torch.tensor(present, dtype=torch.float32)
                m = mean_t[tok]
                s = std_t[tok].clamp(min=1e-6)
                z = ((raw_nums_t - m) / s).clamp(min=-10.0, max=10.0)
                z = torch.where(present_t > 0, z, torch.zeros_like(z))
                numeric_features[i, :l, 0] = z
                numeric_features[i, :l, 1] = present_t
        return {
            "tokens": tokens,
            "time_features": time_features,
            "numeric_features": numeric_features,
            "mask": mask,
            "lengths": lengths,
            "labels": labels,
            "row_ids": row_ids,
            "subject_ids": subject_ids,
        }
    return collate


def maybe_subsample(df: pd.DataFrame, max_examples: int, seed: int) -> pd.DataFrame:
    if max_examples <= 0 or len(df) <= max_examples:
        return df.reset_index(drop=True)
    return df.sample(n=max_examples, random_state=seed).reset_index(drop=True)


def make_loaders(df: pd.DataFrame, args: argparse.Namespace, mean: np.ndarray, std: np.ndarray):
    split_dfs = {}
    for split in ["train", "tuning", "held_out"]:
        part = df[df["split"] == split].copy()
        cap = args.max_train_examples if split == "train" else args.max_eval_examples
        split_dfs[split] = maybe_subsample(part, cap, args.seed)
    collate_fn = make_collate_fn(mean, std, args.use_numeric)
    loaders = {
        split: DataLoader(
            EHRSequenceDataset(part, max_len=args.max_len),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        for split, part in split_dfs.items()
    }
    return loaders, split_dfs


class Time2VecEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
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


class EventEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float, use_numeric: bool):
        super().__init__()
        self.use_numeric = use_numeric
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.time_proj = Time2VecEncoder(2, emb_dim)
        self.numeric_proj = nn.Sequential(nn.Linear(2, emb_dim), nn.GELU(), nn.LayerNorm(emb_dim)) if use_numeric else None
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, time_features, numeric_features):
        x = self.token_emb(tokens) + self.time_proj(time_features)
        if self.numeric_proj is not None:
            x = x + self.numeric_proj(numeric_features)
        return self.dropout(self.norm(x))


def reverse_by_lengths(x, lengths):
    out = x.clone()
    for i, l in enumerate(lengths.tolist()):
        out[i, :l] = torch.flip(x[i, :l], dims=[0])
    return out


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, rnn_type: str, num_layers: int, args: argparse.Namespace):
        super().__init__()
        self.event_emb = EventEmbedding(vocab_size, args.emb_dim, args.dropout, args.use_numeric)
        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            args.emb_dim,
            args.hidden_dim,
            num_layers=num_layers,
            dropout=args.dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(args.hidden_dim), nn.Dropout(args.dropout), nn.Linear(args.hidden_dim, 1))

    def forward(self, tokens, time_features, numeric_features, mask, lengths):
        x = self.event_emb(tokens, time_features, numeric_features)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        return self.head(h[-1]).squeeze(-1)


class RetainLiteClassifier(nn.Module):
    def __init__(self, vocab_size: int, args: argparse.Namespace):
        super().__init__()
        self.event_emb = EventEmbedding(vocab_size, args.emb_dim, args.dropout, args.use_numeric)
        self.alpha_gru = nn.GRU(args.emb_dim, args.hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(args.emb_dim, args.hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(args.hidden_dim, 1)
        self.beta_fc = nn.Linear(args.hidden_dim, args.emb_dim)
        self.head = nn.Sequential(nn.LayerNorm(args.emb_dim), nn.Dropout(args.dropout), nn.Linear(args.emb_dim, 1))

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


def make_model(vocab_size: int, args: argparse.Namespace) -> nn.Module:
    if args.model == "GRU_1L":
        return RNNClassifier(vocab_size, "GRU", 1, args)
    if args.model == "GRU_2L":
        return RNNClassifier(vocab_size, "GRU", 2, args)
    if args.model == "LSTM_1L":
        return RNNClassifier(vocab_size, "LSTM", 1, args)
    if args.model == "LSTM_2L":
        return RNNClassifier(vocab_size, "LSTM", 2, args)
    if args.model == "RETAIN_lite":
        return RetainLiteClassifier(vocab_size, args)
    raise ValueError(args.model)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_y, all_row_ids, all_subject_ids = [], [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            logits = model(
                batch_dev["tokens"],
                batch_dev["time_features"],
                batch_dev["numeric_features"],
                batch_dev["mask"],
                batch_dev["lengths"],
            )
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(batch["labels"].numpy())
            all_row_ids.append(batch["row_ids"].numpy())
            all_subject_ids.append(batch["subject_ids"].numpy())
    elapsed = time.perf_counter() - start
    n = sum(len(x) for x in all_y)
    return {
        "logits": np.concatenate(all_logits),
        "y_true": np.concatenate(all_y).astype(int),
        "row_id": np.concatenate(all_row_ids).astype(int),
        "subject_id": np.concatenate(all_subject_ids).astype(int),
        "inference_seconds": elapsed,
        "inference_examples_per_second": n / elapsed if elapsed > 0 else np.nan,
    }


def compute_ranking_metrics(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    return {
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob, labels=[0, 1]),
        "event_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def compute_topk_metrics(y_true, y_prob, top_fracs: list[float]) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if n else np.nan
    order = np.argsort(-y_prob)
    rows = []
    for frac in top_fracs:
        k = max(1, int(np.ceil(n * frac)))
        idx = order[:k]
        selected_events = int(y_true[idx].sum())
        event_rate = selected_events / k
        rows.append({
            "top_frac": frac,
            "top_k": k,
            "events_in_top_k": selected_events,
            "top_k_event_rate": event_rate,
            "top_k_precision": event_rate,
            "top_k_lift": event_rate / base_rate if base_rate > 0 else np.nan,
            "event_capture": selected_events / n_pos if n_pos > 0 else np.nan,
            "base_event_rate": base_rate,
            "n": n,
            "n_positive": n_pos,
        })
    return pd.DataFrame(rows)


def fit_platt(logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(logits).reshape(-1, 1), np.asarray(y).astype(int))
    return lr


def apply_platt(platt: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    return platt.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


def train_model(model: nn.Module, loaders: dict[str, DataLoader], args: argparse.Namespace, device: torch.device):
    train_labels = []
    for batch in loaders["train"]:
        train_labels.append(batch["labels"].numpy())
    y_train = np.concatenate(train_labels).astype(int)
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - y_train.sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_tuning_auprc = -np.inf
    bad_epochs = 0
    history_rows = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["time_features"], batch["numeric_features"], batch["mask"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        tuning_pred = run_inference(model, loaders["tuning"], device)
        p_tuning = sigmoid_np(tuning_pred["logits"])
        tuning_auprc = average_precision_score(tuning_pred["y_true"], p_tuning)
        tuning_auroc = roc_auc_score(tuning_pred["y_true"], p_tuning) if len(np.unique(tuning_pred["y_true"])) > 1 else np.nan
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "tuning_auprc": float(tuning_auprc),
            "tuning_auroc": float(tuning_auroc),
        }
        history_rows.append(row)
        print(f"epoch={epoch:02d} loss={row['train_loss']:.4f} tuning_auprc={tuning_auprc:.4f} tuning_auroc={tuning_auroc:.4f}")
        if tuning_auprc > best_tuning_auprc + 1e-5:
            best_tuning_auprc = tuning_auprc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break
    train_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    history["best_epoch"] = best_epoch
    history["train_seconds_total"] = train_seconds
    return model, history, train_seconds, best_epoch


def evaluate(model: nn.Module, loaders: dict[str, DataLoader], args: argparse.Namespace, device: torch.device):
    tuning_pred = run_inference(model, loaders["tuning"], device)
    heldout_pred = run_inference(model, loaders["held_out"], device)
    platt = fit_platt(tuning_pred["logits"], tuning_pred["y_true"])
    rows, pred_frames, topk_frames = [], [], []
    for calibration in ["raw", "platt"]:
        if calibration == "raw":
            p_tuning = sigmoid_np(tuning_pred["logits"])
            p_heldout = sigmoid_np(heldout_pred["logits"])
        else:
            p_tuning = apply_platt(platt, tuning_pred["logits"])
            p_heldout = apply_platt(platt, heldout_pred["logits"])
        tuning_metrics = compute_ranking_metrics(tuning_pred["y_true"], p_tuning)
        heldout_metrics = compute_ranking_metrics(heldout_pred["y_true"], p_heldout)
        row = {
            "task": args.task,
            "version": args.version,
            "model": args.model + ("_numeric" if args.use_numeric else ""),
            "base_model": args.model,
            "use_numeric": args.use_numeric,
            "seed": args.seed,
            "calibration": calibration,
            "max_len": args.max_len,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "n_tuning": len(tuning_pred["y_true"]),
            "n_heldout": len(heldout_pred["y_true"]),
            "inference_seconds_heldout": heldout_pred["inference_seconds"],
            "inference_examples_per_second_heldout": heldout_pred["inference_examples_per_second"],
        }
        row.update({f"tuning_{k}": v for k, v in tuning_metrics.items()})
        row.update({f"heldout_{k}": v for k, v in heldout_metrics.items()})
        rows.append(row)
        pred_frames.append(pd.DataFrame({
            "task": args.task,
            "version": args.version,
            "model": row["model"],
            "base_model": args.model,
            "use_numeric": args.use_numeric,
            "seed": args.seed,
            "calibration": calibration,
            "row_id": heldout_pred["row_id"],
            "subject_id": heldout_pred["subject_id"],
            "y_true": heldout_pred["y_true"],
            "logit": heldout_pred["logits"],
            "risk": p_heldout,
        }))
        topk = compute_topk_metrics(heldout_pred["y_true"], p_heldout, args.top_fracs)
        for c, v in [
            ("task", args.task),
            ("version", args.version),
            ("model", row["model"]),
            ("base_model", args.model),
            ("use_numeric", args.use_numeric),
            ("seed", args.seed),
            ("calibration", calibration),
        ]:
            topk.insert(0, c, v)
        topk_frames.append(topk)
    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True), pd.concat(topk_frames, ignore_index=True)



# --------------------------------------------------------------------------------------
# ClearML / remote data helpers
# --------------------------------------------------------------------------------------


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    for key in ["sequence_dir", "results_dir"]:
        if key in cfg:
            cfg[key] = str(cfg[key])
    return cfg


def _to_bool(x: Any) -> bool:
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    return bool(x)


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path_keys = {"sequence_dir", "results_dir"}
    int_keys = {"seed", "max_len", "batch_size", "num_workers", "epochs", "patience", "emb_dim", "hidden_dim", "max_train_examples", "max_eval_examples", "numeric_min_count"}
    float_keys = {"lr", "weight_decay", "grad_clip", "dropout"}
    bool_keys = {"use_numeric", "save_checkpoint"}
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

    # This prevents ClearML from freezing the local environment into a broken package list.
    if not remote_agent_run:
        Task.force_requirements_env_freeze(False, "requirements.txt")

    task_name = args.clearml_task_name or f"compression__{args.task}__{args.version}__{args.model}{'_numeric' if args.use_numeric else ''}__seed{args.seed}"

    if remote_agent_run:
        task = Task.current_task()
        if task is None:
            task = Task.init(
                project_name=args.clearml_project,
                task_name=task_name,
                output_uri=args.clearml_output_uri or None,
                auto_connect_arg_parser=False,
            )
    else:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=task_name,
            output_uri=args.clearml_output_uri or None,
            auto_connect_arg_parser=False,
        )

    connected = dict(task.connect(build_clearml_config(args)))
    sync_args_from_clearml_config(args, connected)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task = {args.task}")
    print(f"  version = {args.version}")
    print(f"  model = {args.model}")
    print(f"  use_numeric = {args.use_numeric}")
    print(f"  seed = {args.seed}")
    print(f"  sequence_dir = {args.sequence_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
    print(f"  results_dir = {args.results_dir}")
    print(f"  device = {args.device}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


def _expected_sequence_files(sequence_dir: Path, task: str, version: str) -> list[Path]:
    return [sequence_dir / task / version / "examples.parquet", sequence_dir / task / version / "vocab.json"]


def _find_extracted_sequence_root(extract_dir: Path, task: str, version: str) -> Path:
    candidates = [extract_dir, extract_dir / "ehrshot_sequence_datasets_compression_v2"]
    candidates += [p for p in extract_dir.iterdir() if p.is_dir()]
    for cand in candidates:
        if all(p.exists() for p in _expected_sequence_files(cand, task, version)):
            return cand
    matches = list(extract_dir.rglob(f"{task}/{version}/examples.parquet"))
    if matches:
        return matches[0].parents[2]
    raise FileNotFoundError(f"Could not find extracted sequence dataset for {task}/{version} under {extract_dir}")


def maybe_download_sequence_from_s3(args: argparse.Namespace) -> Path | None:
    if not args.sequence_data_s3_prefix:
        return None
    if all(p.exists() for p in _expected_sequence_files(args.sequence_dir, args.task, args.version)):
        print(f"Using local sequence dataset: {args.sequence_dir}")
        return args.sequence_dir

    from clearml import StorageManager

    prefix = args.sequence_data_s3_prefix.rstrip("/")
    local_dir = args.sequence_dir / args.task / args.version
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["examples.parquet", "vocab.json"]:
        dst = local_dir / fname
        if dst.exists():
            continue
        remote_url = f"{prefix}/{args.task}/{args.version}/{fname}"
        print(f"Downloading {remote_url}")
        local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))
        if local_copy.is_dir():
            matches = list(local_copy.rglob(fname))
            if not matches:
                raise FileNotFoundError(f"StorageManager returned dir without {fname}: {local_copy}")
            local_copy = matches[0]
        shutil.copy2(local_copy, dst)
    return args.sequence_dir


def prepare_sequence_dataset(args: argparse.Namespace) -> Path:
    if all(p.exists() for p in _expected_sequence_files(args.sequence_dir, args.task, args.version)):
        print(f"Using local sequence dataset: {args.sequence_dir}")
        return args.sequence_dir
    root = maybe_download_sequence_from_s3(args)
    if root is not None:
        return root
    missing = [str(p) for p in _expected_sequence_files(args.sequence_dir, args.task, args.version) if not p.exists()]
    raise FileNotFoundError(
        "Sequence dataset is missing. Provide --sequence-data-s3-prefix. "
        f"Missing files: {missing}"
    )


def main() -> None:
    args = parse_args()
    clearml_task = maybe_init_clearml(args)
    set_seed(args.seed)
    device = get_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    args.sequence_dir = prepare_sequence_dataset(args)

    print("DEVICE:", device)
    print("ARGS:", vars(args))
    df = load_sequence_examples(args.sequence_dir, args.task, args.version)
    vocab = load_vocab(args.sequence_dir, args.task, args.version)
    vocab_size = len(vocab)
    df_train = df[df["split"] == "train"].copy()
    mean, std, numeric_stats_df = compute_token_numeric_stats(df_train, vocab_size, args.numeric_min_count)
    loaders, split_dfs = make_loaders(df, args, mean, std)
    model = make_model(vocab_size, args)

    model, history_df, train_seconds, best_epoch = train_model(model, loaders, args, device)
    result_df, pred_df, topk_df = evaluate(model, loaders, args, device)
    result_df["train_seconds_total"] = train_seconds
    result_df["best_epoch"] = best_epoch

    exp_name = f"{args.task}__{args.version}__{args.model}{'_numeric' if args.use_numeric else ''}__seed{args.seed}"
    result_path = args.results_dir / f"{exp_name}__metrics.csv"
    pred_path = args.results_dir / f"{exp_name}__heldout_predictions.csv"
    topk_path = args.results_dir / f"{exp_name}__topk.csv"
    hist_path = args.results_dir / f"{exp_name}__history.csv"
    num_path = args.results_dir / f"{exp_name}__numeric_stats.csv"

    result_df.to_csv(result_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    history_df.to_csv(hist_path, index=False)
    numeric_stats_df.to_csv(num_path, index=False)

    if args.save_checkpoint:
        ckpt_path = ckpt_dir / f"{exp_name}.pt"
        torch.save({
            "task": args.task,
            "version": args.version,
            "model": args.model,
            "use_numeric": args.use_numeric,
            "seed": args.seed,
            "vocab_size": vocab_size,
            "max_len": args.max_len,
            "state_dict": model.state_dict(),
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)
    else:
        ckpt_path = None

    print("Saved metrics:", result_path)
    print("Saved predictions:", pred_path)
    print("Saved top-k:", topk_path)
    print(result_df[["task", "version", "model", "seed", "calibration", "heldout_auroc", "heldout_auprc", "heldout_brier", "heldout_logloss"]].to_string(index=False))

    if clearml_task is not None:
        clearml_task.upload_artifact("metrics", artifact_object=result_path)
        clearml_task.upload_artifact("heldout_predictions", artifact_object=pred_path)
        clearml_task.upload_artifact("topk", artifact_object=topk_path)
        clearml_task.upload_artifact("history", artifact_object=hist_path)
        clearml_task.upload_artifact("numeric_stats", artifact_object=num_path)
        if ckpt_path is not None:
            clearml_task.upload_artifact("checkpoint", artifact_object=ckpt_path)


if __name__ == "__main__":
    main()
