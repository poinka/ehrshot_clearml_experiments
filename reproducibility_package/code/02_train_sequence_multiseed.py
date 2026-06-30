from __future__ import annotations

"""
02_train_sequence_multiseed_repro.py

Назначение:
    Единый reproducibility-trainer для sequence и numeric-sequence моделей.

Что делает:
    1. Читает список запусков из configs/sequence_final_runs.json.
    2. Для каждого run config и каждого seed обучает sequence-модель.
    3. Использует Time2VecEncoder для временных признаков.
    4. По флагу numeric_on включает или выключает numeric features.
    5. Сохраняет predictions / metrics / top-k / history / configs в едином формате.

    Используем нормализованные имена:
        example_id
        subject_id
        y_true
        pred_proba
        logit
        task
        model_name
        model_family
        representation
        compression_version
        numeric_on
        seed
        split
        calibration
"""

import argparse
import json
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from clearml_multiseed_scripts.reproducibility_package.code.common_ehrshot_eval import (
    binary_ranking_metrics,
    parse_int_list,
    set_global_seed,
    topk_metrics,
)


# -----------------------------------------------------------------------------
# 1. Константы
# -----------------------------------------------------------------------------

PAD_ID = 0
UNK_ID = 1

REQUIRED_PREDICTION_COLUMNS = [
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
    "y_true",
    "pred_proba",
    "logit",
]

DEFAULT_CHECKPOINT_S3_PREFIX = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
    "checkpoints"
)


def safe_path_part(value: object) -> str:
    """
    Делает строку безопасной для использования как часть MinIO/S3 path.

    Например:
        "GRU 2L" -> "GRU_2L"
        "a/b"    -> "a__b"

    Это нужно, чтобы model_name / representation случайно не ломали структуру папок.
    """
    return str(value).replace("/", "__").replace(" ", "_")


def upload_file_to_minio(local_path: Path, remote_url: str) -> str:
    """
    Загружает локальный файл в MinIO/S3 через ClearML StorageManager.

    Возвращает remote_url, чтобы потом сохранить его в configs.csv.

    Если remote_url пустой, upload пропускается.
    """
    if not remote_url:
        return ""

    from clearml import StorageManager

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    print(f"Uploading checkpoint: {local_path} -> {remote_url}")

    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )

    return remote_url

# -----------------------------------------------------------------------------
# 2. CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Аргументы запуска.

    Пример локального запуска:
        python code/02_train_sequence_multiseed_repro.py \\
            --run-config configs/sequence_final_runs.json \\
            --sequence-data-dir ehrshot_sequence_datasets \\
            --output-dir ehrshot_sequence_multiseed_repro

    Пример запуска через ClearML:
        python code/02_train_sequence_multiseed_repro.py \\
            --run-config configs/sequence_final_runs.json \\
            --sequence-data-s3-prefix s3://.../ehrshot_sequence_datasets \\
            --enable-clearml \\
            --execute-remotely \\
            --clearml-queue gpu_40
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("configs/sequence_final_runs.json"),
        help="JSON-файл со списком финальных sequence запусков.",
    )

    parser.add_argument(
        "--sequence-data-dir",
        type=Path,
        default=Path("ehrshot_sequence_datasets"),
        help="Локальная папка с sequence datasets: <task>/<compression_version>/examples.parquet и vocab.json.",
    )

    parser.add_argument(
        "--sequence-data-s3-prefix",
        type=str,
        default="",
        help="MinIO/S3 prefix с sequence datasets. Используется, если данных нет локально.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_sequence_multiseed_repro"),
        help="Куда сохранять metrics, predictions, configs, history.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Опционально переопределяет seeds из JSON, например 42,43,44.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Устройство для обучения. auto выбирает cuda, иначе cpu.",
    )

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument(
        "--numeric-min-count",
        type=int,
        default=3,
        help="Минимальное число observed numeric values на token для расчета mean/std.",
    )

    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help=(
            "Не сохранять checkpoints. "
            "По умолчанию checkpoints сохраняются локально и загружаются в MinIO."
        ),
    )

    parser.add_argument(
        "--checkpoint-s3-prefix",
        type=str,
        default=DEFAULT_CHECKPOINT_S3_PREFIX,
        help=(
            "MinIO/S3 prefix для checkpoints. "
            "Внутри него будет создана структура по run_set/task/model/seed."
        ),
)

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
        default="sequence_multiseed_repro_time2vec",
    )
    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3. Config
# -----------------------------------------------------------------------------

def read_json(path: Path) -> dict[str, Any]:
    """
    Читает JSON config.

    Для reproducibility важно, чтобы список запусков лежал отдельно от кода.
    """
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_seeds(args: argparse.Namespace, run_config: dict[str, Any]) -> list[int]:
    """
    Берет seeds либо из CLI, либо из JSON.

    Приоритет:
        1. --seeds из CLI;
        2. поле "seeds" из config.
    """
    if args.seeds.strip():
        return parse_int_list(args.seeds)

    seeds = run_config.get("seeds", [])
    if not seeds:
        raise ValueError("No seeds provided: either pass --seeds or set seeds in run config.")

    return [int(seed) for seed in seeds]


def normalize_run_cfg(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Приводит один run config к единому виду.

    Ожидаемые поля:
        task
        model_name
        model_family
        representation
        compression_version
        numeric_on
        max_len
        batch_size

    base_model_name:
        нужен для numeric-моделей, чтобы понимать,
        какая базовая архитектура используется:
            GRU_2L_numeric -> GRU_2L
            RETAIN_lite_numeric -> RETAIN_lite
    """
    cfg = dict(raw_cfg)

    required = {
        "task",
        "model_name",
        "model_family",
        "representation",
        "compression_version",
        "numeric_on",
        "max_len",
        "batch_size",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Run config is missing fields: {missing}. Config: {cfg}")

    cfg["numeric_on"] = bool(cfg["numeric_on"])
    cfg["max_len"] = int(cfg["max_len"])
    cfg["batch_size"] = int(cfg["batch_size"])

    if "base_model_name" not in cfg:
        cfg["base_model_name"] = str(cfg["model_name"]).replace("_numeric", "")

    return cfg


# -----------------------------------------------------------------------------
# 4. Device / загрузка данных
# -----------------------------------------------------------------------------

def get_device(device_arg: str) -> torch.device:
    """
    Выбирает устройство.

    В remote ClearML обычно хотим CUDA, если она доступна.
    MPS специально не выбираем в auto, чтобы не получить неожиданные отличия
    и замедления на Mac.
    """
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def to_plain_list(x: Any) -> list:
    """
    Приводит значение из parquet к обычному Python list.

    В parquet list-like колонки иногда читаются как:
        - list;
        - numpy.ndarray;
        - None;
        - NaN;
        - другой iterable.

    Эта функция делает обработку устойчивой.
    """
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

    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)

    return []


def expected_sequence_files(
    sequence_data_dir: Path,
    task: str,
    compression_version: str,
) -> list[Path]:
    """
    Возвращает ожидаемые файлы для одного sequence dataset.
    """
    return [
        sequence_data_dir / task / compression_version / "examples.parquet",
        sequence_data_dir / task / compression_version / "vocab.json",
    ]


def maybe_download_sequence_datasets_from_s3_prefix(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    run_cfgs: list[dict[str, Any]],
) -> Path:
    """
    Проверяет, есть ли все sequence datasets локально.

    Если чего-то не хватает и передан --sequence-data-s3-prefix,
    скачивает недостающие examples.parquet и vocab.json из MinIO/S3.

    Ожидаемая remote-структура:
        <prefix>/<task>/<compression_version>/examples.parquet
        <prefix>/<task>/<compression_version>/vocab.json
    """
    required_pairs = sorted(
        {
            (cfg["task"], cfg["compression_version"])
            for cfg in run_cfgs
        }
    )

    missing_files = []
    for task, compression_version in required_pairs:
        for path in expected_sequence_files(sequence_data_dir, task, compression_version):
            if not path.exists():
                missing_files.append(path)

    if not missing_files:
        print(f"Using local sequence datasets: {sequence_data_dir}")
        return sequence_data_dir

    if not sequence_data_s3_prefix:
        raise FileNotFoundError(
            "Sequence dataset files are missing locally and --sequence-data-s3-prefix is not provided. "
            f"Missing files: {[str(p) for p in missing_files[:20]]}"
        )

    from clearml import StorageManager

    prefix = sequence_data_s3_prefix.rstrip("/")

    for task, compression_version in required_pairs:
        local_dir = sequence_data_dir / task / compression_version
        local_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["examples.parquet", "vocab.json"]:
            dst = local_dir / filename

            if dst.exists():
                continue

            remote_url = f"{prefix}/{task}/{compression_version}/{filename}"
            print(f"Downloading {remote_url}")

            local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))

            if local_copy.is_dir():
                matches = list(local_copy.rglob(filename))
                if not matches:
                    raise FileNotFoundError(
                        f"StorageManager returned dir without {filename}: {local_copy}"
                    )
                local_copy = matches[0]

            if not local_copy.exists():
                raise FileNotFoundError(
                    f"StorageManager returned missing local file: {local_copy}"
                )

            print(f"Copying {local_copy} -> {dst}")
            shutil.copy2(local_copy, dst)

    return sequence_data_dir


def load_sequence_examples(
    sequence_data_dir: Path,
    task: str,
    compression_version: str,
) -> pd.DataFrame:
    """
    Загружает examples.parquet для одной задачи и версии сжатия.
    """
    path = sequence_data_dir / task / compression_version / "examples.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Sequence examples not found: {path}")

    df = pd.read_parquet(path)

    required = {
        "row_id",
        "subject_id",
        "label",
        "split",
        "token_ids",
        "days_before_prediction",
        "delta_days",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sequence examples missing required columns: {missing}")

    return df


def load_vocab(
    sequence_data_dir: Path,
    task: str,
    compression_version: str,
) -> dict[str, int]:
    """
    Загружает vocab.json для одной задачи и версии сжатия.
    """
    path = sequence_data_dir / task / compression_version / "vocab.json"

    if not path.exists():
        raise FileNotFoundError(f"Vocab not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 5. Dataset / collate
# -----------------------------------------------------------------------------

class EHRSequenceDataset(Dataset):
    """
    Dataset для одной строки prediction example.

    Одна строка:
        один prediction example, то есть история пациента до prediction_time.

    Возвращаем:
        example_id      — id строки в examples.parquet;
        subject_id      — id пациента;
        tokens          — последние max_len token_ids;
        days_before     — сколько дней до prediction_time;
        delta_days      — gap между соседними событиями;
        numeric_values  — численные значения, если они есть;
        label           — target.
    """

    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True)
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        token_ids = to_plain_list(row["token_ids"])
        days_before = to_plain_list(row["days_before_prediction"])
        delta_days = to_plain_list(row["delta_days"])
        numeric_values = to_plain_list(row.get("numeric_values", []))

        # Если у пациента по какой-то причине пустая история,
        # подставляем UNK token, чтобы RNN могла получить непустую последовательность.
        if len(token_ids) == 0:
            token_ids = [UNK_ID]
            days_before = [0.0]
            delta_days = [0.0]
            numeric_values = [np.nan]

        n = len(token_ids)

        # Выравниваем длины вспомогательных списков до длины token_ids.
        if len(days_before) < n:
            days_before += [0.0] * (n - len(days_before))

        if len(delta_days) < n:
            delta_days += [0.0] * (n - len(delta_days))

        if len(numeric_values) < n:
            numeric_values += [np.nan] * (n - len(numeric_values))

        return {
            "example_id": int(row["row_id"]),
            "subject_id": int(row["subject_id"]),
            "tokens": token_ids[-self.max_len:],
            "days_before": days_before[-self.max_len:],
            "delta_days": delta_days[-self.max_len:],
            "numeric_values": numeric_values[-self.max_len:],
            "label": int(row["label"]),
        }


def compute_token_numeric_stats(
    df_train: pd.DataFrame,
    vocab_size: int,
    min_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Считает mean/std numeric values по каждому token_id на train split.

    Почему только train:
        Нельзя использовать tuning/held_out для расчета нормализации,
        иначе будет leakage.

    Для token_id, где мало numeric values:
        mean = 0
        std = 1

    numeric_features потом состоят из двух каналов:
        1. z-score numeric value;
        2. present flag: был ли numeric value наблюдаемым.
    """
    count = np.zeros(vocab_size, dtype=np.float64)
    sum_x = np.zeros(vocab_size, dtype=np.float64)
    sum_x2 = np.zeros(vocab_size, dtype=np.float64)

    for _, row in df_train.iterrows():
        token_ids = to_plain_list(row["token_ids"])
        numeric_values = to_plain_list(row.get("numeric_values", []))

        if len(numeric_values) < len(token_ids):
            numeric_values += [np.nan] * (len(token_ids) - len(numeric_values))

        for token_id, value in zip(token_ids, numeric_values):
            try:
                token_id = int(token_id)
            except Exception:
                continue

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

    ok = count >= min_count

    mean[ok] = (sum_x[ok] / count[ok]).astype(np.float32)
    var = np.zeros(vocab_size, dtype=np.float64)
    var[ok] = np.maximum(
        sum_x2[ok] / count[ok] - mean[ok].astype(np.float64) ** 2,
        1e-6,
    )
    std[ok] = np.sqrt(var[ok]).astype(np.float32)

    stats_df = pd.DataFrame(
        {
            "token_id": np.arange(vocab_size),
            "numeric_count_train": count.astype(int),
            "numeric_mean_train": mean,
            "numeric_std_train": std,
            "has_enough_numeric_stats": ok,
        }
    )

    return mean, std, stats_df


def make_collate_fn(
    token_numeric_mean: np.ndarray,
    token_numeric_std: np.ndarray,
    numeric_on: bool,
):
    """
    Создает collate_fn для DataLoader.

    Делает padding batch до max length внутри batch.

    На выходе:
        tokens: [batch, time]
        time_features: [batch, time, 2]
            - log1p(days_before_prediction)
            - log1p(delta_days)
        numeric_features: [batch, time, 2]
            - normalized numeric value
            - present flag
        mask: [batch, time]
        lengths: [batch]
        labels: [batch]
    """
    mean_t = torch.tensor(token_numeric_mean, dtype=torch.float32)
    std_t = torch.tensor(token_numeric_std, dtype=torch.float32)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        lengths = torch.tensor([len(x["tokens"]) for x in batch], dtype=torch.long)
        max_batch_len = int(lengths.max().item())

        tokens = torch.full(
            (batch_size, max_batch_len),
            PAD_ID,
            dtype=torch.long,
        )
        time_features = torch.zeros(
            (batch_size, max_batch_len, 2),
            dtype=torch.float32,
        )
        numeric_features = torch.zeros(
            (batch_size, max_batch_len, 2),
            dtype=torch.float32,
        )
        mask = torch.zeros(
            (batch_size, max_batch_len),
            dtype=torch.bool,
        )
        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.float32,
        )
        example_ids = torch.tensor(
            [x["example_id"] for x in batch],
            dtype=torch.long,
        )
        subject_ids = torch.tensor(
            [x["subject_id"] for x in batch],
            dtype=torch.long,
        )

        for i, item in enumerate(batch):
            length = len(item["tokens"])

            tok = torch.tensor(item["tokens"], dtype=torch.long)
            tokens[i, :length] = tok
            mask[i, :length] = True

            days_before = torch.tensor(
                item["days_before"],
                dtype=torch.float32,
            ).clamp(min=0)

            delta_days = torch.tensor(
                item["delta_days"],
                dtype=torch.float32,
            ).clamp(min=0)

            time_features[i, :length, 0] = torch.log1p(days_before)
            time_features[i, :length, 1] = torch.log1p(delta_days)

            if numeric_on:
                raw_nums = []
                present = []

                for value in item["numeric_values"]:
                    try:
                        x = float(value)
                        ok = np.isfinite(x)
                    except Exception:
                        x = 0.0
                        ok = False

                    raw_nums.append(x if ok else 0.0)
                    present.append(1.0 if ok else 0.0)

                raw_nums_t = torch.tensor(raw_nums, dtype=torch.float32)
                present_t = torch.tensor(present, dtype=torch.float32)

                z = (
                    (raw_nums_t - mean_t[tok])
                    / std_t[tok].clamp(min=1e-6)
                ).clamp(min=-10.0, max=10.0)

                z = torch.where(
                    present_t > 0,
                    z,
                    torch.zeros_like(z),
                )

                numeric_features[i, :length, 0] = z
                numeric_features[i, :length, 1] = present_t

        return {
            "tokens": tokens,
            "time_features": time_features,
            "numeric_features": numeric_features,
            "mask": mask,
            "lengths": lengths,
            "labels": labels,
            "example_ids": example_ids,
            "subject_ids": subject_ids,
        }

    return collate


def make_loaders(
    df: pd.DataFrame,
    run_cfg: dict[str, Any],
    args: argparse.Namespace,
    token_numeric_mean: np.ndarray,
    token_numeric_std: np.ndarray,
    seed: int,
) -> dict[str, DataLoader]:
    """
    Создает train/tuning/held_out DataLoader.

    Важно:
        split уже задан в examples.parquet.
        Здесь мы НЕ пересоздаем split.
    """
    split_dfs = {
        split: df[df["split"] == split].copy().reset_index(drop=True)
        for split in ["train", "tuning", "held_out"]
    }

    for split, part in split_dfs.items():
        if part.empty:
            raise ValueError(
                f"Empty split={split} for task={run_cfg['task']} "
                f"compression={run_cfg['compression_version']}"
            )

    generator = torch.Generator()
    generator.manual_seed(seed)

    collate_fn = make_collate_fn(
        token_numeric_mean=token_numeric_mean,
        token_numeric_std=token_numeric_std,
        numeric_on=bool(run_cfg["numeric_on"]),
    )

    loaders = {
        "train": DataLoader(
            EHRSequenceDataset(split_dfs["train"], max_len=run_cfg["max_len"]),
            batch_size=run_cfg["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        ),
        "tuning": DataLoader(
            EHRSequenceDataset(split_dfs["tuning"], max_len=run_cfg["max_len"]),
            batch_size=run_cfg["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        ),
        "held_out": DataLoader(
            EHRSequenceDataset(split_dfs["held_out"], max_len=run_cfg["max_len"]),
            batch_size=run_cfg["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        ),
    }

    return loaders


# -----------------------------------------------------------------------------
# 6. Модель
# -----------------------------------------------------------------------------

class Time2VecEncoder(nn.Module):
    """
    Time2Vec-проекция временных признаков.

    Вход:
        time_features: [batch, time, 2]
            1. log1p(days_before_prediction)
            2. log1p(delta_days)

    Выход:
        [batch, time, emb_dim]

    Идея:
        Часть измерений линейная, часть периодическая через sin.
        Это дает модели более гибкое представление времени, чем простой Linear.
    """

    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()

        linear_dim = max(1, out_dim // 2)
        periodic_dim = out_dim - linear_dim

        self.linear = nn.Linear(input_dim, linear_dim)
        self.periodic = (
            nn.Linear(input_dim, periodic_dim)
            if periodic_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.linear(x)]

        if self.periodic is not None:
            parts.append(torch.sin(self.periodic(x)))

        return torch.cat(parts, dim=-1)


class EventEmbedding(nn.Module):
    """
    Общее event embedding для code-only и numeric sequence.

    Всегда используем:
        token embedding
        + Time2Vec time embedding

    Если numeric_on=True, дополнительно используем:
        + numeric projection

    Это и есть основная унификация:
        один класс работает для обеих веток.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        dropout: float,
        numeric_on: bool,
    ):
        super().__init__()

        self.numeric_on = bool(numeric_on)

        self.token_emb = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=PAD_ID,
        )

        self.time_proj = Time2VecEncoder(
            input_dim=2,
            out_dim=emb_dim,
        )

        self.numeric_proj = (
            nn.Sequential(
                nn.Linear(2, emb_dim),
                nn.GELU(),
                nn.LayerNorm(emb_dim),
            )
            if self.numeric_on
            else None
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        time_features: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_emb(tokens) + self.time_proj(time_features)

        if self.numeric_proj is not None:
            x = x + self.numeric_proj(numeric_features)

        return self.dropout(self.norm(x))


class RNNClassifier(nn.Module):
    """
    GRU/LSTM classifier.

    Поддерживает:
        GRU_1L
        GRU_2L
        LSTM_1L
        LSTM_2L

    Использует последнее hidden state как representation всей истории.
    """

    def __init__(
        self,
        vocab_size: int,
        rnn_type: str,
        num_layers: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        numeric_on: bool,
    ):
        super().__init__()

        self.event_emb = EventEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dropout=dropout,
            numeric_on=numeric_on,
        )

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM

        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        time_features: torch.Tensor,
        numeric_features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.event_emb(tokens, time_features, numeric_features)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, h = self.rnn(packed)

        if isinstance(h, tuple):
            h = h[0]

        last_hidden = h[-1]

        return self.head(last_hidden).squeeze(-1)


def reverse_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Разворачивает последовательность отдельно для каждого example
    только внутри настоящей длины, не трогая padding.

    Нужно для RETAIN, потому что он читает историю в обратном времени.
    """
    out = x.clone()

    for i, length in enumerate(lengths.tolist()):
        out[i, :length] = x[i, :length].flip(0)

    return out


class RetainLiteClassifier(nn.Module):
    """
    Упрощенный RETAIN-like classifier.

    Идея:
        1. Разворачиваем последовательность во времени.
        2. alpha_gru дает attention weights по событиям.
        3. beta_gru дает feature-wise modulation.
        4. Суммируем weighted context и отправляем в head.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        numeric_on: bool,
    ):
        super().__init__()

        self.event_emb = EventEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dropout=dropout,
            numeric_on=numeric_on,
        )

        self.alpha_gru = nn.GRU(
            emb_dim,
            hidden_dim,
            batch_first=True,
        )
        self.beta_gru = nn.GRU(
            emb_dim,
            hidden_dim,
            batch_first=True,
        )

        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, emb_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        time_features: torch.Tensor,
        numeric_features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.event_emb(tokens, time_features, numeric_features)

        x_rev = reverse_by_lengths(x, lengths)
        mask_rev = reverse_by_lengths(
            mask.unsqueeze(-1).float(),
            lengths,
        ).squeeze(-1).bool()

        alpha_h, _ = self.alpha_gru(x_rev)
        beta_h, _ = self.beta_gru(x_rev)

        alpha_logits = self.alpha_fc(alpha_h).squeeze(-1)
        alpha_logits = alpha_logits.masked_fill(~mask_rev, -1e9)
        alpha = torch.softmax(alpha_logits, dim=1)

        beta = torch.tanh(self.beta_fc(beta_h))

        context = torch.sum(
            alpha.unsqueeze(-1) * beta * x_rev,
            dim=1,
        )

        return self.head(context).squeeze(-1)


def make_model(
    run_cfg: dict[str, Any],
    vocab_size: int,
    args: argparse.Namespace,
) -> nn.Module:
    """
    Создает модель по base_model_name.

    В config model_name может быть:
        GRU_2L_numeric

    Но фактическая архитектура:
        base_model_name = GRU_2L
        numeric_on = True

    Это лучше, чем иметь отдельные классы GRU_2L_numeric / GRU_2L.
    """
    base_model_name = run_cfg["base_model_name"]
    numeric_on = bool(run_cfg["numeric_on"])

    if base_model_name == "GRU_1L":
        return RNNClassifier(
            vocab_size=vocab_size,
            rnn_type="GRU",
            num_layers=1,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            numeric_on=numeric_on,
        )

    if base_model_name == "GRU_2L":
        return RNNClassifier(
            vocab_size=vocab_size,
            rnn_type="GRU",
            num_layers=2,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            numeric_on=numeric_on,
        )

    if base_model_name == "LSTM_1L":
        return RNNClassifier(
            vocab_size=vocab_size,
            rnn_type="LSTM",
            num_layers=1,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            numeric_on=numeric_on,
        )

    if base_model_name == "LSTM_2L":
        return RNNClassifier(
            vocab_size=vocab_size,
            rnn_type="LSTM",
            num_layers=2,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            numeric_on=numeric_on,
        )

    if base_model_name == "RETAIN_lite":
        return RetainLiteClassifier(
            vocab_size=vocab_size,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            numeric_on=numeric_on,
        )

    raise ValueError(f"Unknown base_model_name: {base_model_name}")


# -----------------------------------------------------------------------------
# 7. Train / inference / evaluation
# -----------------------------------------------------------------------------

def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Переносит все tensor-значения batch на нужное устройство.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """
    Стабильный sigmoid для numpy.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Прогоняет модель на одном split и возвращает logits + ids + y_true.
    """
    model.eval()

    out = {
        "logits": [],
        "y_true": [],
        "example_id": [],
        "subject_id": [],
    }

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

            out["logits"].append(logits.detach().cpu().numpy())
            out["y_true"].append(batch["labels"].numpy())
            out["example_id"].append(batch["example_ids"].numpy())
            out["subject_id"].append(batch["subject_ids"].numpy())

    elapsed = time.perf_counter() - start
    n_examples = sum(len(x) for x in out["y_true"])

    return {
        "logits": np.concatenate(out["logits"]).astype(float),
        "y_true": np.concatenate(out["y_true"]).astype(int),
        "example_id": np.concatenate(out["example_id"]).astype(int),
        "subject_id": np.concatenate(out["subject_id"]).astype(int),
        "inference_seconds": float(elapsed),
        "inference_examples_per_second": (
            float(n_examples / elapsed)
            if elapsed > 0
            else np.nan
        ),
    }


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, float, int]:
    """
    Обучает модель с early stopping по tuning AUPRC.

    Loss:
        BCEWithLogitsLoss(pos_weight=n_neg / n_pos)

    Почему pos_weight:
        Задачи несбалансированы, поэтому положительный класс надо усилить.
    """
    model.to(device)

    train_labels = []
    for batch in loaders["train"]:
        train_labels.append(batch["labels"].numpy())

    y_train = np.concatenate(train_labels).astype(int)

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - y_train.sum()))

    pos_weight = torch.tensor(
        [n_neg / n_pos],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_auprc = -np.inf
    best_epoch = 0
    bad_epochs = 0

    history_rows = []
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in loaders["train"]:
            batch_dev = move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                batch_dev["tokens"],
                batch_dev["time_features"],
                batch_dev["numeric_features"],
                batch_dev["mask"],
                batch_dev["lengths"],
            )

            loss = criterion(logits, batch_dev["labels"])
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        tuning_pred = run_inference(model, loaders["tuning"], device)
        tuning_prob = sigmoid_np(tuning_pred["logits"])

        tuning_metrics = binary_ranking_metrics(
            tuning_pred["y_true"],
            tuning_prob,
        )

        current_auprc = tuning_metrics["auprc"]

        if current_auprc > best_auprc + 1e-5:
            best_auprc = current_auprc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "tuning_auroc": tuning_metrics["auroc"],
                "tuning_auprc": tuning_metrics["auprc"],
                "tuning_brier": tuning_metrics["brier"],
                "tuning_logloss": tuning_metrics["logloss"],
                "best_epoch_so_far": best_epoch,
                "best_tuning_auprc_so_far": best_auprc,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={np.mean(losses):.4f} "
            f"tuning_auprc={tuning_metrics['auprc']:.4f} "
            f"tuning_auroc={tuning_metrics['auroc']:.4f} "
            f"best_epoch={best_epoch}"
        )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break

    train_seconds = time.perf_counter() - train_start

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    history_df["best_epoch"] = best_epoch
    history_df["train_seconds_total"] = train_seconds

    return model, history_df, float(train_seconds), int(best_epoch)


def fit_platt_from_logits(
    tuning_logits: np.ndarray,
    y_tuning: np.ndarray,
    seed: int,
) -> LogisticRegression:
    """
    Обучает Platt calibration на tuning logits.

    Важно:
        calibration обучается на tuning, а оценивается на held_out.
    """
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )

    calibrator.fit(
        np.asarray(tuning_logits).reshape(-1, 1),
        np.asarray(y_tuning).astype(int),
    )

    return calibrator


def apply_platt(
    calibrator: LogisticRegression,
    logits: np.ndarray,
) -> np.ndarray:
    """
    Применяет Platt calibration к logits.
    """
    return calibrator.predict_proba(
        np.asarray(logits).reshape(-1, 1),
    )[:, 1]


def evaluate_with_platt(
    run_cfg: dict[str, Any],
    seed: int,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    train_seconds: float,
    best_epoch: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Считает raw и platt predictions на held_out.

    Сохраняет metrics/predictions/top-k в нормализованной схеме.
    """
    tuning = run_inference(model, loaders["tuning"], device)
    heldout = run_inference(model, loaders["held_out"], device)

    calibrator = fit_platt_from_logits(
        tuning_logits=tuning["logits"],
        y_tuning=tuning["y_true"],
        seed=seed,
    )

    probs = {
        "raw": sigmoid_np(heldout["logits"]),
        "platt": apply_platt(calibrator, heldout["logits"]),
    }

    result_rows = []
    pred_frames = []
    topk_frames = []

    for calibration, pred_proba in probs.items():
        metrics = binary_ranking_metrics(
            heldout["y_true"],
            pred_proba,
        )

        result_rows.append(
            {
                "task": run_cfg["task"],
                "model_family": run_cfg["model_family"],
                "model_name": run_cfg["model_name"],
                "representation": run_cfg["representation"],
                "compression_version": run_cfg["compression_version"],
                "numeric_on": bool(run_cfg["numeric_on"]),
                "calibration": calibration,
                "seed": int(seed),
                "split": "held_out",

                "max_len": int(run_cfg["max_len"]),
                "batch_size": int(run_cfg["batch_size"]),
                "emb_dim": int(args.emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "best_epoch": int(best_epoch),
                "train_seconds_total": float(train_seconds),
                "inference_seconds_heldout": heldout["inference_seconds"],
                "inference_examples_per_second_heldout": heldout[
                    "inference_examples_per_second"
                ],

                **{
                    f"heldout_{metric_name}": value
                    for metric_name, value in metrics.items()
                },
            }
        )

        pred_df = pd.DataFrame(
            {
                "task": run_cfg["task"],
                "model_family": run_cfg["model_family"],
                "model_name": run_cfg["model_name"],
                "representation": run_cfg["representation"],
                "compression_version": run_cfg["compression_version"],
                "numeric_on": bool(run_cfg["numeric_on"]),
                "calibration": calibration,
                "seed": int(seed),
                "split": "held_out",
                "example_id": heldout["example_id"].astype(int),
                "subject_id": heldout["subject_id"].astype(int),
                "y_true": heldout["y_true"].astype(int),
                "pred_proba": pred_proba.astype(float),
                "logit": heldout["logits"].astype(float),            
            }
        )

        missing_prediction_cols = set(REQUIRED_PREDICTION_COLUMNS) - set(pred_df.columns)
        if missing_prediction_cols:
            raise ValueError(
                f"Prediction schema is missing columns: {missing_prediction_cols}"
            )

        pred_frames.append(pred_df[REQUIRED_PREDICTION_COLUMNS])

        topk = topk_metrics(
            heldout["y_true"],
            pred_proba,
        )

        topk.insert(0, "split", "held_out")
        topk.insert(0, "seed", int(seed))
        topk.insert(0, "calibration", calibration)
        topk.insert(0, "numeric_on", bool(run_cfg["numeric_on"]))
        topk.insert(0, "compression_version", run_cfg["compression_version"])
        topk.insert(0, "representation", run_cfg["representation"])
        topk.insert(0, "model_name", run_cfg["model_name"])
        topk.insert(0, "model_family", run_cfg["model_family"])
        topk.insert(0, "task", run_cfg["task"])

        topk_frames.append(topk)

    return (
        pd.DataFrame(result_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(topk_frames, ignore_index=True),
    )


# -----------------------------------------------------------------------------
# 8. Один run
# -----------------------------------------------------------------------------

def build_config_row(
    run_cfg: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    checkpoint_path: str | None,
    checkpoint_s3_url: str | None,
    clearml_task_id: str | None,
) -> dict[str, Any]:
    """
    Формирует одну строку configs.csv.

    Это нужно для reproducibility package:
        predictions показывают, что модель предсказала;
        configs показывают, как именно модель была обучена.
    """
    return {
        "task": run_cfg["task"],
        "model_family": run_cfg["model_family"],
        "model_name": run_cfg["model_name"],
        "representation": run_cfg["representation"],
        "compression_version": run_cfg["compression_version"],
        "numeric_on": bool(run_cfg["numeric_on"]),

        "seed": int(seed),
        "split": "held_out",

        "max_len": int(run_cfg["max_len"]),
        "batch_size": int(run_cfg["batch_size"]),
        "emb_dim": int(args.emb_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "grad_clip": float(args.grad_clip),
        "numeric_min_count": int(args.numeric_min_count),

        "dataset_path_or_version": str(
            args.sequence_data_dir
            / run_cfg["task"]
            / run_cfg["compression_version"]
        ),
        "checkpoint_path": checkpoint_path or "",
        "checkpoint_s3_url": checkpoint_s3_url or "",
        "output_dir": str(args.output_dir),

        "code_path": "code/02_train_sequence_multiseed_repro.py",
        "config_path": str(args.run_config),
        "notebook_path": "",
        "clearml_task_id": clearml_task_id or "",
    }


def build_sequence_checkpoint_remote_url(
    checkpoint_s3_prefix: str,
    run_set: str,
    run_cfg: dict[str, Any],
    seed: int,
) -> str:
    """
    Строит MinIO/S3 path для sequence checkpoint.

    Итоговая структура:
        <checkpoint_s3_prefix>/
            <run_set>/
              <task>/
                <model_family>/
                  <model_name>_<compression_version>_seed<seed>_model.pt
    """
    if not checkpoint_s3_prefix:
        return ""

    filename = "_".join(
        [
            safe_path_part(run_cfg["model_name"]),
            safe_path_part(run_cfg["compression_version"]),
            f"seed{int(seed)}",
            "model.pt",
        ]
    )

    parts = [
        checkpoint_s3_prefix.rstrip("/"),
        safe_path_part(run_set),
        safe_path_part(run_cfg["task"]),
        safe_path_part(run_cfg["model_family"]),
        filename,
    ]

    return "/".join(parts)


def run_one(
    run_cfg: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    clearml_task_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Запускает один эксперимент:
        task + compression_version + model + numeric_on + seed.
    """
    set_global_seed(seed)

    df = load_sequence_examples(
        sequence_data_dir=args.sequence_data_dir,
        task=run_cfg["task"],
        compression_version=run_cfg["compression_version"],
    )

    vocab = load_vocab(
        sequence_data_dir=args.sequence_data_dir,
        task=run_cfg["task"],
        compression_version=run_cfg["compression_version"],
    )

    vocab_size = len(vocab)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)

    token_mean, token_std, token_stats = compute_token_numeric_stats(
        df_train=train_df,
        vocab_size=vocab_size,
        min_count=args.numeric_min_count,
    )

    token_stats.insert(0, "seed", int(seed))
    token_stats.insert(0, "numeric_on", bool(run_cfg["numeric_on"]))
    token_stats.insert(0, "compression_version", run_cfg["compression_version"])
    token_stats.insert(0, "representation", run_cfg["representation"])
    token_stats.insert(0, "model_name", run_cfg["model_name"])
    token_stats.insert(0, "model_family", run_cfg["model_family"])
    token_stats.insert(0, "task", run_cfg["task"])

    loaders = make_loaders(
        df=df,
        run_cfg=run_cfg,
        args=args,
        token_numeric_mean=token_mean,
        token_numeric_std=token_std,
        seed=seed,
    )

    model = make_model(
        run_cfg=run_cfg,
        vocab_size=vocab_size,
        args=args,
    )

    model, history_df, train_seconds, best_epoch = train_model(
        model=model,
        loaders=loaders,
        args=args,
        device=device,
    )

    result_df, pred_df, topk_df = evaluate_with_platt(
        run_cfg=run_cfg,
        seed=seed,
        model=model,
        loaders=loaders,
        args=args,
        device=device,
        train_seconds=train_seconds,
        best_epoch=best_epoch,
    )

    history_df.insert(0, "seed", int(seed))
    history_df.insert(0, "numeric_on", bool(run_cfg["numeric_on"]))
    history_df.insert(0, "compression_version", run_cfg["compression_version"])
    history_df.insert(0, "representation", run_cfg["representation"])
    history_df.insert(0, "model_name", run_cfg["model_name"])
    history_df.insert(0, "model_family", run_cfg["model_family"])
    history_df.insert(0, "task", run_cfg["task"])

    stem = (
        f"{run_cfg['task']}__"
        f"{run_cfg['compression_version']}__"
        f"{run_cfg['model_name']}__"
        f"seed{seed}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.output_dir / f"{stem}__metrics.csv"
    pred_path = args.output_dir / f"{stem}__heldout_predictions.csv"
    topk_path = args.output_dir / f"{stem}__topk.csv"
    history_path = args.output_dir / f"{stem}__history.csv"
    token_stats_path = args.output_dir / f"{stem}__token_numeric_stats.csv"

    result_df.to_csv(result_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    history_df.to_csv(history_path, index=False)
    token_stats.to_csv(token_stats_path, index=False)

    checkpoint_path = None
    checkpoint_s3_url = None

    # По умолчанию checkpoints сохраняем.
    # Отключить можно только явно через --no-save-checkpoints.
    if not args.no_save_checkpoints:
        local_ckpt_dir = (
            args.output_dir
            / "checkpoints"
            / safe_path_part(args.run_set)
            / safe_path_part(run_cfg["task"])
            / safe_path_part(run_cfg["model_family"])
        )
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_filename = "_".join(
            [
                safe_path_part(run_cfg["model_name"]),
                safe_path_part(run_cfg["compression_version"]),
                f"seed{int(seed)}",
                "model.pt",
            ]
        )

        ckpt_path = local_ckpt_dir / ckpt_filename

        torch.save(
            {
                # Идентификация запуска.
                "task": run_cfg["task"],
                "model_family": run_cfg["model_family"],
                "model_name": run_cfg["model_name"],
                "base_model_name": run_cfg["base_model_name"],
                "representation": run_cfg["representation"],
                "compression_version": run_cfg["compression_version"],
                "numeric_on": bool(run_cfg["numeric_on"]),
                "seed": int(seed),

                # Параметры модели.
                "vocab_size": int(vocab_size),
                "max_len": int(run_cfg["max_len"]),
                "emb_dim": int(args.emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),

                # Numeric normalization.
                # Даже для numeric_on=False можно сохранить,
                # чтобы checkpoint schema была одинаковой.
                "numeric_min_count": int(args.numeric_min_count),
                "token_numeric_mean": token_mean,
                "token_numeric_std": token_std,

                # Весовая матрица модели.
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )

        checkpoint_path = str(ckpt_path)

        remote_ckpt_url = build_sequence_checkpoint_remote_url(
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            run_set=args.run_set,
            run_cfg=run_cfg,
            seed=seed,
        )

        checkpoint_s3_url = upload_file_to_minio(
            local_path=ckpt_path,
            remote_url=remote_ckpt_url,
        )

    config_row = build_config_row(
        run_cfg=run_cfg,
        seed=seed,
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_s3_url=checkpoint_s3_url,
        clearml_task_id=clearml_task_id,
    )

    print("Saved metrics:", result_path)
    print("Saved predictions:", pred_path)

    preview = result_df[
        [
            "task",
            "model_name",
            "compression_version",
            "numeric_on",
            "seed",
            "calibration",
            "heldout_auroc",
            "heldout_auprc",
            "heldout_brier",
            "heldout_logloss",
        ]
    ]
    print(preview.to_string(index=False))

    return result_df, pred_df, topk_df, history_df, token_stats, config_row


# -----------------------------------------------------------------------------
# 9. ClearML
# -----------------------------------------------------------------------------

def is_clearml_agent_run() -> bool:
    """
    Проверяет, запущен ли код внутри ClearML agent.
    """
    return bool(
        os.environ.get("CLEARML_TASK_ID")
        or os.environ.get("TRAINS_TASK_ID")
    )


def build_clearml_config(
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Готовит параметры для ClearML.

    Path переводим в str, потому что ClearML может плохо сериализовать Path.
    """
    cfg = vars(args).copy()

    for key in ["run_config", "sequence_data_dir", "output_dir"]:
        if key in cfg:
            cfg[key] = str(cfg[key])

    cfg["run_config_json"] = run_config

    return cfg


def _to_bool(x: Any) -> bool:
    """
    Безопасное приведение к bool для параметров, пришедших из ClearML.
    """
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}

    return bool(x)


def sync_args_from_clearml_config(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    """
    На remote ClearML agent аргументы могут прийти не из CLI,
    а из сохраненного task config.

    Эта функция синхронизирует ClearML config обратно в argparse Namespace.
    """
    path_keys = {
        "run_config",
        "sequence_data_dir",
        "output_dir",
    }
    int_keys = {
        "num_workers",
        "epochs",
        "patience",
        "emb_dim",
        "hidden_dim",
        "numeric_min_count",
    }
    float_keys = {
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "dropout",
    }
    bool_keys = {
        "no_save_checkpoints",
    }

    skip_keys = {
        "enable_clearml",
        "execute_remotely",
        "run_config_json",
    }

    for key, value in dict(cfg).items():
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


def maybe_init_clearml(
    args: argparse.Namespace,
    run_config: dict[str, Any],
):
    """
    Инициализирует ClearML.

    Режимы:
        1. Локально без ClearML:
            --enable-clearml не указан, ничего не делаем.

        2. Локально с постановкой в очередь:
            --enable-clearml --execute-remotely

        3. Уже внутри ClearML agent:
            берем текущий task.
    """
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
            build_clearml_config(args, run_config),
        )
    )

    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  clearml_task_id = {task.id}")
    print(f"  run_config = {args.run_config}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  device = {args.device}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")

        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task


# -----------------------------------------------------------------------------
# 10. Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_config = read_json(args.run_config)
    args.run_set = str(
        run_config.get(
            "run_set",
            "sequence_final_repro_time2vec_v1",
        )
    )
    run_cfgs = [
        normalize_run_cfg(raw_cfg)
        for raw_cfg in run_config.get("runs", [])
    ]

    if not run_cfgs:
        raise ValueError("Run config contains no runs.")

    seeds = get_seeds(args, run_config)

    clearml_task = maybe_init_clearml(args, run_config)
    clearml_task_id = clearml_task.id if clearml_task is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.sequence_data_dir = maybe_download_sequence_datasets_from_s3_prefix(
        sequence_data_dir=args.sequence_data_dir,
        sequence_data_s3_prefix=args.sequence_data_s3_prefix,
        run_cfgs=run_cfgs,
    )

    device = get_device(args.device)

    print("=" * 100)
    print("SEQUENCE REPRO TRAINING")
    print(f"DEVICE: {device}")
    print(f"RUN SET: {run_config.get('run_set', '')}")
    print(f"N RUN CONFIGS: {len(run_cfgs)}")
    print(f"SEEDS: {seeds}")
    print("=" * 100)

    all_results = []
    all_predictions = []
    all_topk = []
    all_history = []
    all_token_stats = []
    all_configs = []

    total_runs = len(run_cfgs) * len(seeds)
    run_index = 0

    for run_cfg in run_cfgs:
        for seed in seeds:
            run_index += 1

            print("=" * 100)
            print(f"[{run_index}/{total_runs}]")
            print(
                f"task={run_cfg['task']} | "
                f"model={run_cfg['model_name']} | "
                f"compression={run_cfg['compression_version']} | "
                f"numeric_on={run_cfg['numeric_on']} | "
                f"seed={seed}"
            )

            (
                result_df,
                pred_df,
                topk_df,
                history_df,
                token_stats_df,
                config_row,
            ) = run_one(
                run_cfg=run_cfg,
                seed=seed,
                args=args,
                device=device,
                clearml_task_id=clearml_task_id,
            )

            all_results.append(result_df)
            all_predictions.append(pred_df)
            all_topk.append(topk_df)
            all_history.append(history_df)
            all_token_stats.append(token_stats_df)
            all_configs.append(config_row)

            # Сохраняем агрегаты после каждого run,
            # чтобы при падении задачи частичный результат не потерялся.
            pd.concat(all_results, ignore_index=True).to_csv(
                args.output_dir / "sequence_multiseed_results.csv",
                index=False,
            )
            pd.concat(all_predictions, ignore_index=True).to_csv(
                args.output_dir / "sequence_multiseed_heldout_predictions.csv",
                index=False,
            )
            pd.concat(all_topk, ignore_index=True).to_csv(
                args.output_dir / "sequence_multiseed_topk.csv",
                index=False,
            )
            pd.concat(all_history, ignore_index=True).to_csv(
                args.output_dir / "sequence_multiseed_history.csv",
                index=False,
            )
            pd.concat(all_token_stats, ignore_index=True).to_csv(
                args.output_dir / "sequence_token_numeric_stats.csv",
                index=False,
            )
            pd.DataFrame(all_configs).to_csv(
                args.output_dir / "sequence_multiseed_configs.csv",
                index=False,
            )

    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    topk_df = pd.concat(all_topk, ignore_index=True)
    history_df = pd.concat(all_history, ignore_index=True)
    token_stats_df = pd.concat(all_token_stats, ignore_index=True)
    configs_df = pd.DataFrame(all_configs)

    results_path = args.output_dir / "sequence_multiseed_results.csv"
    predictions_path = args.output_dir / "sequence_multiseed_heldout_predictions.csv"
    topk_path = args.output_dir / "sequence_multiseed_topk.csv"
    history_path = args.output_dir / "sequence_multiseed_history.csv"
    token_stats_path = args.output_dir / "sequence_token_numeric_stats.csv"
    configs_path = args.output_dir / "sequence_multiseed_configs.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    history_df.to_csv(history_path, index=False)
    token_stats_df.to_csv(token_stats_path, index=False)
    configs_df.to_csv(configs_path, index=False)

    print("=" * 100)
    print("DONE")
    print(f"Saved results: {results_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved top-k: {topk_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved token numeric stats: {token_stats_path}")
    print(f"Saved configs: {configs_path}")

    print("\nMetrics preview:")
    preview_cols = [
        "task",
        "model_name",
        "compression_version",
        "numeric_on",
        "seed",
        "calibration",
        "heldout_auroc",
        "heldout_auprc",
        "heldout_brier",
        "heldout_logloss",
    ]
    print(
        results_df[preview_cols]
        .sort_values(
            [
                "task",
                "model_name",
                "compression_version",
                "numeric_on",
                "seed",
                "calibration",
            ]
        )
        .to_string(index=False)
    )

    if clearml_task is not None:
        clearml_task.upload_artifact("sequence_multiseed_results", results_df)
        clearml_task.upload_artifact("sequence_multiseed_predictions", predictions_df)
        clearml_task.upload_artifact("sequence_multiseed_topk", topk_df)
        clearml_task.upload_artifact("sequence_multiseed_history", history_df)
        clearml_task.upload_artifact("sequence_token_numeric_stats", token_stats_df)
        clearml_task.upload_artifact("sequence_multiseed_configs", configs_df)


if __name__ == "__main__":
    main()