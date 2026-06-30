from __future__ import annotations

"""
00_build_sequence_datasets_from_meds.py

Назначение:
    Собрать sequence datasets из локальной папки EHRSHOT_MEDS в формате,
    который потом используют sequence training scripts.

Что делает:
    1. Читает MEDS/EHRSHOT parquet из локальной папки.
    2. Читает labels для задач guo_readmission и guo_icu.
    3. Читает subject_splits.parquet.
    4. Строит patient history до prediction_time.
    5. Строит несколько версий sequence datasets:
        - raw
        - compressed_dedup
        - compressed_first_last
        - condition_era_90
        - condition_era_180
        - state_duration_90
        - state_duration_180
        - compressed_hybrid         # гибрид first-last / condition-era
    6. Для каждой task/version сохраняет:
        - examples.parquet
        - vocab.json
        - code_counts_train.csv
        - metadata.json
    7. Сохраняет summary:
        - all_compression_version_summary.csv
        - sequence_dataset_sanity.csv
    8. Загружает итоговую папку в MinIO/S3 через ClearML StorageManager.

Ожидаемая локальная структура:
    EHRSHOT_MEDS/
      data/data.parquet
      metadata/subject_splits.parquet
      labels/guo_readmission/labels.parquet
      labels/guo_icu/labels.parquet

    ehrshot_copy_forwarding_audit/
      strong_empirical_chronic_like_diagnosis_codes.csv
      possible_empirical_chronic_like_diagnosis_codes.csv  # optional

Итоговая структура:
    ehrshot_sequence_datasets/
      guo_readmission/
        raw/
          examples.parquet
          vocab.json
          code_counts_train.csv
          metadata.json
        compressed_dedup/
          ...
      guo_icu/
        raw/
          ...

Пример локального запуска из папки ноутбука:
    python 00_build_sequence_datasets_from_meds.py \\
      --notebook-root . \\
      --ehrshot-root EHRSHOT_MEDS \\
      --audit-dir ehrshot_copy_forwarding_audit \\
      --output-dir ehrshot_sequence_datasets \\
      --rebuild

Пример запуска с upload в MinIO:
    python 00_build_sequence_datasets_from_meds.py \\
      --notebook-root . \\
      --ehrshot-root EHRSHOT_MEDS \\
      --audit-dir ehrshot_copy_forwarding_audit \\
      --output-dir ehrshot_sequence_datasets \\
      --output-s3-prefix s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_sequence_datasets \\
      --rebuild
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


# -----------------------------------------------------------------------------
# 1. Константы
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
}

PAD_ID = SPECIAL_TOKENS["<PAD>"]
UNK_ID = SPECIAL_TOKENS["<UNK>"]
CLS_ID = SPECIAL_TOKENS["<CLS>"]

DEFAULT_TASKS = [
    "guo_readmission",
    "guo_icu",
]

DEFAULT_SEQUENCE_VERSIONS = [
    "raw",
    "compressed_dedup",
    "compressed_first_last",
    "condition_era_90",
    "condition_era_180",
    "state_duration_90",
    "state_duration_180",
]

# Алиасы нужны, чтобы training config мог использовать более короткие имена.
# compressed_condition_era будет физически отдельной папкой, но логика = condition_era_90.
OPTIONAL_SEQUENCE_VERSIONS = [
    "compressed_hybrid",
]

DIAGNOSIS_LIKE_FAMILIES = [
    "SNOMED",
    "ICD10CM",
    "ICD9CM",
    "ICD10",
    "ICD9",
]

STANDARD_LONG_COLS = [
    "task",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
    "time",
    "code",
    "numeric_value",
    "text_value",
    "days_before_prediction",
    "is_compression_token",
]

EXAMPLE_KEY_COLS = [
    "task",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
]

COMPRESSION_GROUP_COLS = EXAMPLE_KEY_COLS + [
    "code",
    "compression_bucket",
]

CODE_SUMMARY_COLS = EXAMPLE_KEY_COLS + [
    "code",
]


# -----------------------------------------------------------------------------
# 2. CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--notebook-root",
        type=Path,
        default=Path("."),
        help=(
            "Папка, из которой раньше запускался ноутбук. "
            "Относительные пути ehrshot-root/audit-dir/output-dir считаются от нее."
        ),
    )

    parser.add_argument(
        "--ehrshot-root",
        type=Path,
        default=Path("EHRSHOT_MEDS"),
        help="Локальная папка с EHRSHOT_MEDS.",
    )

    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("ehrshot_copy_forwarding_audit"),
        help="Папка с chronic-like code whitelist из copy-forwarding audit.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_sequence_datasets"),
        help="Локальная папка для сохранения sequence datasets.",
    )

    parser.add_argument(
        "--output-s3-prefix",
        type=str,
        default=(
            "s3://api.blackhole2.ai.innopolis.university:443/"
            "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/"
            "ehrshot_sequence_datasets"
        ),
        help=(
            "MinIO/S3 prefix для загрузки итоговых датасетов. "
            "Внутри будет структура <task>/<version>/examples.parquet."
        ),
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks, например guo_readmission,guo_icu.",
    )

    parser.add_argument(
        "--versions",
        type=str,
        default=",".join(DEFAULT_SEQUENCE_VERSIONS),
        help=(
            "Comma-separated sequence versions. "
            "Можно добавить compressed_condition_era,compressed_hybrid."
        ),
    )

    parser.add_argument(
        "--chronic-code-mode",
        type=str,
        default="strong_only",
        choices=["strong_only", "strong_plus_possible"],
        help="Какие chronic-like codes брать из audit результатов.",
    )

    parser.add_argument(
        "--keep-code-strings",
        action="store_true",
        default=True,
        help="Сохранять исходные code strings в examples.parquet вместе с token_ids.",
    )

    parser.add_argument(
        "--drop-code-strings",
        action="store_true",
        help="Не сохранять codes в examples.parquet, чтобы уменьшить размер parquet.",
    )

    parser.add_argument(
        "--include-prediction-time",
        action="store_true",
        default=True,
        help="Включать события time <= prediction_time.",
    )

    parser.add_argument(
        "--exclude-prediction-time",
        action="store_true",
        help="Использовать только события time < prediction_time.",
    )

    parser.add_argument(
        "--visit-anchor-max-days",
        type=int,
        default=30,
        help="Максимальный разрыв от Visit anchor до события для reconstructed visit.",
    )

    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=0,
        help="0 = не ограничивать vocab; иначе максимум token vocabulary size включая special tokens.",
    )

    parser.add_argument(
        "--min-code-count",
        type=int,
        default=1,
        help="Минимальная train frequency для попадания code в vocab.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересобрать datasets даже если files уже существуют.",
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
        help="Поставить ClearML task в очередь и остановить локальное выполнение.",
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
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
        help="ClearML project name.",
    )

    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="build_sequence_datasets_from_meds",
        help="ClearML task name.",
    )

    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
        help="ClearML output URI.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3. Вспомогательные функции
# -----------------------------------------------------------------------------

def resolve_path(root: Path, path: Path) -> Path:
    """
    Если path относительный, считаем его относительно notebook-root.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def to_jsonable(x: Any) -> Any:
    """
    Приводит numpy/pandas types к обычным Python types для json.dump.
    """
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, list):
        return [to_jsonable(v) for v in x]

    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if pd.isna(x) if not isinstance(x, (list, dict, tuple)) else False:
        return None

    return x


def as_python_list(x: Any) -> list:
    """
    Приводит list-like parquet values к обычному Python list.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
    return []


def safe_path_part(value: object) -> str:
    """
    Безопасная часть пути для MinIO/S3.
    """
    return str(value).replace("/", "__").replace(" ", "_")


def upload_file_to_minio(local_path: Path, remote_url: str) -> str:
    """
    Загружает один файл в MinIO/S3 через ClearML StorageManager.
    """
    if not remote_url:
        return ""

    from clearml import StorageManager

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    print(f"Uploading: {local_path} -> {remote_url}")

    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )

    return remote_url


def upload_tree_to_minio(local_root: Path, s3_prefix: str) -> pd.DataFrame:
    """
    Загружает всю папку local_root в MinIO/S3.

    Layout сохраняется:
        local_root/guo_icu/raw/examples.parquet
    становится:
        <s3_prefix>/guo_icu/raw/examples.parquet
    """
    if not s3_prefix:
        return pd.DataFrame()

    local_root = Path(local_root)

    rows = []
    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(local_root).as_posix()
        remote_url = f"{s3_prefix.rstrip('/')}/{rel}"

        upload_file_to_minio(path, remote_url)

        rows.append(
            {
                "local_path": str(path),
                "relative_path": rel,
                "s3_url": remote_url,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. ClearML
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


def build_clearml_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    for key in ["notebook_root", "ehrshot_root", "audit_dir", "output_dir"]:
        cfg[key] = str(cfg[key])
    return cfg


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path_keys = {
        "notebook_root",
        "ehrshot_root",
        "audit_dir",
        "output_dir",
    }
    int_keys = {
        "visit_anchor_max_days",
        "max_vocab_size",
        "min_code_count",
    }
    bool_keys = {
        "keep_code_strings",
        "drop_code_strings",
        "include_prediction_time",
        "exclude_prediction_time",
        "rebuild",
        "skip_upload",
    }
    skip_keys = {
        "enable_clearml",
        "execute_remotely",
    }

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

    connected_cfg = dict(task.connect(build_clearml_config(args)))
    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  task_id = {task.id}")
    print(f"  notebook_root = {args.notebook_root}")
    print(f"  ehrshot_root = {args.ehrshot_root}")
    print(f"  audit_dir = {args.audit_dir}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  output_s3_prefix = {args.output_s3_prefix}")
    print(f"  tasks = {args.tasks}")
    print(f"  versions = {args.versions}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(
            queue_name=args.clearml_queue,
            exit_process=True,
        )

    return task


# -----------------------------------------------------------------------------
# 5. Builder class
# -----------------------------------------------------------------------------

class SequenceDatasetBuilder:
    """
    Сборщик sequence datasets из EHRSHOT_MEDS.

    Важный принцип:
        vocab строится только на train split.
        compression whitelist берется из audit результатов.
        все events фильтруются по time <= prediction_time или < prediction_time.
    """

    def __init__(
        self,
        ehrshot_root: Path,
        audit_dir: Path,
        output_dir: Path,
        tasks: list[str],
        versions: list[str],
        chronic_code_mode: str,
        keep_code_strings: bool,
        include_prediction_time: bool,
        visit_anchor_max_days: int,
        max_vocab_size: int | None,
        min_code_count: int,
        rebuild: bool,
    ):
        self.ehrshot_root = Path(ehrshot_root)
        self.audit_dir = Path(audit_dir)
        self.output_dir = Path(output_dir)
        self.tasks = tasks
        self.versions = versions
        self.chronic_code_mode = chronic_code_mode
        self.keep_code_strings = keep_code_strings
        self.include_prediction_time = include_prediction_time
        self.visit_anchor_max_days = int(visit_anchor_max_days)
        self.max_vocab_size = max_vocab_size
        self.min_code_count = int(min_code_count)
        self.rebuild = bool(rebuild)

        self.data_path = self.ehrshot_root / "data" / "data.parquet"
        self.splits_path = self.ehrshot_root / "metadata" / "subject_splits.parquet"
        self.labels_dir = self.ehrshot_root / "labels"

        self._check_input_paths()

        self.events_lf_raw = pl.scan_parquet(str(self.data_path))
        self.event_schema = self.events_lf_raw.collect_schema().names()

        self.splits = pl.read_parquet(self.splits_path)
        self.task_to_label_file = {
            p.parent.name: p
            for p in sorted(self.labels_dir.glob("*/labels.parquet"))
        }

        for task_name in self.tasks:
            if task_name not in self.task_to_label_file:
                raise FileNotFoundError(f"Label file not found for task={task_name}")

        self.chronic_code_whitelist = self.load_chronic_code_whitelist()
        self.visit_anchors_lf = self.build_visit_anchors_lf()

    def _check_input_paths(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Не найден data.parquet: {self.data_path}")

        if not self.splits_path.exists():
            raise FileNotFoundError(f"Не найден subject_splits.parquet: {self.splits_path}")

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Не найдена labels dir: {self.labels_dir}")

        if not self.audit_dir.exists():
            raise FileNotFoundError(f"Не найдена audit dir: {self.audit_dir}")

    def load_chronic_code_whitelist(self) -> set[str]:
        strong_path = self.audit_dir / "strong_empirical_chronic_like_diagnosis_codes.csv"
        possible_path = self.audit_dir / "possible_empirical_chronic_like_diagnosis_codes.csv"

        if not strong_path.exists():
            raise FileNotFoundError(
                f"Не найден файл {strong_path}. "
                "Сначала нужно запустить copy-forwarding audit notebook."
            )

        strong = pd.read_csv(strong_path)
        chronic_codes = set(strong["code"].astype(str))

        if self.chronic_code_mode == "strong_plus_possible":
            if not possible_path.exists():
                raise FileNotFoundError(f"Не найден файл {possible_path}")
            possible = pd.read_csv(possible_path)
            chronic_codes |= set(possible["code"].astype(str))

        print(f"Chronic-like codes loaded: {len(chronic_codes)}")
        return chronic_codes

    def detect_label_col(self, lf: pl.LazyFrame) -> str:
        candidate_cols = [
            "boolean_value",
            "integer_value",
            "float_value",
            "categorical_value",
            "label",
            "value",
        ]

        schema_cols = lf.collect_schema().names()
        existing = [c for c in candidate_cols if c in schema_cols]

        if not existing:
            raise ValueError(f"No known label column found. Schema: {schema_cols}")

        non_null_counts = (
            lf
            .select([pl.col(c).is_not_null().sum().alias(c) for c in existing])
            .collect()
            .to_pandas()
            .iloc[0]
        )

        non_null_cols = non_null_counts[non_null_counts > 0]

        if len(non_null_cols) == 0:
            raise ValueError(f"All candidate label columns are null. Existing: {existing}")

        return str(non_null_cols.index[0])

    def load_task_labels(self, task_name: str) -> pl.DataFrame:
        label_path = self.task_to_label_file[task_name]
        lf = pl.scan_parquet(str(label_path))
        label_col = self.detect_label_col(lf)

        labels = (
            pl.read_parquet(label_path)
            .with_row_index("row_id")
            .with_columns(
                [
                    pl.col(label_col).cast(pl.Int8).alias("label"),
                    pl.lit(task_name).alias("task"),
                ]
            )
            .join(self.splits, on="subject_id", how="left")
            .select(
                [
                    "task",
                    "row_id",
                    "subject_id",
                    "prediction_time",
                    "label",
                    "split",
                ]
            )
        )

        missing_split = labels.filter(pl.col("split").is_null()).height
        if missing_split > 0:
            raise ValueError(f"{task_name}: {missing_split} labels have missing split.")

        return labels

    def build_visit_anchors_lf(self) -> pl.LazyFrame:
        """
        Reconstructed visits:
            берем Visit/... коды, если есть omop_table — только visit_occurrence.
        """
        visit_lf = (
            self.events_lf_raw
            .filter(pl.col("code").cast(pl.Utf8).str.starts_with("Visit/"))
        )

        if "omop_table" in self.event_schema:
            visit_lf = visit_lf.filter(pl.col("omop_table") == "visit_occurrence")

        visit_anchors_lf = (
            visit_lf
            .select(
                [
                    pl.col("subject_id"),
                    pl.col("time").alias("visit_time"),
                    pl.col("code").cast(pl.Utf8).alias("visit_code"),
                ]
            )
            .group_by(["subject_id", "visit_time"])
            .agg(
                [
                    pl.col("visit_code").unique().sort().alias("visit_codes"),
                    pl.len().alias("n_visit_rows_at_time"),
                ]
            )
            .sort(["subject_id", "visit_time"])
            .with_columns(
                [
                    (
                        pl.col("visit_time").cum_count().over("subject_id") - 1
                    )
                    .cast(pl.Int32)
                    .alias("reconstructed_visit_idx"),

                    pl.col("visit_time")
                    .shift(-1)
                    .over("subject_id")
                    .alias("next_visit_time"),
                ]
            )
        )

        return visit_anchors_lf

    def base_events_lf(self) -> pl.LazyFrame:
        exprs = [
            pl.col("subject_id"),
            pl.col("time"),
            pl.col("code").cast(pl.Utf8).alias("code"),
        ]

        if "numeric_value" in self.event_schema:
            exprs.append(pl.col("numeric_value").cast(pl.Float32).alias("numeric_value"))
        else:
            exprs.append(pl.lit(None).cast(pl.Float32).alias("numeric_value"))

        if "text_value" in self.event_schema:
            exprs.append(pl.col("text_value").cast(pl.Utf8).alias("text_value"))
        else:
            exprs.append(pl.lit(None).cast(pl.Utf8).alias("text_value"))

        base = (
            self.events_lf_raw
            .select(exprs)
            .with_columns(
                [
                    pl.col("code")
                    .str.extract(r"^([^/]+)", 1)
                    .fill_null("UNKNOWN")
                    .alias("code_family"),

                    pl.col("time")
                    .dt.date()
                    .cast(pl.Utf8)
                    .alias("event_day"),
                ]
            )
            .sort(["subject_id", "time"])
        )

        base_with_visit = (
            base
            .join_asof(
                self.visit_anchors_lf.sort(["subject_id", "visit_time"]),
                left_on="time",
                right_on="visit_time",
                by="subject_id",
                strategy="backward",
            )
            .with_columns(
                [
                    (
                        (pl.col("time") - pl.col("visit_time"))
                        .dt.total_hours()
                        / 24
                    )
                    .cast(pl.Float32)
                    .alias("days_since_visit_anchor")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("reconstructed_visit_idx").is_not_null()
                        & pl.col("days_since_visit_anchor").is_not_null()
                        & (pl.col("days_since_visit_anchor") >= 0)
                        & (pl.col("days_since_visit_anchor") <= self.visit_anchor_max_days)
                    )
                    .alias("has_valid_reconstructed_visit")
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("has_valid_reconstructed_visit"))
                    .then(
                        pl.concat_str(
                            [
                                pl.lit("reconstructed_visit="),
                                pl.col("reconstructed_visit_idx").cast(pl.Utf8),
                            ]
                        )
                    )
                    .otherwise(
                        pl.concat_str(
                            [
                                pl.lit("day="),
                                pl.col("event_day"),
                            ]
                        )
                    )
                    .alias("compression_bucket")
                ]
            )
        )

        return base_with_visit

    def make_history_lf(self, labels: pl.DataFrame) -> pl.LazyFrame:
        if self.include_prediction_time:
            op = pl.col("time") <= pl.col("prediction_time")
        else:
            op = pl.col("time") < pl.col("prediction_time")

        return (
            labels.lazy()
            .join(self.base_events_lf(), on="subject_id", how="inner")
            .filter(op)
            .with_columns(
                [
                    (
                        pl.col("prediction_time") - pl.col("time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("days_before_prediction")
                ]
            )
        )

    def compressible_chronic_code_expr(self) -> pl.Expr:
        return pl.col("code").is_in(list(self.chronic_code_whitelist))

    # -------------------------------------------------------------------------
    # Compression versions
    # -------------------------------------------------------------------------

    def build_raw_long_lf(self, labels: pl.DataFrame) -> pl.LazyFrame:
        return (
            self.make_history_lf(labels)
            .with_columns(pl.lit(False).alias("is_compression_token"))
            .select(STANDARD_LONG_COLS)
        )

    def prepare_chronic_dedup_lf(
        self,
        labels: pl.DataFrame,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        hist = (
            self.make_history_lf(labels)
            .with_columns(
                self.compressible_chronic_code_expr()
                .fill_null(False)
                .alias("is_compressible_chronic")
            )
        )

        non_compressed = (
            hist
            .filter(~pl.col("is_compressible_chronic"))
            .with_columns(pl.lit(False).alias("is_compression_token"))
            .select(STANDARD_LONG_COLS)
        )

        chronic_for_compression = hist.filter(pl.col("is_compressible_chronic"))

        chronic_dedup = (
            chronic_for_compression
            .sort(COMPRESSION_GROUP_COLS + ["time", "code"])
            .group_by(COMPRESSION_GROUP_COLS)
            .agg(
                [
                    pl.col("time").min().alias("time"),
                    pl.col("numeric_value").sort_by("time").first().alias("numeric_value"),
                    pl.col("text_value").sort_by("time").first().alias("text_value"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("prediction_time") - pl.col("time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("days_before_prediction"),

                    pl.lit(False).alias("is_compression_token"),
                ]
            )
            .select(STANDARD_LONG_COLS)
        )

        return non_compressed, chronic_dedup

    def build_compressed_dedup_long_lf(self, labels: pl.DataFrame) -> pl.LazyFrame:
        non_compressed, chronic_dedup = self.prepare_chronic_dedup_lf(labels)

        return (
            pl.concat([non_compressed, chronic_dedup], how="vertical_relaxed")
            .sort(["row_id", "time", "code"])
        )

    def compression_token_from_summary(
        self,
        summary_lf: pl.LazyFrame,
        token_prefix: str | None,
        time_col: str,
        numeric_expr: pl.Expr | None = None,
        code_expr: pl.Expr | None = None,
    ) -> pl.LazyFrame:
        if numeric_expr is None:
            numeric_expr = pl.lit(None).cast(pl.Float32)

        if code_expr is None:
            if token_prefix is None:
                raise ValueError("Either token_prefix or code_expr must be provided.")
            code_expr = pl.concat_str(
                [
                    pl.lit(token_prefix),
                    pl.col("code"),
                ]
            )

        return (
            summary_lf
            .with_columns(
                [
                    pl.col(time_col).alias("time"),
                    code_expr.alias("code"),
                    numeric_expr.cast(pl.Float32).alias("numeric_value"),
                    pl.lit(None).cast(pl.Utf8).alias("text_value"),
                    (
                        pl.col("prediction_time") - pl.col(time_col)
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("days_before_prediction"),
                    pl.lit(True).alias("is_compression_token"),
                ]
            )
            .select(STANDARD_LONG_COLS)
        )

    def build_compressed_first_last_long_lf(self, labels: pl.DataFrame) -> pl.LazyFrame:
        non_compressed, chronic_dedup = self.prepare_chronic_dedup_lf(labels)

        summary = (
            chronic_dedup
            .group_by(CODE_SUMMARY_COLS)
            .agg(
                [
                    pl.len().alias("n_dx_points"),
                    pl.col("time").min().alias("first_time"),
                    pl.col("time").max().alias("last_time"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("last_time") - pl.col("first_time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("span_days")
                ]
            )
        )

        compress_condition = pl.col("n_dx_points") >= 2

        normal_keys = summary.filter(~compress_condition).select(CODE_SUMMARY_COLS)
        compressed_summary = summary.filter(compress_condition)

        normal_dx = (
            chronic_dedup
            .join(normal_keys, on=CODE_SUMMARY_COLS, how="inner")
            .select(STANDARD_LONG_COLS)
        )

        fl_first = self.compression_token_from_summary(
            compressed_summary,
            token_prefix="DX_FIRST/",
            time_col="first_time",
            numeric_expr=pl.lit(0.0),
        )

        fl_last = self.compression_token_from_summary(
            compressed_summary,
            token_prefix="DX_LAST/",
            time_col="last_time",
            numeric_expr=pl.col("span_days"),
        )

        return (
            pl.concat([non_compressed, normal_dx, fl_first, fl_last], how="vertical_relaxed")
            .sort(["row_id", "time", "code"])
        )

    def add_era_ids_to_dedup(
        self,
        chronic_dedup: pl.LazyFrame,
        max_gap_days: int,
    ) -> pl.LazyFrame:
        return (
            chronic_dedup
            .sort(CODE_SUMMARY_COLS + ["time", "code"])
            .with_columns(
                [
                    pl.col("time")
                    .shift(1)
                    .over(CODE_SUMMARY_COLS)
                    .alias("prev_dx_time")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("time") - pl.col("prev_dx_time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("gap_from_prev_dx_days")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("prev_dx_time").is_null()
                        | (pl.col("gap_from_prev_dx_days") > max_gap_days)
                    )
                    .cast(pl.Int32)
                    .alias("new_era_flag")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("new_era_flag")
                        .cum_sum()
                        .over(CODE_SUMMARY_COLS)
                        - 1
                    )
                    .cast(pl.Int32)
                    .alias("era_idx")
                ]
            )
        )

    def build_era_summary_lf(
        self,
        chronic_dedup: pl.LazyFrame,
        max_gap_days: int,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        dx_with_era = self.add_era_ids_to_dedup(
            chronic_dedup=chronic_dedup,
            max_gap_days=max_gap_days,
        )

        era_key_cols = CODE_SUMMARY_COLS + ["era_idx"]

        era_summary = (
            dx_with_era
            .group_by(era_key_cols)
            .agg(
                [
                    pl.len().alias("n_dx_points"),
                    pl.col("time").min().alias("first_time"),
                    pl.col("time").max().alias("last_time"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("last_time") - pl.col("first_time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("span_days"),

                    (
                        pl.col("prediction_time") - pl.col("last_time")
                    )
                    .dt.total_days()
                    .cast(pl.Float32)
                    .alias("recency_days"),
                ]
            )
        )

        return dx_with_era, era_summary

    @staticmethod
    def count_bin_expr(n_expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(n_expr <= 1)
            .then(pl.lit("1"))
            .when(n_expr == 2)
            .then(pl.lit("2"))
            .when(n_expr <= 5)
            .then(pl.lit("3_5"))
            .when(n_expr <= 10)
            .then(pl.lit("6_10"))
            .otherwise(pl.lit("gt10"))
        )

    @staticmethod
    def duration_bin_expr(days_expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(days_expr <= 0)
            .then(pl.lit("0"))
            .when(days_expr <= 30)
            .then(pl.lit("1_30"))
            .when(days_expr <= 90)
            .then(pl.lit("31_90"))
            .when(days_expr <= 180)
            .then(pl.lit("91_180"))
            .when(days_expr <= 365)
            .then(pl.lit("181_365"))
            .when(days_expr <= 730)
            .then(pl.lit("366_730"))
            .otherwise(pl.lit("gt730"))
        )

    @staticmethod
    def recency_bin_expr(days_expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(days_expr <= 7)
            .then(pl.lit("0_7"))
            .when(days_expr <= 30)
            .then(pl.lit("8_30"))
            .when(days_expr <= 90)
            .then(pl.lit("31_90"))
            .when(days_expr <= 180)
            .then(pl.lit("91_180"))
            .when(days_expr <= 365)
            .then(pl.lit("181_365"))
            .otherwise(pl.lit("gt365"))
        )

    def build_condition_era_long_lf(
        self,
        labels: pl.DataFrame,
        max_gap_days: int,
    ) -> pl.LazyFrame:
        non_compressed, chronic_dedup = self.prepare_chronic_dedup_lf(labels)
        dx_with_era, era_summary = self.build_era_summary_lf(chronic_dedup, max_gap_days)

        era_key_cols = CODE_SUMMARY_COLS + ["era_idx"]

        normal_keys = (
            era_summary
            .filter(pl.col("n_dx_points") <= 1)
            .select(era_key_cols)
        )

        compressed_summary = era_summary.filter(pl.col("n_dx_points") >= 2)

        normal_dx = (
            dx_with_era
            .join(normal_keys, on=era_key_cols, how="inner")
            .select(STANDARD_LONG_COLS)
        )

        prefix = f"DX_ERA{max_gap_days}"

        era_start = self.compression_token_from_summary(
            compressed_summary,
            token_prefix=f"{prefix}_START/",
            time_col="first_time",
            numeric_expr=pl.lit(0.0),
        )

        era_end = self.compression_token_from_summary(
            compressed_summary,
            token_prefix=f"{prefix}_END/",
            time_col="last_time",
            numeric_expr=pl.col("span_days"),
        )

        era_count = self.compression_token_from_summary(
            compressed_summary,
            token_prefix=None,
            time_col="last_time",
            numeric_expr=pl.col("n_dx_points").cast(pl.Float32),
            code_expr=pl.concat_str(
                [
                    pl.lit(f"{prefix}_COUNT_BIN/"),
                    self.count_bin_expr(pl.col("n_dx_points")),
                    pl.lit("/"),
                    pl.col("code"),
                ]
            ),
        )

        return (
            pl.concat(
                [non_compressed, normal_dx, era_start, era_end, era_count],
                how="vertical_relaxed",
            )
            .sort(["row_id", "time", "code"])
        )

    def build_state_duration_long_lf(
        self,
        labels: pl.DataFrame,
        max_gap_days: int,
    ) -> pl.LazyFrame:
        non_compressed, chronic_dedup = self.prepare_chronic_dedup_lf(labels)
        dx_with_era, era_summary = self.build_era_summary_lf(chronic_dedup, max_gap_days)

        era_key_cols = CODE_SUMMARY_COLS + ["era_idx"]

        normal_keys = (
            era_summary
            .filter(pl.col("n_dx_points") <= 1)
            .select(era_key_cols)
        )

        state_summary = era_summary.filter(pl.col("n_dx_points") >= 2)

        normal_dx = (
            dx_with_era
            .join(normal_keys, on=era_key_cols, how="inner")
            .select(STANDARD_LONG_COLS)
        )

        prefix = f"DX_STATE{max_gap_days}"

        onset = self.compression_token_from_summary(
            state_summary,
            token_prefix=f"{prefix}_ONSET/",
            time_col="first_time",
            numeric_expr=pl.lit(0.0),
        )

        active = self.compression_token_from_summary(
            state_summary,
            token_prefix=f"{prefix}_ACTIVE/",
            time_col="last_time",
            numeric_expr=pl.col("n_dx_points").cast(pl.Float32),
        )

        duration = self.compression_token_from_summary(
            state_summary,
            token_prefix=None,
            time_col="last_time",
            numeric_expr=pl.col("span_days"),
            code_expr=pl.concat_str(
                [
                    pl.lit(f"{prefix}_DURATION_BIN/"),
                    self.duration_bin_expr(pl.col("span_days")),
                    pl.lit("/"),
                    pl.col("code"),
                ]
            ),
        )

        count = self.compression_token_from_summary(
            state_summary,
            token_prefix=None,
            time_col="last_time",
            numeric_expr=pl.col("n_dx_points").cast(pl.Float32),
            code_expr=pl.concat_str(
                [
                    pl.lit(f"{prefix}_COUNT_BIN/"),
                    self.count_bin_expr(pl.col("n_dx_points")),
                    pl.lit("/"),
                    pl.col("code"),
                ]
            ),
        )

        recency = self.compression_token_from_summary(
            state_summary,
            token_prefix=None,
            time_col="last_time",
            numeric_expr=pl.col("recency_days"),
            code_expr=pl.concat_str(
                [
                    pl.lit(f"{prefix}_RECENCY_BIN/"),
                    self.recency_bin_expr(pl.col("recency_days")),
                    pl.lit("/"),
                    pl.col("code"),
                ]
            ),
        )

        return (
            pl.concat(
                [non_compressed, normal_dx, onset, active, duration, count, recency],
                how="vertical_relaxed",
            )
            .sort(["row_id", "time", "code"])
        )

    def build_compressed_hybrid_long_lf(
        self,
        labels: pl.DataFrame,
        max_gap_days: int = 90,
    ) -> pl.LazyFrame:
        """
        Hybrid compression:
            1. сначала dedup внутри reconstructed visit/day;
            2. для single mention оставляем обычный event;
            3. для коротких повторов делаем DX_FIRST / DX_LAST;
            4. для устойчивых eras делаем DX_ERA_START / DX_ERA_END / COUNT_BIN.

        Это аккуратная реализация идеи hybrid из markdown-описания ноутбука.
        """
        non_compressed, chronic_dedup = self.prepare_chronic_dedup_lf(labels)
        dx_with_era, era_summary = self.build_era_summary_lf(chronic_dedup, max_gap_days)

        era_key_cols = CODE_SUMMARY_COLS + ["era_idx"]

        normal_keys = (
            era_summary
            .filter(pl.col("n_dx_points") <= 1)
            .select(era_key_cols)
        )

        first_last_summary = (
            era_summary
            .filter(
                (pl.col("n_dx_points") >= 2)
                & ~(
                    (pl.col("n_dx_points") >= 3)
                    & (pl.col("span_days") >= 30)
                )
            )
        )

        era_summary_compressed = (
            era_summary
            .filter(
                (pl.col("n_dx_points") >= 3)
                & (pl.col("span_days") >= 30)
            )
        )

        normal_dx = (
            dx_with_era
            .join(normal_keys, on=era_key_cols, how="inner")
            .select(STANDARD_LONG_COLS)
        )

        fl_first = self.compression_token_from_summary(
            first_last_summary,
            token_prefix="DX_FIRST/",
            time_col="first_time",
            numeric_expr=pl.lit(0.0),
        )

        fl_last = self.compression_token_from_summary(
            first_last_summary,
            token_prefix="DX_LAST/",
            time_col="last_time",
            numeric_expr=pl.col("span_days"),
        )

        prefix = f"DX_HYBRID_ERA{max_gap_days}"

        era_start = self.compression_token_from_summary(
            era_summary_compressed,
            token_prefix=f"{prefix}_START/",
            time_col="first_time",
            numeric_expr=pl.lit(0.0),
        )

        era_end = self.compression_token_from_summary(
            era_summary_compressed,
            token_prefix=f"{prefix}_END/",
            time_col="last_time",
            numeric_expr=pl.col("span_days"),
        )

        era_count = self.compression_token_from_summary(
            era_summary_compressed,
            token_prefix=None,
            time_col="last_time",
            numeric_expr=pl.col("n_dx_points").cast(pl.Float32),
            code_expr=pl.concat_str(
                [
                    pl.lit(f"{prefix}_COUNT_BIN/"),
                    self.count_bin_expr(pl.col("n_dx_points")),
                    pl.lit("/"),
                    pl.col("code"),
                ]
            ),
        )

        return (
            pl.concat(
                [
                    non_compressed,
                    normal_dx,
                    fl_first,
                    fl_last,
                    era_start,
                    era_end,
                    era_count,
                ],
                how="vertical_relaxed",
            )
            .sort(["row_id", "time", "code"])
        )

    def build_long_lf(self, labels: pl.DataFrame, version: str) -> pl.LazyFrame:
        if version == "raw":
            return self.build_raw_long_lf(labels)

        if version == "compressed_dedup":
            return self.build_compressed_dedup_long_lf(labels)

        if version == "compressed_first_last":
            return self.build_compressed_first_last_long_lf(labels)

        if version == "condition_era_90":
            return self.build_condition_era_long_lf(labels, max_gap_days=90)

        if version == "condition_era_180":
            return self.build_condition_era_long_lf(labels, max_gap_days=180)

        if version == "state_duration_90":
            return self.build_state_duration_long_lf(labels, max_gap_days=90)

        if version == "state_duration_180":
            return self.build_state_duration_long_lf(labels, max_gap_days=180)

        if version == "compressed_condition_era":
            return self.build_condition_era_long_lf(labels, max_gap_days=90)

        if version == "compressed_hybrid":
            return self.build_compressed_hybrid_long_lf(labels, max_gap_days=90)

        raise ValueError(f"Unknown version: {version}")

    # -------------------------------------------------------------------------
    # Long -> sequences
    # -------------------------------------------------------------------------

    def build_vocab_from_train(
        self,
        long_lf: pl.LazyFrame,
    ) -> tuple[dict[str, int], pd.DataFrame]:
        counts = (
            long_lf
            .filter(pl.col("split") == "train")
            .group_by("code")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= self.min_code_count)
            .sort("n", descending=True)
            .collect()
            .to_pandas()
        )

        if self.max_vocab_size is not None:
            counts = counts.head(max(0, self.max_vocab_size - len(SPECIAL_TOKENS)))

        vocab = dict(SPECIAL_TOKENS)

        for code in counts["code"].astype(str).tolist():
            if code not in vocab:
                vocab[code] = len(vocab)

        return vocab, counts

    def long_to_sequences(
        self,
        long_lf: pl.LazyFrame,
        labels: pl.DataFrame,
        vocab: dict[str, int],
    ) -> pd.DataFrame:
        vocab_df = pl.DataFrame(
            {
                "code": list(vocab.keys()),
                "token_id": list(vocab.values()),
            }
        )

        seq_lf = (
            long_lf
            .join(vocab_df.lazy(), on="code", how="left")
            .with_columns(
                [
                    pl.col("token_id")
                    .fill_null(UNK_ID)
                    .cast(pl.Int32)
                    .alias("token_id")
                ]
            )
            .sort(["row_id", "time", "code"])
            .with_columns(
                [
                    pl.col("time")
                    .diff()
                    .over("row_id")
                    .dt.total_days()
                    .fill_null(0)
                    .cast(pl.Float32)
                    .alias("delta_days")
                ]
            )
        )

        agg_exprs = [
            pl.col("token_id").alias("token_ids"),
            pl.col("days_before_prediction").cast(pl.Float32).alias("days_before_prediction"),
            pl.col("delta_days").cast(pl.Float32).alias("delta_days"),
            pl.col("numeric_value").cast(pl.Float32).alias("numeric_values"),
            pl.len().cast(pl.Int32).alias("seq_len"),
            pl.col("token_id").n_unique().cast(pl.Int32).alias("n_unique_tokens"),
            pl.col("is_compression_token").sum().cast(pl.Int32).alias("n_compression_events"),
        ]

        if self.keep_code_strings:
            agg_exprs.insert(1, pl.col("code").alias("codes"))

        seqs = (
            seq_lf
            .group_by(
                [
                    "task",
                    "row_id",
                    "subject_id",
                    "prediction_time",
                    "label",
                    "split",
                ],
                maintain_order=True,
            )
            .agg(agg_exprs)
            .collect()
            .to_pandas()
        )

        label_cols = ["task", "row_id", "subject_id", "prediction_time", "label", "split"]

        out = labels.to_pandas()[label_cols].merge(
            seqs,
            on=label_cols,
            how="left",
        )

        list_cols = ["token_ids", "days_before_prediction", "delta_days", "numeric_values"]
        if self.keep_code_strings:
            list_cols.append("codes")

        for col in list_cols:
            out[col] = out[col].apply(as_python_list)

        out["seq_len"] = out["token_ids"].apply(len).astype(np.int32)
        out["n_unique_tokens"] = out["token_ids"].apply(lambda xs: len(set(xs))).astype(np.int32)
        out["n_compression_events"] = out["n_compression_events"].fillna(0).astype(np.int32)

        return out

    def is_diagnosis_like_code_str(self, code: str) -> bool:
        code = str(code)
        family = code.split("/", 1)[0]
        return family in DIAGNOSIS_LIKE_FAMILIES or code in self.chronic_code_whitelist

    def add_sequence_audit_columns(
        self,
        seq_df: pd.DataFrame,
        task_name: str,
        version: str,
    ) -> pd.DataFrame:
        seq_df = seq_df.copy()

        seq_df["n_synthetic_events"] = (
            seq_df["n_compression_events"]
            .fillna(0)
            .astype(np.int32)
        )

        if version == "raw":
            seq_df["raw_seq_len"] = seq_df["seq_len"].astype(np.int32)
            seq_df["n_events_removed_vs_raw"] = 0

            if "codes" in seq_df.columns:
                seq_df["n_diagnosis_like_events_raw"] = seq_df["codes"].apply(
                    lambda xs: sum(
                        self.is_diagnosis_like_code_str(c)
                        for c in as_python_list(xs)
                    )
                ).astype(np.int32)

                seq_df["n_compressible_chronic_events_raw"] = seq_df["codes"].apply(
                    lambda xs: sum(
                        str(c) in self.chronic_code_whitelist
                        for c in as_python_list(xs)
                    )
                ).astype(np.int32)
            else:
                seq_df["n_diagnosis_like_events_raw"] = 0
                seq_df["n_compressible_chronic_events_raw"] = 0

            return seq_df

        raw_path = self.output_dir / task_name / "raw" / "examples.parquet"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw dataset must be built before compressed versions. Missing: {raw_path}"
            )

        raw_df = pd.read_parquet(raw_path)

        needed_cols = [
            "row_id",
            "raw_seq_len",
            "n_diagnosis_like_events_raw",
            "n_compressible_chronic_events_raw",
        ]

        missing = [c for c in needed_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(
                f"Raw dataset does not contain audit columns {missing}. "
                "Rebuild raw first."
            )

        raw_ref = raw_df[needed_cols].copy()

        seq_df = seq_df.merge(
            raw_ref,
            on="row_id",
            how="left",
            validate="one_to_one",
        )

        seq_df["raw_seq_len"] = (
            seq_df["raw_seq_len"]
            .fillna(seq_df["seq_len"])
            .astype(np.int32)
        )

        seq_df["n_events_removed_vs_raw"] = (
            seq_df["raw_seq_len"].astype(int)
            - seq_df["seq_len"].astype(int)
        ).astype(np.int32)

        seq_df["n_diagnosis_like_events_raw"] = (
            seq_df["n_diagnosis_like_events_raw"]
            .fillna(0)
            .astype(np.int32)
        )

        seq_df["n_compressible_chronic_events_raw"] = (
            seq_df["n_compressible_chronic_events_raw"]
            .fillna(0)
            .astype(np.int32)
        )

        return seq_df

    def save_sequence_dataset(
        self,
        task_name: str,
        version: str,
        labels: pl.DataFrame,
    ) -> dict[str, Any]:
        out_dir = self.output_dir / task_name / version
        out_dir.mkdir(parents=True, exist_ok=True)

        examples_path = out_dir / "examples.parquet"
        vocab_path = out_dir / "vocab.json"
        counts_path = out_dir / "code_counts_train.csv"
        metadata_path = out_dir / "metadata.json"

        if (
            examples_path.exists()
            and vocab_path.exists()
            and metadata_path.exists()
            and not self.rebuild
        ):
            print(f"[{task_name} | {version}] already exists, skip: {examples_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print("\n" + "=" * 100)
        print(f"Building sequence dataset: task={task_name} | version={version}")

        long_lf = self.build_long_lf(labels=labels, version=version)
        vocab, code_counts = self.build_vocab_from_train(long_lf)
        seq_df = self.long_to_sequences(long_lf, labels, vocab)
        seq_df = self.add_sequence_audit_columns(
            seq_df=seq_df,
            task_name=task_name,
            version=version,
        )

        non_empty_days = [
            d
            for xs in seq_df["days_before_prediction"]
            for d in as_python_list(xs)
        ]
        min_days_before = float(np.min(non_empty_days)) if non_empty_days else np.nan

        if not np.isnan(min_days_before) and min_days_before < -1e-6:
            raise ValueError(
                f"Found future events: task={task_name}, version={version}, "
                f"min_days_before={min_days_before}"
            )

        seq_df.to_parquet(examples_path, index=False)
        code_counts.to_csv(counts_path, index=False)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        split_summary = (
            seq_df
            .groupby("split")
            .agg(
                n_examples=("label", "size"),
                n_patients=("subject_id", "nunique"),
                n_positive=("label", "sum"),
                event_rate=("label", "mean"),
                mean_seq_len=("seq_len", "mean"),
                median_seq_len=("seq_len", "median"),
                p90_seq_len=("seq_len", lambda x: float(np.quantile(x, 0.90))),
                max_seq_len=("seq_len", "max"),
                mean_compression_events=("n_compression_events", "mean"),
                mean_events_removed_vs_raw=("n_events_removed_vs_raw", "mean"),
                p90_events_removed_vs_raw=(
                    "n_events_removed_vs_raw",
                    lambda x: float(np.quantile(x, 0.90)),
                ),
                mean_synthetic_events=("n_synthetic_events", "mean"),
                mean_compressible_chronic_events_raw=(
                    "n_compressible_chronic_events_raw",
                    "mean",
                ),
            )
            .reset_index()
        )

        metadata = {
            "task": task_name,
            "version": version,
            "n_examples": int(len(seq_df)),
            "n_patients": int(seq_df["subject_id"].nunique()),
            "n_positive": int(seq_df["label"].sum()),
            "event_rate": float(seq_df["label"].mean()),
            "vocab_size": int(len(vocab)),
            "mean_seq_len": float(seq_df["seq_len"].mean()),
            "median_seq_len": float(seq_df["seq_len"].median()),
            "p90_seq_len": float(np.quantile(seq_df["seq_len"], 0.90)),
            "max_seq_len": int(seq_df["seq_len"].max()) if len(seq_df) else 0,
            "mean_events_removed_vs_raw": float(seq_df["n_events_removed_vs_raw"].mean()),
            "median_events_removed_vs_raw": float(seq_df["n_events_removed_vs_raw"].median()),
            "p90_events_removed_vs_raw": float(np.quantile(seq_df["n_events_removed_vs_raw"], 0.90)),
            "mean_synthetic_events": float(seq_df["n_synthetic_events"].mean()),
            "mean_compressible_chronic_events_raw": float(seq_df["n_compressible_chronic_events_raw"].mean()),
            "min_days_before_prediction": min_days_before,
            "include_prediction_time": bool(self.include_prediction_time),
            "keep_code_strings": bool(self.keep_code_strings),
            "max_vocab_size": self.max_vocab_size,
            "min_code_count": int(self.min_code_count),
            "compression": {
                "compression_scope": "empirical_chronic_like_codes_only",
                "visit_bucket_source": "reconstructed_from_Visit_codes_with_day_fallback",
                "visit_anchor_max_days": int(self.visit_anchor_max_days),
                "chronic_code_mode": self.chronic_code_mode,
                "n_chronic_codes_in_whitelist": int(len(self.chronic_code_whitelist)),
                "versions": {
                    "raw": "no compression",
                    "compressed_dedup": "deduplicate same chronic code inside reconstructed visit/day bucket",
                    "compressed_first_last": "after dedup, replace repeated chronic code by DX_FIRST and DX_LAST",
                    "condition_era_90": "after dedup, split mentions into eras with max gap 90 days",
                    "condition_era_180": "after dedup, split mentions into eras with max gap 180 days",
                    "state_duration_90": "state-duration representation with max gap 90 days",
                    "state_duration_180": "state-duration representation with max gap 180 days",
                    "compressed_condition_era": "alias of condition_era_90",
                    "compressed_hybrid": "hybrid first-last / condition-era compression with 90-day era gap",
                },
            },
            "files": {
                "examples": str(examples_path),
                "vocab": str(vocab_path),
                "code_counts_train": str(counts_path),
                "metadata": str(metadata_path),
            },
            "split_summary": split_summary.to_dict(orient="records"),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metadata), f, ensure_ascii=False, indent=2)

        print("Saved:", examples_path)
        print("Saved:", vocab_path)
        print("Saved:", counts_path)
        print("Saved:", metadata_path)
        print(split_summary.to_string(index=False))

        return metadata

    def build_sanity_table(self) -> pd.DataFrame:
        rows = []

        for task_name in self.tasks:
            raw_path = self.output_dir / task_name / "raw" / "examples.parquet"
            if not raw_path.exists():
                continue

            raw_df = pd.read_parquet(raw_path)
            raw_keys = raw_df[
                ["row_id", "subject_id", "label", "split", "seq_len"]
            ].rename(columns={"seq_len": "raw_len_check"})

            for version in self.versions:
                path = self.output_dir / task_name / version / "examples.parquet"
                if not path.exists():
                    continue

                df = pd.read_parquet(path)

                merged = df.merge(
                    raw_keys,
                    on=["row_id", "subject_id", "label", "split"],
                    how="left",
                    validate="one_to_one",
                )

                days_values = [
                    d
                    for xs in df["days_before_prediction"]
                    for d in as_python_list(xs)
                ]

                rows.append(
                    {
                        "task": task_name,
                        "version": version,
                        "n_examples": int(len(df)),
                        "n_patients": int(df["subject_id"].nunique()),
                        "n_positive": int(df["label"].sum()),
                        "event_rate": float(df["label"].mean()),
                        "mean_seq_len": float(df["seq_len"].mean()),
                        "p90_seq_len": float(np.quantile(df["seq_len"], 0.90)),
                        "mean_removed_vs_raw": float(
                            (merged["raw_len_check"] - merged["seq_len"]).mean()
                        ),
                        "p90_removed_vs_raw": float(
                            np.quantile(
                                merged["raw_len_check"] - merged["seq_len"],
                                0.90,
                            )
                        ),
                        "mean_synthetic_events": float(df["n_synthetic_events"].mean()),
                        "min_days_before_prediction": (
                            float(np.min(days_values))
                            if days_values
                            else np.nan
                        ),
                        "has_future_events": bool(
                            len(days_values) > 0 and np.min(days_values) < -1e-6
                        ),
                    }
                )

        sanity_df = pd.DataFrame(rows)

        if len(sanity_df) and sanity_df["has_future_events"].any():
            raise ValueError("Found future events in some sequence dataset.")

        return sanity_df

    def build_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Собирает все task/version datasets.

        raw всегда строим первым для каждой task, потому что compressed versions
        используют raw для audit columns.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []

        ordered_versions = ["raw"] + [v for v in self.versions if v != "raw"]

        labels_by_task = {
            task_name: self.load_task_labels(task_name)
            for task_name in self.tasks
        }

        for task_name, labels in labels_by_task.items():
            for version in ordered_versions:
                metadata = self.save_sequence_dataset(
                    task_name=task_name,
                    version=version,
                    labels=labels,
                )
                metadata_rows.append(metadata)

        summary_rows = []
        for m in metadata_rows:
            summary_rows.append(
                {
                    "task": m["task"],
                    "version": m["version"],
                    "n_examples": m["n_examples"],
                    "n_patients": m["n_patients"],
                    "n_positive": m["n_positive"],
                    "event_rate": m["event_rate"],
                    "vocab_size": m["vocab_size"],
                    "mean_seq_len": m["mean_seq_len"],
                    "median_seq_len": m["median_seq_len"],
                    "p90_seq_len": m["p90_seq_len"],
                    "max_seq_len": m["max_seq_len"],
                    "mean_events_removed_vs_raw": m["mean_events_removed_vs_raw"],
                    "median_events_removed_vs_raw": m["median_events_removed_vs_raw"],
                    "p90_events_removed_vs_raw": m["p90_events_removed_vs_raw"],
                    "mean_synthetic_events": m["mean_synthetic_events"],
                    "mean_compressible_chronic_events_raw": m[
                        "mean_compressible_chronic_events_raw"
                    ],
                    "min_days_before_prediction": m["min_days_before_prediction"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / "all_compression_version_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        sanity_df = self.build_sanity_table()
        sanity_path = self.output_dir / "sequence_dataset_sanity.csv"
        sanity_df.to_csv(sanity_path, index=False)

        print("\nSaved summary:", summary_path)
        print("Saved sanity:", sanity_path)

        print("\nSummary preview:")
        print(summary_df.to_string(index=False))

        print("\nSanity preview:")
        print(sanity_df.to_string(index=False))

        return summary_df, sanity_df


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    task = maybe_init_clearml(args)

    notebook_root = Path(args.notebook_root).resolve()

    args.ehrshot_root = resolve_path(notebook_root, args.ehrshot_root)
    args.audit_dir = resolve_path(notebook_root, args.audit_dir)
    args.output_dir = resolve_path(notebook_root, args.output_dir)

    tasks = parse_csv_list(args.tasks)
    versions = parse_csv_list(args.versions)

    keep_code_strings = bool(args.keep_code_strings)
    if args.drop_code_strings:
        keep_code_strings = False

    include_prediction_time = bool(args.include_prediction_time)
    if args.exclude_prediction_time:
        include_prediction_time = False

    max_vocab_size = args.max_vocab_size if args.max_vocab_size > 0 else None

    print("=" * 100)
    print("BUILD SEQUENCE DATASETS FROM MEDS")
    print(f"notebook_root: {notebook_root}")
    print(f"ehrshot_root: {args.ehrshot_root}")
    print(f"audit_dir: {args.audit_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"tasks: {tasks}")
    print(f"versions: {versions}")
    print(f"keep_code_strings: {keep_code_strings}")
    print(f"include_prediction_time: {include_prediction_time}")
    print(f"output_s3_prefix: {args.output_s3_prefix}")
    print("=" * 100)

    builder = SequenceDatasetBuilder(
        ehrshot_root=args.ehrshot_root,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        tasks=tasks,
        versions=versions,
        chronic_code_mode=args.chronic_code_mode,
        keep_code_strings=keep_code_strings,
        include_prediction_time=include_prediction_time,
        visit_anchor_max_days=args.visit_anchor_max_days,
        max_vocab_size=max_vocab_size,
        min_code_count=args.min_code_count,
        rebuild=args.rebuild,
    )

    summary_df, sanity_df = builder.build_all()

    upload_manifest_df = pd.DataFrame()
    upload_manifest_path = args.output_dir / "minio_upload_manifest.csv"

    if not args.skip_upload:
        upload_manifest_df = upload_tree_to_minio(
            local_root=args.output_dir,
            s3_prefix=args.output_s3_prefix,
        )
        upload_manifest_df.to_csv(upload_manifest_path, index=False)
        print("Saved upload manifest:", upload_manifest_path)
    else:
        print("Skip MinIO upload because --skip-upload is set.")

    if task is not None:
        task.upload_artifact("sequence_dataset_summary", summary_df)
        task.upload_artifact("sequence_dataset_sanity", sanity_df)
        if len(upload_manifest_df):
            task.upload_artifact("minio_upload_manifest", upload_manifest_df)
        task.upload_artifact("sequence_dataset_output_dir", str(args.output_dir))

    print("=" * 100)
    print("DONE")
    print(f"Local output: {args.output_dir}")
    if not args.skip_upload:
        print(f"MinIO/S3 prefix: {args.output_s3_prefix}")


if __name__ == "__main__":
    main()