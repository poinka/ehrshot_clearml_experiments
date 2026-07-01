from __future__ import annotations

"""
split_audit.py

Назначение:
    Проверить, что split в EHRSHOT reproducibility package сделан корректно.

Что проверяем:
    1. Нет ли patient overlap между train / tuning / held_out внутри каждого датасета.
    2. Не попал ли один subject_id в разные split внутри одного источника.
    3. Не попал ли один row_id / prediction example в разные split.
    4. Совпадает ли split в tabular / sequence датасетах с официальным
       EHRSHOT_MEDS/metadata/subject_splits.parquet.
    5. Совпадает ли mapping row_id -> subject_id / label / split / prediction_time
       между:
          - label reference из EHRSHOT_MEDS;
          - tabular <= prediction_time cache;
          - tabular < prediction_time cache;
          - sequence raw/compressed versions.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Константы
# -----------------------------------------------------------------------------

DEFAULT_TASKS = ["guo_readmission", "guo_icu"]

EXPECTED_SPLITS = ["train", "tuning", "held_out"]

# Колонки, которые должны быть в финальной нормализованной таблице
# для split audit.
CORE_COLUMNS = [
    "source_type",
    "source_name",
    "task",
    "version",
    "time_rule",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
    "path",
]


# -----------------------------------------------------------------------------
# 2. CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ehrshot-root",
        type=Path,
        default=Path("EHRSHOT_MEDS"),
        help="Папка с EHRSHOT_MEDS: metadata/subject_splits.parquet и labels/<task>/labels.parquet.",
    )

    parser.add_argument(
        "--tabular-cache-dir",
        type=Path,
        default=Path("ehrshot_baseline_cache"),
        help="Папка с tabular feature cache parquet.",
    )

    parser.add_argument(
        "--sequence-data-dir",
        type=Path,
        default=Path("ehrshot_sequence_datasets"),
        help="Папка с sequence datasets: <task>/<version>/examples.parquet.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_split_leakage_audit"),
        help="Куда сохранить audit CSV и markdown summary.",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated список задач.",
    )

    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Завершать скрипт с ошибкой, если найдены критичные split issues.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3. Маленькие helper-функции
# -----------------------------------------------------------------------------

def parse_csv_list(x: str) -> list[str]:
    """
    Превращает строку вида 'a,b,c' в список ['a', 'b', 'c'].
    """
    return [v.strip() for v in str(x).split(",") if v.strip()]


def ensure_output_dir(path: Path) -> Path:
    """
    Создает output directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Сохраняет CSV и создает родительскую папку, если ее нет.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_unique_join(values: Any, max_items: int = 20) -> str:
    """
    Красиво склеивает уникальные значения в строку.

    Нужно для колонок вроде:
        splits = train|held_out
        sources = tabular__...|sequence__...
    """
    s = pd.Series(values).dropna().astype(str)
    vals = sorted(s.unique().tolist())

    if len(vals) > max_items:
        vals = vals[:max_items] + [f"...(+{len(vals) - max_items})"]

    return "|".join(vals)


def safe_sample_join(values: Any, max_items: int = 10) -> str:
    """
    Сохраняет небольшой sample значений для диагностики.
    """
    s = pd.Series(values).dropna().astype(str)
    vals = s.drop_duplicates().head(max_items).tolist()
    return "|".join(vals)


def normalize_prediction_time_col(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает prediction_time как строку.

    Для split audit нам не нужно делать временную арифметику.
    Важно только сравнить, совпадает ли prediction_time между источниками.
    """
    if "prediction_time" not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    return df["prediction_time"].astype("string")


def detect_label_col(df: pd.DataFrame) -> str:
    """
    Находит колонку с label в EHRSHOT labels.parquet.

    В разных label files целевая переменная может называться по-разному.
    Для guo_readmission / guo_icu обычно подходит boolean_value или label.
    """
    candidates = [
        "boolean_value",
        "integer_value",
        "float_value",
        "categorical_value",
        "label",
        "value",
    ]

    existing = [c for c in candidates if c in df.columns]

    if not existing:
        raise ValueError(
            f"Не нашла label column. Доступные колонки: {list(df.columns)}"
        )

    # Берем первую колонку-кандидат, где есть хотя бы одно непустое значение.
    for col in existing:
        if df[col].notna().sum() > 0:
            return col

    raise ValueError(
        f"Все candidate label columns пустые. Кандидаты: {existing}"
    )


def normalize_label_values(s: pd.Series) -> pd.Series:
    """
    Приводит label к 0/1, насколько это возможно.
    """
    if s.dtype == bool:
        return s.astype("Int64")

    out = pd.to_numeric(s, errors="coerce")

    # На случай, если label был строкой True/False.
    if out.isna().all():
        lower = s.astype(str).str.lower()
        out = lower.map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        )

    return out.astype("Int64")


def detect_split_col(df: pd.DataFrame) -> str:
    """
    Находит колонку split в subject_splits.parquet.
    Обычно она называется split.
    """
    if "split" in df.columns:
        return "split"

    candidates = [c for c in df.columns if "split" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Не нашла split column в subject_splits.parquet. "
        f"Колонки: {list(df.columns)}"
    )


def normalize_core_frame(
    df: pd.DataFrame,
    *,
    source_type: str,
    source_name: str,
    task: str,
    version: str,
    time_rule: str,
    path: Path,
) -> pd.DataFrame:
    """
    Приводит любой источник к единой схеме split audit.

    На вход может прийти:
        - label reference;
        - tabular feature cache;
        - sequence examples.parquet.

    На выходе всегда есть:
        source_type, source_name, task, version, time_rule,
        row_id, subject_id, prediction_time, label, split, path.
    """
    required = {"row_id", "subject_id", "label", "split"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{source_name}: не хватает обязательных колонок: {missing}. "
            f"path={path}"
        )

    out = pd.DataFrame(
        {
            "source_type": source_type,
            "source_name": source_name,
            "task": task,
            "version": version,
            "time_rule": time_rule,
            "row_id": pd.to_numeric(df["row_id"], errors="coerce").astype("Int64"),
            "subject_id": pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64"),
            "prediction_time": normalize_prediction_time_col(df),
            "label": normalize_label_values(df["label"]),
            "split": df["split"].astype("string"),
            "path": str(path),
        }
    )

    return out[CORE_COLUMNS].copy()


# -----------------------------------------------------------------------------
# 4. Загрузка official EHRSHOT splits и labels
# -----------------------------------------------------------------------------

def load_official_subject_splits(ehrshot_root: Path) -> pd.DataFrame:
    """
    Читает EHRSHOT_MEDS/metadata/subject_splits.parquet.

    Это главный источник истины для patient-level split.
    """
    path = Path(ehrshot_root) / "metadata" / "subject_splits.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Не найден official split file: {path}")

    df = pd.read_parquet(path)

    if "subject_id" not in df.columns:
        raise ValueError(
            f"В official split file нет subject_id. path={path}, columns={list(df.columns)}"
        )

    split_col = detect_split_col(df)

    out = (
        df[["subject_id", split_col]]
        .rename(columns={split_col: "official_split"})
        .copy()
    )

    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    out["official_split"] = out["official_split"].astype("string")

    # На всякий случай убираем дубли exact same rows.
    out = out.drop_duplicates(["subject_id", "official_split"]).reset_index(drop=True)

    # Один subject_id должен иметь ровно один official split.
    bad = (
        out.groupby("subject_id", dropna=False)["official_split"]
        .nunique(dropna=True)
        .reset_index(name="n_official_splits")
    )
    bad = bad[bad["n_official_splits"] > 1]

    if not bad.empty:
        raise ValueError(
            "В official subject_splits.parquet один subject_id встречается в нескольких split. "
            f"Примеры:\n{bad.head(20).to_string(index=False)}"
        )

    return out


def load_label_reference(
    ehrshot_root: Path,
    tasks: list[str],
    official_splits: pd.DataFrame,
) -> pd.DataFrame:
    """
    Загружает labels.parquet для каждой задачи и добавляет official split.

    Это нужно как reference:
        row_id -> subject_id / prediction_time / label / split.

    Важно:
        row_id здесь создается так же, как в builder:
        обычный индекс строки labels.parquet, начиная с 0.
    """
    frames: list[pd.DataFrame] = []

    labels_root = Path(ehrshot_root) / "labels"

    for task in tasks:
        path = labels_root / task / "labels.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Не найден labels file для task={task}: {path}")

        df = pd.read_parquet(path).reset_index(drop=True)

        if "subject_id" not in df.columns:
            raise ValueError(f"{path}: нет subject_id")

        if "prediction_time" not in df.columns:
            raise ValueError(f"{path}: нет prediction_time")

        label_col = detect_label_col(df)

        ref = pd.DataFrame(
            {
                "row_id": np.arange(len(df), dtype=np.int64),
                "subject_id": pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64"),
                "prediction_time": df["prediction_time"].astype("string"),
                "label": normalize_label_values(df[label_col]),
            }
        )

        ref = ref.merge(
            official_splits,
            on="subject_id",
            how="left",
            validate="many_to_one",
        )

        missing_split = ref["official_split"].isna().sum()
        if missing_split > 0:
            raise ValueError(
                f"{task}: {missing_split} label rows не нашли official split по subject_id."
            )

        ref = ref.rename(columns={"official_split": "split"})

        core = normalize_core_frame(
            ref,
            source_type="label_reference",
            source_name=f"label_reference__{task}",
            task=task,
            version="labels",
            time_rule="not_applicable",
            path=path,
        )

        frames.append(core)

    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# 5. Загрузка tabular feature cache
# -----------------------------------------------------------------------------

def infer_tabular_task_from_filename(path: Path, tasks: list[str]) -> str | None:
    """
    Определяет task по имени tabular parquet.
    """
    name = path.name

    for task in tasks:
        if name.startswith(task + "_"):
            return task

    return None


def infer_tabular_time_rule(path: Path) -> str:
    """
    Определяет, какой cutoff rule использовался в tabular cache.

    По твоему naming:
        *_lt_prediction_time_*  -> event_time < prediction_time
        без lt_prediction_time  -> event_time <= prediction_time
    """
    name = path.name

    if "_lt_prediction_time_" in name:
        return "event_time < prediction_time"

    return "event_time <= prediction_time"


def infer_tabular_version(path: Path) -> str:
    """
    Человеческое имя версии tabular cache.
    """
    if "_lt_prediction_time_" in path.name:
        return "tabular_lt_prediction_time"

    return "tabular_leq_prediction_time"


def load_tabular_sources(
    tabular_cache_dir: Path,
    tasks: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает все tabular parquet из ehrshot_baseline_cache.

    Ожидаемые файлы:
        guo_icu_features_top500_num40.parquet
        guo_icu_lt_prediction_time_features_top500_num40.parquet
        guo_readmission_features_top500_num40.parquet
        guo_readmission_lt_prediction_time_features_top500_num40.parquet
    """
    tabular_cache_dir = Path(tabular_cache_dir)

    if not tabular_cache_dir.exists():
        raise FileNotFoundError(f"Не найдена tabular cache dir: {tabular_cache_dir}")

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    parquet_files = sorted(tabular_cache_dir.glob("*.parquet"))

    for path in parquet_files:
        task = infer_tabular_task_from_filename(path, tasks)

        # Пропускаем чужие parquet, если они есть в папке.
        if task is None:
            continue

        version = infer_tabular_version(path)
        time_rule = infer_tabular_time_rule(path)
        source_name = f"tabular__{task}__{version}"

        df = pd.read_parquet(path)

        # Если task нет в parquet, добавляем из имени файла.
        if "task" not in df.columns:
            df["task"] = task

        # Проверяем, что файл не перемешал задачи.
        observed_tasks = sorted(df["task"].astype(str).unique().tolist())
        if observed_tasks != [task]:
            raise ValueError(
                f"{path}: ожидали task={task}, но внутри observed_tasks={observed_tasks}"
            )

        core = normalize_core_frame(
            df,
            source_type="tabular",
            source_name=source_name,
            task=task,
            version=version,
            time_rule=time_rule,
            path=path,
        )
        frames.append(core)

        manifest_rows.append(
            {
                "source_type": "tabular",
                "source_name": source_name,
                "task": task,
                "version": version,
                "time_rule": time_rule,
                "path": str(path),
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "has_prediction_time": bool("prediction_time" in df.columns),
                "columns_sample": safe_sample_join(df.columns, max_items=30),
            }
        )

    if not frames:
        raise FileNotFoundError(
            f"В {tabular_cache_dir} не найдено tabular parquet для tasks={tasks}"
        )

    manifest = pd.DataFrame(manifest_rows)
    return pd.concat(frames, ignore_index=True), manifest


# -----------------------------------------------------------------------------
# 6. Загрузка sequence examples
# -----------------------------------------------------------------------------

def load_sequence_sources(
    sequence_data_dir: Path,
    tasks: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает все sequence examples.parquet:

        ehrshot_sequence_datasets/<task>/<version>/examples.parquet
    """
    sequence_data_dir = Path(sequence_data_dir)

    if not sequence_data_dir.exists():
        raise FileNotFoundError(f"Не найдена sequence data dir: {sequence_data_dir}")

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for task in tasks:
        task_dir = sequence_data_dir / task

        if not task_dir.exists():
            raise FileNotFoundError(f"Не найдена папка sequence task={task}: {task_dir}")

        version_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir()])

        for version_dir in version_dirs:
            version = version_dir.name
            path = version_dir / "examples.parquet"

            if not path.exists():
                continue

            source_name = f"sequence__{task}__{version}"

            df = pd.read_parquet(path)

            # В sequence examples task может быть уже внутри,
            # но для надежности добавляем или проверяем.
            if "task" not in df.columns:
                df["task"] = task

            observed_tasks = sorted(df["task"].astype(str).unique().tolist())
            if observed_tasks != [task]:
                raise ValueError(
                    f"{path}: ожидали task={task}, но внутри observed_tasks={observed_tasks}"
                )

            core = normalize_core_frame(
                df,
                source_type="sequence",
                source_name=source_name,
                task=task,
                version=version,
                time_rule="inherits_from_sequence_builder",
                path=path,
            )
            frames.append(core)

            manifest_rows.append(
                {
                    "source_type": "sequence",
                    "source_name": source_name,
                    "task": task,
                    "version": version,
                    "time_rule": "inherits_from_sequence_builder",
                    "path": str(path),
                    "n_rows": int(len(df)),
                    "n_columns": int(df.shape[1]),
                    "has_prediction_time": bool("prediction_time" in df.columns),
                    "has_seq_len": bool("seq_len" in df.columns),
                    "has_days_before_prediction": bool("days_before_prediction" in df.columns),
                    "columns_sample": safe_sample_join(df.columns, max_items=30),
                }
            )

    if not frames:
        raise FileNotFoundError(
            f"В {sequence_data_dir} не найдено sequence examples.parquet для tasks={tasks}"
        )

    manifest = pd.DataFrame(manifest_rows)
    return pd.concat(frames, ignore_index=True), manifest


# -----------------------------------------------------------------------------
# 7. Split audit checks
# -----------------------------------------------------------------------------

def make_split_value_audit(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет, какие split values встречаются в каждом источнике.
    """
    out = (
        core.groupby(
            ["source_type", "source_name", "task", "version", "time_rule", "split"],
            dropna=False,
        )
        .agg(n_rows=("row_id", "size"))
        .reset_index()
    )

    out["is_expected_split"] = out["split"].isin(EXPECTED_SPLITS)

    return out.sort_values(
        ["source_type", "task", "source_name", "split"]
    ).reset_index(drop=True)


def make_split_summary(core: pd.DataFrame) -> pd.DataFrame:
    """
    Сводка размера split по каждому источнику.
    """
    out = (
        core.groupby(
            ["source_type", "source_name", "task", "version", "time_rule", "split"],
            dropna=False,
        )
        .agg(
            n_examples=("row_id", "size"),
            n_unique_row_id=("row_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_positive=("label", "sum"),
            event_rate=("label", "mean"),
            n_missing_subject_id=("subject_id", lambda x: int(x.isna().sum())),
            n_missing_label=("label", lambda x: int(x.isna().sum())),
            n_missing_prediction_time=("prediction_time", lambda x: int(x.isna().sum())),
        )
        .reset_index()
    )

    out["n_positive"] = out["n_positive"].astype("Int64")

    return out.sort_values(
        ["source_type", "task", "source_name", "split"]
    ).reset_index(drop=True)


def make_duplicate_example_issues(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет, нет ли дублей prediction example внутри одного source.

    В норме для каждого source_name + task + row_id должна быть одна строка.
    """
    grouped = (
        core.groupby(["source_type", "source_name", "task", "row_id"], dropna=False)
        .agg(
            n_rows=("row_id", "size"),
            n_subjects=("subject_id", "nunique"),
            splits=("split", safe_unique_join),
            labels=("label", safe_unique_join),
            paths=("path", safe_unique_join),
        )
        .reset_index()
    )

    issues = grouped[grouped["n_rows"] > 1].copy()
    return issues.sort_values(["source_type", "task", "source_name", "row_id"]).reset_index(drop=True)


def make_subject_overlap(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет patient overlap между split внутри каждого source.

    Для каждого source_name и task считаем:
        train ∩ tuning
        train ∩ held_out
        tuning ∩ held_out

    Критерий OK:
        n_overlapping_subjects == 0 для всех пар.
    """
    rows: list[dict[str, Any]] = []

    group_cols = ["source_type", "source_name", "task", "version", "time_rule"]

    for key, group in core.groupby(group_cols, dropna=False):
        source_type, source_name, task, version, time_rule = key

        split_values = sorted(group["split"].dropna().astype(str).unique().tolist())

        # Проверяем все реальные пары split, но порядок делаем ожидаемым.
        ordered = [s for s in EXPECTED_SPLITS if s in split_values]
        ordered += [s for s in split_values if s not in ordered]

        subjects_by_split = {
            split: set(
                group.loc[group["split"].astype(str) == split, "subject_id"]
                .dropna()
                .astype(int)
                .tolist()
            )
            for split in ordered
        }

        for split_a, split_b in itertools.combinations(ordered, 2):
            overlap = sorted(subjects_by_split[split_a] & subjects_by_split[split_b])

            rows.append(
                {
                    "source_type": source_type,
                    "source_name": source_name,
                    "task": task,
                    "version": version,
                    "time_rule": time_rule,
                    "split_a": split_a,
                    "split_b": split_b,
                    "split_pair": f"{split_a}__vs__{split_b}",
                    "n_subjects_split_a": len(subjects_by_split[split_a]),
                    "n_subjects_split_b": len(subjects_by_split[split_b]),
                    "n_overlapping_subjects": len(overlap),
                    "overlapping_subject_ids_sample": "|".join(map(str, overlap[:20])),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["source_type", "task", "source_name", "split_pair"]
    ).reset_index(drop=True)


def make_subject_multi_split_issues(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет, есть ли subject_id, который внутри одного source попал в несколько split.
    """
    grouped = (
        core.dropna(subset=["subject_id"])
        .groupby(["source_type", "source_name", "task", "subject_id"], dropna=False)
        .agg(
            n_splits=("split", lambda x: x.dropna().astype(str).nunique()),
            splits=("split", safe_unique_join),
            n_examples=("row_id", "size"),
            row_ids_sample=("row_id", safe_sample_join),
            paths=("path", safe_unique_join),
        )
        .reset_index()
    )

    issues = grouped[grouped["n_splits"] > 1].copy()

    return issues.sort_values(
        ["source_type", "task", "source_name", "subject_id"]
    ).reset_index(drop=True)


def make_row_multi_split_issues(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет, есть ли row_id, который внутри одного source попал в несколько split.

    В норме row_id — это prediction example.
    Один prediction example не должен быть одновременно train и held_out.
    """
    grouped = (
        core.dropna(subset=["row_id"])
        .groupby(["source_type", "source_name", "task", "row_id"], dropna=False)
        .agg(
            n_splits=("split", lambda x: x.dropna().astype(str).nunique()),
            splits=("split", safe_unique_join),
            n_subjects=("subject_id", "nunique"),
            subject_ids=("subject_id", safe_unique_join),
            labels=("label", safe_unique_join),
            paths=("path", safe_unique_join),
        )
        .reset_index()
    )

    issues = grouped[grouped["n_splits"] > 1].copy()

    return issues.sort_values(
        ["source_type", "task", "source_name", "row_id"]
    ).reset_index(drop=True)


def make_official_split_mismatch(
    core: pd.DataFrame,
    official_splits: pd.DataFrame,
) -> pd.DataFrame:
    """
    Проверяет, совпадает ли split в каждом source с official subject_splits.parquet.
    """
    merged = core.merge(
        official_splits,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    merged["has_official_split"] = merged["official_split"].notna()
    merged["split_matches_official"] = (
        merged["split"].astype("string") == merged["official_split"].astype("string")
    )

    issues = merged[
        (~merged["has_official_split"])
        | (~merged["split_matches_official"])
    ].copy()

    keep_cols = [
        "source_type",
        "source_name",
        "task",
        "version",
        "row_id",
        "subject_id",
        "split",
        "official_split",
        "has_official_split",
        "split_matches_official",
        "path",
    ]

    return issues[keep_cols].sort_values(
        ["source_type", "task", "source_name", "row_id"]
    ).reset_index(drop=True)


def make_cross_source_subject_split_issues(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет split consistency для subject_id между всеми источниками.

    Например:
        в tabular subject_id=123 train,
        а в sequence subject_id=123 held_out.

    Такое было бы критичным leakage issue.
    """
    tmp = (
        core.dropna(subset=["subject_id", "split"])
        [["task", "subject_id", "split", "source_type", "source_name", "version", "path"]]
        .drop_duplicates()
        .copy()
    )

    grouped = (
        tmp.groupby(["task", "subject_id"], dropna=False)
        .agg(
            n_splits=("split", lambda x: x.dropna().astype(str).nunique()),
            splits=("split", safe_unique_join),
            n_sources=("source_name", "nunique"),
            sources=("source_name", safe_unique_join),
            source_types=("source_type", safe_unique_join),
            versions=("version", safe_unique_join),
            paths=("path", safe_unique_join),
        )
        .reset_index()
    )

    issues = grouped[grouped["n_splits"] > 1].copy()

    return issues.sort_values(["task", "subject_id"]).reset_index(drop=True)


def make_row_mapping_consistency_issues(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет, что один и тот же task + row_id имеет одинаковый mapping
    во всех источниках:

        row_id -> subject_id
        row_id -> label
        row_id -> split
        row_id -> prediction_time

    Важно:
        prediction_time проверяем только по non-null значениям.
        Если один источник не содержит prediction_time, это не считается
        автоматическим mismatch.
    """
    tmp = core.copy()

    tmp["prediction_time_cmp"] = tmp["prediction_time"].astype("string")
    tmp.loc[tmp["prediction_time_cmp"].isna(), "prediction_time_cmp"] = pd.NA

    dedup = tmp[
        [
            "task",
            "row_id",
            "subject_id",
            "label",
            "split",
            "prediction_time_cmp",
            "source_type",
            "source_name",
            "version",
            "path",
        ]
    ].drop_duplicates()

    grouped = (
        dedup.groupby(["task", "row_id"], dropna=False)
        .agg(
            n_sources=("source_name", "nunique"),
            sources=("source_name", safe_unique_join),
            source_types=("source_type", safe_unique_join),
            versions=("version", safe_unique_join),
            n_subject_ids=("subject_id", "nunique"),
            subject_ids=("subject_id", safe_unique_join),
            n_labels=("label", "nunique"),
            labels=("label", safe_unique_join),
            n_splits=("split", lambda x: x.dropna().astype(str).nunique()),
            splits=("split", safe_unique_join),
            n_prediction_times_non_null=(
                "prediction_time_cmp",
                lambda x: x.dropna().astype(str).nunique(),
            ),
            prediction_times=("prediction_time_cmp", safe_unique_join),
            paths=("path", safe_unique_join),
        )
        .reset_index()
    )

    issues = grouped[
        (grouped["n_subject_ids"] > 1)
        | (grouped["n_labels"] > 1)
        | (grouped["n_splits"] > 1)
        | (grouped["n_prediction_times_non_null"] > 1)
    ].copy()

    return issues.sort_values(["task", "row_id"]).reset_index(drop=True)


def make_coverage_against_labels(core: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет покрытие каждого source относительно label_reference.

    Для каждого source считаем:
        - сколько examples есть в label_reference;
        - сколько examples есть в source;
        - сколько label examples отсутствует в source;
        - сколько source examples отсутствует в label_reference.

    Ключ сравнения:
        task + row_id + subject_id

    prediction_time и label отдельно проверяются в row_mapping_consistency_issues.
    """
    labels = core[core["source_type"] == "label_reference"].copy()

    if labels.empty:
        return pd.DataFrame()

    label_keys = labels[["task", "row_id", "subject_id"]].drop_duplicates()

    rows: list[dict[str, Any]] = []

    for source_name, src in core[core["source_type"] != "label_reference"].groupby("source_name"):
        source_type = str(src["source_type"].iloc[0])
        task = str(src["task"].iloc[0])
        version = str(src["version"].iloc[0])
        time_rule = str(src["time_rule"].iloc[0])
        path = str(src["path"].iloc[0])

        src_keys = src[["task", "row_id", "subject_id"]].drop_duplicates()

        labels_for_task = label_keys[label_keys["task"] == task].copy()

        left = labels_for_task.merge(
            src_keys,
            on=["task", "row_id", "subject_id"],
            how="left",
            indicator=True,
        )
        missing_in_source = left[left["_merge"] == "left_only"]

        right = src_keys.merge(
            labels_for_task,
            on=["task", "row_id", "subject_id"],
            how="left",
            indicator=True,
        )
        extra_in_source = right[right["_merge"] == "left_only"]

        rows.append(
            {
                "source_type": source_type,
                "source_name": source_name,
                "task": task,
                "version": version,
                "time_rule": time_rule,
                "path": path,
                "n_label_examples": int(len(labels_for_task)),
                "n_source_examples": int(len(src_keys)),
                "n_missing_label_examples_in_source": int(len(missing_in_source)),
                "n_extra_source_examples_not_in_labels": int(len(extra_in_source)),
                "missing_row_ids_sample": safe_sample_join(missing_in_source["row_id"]),
                "extra_row_ids_sample": safe_sample_join(extra_in_source["row_id"]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["source_type", "task", "source_name"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 8. Status table и summary
# -----------------------------------------------------------------------------

def build_status_table(
    split_value_audit: pd.DataFrame,
    duplicate_examples: pd.DataFrame,
    subject_overlap: pd.DataFrame,
    subject_multi_split: pd.DataFrame,
    row_multi_split: pd.DataFrame,
    official_mismatch: pd.DataFrame,
    cross_source_subject_issues: pd.DataFrame,
    row_mapping_issues: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Собирает компактный статус по всем split checks.
    """
    rows: list[dict[str, Any]] = []

    def add(check_name: str, status: str, n_issues: int, artifact: str, comment: str) -> None:
        rows.append(
            {
                "check_name": check_name,
                "status": status,
                "n_issues": int(n_issues),
                "artifact": artifact,
                "comment": comment,
            }
        )

    invalid_split_rows = split_value_audit[~split_value_audit["is_expected_split"]]
    add(
        check_name="split_values_are_expected",
        status="OK" if invalid_split_rows.empty else "FAIL",
        n_issues=len(invalid_split_rows),
        artifact="split_value_audit.csv",
        comment="Разрешены только train/tuning/held_out.",
    )

    add(
        check_name="duplicate_prediction_examples_inside_source",
        status="OK" if duplicate_examples.empty else "FAIL",
        n_issues=len(duplicate_examples),
        artifact="duplicate_prediction_example_issues.csv",
        comment="Внутри одного source не должно быть дублей source+task+row_id.",
    )

    n_overlap_subjects = int(subject_overlap["n_overlapping_subjects"].sum()) if not subject_overlap.empty else 0
    add(
        check_name="patient_overlap_between_splits_inside_source",
        status="OK" if n_overlap_subjects == 0 else "FAIL",
        n_issues=n_overlap_subjects,
        artifact="subject_split_overlap.csv",
        comment="Критичная проверка: один patient не должен быть одновременно в train/tuning/held_out.",
    )

    add(
        check_name="subject_single_split_inside_source",
        status="OK" if subject_multi_split.empty else "FAIL",
        n_issues=len(subject_multi_split),
        artifact="subject_multi_split_issues.csv",
        comment="Один subject_id внутри одного source должен иметь ровно один split.",
    )

    add(
        check_name="row_id_single_split_inside_source",
        status="OK" if row_multi_split.empty else "FAIL",
        n_issues=len(row_multi_split),
        artifact="row_multi_split_issues.csv",
        comment="Один prediction example row_id внутри одного source должен иметь ровно один split.",
    )

    add(
        check_name="source_split_matches_official_subject_splits",
        status="OK" if official_mismatch.empty else "FAIL",
        n_issues=len(official_mismatch),
        artifact="official_split_mismatch.csv",
        comment="Split в tabular/sequence/labels должен совпадать с EHRSHOT_MEDS metadata/subject_splits.parquet.",
    )

    add(
        check_name="cross_source_subject_split_consistency",
        status="OK" if cross_source_subject_issues.empty else "FAIL",
        n_issues=len(cross_source_subject_issues),
        artifact="cross_source_subject_split_issues.csv",
        comment="Один subject_id не должен иметь разные split в разных источниках.",
    )

    add(
        check_name="cross_source_row_mapping_consistency",
        status="OK" if row_mapping_issues.empty else "FAIL",
        n_issues=len(row_mapping_issues),
        artifact="row_mapping_consistency_issues.csv",
        comment="task+row_id должен одинаково маппиться в subject_id/label/split/prediction_time.",
    )

    if coverage.empty:
        add(
            check_name="coverage_against_label_reference",
            status="WARN",
            n_issues=1,
            artifact="coverage_against_label_reference.csv",
            comment="Не удалось построить coverage table.",
        )
    else:
        bad_coverage = coverage[
            (coverage["n_missing_label_examples_in_source"] > 0)
            | (coverage["n_extra_source_examples_not_in_labels"] > 0)
        ]
        add(
            check_name="coverage_against_label_reference",
            status="OK" if bad_coverage.empty else "WARN",
            n_issues=len(bad_coverage),
            artifact="coverage_against_label_reference.csv",
            comment="WARN, а не FAIL: иногда source может быть намеренно отфильтрован, но это нужно явно объяснить.",
        )

    return pd.DataFrame(rows)


def write_markdown_summary(
    output_dir: Path,
    status: pd.DataFrame,
    split_summary: pd.DataFrame,
    subject_overlap: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    """
    Пишет короткий one-page markdown summary.
    """
    output_dir = Path(output_dir)

    n_fail = int((status["status"] == "FAIL").sum())
    n_warn = int((status["status"] == "WARN").sum())

    if n_fail > 0:
        overall = "FAIL"
    elif n_warn > 0:
        overall = "WARN"
    else:
        overall = "OK"

    status_md = status.to_markdown(index=False)

    compact_split = (
        split_summary[
            [
                "source_type",
                "source_name",
                "task",
                "version",
                "split",
                "n_examples",
                "n_subjects",
                "n_positive",
                "event_rate",
            ]
        ]
        .sort_values(["source_type", "task", "source_name", "split"])
        .head(80)
    )

    compact_split_md = compact_split.to_markdown(index=False)

    if not subject_overlap.empty:
        max_overlap = int(subject_overlap["n_overlapping_subjects"].max())
        total_overlap = int(subject_overlap["n_overlapping_subjects"].sum())
    else:
        max_overlap = 0
        total_overlap = 0

    if coverage.empty:
        coverage_md = "_Coverage table is empty._"
    else:
        coverage_md = coverage[
            [
                "source_type",
                "source_name",
                "task",
                "version",
                "n_label_examples",
                "n_source_examples",
                "n_missing_label_examples_in_source",
                "n_extra_source_examples_not_in_labels",
            ]
        ].to_markdown(index=False)

    text = f"""# Split / Leakage Audit: patient split checks

            ## Overall status

            **Overall:** `{overall}`

            - Number of failed checks: `{n_fail}`
            - Number of warning checks: `{n_warn}`
            - Total overlapping subjects across split pairs inside sources: `{total_overlap}`
            - Max overlapping subjects in one source/split pair: `{max_overlap}`

            ## Status table

            {status_md}

            ## Split summary preview

            {compact_split_md}

            ## Coverage against label reference

            {coverage_md}

            ## Interpretation rule

            For split audit, the critical condition is:

            ```text
            For each task and each data source:
            subject_id must belong to exactly one of train / tuning / held_out.
            For final reporting:

            OK means the check passed directly from saved datasets.
            WARN means the situation may be acceptable, but needs a written explanation.
            FAIL means there is a likely split/leakage issue that should be fixed before using metrics.
            """
    path = output_dir / "split_audit_summary.md"
    path.write_text(text, encoding="utf-8")

# -----------------------------------------------------------------------------
# 9. Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    tasks = parse_csv_list(args.tasks)

    print("=" * 100)
    print("SPLIT / LEAKAGE AUDIT: PATIENT SPLIT CHECKS")
    print(f"ehrshot_root       = {args.ehrshot_root}")
    print(f"tabular_cache_dir  = {args.tabular_cache_dir}")
    print(f"sequence_data_dir  = {args.sequence_data_dir}")
    print(f"output_dir         = {output_dir}")
    print(f"tasks              = {tasks}")
    print("=" * 100)

    # 1. Official split.
    official_splits = load_official_subject_splits(args.ehrshot_root)
    write_csv(official_splits, output_dir / "official_subject_splits_used.csv")

    # 2. Label reference.
    label_reference = load_label_reference(
        ehrshot_root=args.ehrshot_root,
        tasks=tasks,
        official_splits=official_splits,
    )

    # 3. Tabular sources.
    tabular_core, tabular_manifest = load_tabular_sources(
        tabular_cache_dir=args.tabular_cache_dir,
        tasks=tasks,
    )

    # 4. Sequence sources.
    sequence_core, sequence_manifest = load_sequence_sources(
        sequence_data_dir=args.sequence_data_dir,
        tasks=tasks,
    )

    # 5. Общий core dataframe для всех проверок.
    core = pd.concat(
        [label_reference, tabular_core, sequence_core],
        ignore_index=True,
    )

    manifest = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "source_type": "label_reference",
                        "source_name": f"label_reference__{task}",
                        "task": task,
                        "version": "labels",
                        "time_rule": "not_applicable",
                        "path": str(Path(args.ehrshot_root) / "labels" / task / "labels.parquet"),
                        "n_rows": int((label_reference["task"] == task).sum()),
                        "n_columns": None,
                        "has_prediction_time": True,
                        "columns_sample": "row_id|subject_id|prediction_time|label|split",
                    }
                    for task in tasks
                ]
            ),
            tabular_manifest,
            sequence_manifest,
        ],
        ignore_index=True,
    )

    # 6. Сохраняем базовые таблицы.
    write_csv(core, output_dir / "all_split_audit_core_rows.csv")
    write_csv(manifest, output_dir / "dataset_file_manifest.csv")

    # 7. Запускаем проверки.
    split_value_audit = make_split_value_audit(core)
    split_summary = make_split_summary(core)
    duplicate_examples = make_duplicate_example_issues(core)
    subject_overlap = make_subject_overlap(core)
    subject_multi_split = make_subject_multi_split_issues(core)
    row_multi_split = make_row_multi_split_issues(core)
    official_mismatch = make_official_split_mismatch(core, official_splits)
    cross_source_subject_issues = make_cross_source_subject_split_issues(core)
    row_mapping_issues = make_row_mapping_consistency_issues(core)
    coverage = make_coverage_against_labels(core)

    # 8. Сохраняем результаты проверок.
    write_csv(split_value_audit, output_dir / "split_value_audit.csv")
    write_csv(split_summary, output_dir / "split_summary.csv")
    write_csv(duplicate_examples, output_dir / "duplicate_prediction_example_issues.csv")
    write_csv(subject_overlap, output_dir / "subject_split_overlap.csv")
    write_csv(subject_multi_split, output_dir / "subject_multi_split_issues.csv")
    write_csv(row_multi_split, output_dir / "row_multi_split_issues.csv")
    write_csv(official_mismatch, output_dir / "official_split_mismatch.csv")
    write_csv(cross_source_subject_issues, output_dir / "cross_source_subject_split_issues.csv")
    write_csv(row_mapping_issues, output_dir / "row_mapping_consistency_issues.csv")
    write_csv(coverage, output_dir / "coverage_against_label_reference.csv")

    # 9. Status + markdown.
    status = build_status_table(
        split_value_audit=split_value_audit,
        duplicate_examples=duplicate_examples,
        subject_overlap=subject_overlap,
        subject_multi_split=subject_multi_split,
        row_multi_split=row_multi_split,
        official_mismatch=official_mismatch,
        cross_source_subject_issues=cross_source_subject_issues,
        row_mapping_issues=row_mapping_issues,
        coverage=coverage,
    )

    write_csv(status, output_dir / "split_audit_status.csv")

    write_markdown_summary(
        output_dir=output_dir,
        status=status,
        split_summary=split_summary,
        subject_overlap=subject_overlap,
        coverage=coverage,
    )

    # 10. Машиночитаемый JSON summary.
    summary_json = {
        "output_dir": str(output_dir),
        "tasks": tasks,
        "n_sources": int(core["source_name"].nunique()),
        "n_rows_total": int(len(core)),
        "status_counts": status["status"].value_counts().to_dict(),
        "n_failed_checks": int((status["status"] == "FAIL").sum()),
        "n_warning_checks": int((status["status"] == "WARN").sum()),
        "artifacts": sorted([p.name for p in output_dir.iterdir() if p.is_file()]),
    }

    (output_dir / "split_audit_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved audit artifacts:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")

    print("\nStatus:")
    print(status.to_string(index=False))

    n_fail = int((status["status"] == "FAIL").sum())

    if args.fail_on_issues and n_fail > 0:
        raise SystemExit(
            f"Split audit failed: {n_fail} failed checks. "
            f"See {output_dir / 'split_audit_status.csv'}"
        )

if __name__ == "__main__":
    main()