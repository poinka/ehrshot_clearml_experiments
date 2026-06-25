from __future__ import annotations

"""
Build EHRSHOT sequence compression variants from cached sequence datasets.

This file:
- does NOT require raw EHRSHOT_MEDS;
- downloads cached raw sequence examples from MinIO/S3 via ClearML StorageManager;
- optionally downloads chronic-code audit files from MinIO/S3;
- builds new sequence versions under <output-dir>/<task>/<version>/;
- uploads generated datasets back to MinIO/S3;
- supports ClearML remote execution with --enable-clearml --execute-remotely.

Expected cached input layout:
    <sequence-data-s3-prefix>/<task>/raw/examples.parquet
    <sequence-data-s3-prefix>/<task>/raw/vocab.json

Output layout:
    <output-s3-prefix>/<task>/<version>/examples.parquet
    <output-s3-prefix>/<task>/<version>/vocab.json
    <output-s3-prefix>/<task>/<version>/metadata.json
    <output-s3-prefix>/<task>/<version>/compression_audit.csv
"""

import argparse
import json
import math
import os
import shutil
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEFAULT_TASKS = "guo_readmission,guo_icu"
DEFAULT_VERSIONS = "raw,compressed_dedup,compressed_first_last,condition_era_90,condition_era_180,state_duration_90,state_duration_180"

# These defaults match the bucket/path shown in the current MinIO screenshots:
# bucket: pershin-medailab
# path:   pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_multiseed_inputs/...
DEFAULT_S3_ROOT = "s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
DEFAULT_SEQUENCE_S3_PREFIX = f"{DEFAULT_S3_ROOT}/ehrshot_multiseed_inputs/ehrshot_sequence_datasets"
DEFAULT_AUDIT_S3_PREFIX = f"{DEFAULT_S3_ROOT}/ehrshot_multiseed_inputs/ehrshot_copy_forwarding_audit"
DEFAULT_OUTPUT_S3_PREFIX = f"{DEFAULT_S3_ROOT}/ehrshot_multiseed_inputs/ehrshot_sequence_datasets_compression_v2"

SOURCE_FILES = ("examples.parquet", "vocab.json")
AUDIT_CODE_FILES = (
    "strong_empirical_chronic_like_diagnosis_codes.csv",
    "possible_empirical_chronic_like_diagnosis_codes.csv",
)

CHRONIC_DIAGNOSIS_WHITELIST: set[str] | None = None
CHRONIC_DIAGNOSIS_WHITELIST_UPPER: set[str] | None = None

def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_list_cell(x: Any) -> list[Any]:
    """Robust conversion for list-like parquet cells."""
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
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                import ast

                y = ast.literal_eval(s)
                if isinstance(y, (list, tuple)):
                    return list(y)
                return [y]
            except Exception:
                return [x]
        return [x]
    try:
        return list(x)
    except Exception:
        return []


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def as_nonnegative_day(x: Any) -> float:
    v = safe_float(x, 0.0)
    return max(float(v), 0.0)


def normalize_lengths(
    codes: list[Any],
    days: list[Any],
    deltas: list[Any],
    nums: list[Any],
) -> tuple[list[str], list[float], list[float], list[float]]:
    codes = [str(c) for c in codes]
    n = len(codes)
    days_out = [as_nonnegative_day(v) for v in days[:n]] + [0.0] * max(0, n - len(days))
    deltas_out = [as_nonnegative_day(v) for v in deltas[:n]] + [0.0] * max(0, n - len(deltas))
    nums_out = [safe_float(v) for v in nums[:n]] + [float("nan")] * max(0, n - len(nums))
    return codes, days_out, deltas_out, nums_out


def recompute_delta_days(days_before_prediction: list[float]) -> list[float]:
    """
    Cached sequences are ordered old -> recent. days_before_prediction usually decreases.
    Recompute a non-negative time gap between consecutive retained events.
    """
    if not days_before_prediction:
        return []
    out = [0.0]
    prev = float(days_before_prediction[0])
    for cur in days_before_prediction[1:]:
        cur = float(cur)
        out.append(max(prev - cur, 0.0))
        prev = cur
    return out


def read_vocab(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"vocab.json must contain a dict, got {type(obj)} at {path}")

    # Normal expected format: code -> id.
    # Fallback supported: id -> code.
    keys_are_ints = True
    for k in obj.keys():
        try:
            int(k)
        except Exception:
            keys_are_ints = False
            break

    if keys_are_ints:
        return {str(code): int(idx) for idx, code in obj.items()}
    return {str(code): int(idx) for code, idx in obj.items()}


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in vocab.items()}


def ensure_codes_column(df: pd.DataFrame, vocab: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    if "codes" in df.columns:
        df["codes"] = df["codes"].apply(lambda xs: [str(x) for x in parse_list_cell(xs)])
        return df
    if "token_ids" not in df.columns:
        raise ValueError("examples.parquet must contain either `codes` or `token_ids`.")
    inv = invert_vocab(vocab)
    df["codes"] = df["token_ids"].apply(lambda xs: [inv.get(int(t), UNK_TOKEN) for t in parse_list_cell(xs)])
    return df


def set_chronic_diagnosis_whitelist(codes: set[str] | None) -> None:
    """
    Register chronic diagnosis whitelist globally.

    If whitelist is loaded, is_diagnosis_like_code() will use it instead of
    broad string heuristics. This is safer because compression should affect
    only empirically selected chronic-like diagnosis codes.
    """
    global CHRONIC_DIAGNOSIS_WHITELIST
    global CHRONIC_DIAGNOSIS_WHITELIST_UPPER

    if codes is None:
        CHRONIC_DIAGNOSIS_WHITELIST = None
        CHRONIC_DIAGNOSIS_WHITELIST_UPPER = None
        return

    CHRONIC_DIAGNOSIS_WHITELIST = {str(c).strip() for c in codes if pd.notna(c) and str(c).strip()}
    CHRONIC_DIAGNOSIS_WHITELIST_UPPER = {c.upper() for c in CHRONIC_DIAGNOSIS_WHITELIST}

    print(f"Loaded chronic diagnosis whitelist: {len(CHRONIC_DIAGNOSIS_WHITELIST)} codes")


def is_diagnosis_like_code(code: str) -> bool:
    """
    Return True only for codes eligible for chronic-diagnosis compression.

    Priority:
    1. If strong_empirical_chronic_like_diagnosis_codes.csv was loaded,
       use it as a strict whitelist.
    2. If whitelist was not loaded, fall back to conservative cached-code heuristic.
    """
    raw = str(code).strip()
    c = raw.upper()

    if raw in {PAD_TOKEN, UNK_TOKEN} or c in {PAD_TOKEN.upper(), UNK_TOKEN.upper()}:
        return False

    # Main path: strict whitelist from strong_empirical_chronic_like_diagnosis_codes.csv
    if CHRONIC_DIAGNOSIS_WHITELIST is not None:
        return raw in CHRONIC_DIAGNOSIS_WHITELIST or c in CHRONIC_DIAGNOSIS_WHITELIST_UPPER

    # Fallback only if whitelist was not found.
    exclude = (
        "LOINC",
        "RXNORM",
        "CPT4",
        "VISIT",
        "CARE_SITE",
        "DRUG",
        "MEDICATION",
        "PROCEDURE",
        "DEVICE",
        "LAB",
        "MEASUREMENT",
        "OBSERVATION_PERIOD",
        "PERSON/",
        "STATIC",
        "GENDER",
        "RACE",
        "ETHNICITY",
    )
    if any(x in c for x in exclude):
        return False

    include = ("SNOMED", "CONDITION", "DIAG", "DIAGNOS", "ICD", "DX/")
    return any(x in c for x in include)


def duration_bin(days: float) -> str:
    d = float(days)
    if d <= 0:
        return "0"
    if d <= 30:
        return "1_30"
    if d <= 90:
        return "31_90"
    if d <= 180:
        return "91_180"
    if d <= 365:
        return "181_365"
    if d <= 730:
        return "366_730"
    return "gt730"


def count_bin(n: int) -> str:
    n = int(n)
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3_5"
    if n <= 10:
        return "6_10"
    return "gt10"


def recency_bin(days_before_prediction: float) -> str:
    d = float(days_before_prediction)
    if d <= 7:
        return "0_7"
    if d <= 30:
        return "8_30"
    if d <= 90:
        return "31_90"
    if d <= 180:
        return "91_180"
    if d <= 365:
        return "181_365"
    return "gt365"


def make_event(idx: float, code: str, day: float, num: float = float("nan"), synthetic: bool = False) -> dict[str, Any]:
    return {
        "idx": float(idx),
        "code": str(code),
        "day": float(day),
        "num": safe_float(num),
        "synthetic": bool(synthetic),
    }


def dedup_same_day(events: list[dict[str, Any]], compressible: set[str]) -> list[dict[str, Any]]:
    """Approximation of visit/day dedup from cached sequences: same code + same rounded day before prediction."""
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for e in events:
        code = e["code"]
        if code in compressible:
            key = (code, int(round(float(e["day"]))))
            if key in seen:
                continue
            seen.add(key)
        out.append(e)
    return out


def compress_first_last(events: list[dict[str, Any]], compressible: set[str]) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out: list[dict[str, Any]] = []

    for e in events:
        if e["code"] in compressible:
            by_code[e["code"]].append(e)
        else:
            out.append(e)

    for code, es in by_code.items():
        es = sorted(es, key=lambda x: x["idx"])
        if len(es) <= 1:
            out.extend(es)
            continue
        first, last = es[0], es[-1]
        span = max(float(first["day"]) - float(last["day"]), 0.0)
        out.append(make_event(first["idx"], f"DX_FIRST/{code}", first["day"], 0.0, True))
        out.append(make_event(last["idx"], f"DX_LAST/{code}", last["day"], span, True))

    return sorted(out, key=lambda x: x["idx"])


def split_into_eras(es: list[dict[str, Any]], max_gap_days: int) -> list[list[dict[str, Any]]]:
    es = sorted(es, key=lambda x: x["idx"])
    if not es:
        return []
    eras = [[es[0]]]
    for e in es[1:]:
        prev = eras[-1][-1]
        # old -> recent: day_before decreases, so gap = prev_day - current_day
        gap = max(float(prev["day"]) - float(e["day"]), 0.0)
        if gap <= max_gap_days:
            eras[-1].append(e)
        else:
            eras.append([e])
    return eras


def compress_condition_era(events: list[dict[str, Any]], compressible: set[str], max_gap_days: int) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out: list[dict[str, Any]] = []

    for e in events:
        if e["code"] in compressible:
            by_code[e["code"]].append(e)
        else:
            out.append(e)

    for code, es in by_code.items():
        for era in split_into_eras(es, max_gap_days):
            if len(era) <= 1:
                out.extend(era)
                continue
            first, last = era[0], era[-1]
            span = max(float(first["day"]) - float(last["day"]), 0.0)
            out.append(make_event(first["idx"], f"DX_ERA{max_gap_days}_START/{code}", first["day"], 0.0, True))
            out.append(make_event(last["idx"], f"DX_ERA{max_gap_days}_END/{code}", last["day"], span, True))

    return sorted(out, key=lambda x: x["idx"])


def compress_state_duration(events: list[dict[str, Any]], compressible: set[str], max_gap_days: int) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out: list[dict[str, Any]] = []

    for e in events:
        if e["code"] in compressible:
            by_code[e["code"]].append(e)
        else:
            out.append(e)

    eps = 1e-4
    for code, es in by_code.items():
        for era in split_into_eras(es, max_gap_days):
            if len(era) <= 1:
                out.extend(era)
                continue
            first, last = era[0], era[-1]
            span = max(float(first["day"]) - float(last["day"]), 0.0)
            cnt = len(era)
            recency = max(float(last["day"]), 0.0)
            prefix = f"DX_STATE{max_gap_days}"
            out.append(make_event(first["idx"], f"{prefix}_ONSET/{code}", first["day"], 0.0, True))
            # Put state tokens at the last known mention. Small index offsets keep deterministic order.
            out.append(make_event(last["idx"] + eps * 1, f"{prefix}_ACTIVE/{code}", last["day"], float(cnt), True))
            out.append(make_event(last["idx"] + eps * 2, f"{prefix}_DURATION_BIN/{duration_bin(span)}/{code}", last["day"], span, True))
            out.append(make_event(last["idx"] + eps * 3, f"{prefix}_COUNT_BIN/{count_bin(cnt)}/{code}", last["day"], float(cnt), True))
            out.append(make_event(last["idx"] + eps * 4, f"{prefix}_RECENCY_BIN/{recency_bin(recency)}/{code}", last["day"], recency, True))

    return sorted(out, key=lambda x: x["idx"])


def apply_compression_to_row(row: pd.Series, version: str, compressible: set[str]) -> dict[str, Any]:
    codes, days, deltas, nums = normalize_lengths(
        parse_list_cell(row.get("codes", [])),
        parse_list_cell(row.get("days_before_prediction", [])),
        parse_list_cell(row.get("delta_days", [])),
        parse_list_cell(row.get("numeric_values", [])),
    )

    raw_events = [make_event(i, c, d, n, False) for i, (c, d, n) in enumerate(zip(codes, days, nums))]

    if version == "raw":
        out_events = raw_events
    elif version == "compressed_dedup":
        out_events = dedup_same_day(raw_events, compressible)
    elif version == "compressed_first_last":
        out_events = compress_first_last(dedup_same_day(raw_events, compressible), compressible)
    elif version.startswith("condition_era_"):
        gap = int(version.rsplit("_", 1)[-1])
        out_events = compress_condition_era(dedup_same_day(raw_events, compressible), compressible, gap)
    elif version.startswith("state_duration_"):
        gap = int(version.rsplit("_", 1)[-1])
        out_events = compress_state_duration(dedup_same_day(raw_events, compressible), compressible, gap)
    else:
        raise ValueError(f"Unknown compression version: {version}")

    out_codes = [e["code"] for e in out_events]
    out_days = [float(e["day"]) for e in out_events]
    out_nums = [safe_float(e["num"]) for e in out_events]
    out_delta = recompute_delta_days(out_days)

    return {
        "codes": out_codes,
        "days_before_prediction": out_days,
        "delta_days": out_delta,
        "numeric_values": out_nums,
        "seq_len": int(len(out_codes)),
        "n_unique_tokens": int(len(set(out_codes))),
        "raw_seq_len": int(len(codes)),
        "n_events_removed_vs_raw": int(len(codes) - len(out_codes)),
        "n_synthetic_events": int(sum(bool(e["synthetic"]) for e in out_events)),
        "n_diagnosis_like_events_raw": int(sum(is_diagnosis_like_code(c) for c in codes)),
        "n_compressible_chronic_events_raw": int(sum(c in compressible for c in codes)),
    }


def build_vocab_preserving_source(out_df: pd.DataFrame, source_vocab: dict[str, int]) -> dict[str, int]:
    """
    Preserve original token ids for all existing codes and append only new synthetic tokens.
    This keeps raw/compressed model inputs comparable and avoids accidental remapping.
    """
    vocab: dict[str, int] = {}
    used_ids: set[int] = set()

    for code, idx in sorted(source_vocab.items(), key=lambda kv: int(kv[1])):
        idx = int(idx)
        code = str(code)
        if code not in vocab and idx not in used_ids:
            vocab[code] = idx
            used_ids.add(idx)

    # Safety: ensure PAD/UNK exist at expected ids unless already used by source vocab.
    if PAD_TOKEN not in vocab and PAD_ID not in used_ids:
        vocab[PAD_TOKEN] = PAD_ID
        used_ids.add(PAD_ID)
    if UNK_TOKEN not in vocab and UNK_ID not in used_ids:
        vocab[UNK_TOKEN] = UNK_ID
        used_ids.add(UNK_ID)

    next_id = max(used_ids) + 1 if used_ids else 0
    train_df = out_df[out_df["split"] == "train"].copy() if "split" in out_df.columns else out_df.copy()
    counts: dict[str, int] = defaultdict(int)
    for codes in train_df["codes"]:
        for c in parse_list_cell(codes):
            counts[str(c)] += 1

    for code, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if code not in vocab:
            vocab[code] = next_id
            next_id += 1

    return vocab


def add_token_ids(df: pd.DataFrame, vocab: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["token_ids"] = df["codes"].apply(lambda cs: [int(vocab.get(str(c), UNK_ID)) for c in parse_list_cell(cs)])
    return df


def load_cached_source(input_dir: Path, task: str, source_version: str) -> tuple[pd.DataFrame, dict[str, int]]:
    base = input_dir / task / source_version
    examples_path = base / "examples.parquet"
    vocab_path = base / "vocab.json"

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing cached source examples: {examples_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing cached source vocab: {vocab_path}")

    print(f"Reading {examples_path}")
    df = pd.read_parquet(examples_path)
    vocab = read_vocab(vocab_path)
    df = ensure_codes_column(df, vocab)

    if "task" not in df.columns:
        df["task"] = task
    if "numeric_values" not in df.columns:
        df["numeric_values"] = df["codes"].apply(lambda xs: [float("nan")] * len(parse_list_cell(xs)))
    if "delta_days" not in df.columns:
        df["delta_days"] = df["days_before_prediction"].apply(lambda xs: recompute_delta_days([as_nonnegative_day(v) for v in parse_list_cell(xs)]))

    required = {"row_id", "subject_id", "label", "split", "codes", "days_before_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cached examples for task={task} are missing columns: {sorted(missing)}")

    return df, vocab


def infer_persistent_codes(
    df: pd.DataFrame,
    min_subjects: int,
    min_mean_occurrences: float,
    min_span_days: float,
    compress_all_diagnosis_like: bool,
) -> tuple[set[str], pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    per_code_subject: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "min_day": np.inf, "max_day": -np.inf}))

    for _, row in train_df.iterrows():
        subject_id = int(row.get("subject_id", row.get("row_id", 0)))
        codes, days, _, _ = normalize_lengths(
            parse_list_cell(row.get("codes", [])),
            parse_list_cell(row.get("days_before_prediction", [])),
            parse_list_cell(row.get("delta_days", [])),
            parse_list_cell(row.get("numeric_values", [])),
        )
        for code, day in zip(codes, days):
            if not is_diagnosis_like_code(code):
                continue
            s = per_code_subject[code][subject_id]
            s["n"] += 1
            s["min_day"] = min(s["min_day"], float(day))
            s["max_day"] = max(s["max_day"], float(day))

    rows: list[dict[str, Any]] = []
    selected: set[str] = set()

    for code, subject_stats in per_code_subject.items():
        occs = np.asarray([v["n"] for v in subject_stats.values()], dtype=float)
        spans = np.asarray([max(v["max_day"] - v["min_day"], 0.0) for v in subject_stats.values()], dtype=float)
        n_subjects = len(subject_stats)
        mean_occ = float(np.mean(occs)) if len(occs) else 0.0
        median_occ = float(np.median(occs)) if len(occs) else 0.0
        median_span = float(np.median(spans)) if len(spans) else 0.0
        keep = bool(
            compress_all_diagnosis_like
            or (
                n_subjects >= min_subjects
                and mean_occ >= min_mean_occurrences
                and median_span >= min_span_days
            )
        )
        rows.append(
            {
                "code": code,
                "n_subjects": int(n_subjects),
                "n_repeated_subjects": int((occs >= 2).sum()) if len(occs) else 0,
                "mean_occurrences_per_subject": mean_occ,
                "median_occurrences_per_subject": median_occ,
                "median_span_days": median_span,
                "p75_span_days": float(np.quantile(spans, 0.75)) if len(spans) else 0.0,
                "selected_for_compression": keep,
                "selection_source": "inferred_from_cached_train_examples",
            }
        )
        if keep:
            selected.add(code)

    stats = pd.DataFrame(rows)
    if len(stats):
        stats = stats.sort_values(
            ["selected_for_compression", "n_subjects", "mean_occurrences_per_subject", "median_span_days"],
            ascending=[False, False, False, False],
        )
    return selected, stats


def get_code_column(df: pd.DataFrame, path: Path) -> str:
    """
    Find code column in whitelist CSV.
    Prefer `code`, but tolerate common alternative names.
    """
    candidates = [
        "code",
        "concept_code",
        "diagnosis_code",
        "condition_code",
        "medical_code",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"{path} must contain one of these columns: {candidates}. "
        f"Actual columns: {list(df.columns)}"
    )


def load_chronic_codes_from_audit(audit_dir: Path, mode: str) -> tuple[set[str] | None, pd.DataFrame | None]:
    strong_path = audit_dir / "strong_empirical_chronic_like_diagnosis_codes.csv"
    possible_path = audit_dir / "possible_empirical_chronic_like_diagnosis_codes.csv"

    if not strong_path.exists():
        set_chronic_diagnosis_whitelist(None)
        return None, None

    strong = pd.read_csv(strong_path)
    strong_code_col = get_code_column(strong, strong_path)

    strong = strong.copy()
    strong["code"] = strong[strong_code_col].astype(str).str.strip()
    strong = strong[strong["code"].notna() & (strong["code"] != "")]
    strong["selection_source"] = "audit_strong_empirical_chronic_like"

    frames = [strong]
    codes = set(strong["code"].dropna().astype(str))

    if mode == "strong_plus_possible":
        if possible_path.exists():
            possible = pd.read_csv(possible_path)
            possible_code_col = get_code_column(possible, possible_path)

            possible = possible.copy()
            possible["code"] = possible[possible_code_col].astype(str).str.strip()
            possible = possible[possible["code"].notna() & (possible["code"] != "")]
            possible["selection_source"] = "audit_possible_empirical_chronic_like"

            frames.append(possible)
            codes |= set(possible["code"].dropna().astype(str))
        else:
            print(f"WARNING: {possible_path} not found; using strong codes only.")
    elif mode != "strong_only":
        raise ValueError("--chronic-code-mode must be strong_only or strong_plus_possible")

    set_chronic_diagnosis_whitelist(codes)

    stats = pd.concat(frames, ignore_index=True, sort=False)
    stats["selected_for_compression"] = stats["code"].astype(str).isin(codes)

    return codes, stats

def summarize_examples(df: pd.DataFrame, task: str, version: str, n_chronic_codes: int) -> dict[str, Any]:
    lengths = df["seq_len"].astype(float).to_numpy() if "seq_len" in df.columns else np.asarray([len(parse_list_cell(x)) for x in df["codes"]], dtype=float)
    y = df["label"].astype(int).to_numpy() if "label" in df.columns else np.array([], dtype=int)

    summary: dict[str, Any] = {
        "task": task,
        "version": version,
        "n_examples": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "n_positive": int(y.sum()) if len(y) else None,
        "event_rate": float(y.mean()) if len(y) else None,
        "mean_seq_len": float(np.mean(lengths)) if len(lengths) else None,
        "median_seq_len": float(np.median(lengths)) if len(lengths) else None,
        "p90_seq_len": float(np.quantile(lengths, 0.90)) if len(lengths) else None,
        "p95_seq_len": float(np.quantile(lengths, 0.95)) if len(lengths) else None,
        "max_seq_len": int(np.max(lengths)) if len(lengths) else None,
        "mean_events_removed_vs_raw": float(df["n_events_removed_vs_raw"].mean()) if "n_events_removed_vs_raw" in df.columns else None,
        "median_events_removed_vs_raw": float(df["n_events_removed_vs_raw"].median()) if "n_events_removed_vs_raw" in df.columns else None,
        "p90_events_removed_vs_raw": float(df["n_events_removed_vs_raw"].quantile(0.90)) if "n_events_removed_vs_raw" in df.columns else None,
        "mean_synthetic_events": float(df["n_synthetic_events"].mean()) if "n_synthetic_events" in df.columns else None,
        "n_chronic_codes_selected": int(n_chronic_codes),
    }

    if "split" in df.columns:
        for split, g in df.groupby("split"):
            gl = g["seq_len"].astype(float).to_numpy()
            summary[f"{split}_n_examples"] = int(len(g))
            summary[f"{split}_mean_seq_len"] = float(np.mean(gl)) if len(gl) else None
            summary[f"{split}_median_seq_len"] = float(np.median(gl)) if len(gl) else None
            summary[f"{split}_p90_seq_len"] = float(np.quantile(gl, 0.90)) if len(gl) else None
            if "n_events_removed_vs_raw" in g.columns:
                summary[f"{split}_mean_events_removed_vs_raw"] = float(g["n_events_removed_vs_raw"].mean())
                summary[f"{split}_p90_events_removed_vs_raw"] = float(g["n_events_removed_vs_raw"].quantile(0.90))
    return summary


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_download_file(remote_url: str, dst: Path, required: bool = True) -> bool:
    if dst.exists():
        return True
    from clearml import StorageManager

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {remote_url}")
        local_copy = Path(StorageManager.get_local_copy(remote_url=remote_url))
        if not local_copy.exists():
            raise FileNotFoundError(f"StorageManager returned missing local copy: {local_copy}")
        if local_copy.is_dir():
            # Not expected for files, but keep this robust.
            matches = list(local_copy.rglob(dst.name))
            if not matches:
                raise FileNotFoundError(f"Could not find {dst.name} inside {local_copy}")
            local_copy = matches[0]
        shutil.copy2(local_copy, dst)
        return True
    except Exception as e:
        if required:
            raise
        print(f"WARNING: optional download failed for {remote_url}: {repr(e)}")
        return False


def maybe_download_sequence_from_s3(input_dir: Path, s3_prefix: str, tasks: list[str], source_version: str) -> Path:
    expected = [input_dir / task / source_version / fname for task in tasks for fname in SOURCE_FILES]
    if all(p.exists() for p in expected):
        print(f"Using local cached sequence datasets: {input_dir}")
        return input_dir

    if not s3_prefix:
        missing = [str(p) for p in expected if not p.exists()]
        raise FileNotFoundError("Missing cached sequence files and no --sequence-data-s3-prefix was provided. " + str(missing[:20]))

    prefix = s3_prefix.rstrip("/")
    for task in tasks:
        for fname in SOURCE_FILES:
            dst = input_dir / task / source_version / fname
            remote_url = f"{prefix}/{task}/{source_version}/{fname}"
            maybe_download_file(remote_url, dst, required=True)
    return input_dir


def maybe_download_audit_from_s3(audit_dir: Path, audit_s3_prefix: str) -> Path:
    if all((audit_dir / name).exists() for name in AUDIT_CODE_FILES):
        print(f"Using local audit dir: {audit_dir}")
        return audit_dir
    if not audit_s3_prefix:
        print("No --audit-s3-prefix provided; chronic codes will be inferred from cached train examples.")
        return audit_dir

    prefix = audit_s3_prefix.rstrip("/")
    for fname in AUDIT_CODE_FILES:
        maybe_download_file(f"{prefix}/{fname}", audit_dir / fname, required=False)
    return audit_dir


def upload_directory_to_s3(local_dir: Path, output_s3_prefix: str) -> None:
    if not output_s3_prefix:
        print("No --output-s3-prefix provided; generated datasets remain local only.")
        return

    from clearml import StorageManager

    prefix = output_s3_prefix.rstrip("/")
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    print(f"Uploading {len(files)} files to {prefix}")
    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        remote_url = f"{prefix}/{rel}"
        print(f"Uploading {p} -> {remote_url}")
        StorageManager.upload_file(local_file=str(p), remote_url=remote_url, wait_for_upload=True)


def zip_dir(src_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir.parent))
    return zip_path


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def build_clearml_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    for key in ["sequence_data_dir", "audit_dir", "output_dir"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = str(cfg[key])
    return cfg


def sync_args_from_clearml_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path_keys = {"sequence_data_dir", "audit_dir", "output_dir"}
    int_keys = {"min_subjects", "min_span_days", "max_examples_per_task"}
    float_keys = {"min_mean_occurrences"}
    bool_keys = {"compress_all_diagnosis_like", "skip_s3_upload"}
    skip_keys = {"enable_clearml", "execute_remotely"}

    for key, value in cfg.items():
        if key in skip_keys or not hasattr(args, key):
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


def maybe_init_clearml(args: argparse.Namespace, cfg: dict[str, Any]):
    remote_agent_run = is_clearml_agent_run()
    if not args.enable_clearml and not remote_agent_run:
        return None, cfg

    from clearml import Task

    if not remote_agent_run:
        # Same policy as existing project scripts: use checked-in requirements.txt, do not freeze local env.
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

    connected_cfg = dict(task.connect(cfg))
    sync_args_from_clearml_config(args, connected_cfg)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  sequence_data_s3_prefix = {args.sequence_data_s3_prefix}")
    print(f"  audit_dir = {args.audit_dir}")
    print(f"  audit_s3_prefix = {args.audit_s3_prefix}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  output_s3_prefix = {args.output_s3_prefix}")
    print(f"  tasks = {args.tasks}")
    print(f"  versions = {args.versions}")
    print(f"  chronic_code_mode = {args.chronic_code_mode}")
    print(f"  clearml_queue = {args.clearml_queue}")

    if args.execute_remotely and not remote_agent_run:
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task, connected_cfg


def choose_compressible_codes(args: argparse.Namespace, source_df: pd.DataFrame, audit_dir: Path) -> tuple[set[str], pd.DataFrame]:
    audit_codes, audit_stats = load_chronic_codes_from_audit(audit_dir, args.chronic_code_mode)
    if audit_codes is not None:
        # Keep only codes actually present in cached examples to avoid confusing summaries.
        present_codes: set[str] = set()
        for codes in source_df["codes"]:
            present_codes.update(str(c) for c in parse_list_cell(codes))
        selected = {c for c in audit_codes if c in present_codes}
        if audit_stats is not None:
            audit_stats = audit_stats.copy()
            audit_stats["present_in_cached_examples"] = audit_stats["code"].astype(str).isin(present_codes)
            audit_stats["selected_for_compression"] = audit_stats["code"].astype(str).isin(selected)
        print(f"Using chronic-code whitelist from audit: selected={len(selected)}, audit_total={len(audit_codes)}")
        return selected, audit_stats if audit_stats is not None else pd.DataFrame({"code": sorted(selected), "selected_for_compression": True})

    print("WARNING: audit chronic-code whitelist not found. Falling back to inference from cached train examples.")
    selected, stats = infer_persistent_codes(
        source_df,
        min_subjects=args.min_subjects,
        min_mean_occurrences=args.min_mean_occurrences,
        min_span_days=args.min_span_days,
        compress_all_diagnosis_like=args.compress_all_diagnosis_like,
    )
    print(f"Inferred compressible diagnosis-like codes: selected={len(selected)}")
    return selected, stats


def build_one_task(args: argparse.Namespace, task: str, versions: list[str], audit_dir: Path) -> list[dict[str, Any]]:
    print("=" * 100)
    print(f"Building compression variants for task={task}")

    source_df, source_vocab = load_cached_source(args.sequence_data_dir, task, args.source_version)

    if args.max_examples_per_task and args.max_examples_per_task > 0:
        # Debug-only option. Keep deterministic order and class distribution roughly untouched by taking the head.
        source_df = source_df.head(int(args.max_examples_per_task)).copy()
        print(f"DEBUG: limited task={task} to first {len(source_df)} examples")

    compressible, code_stats = choose_compressible_codes(args, source_df, audit_dir)

    task_out = args.output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)
    code_stats.to_csv(task_out / "compressible_code_stats.csv", index=False)

    summaries: list[dict[str, Any]] = []

    for version in versions:
        print("-" * 80)
        print(f"task={task} version={version}")
        rows: list[dict[str, Any]] = []

        for i, (_, row) in enumerate(source_df.iterrows(), start=1):
            if i % 500 == 0:
                print(f"  processed {i}/{len(source_df)} examples")
            comp = apply_compression_to_row(row, version, compressible)
            out_row = row.to_dict()
            out_row.update(comp)
            rows.append(out_row)

        out_df = pd.DataFrame(rows)
        vocab = build_vocab_preserving_source(out_df, source_vocab)
        out_df = add_token_ids(out_df, vocab)

        # Leakage sanity check: cached histories should only contain events before prediction_time.
        min_days = np.inf
        for xs in out_df["days_before_prediction"]:
            vals = [safe_float(v) for v in parse_list_cell(xs)]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                min_days = min(min_days, min(vals))
        if np.isfinite(min_days) and min_days < -1e-6:
            raise AssertionError(f"Found future events after compression: task={task}, version={version}, min_days_before={min_days}")

        preferred_cols = [
            "task",
            "row_id",
            "subject_id",
            "prediction_time",
            "label",
            "split",
            "codes",
            "token_ids",
            "days_before_prediction",
            "delta_days",
            "numeric_values",
            "seq_len",
            "n_unique_tokens",
            "raw_seq_len",
            "n_events_removed_vs_raw",
            "n_synthetic_events",
            "n_diagnosis_like_events_raw",
            "n_compressible_chronic_events_raw",
        ]
        ordered_cols = [c for c in preferred_cols if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_cols]
        out_df = out_df[ordered_cols]

        out_dir = task_out / version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_dir / "examples.parquet", index=False)
        write_json(vocab, out_dir / "vocab.json")

        audit_cols = [
            "task",
            "row_id",
            "subject_id",
            "label",
            "split",
            "raw_seq_len",
            "seq_len",
            "n_events_removed_vs_raw",
            "n_synthetic_events",
            "n_diagnosis_like_events_raw",
            "n_compressible_chronic_events_raw",
        ]
        out_df[[c for c in audit_cols if c in out_df.columns]].to_csv(out_dir / "compression_audit.csv", index=False)

        metadata = summarize_examples(out_df, task=task, version=version, n_chronic_codes=len(compressible))
        metadata.update(
            {
                "source_version": args.source_version,
                "compression_built_from": "cached_sequence_examples",
                "chronic_code_mode": args.chronic_code_mode,
                "vocab_size": int(len(vocab)),
                "source_vocab_size": int(len(source_vocab)),
                "note": (
                    "Built from cached sequence examples, not raw EHRSHOT_MEDS. "
                    "Visit-level dedup is approximated by same-day dedup using rounded days_before_prediction. "
                    "Original token ids are preserved for existing source codes; synthetic compression tokens are appended."
                ),
            }
        )
        write_json(metadata, out_dir / "metadata.json")
        summaries.append(metadata)

        print(
            f"Saved {out_dir} | n={metadata['n_examples']} | "
            f"mean_len={metadata['mean_seq_len']:.2f} | p90_len={metadata['p90_seq_len']:.2f} | "
            f"mean_removed={metadata['mean_events_removed_vs_raw']:.3f}"
        )

    pd.DataFrame(summaries).to_csv(task_out / "compression_version_summary.csv", index=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EHRSHOT compression variants from cached raw sequence examples.")

    parser.add_argument("--sequence-data-dir", type=Path, default=Path("ehrshot_sequence_datasets"))
    parser.add_argument("--sequence-data-s3-prefix", type=str, default=DEFAULT_SEQUENCE_S3_PREFIX)
    parser.add_argument("--source-version", type=str, default="raw")

    parser.add_argument("--audit-dir", type=Path, default=Path("ehrshot_copy_forwarding_audit"))
    parser.add_argument("--audit-s3-prefix", type=str, default=DEFAULT_AUDIT_S3_PREFIX)
    parser.add_argument("--chronic-code-mode", type=str, default="strong_only", choices=["strong_only", "strong_plus_possible"])

    parser.add_argument("--output-dir", type=Path, default=Path("ehrshot_sequence_datasets_compression_v2"))
    parser.add_argument("--output-s3-prefix", type=str, default=DEFAULT_OUTPUT_S3_PREFIX)
    parser.add_argument("--skip-s3-upload", action="store_true")

    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--versions", type=str, default=DEFAULT_VERSIONS)

    # Used only if audit whitelist is absent.
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument("--min-mean-occurrences", type=float, default=2.0)
    parser.add_argument("--min-span-days", type=int, default=180)
    parser.add_argument("--compress-all-diagnosis-like", action="store_true")

    # Debug-only: 0 means full dataset.
    parser.add_argument("--max-examples-per-task", type=int, default=0)

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", type=str, default="cpu")
    parser.add_argument("--clearml-project", type=str, default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT")
    parser.add_argument("--clearml-task-name", type=str, default="build_sequence_compressions_from_cached_v2")
    parser.add_argument("--clearml-output-uri", type=str, default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab")

    args = parser.parse_args()

    clearml_task = None
    try:
        clearml_cfg = build_clearml_config(args)
        clearml_task, _ = maybe_init_clearml(args, clearml_cfg)

        tasks = parse_csv_list(args.tasks)
        versions = parse_csv_list(args.versions)
        if not tasks:
            raise ValueError("No tasks provided")
        if not versions:
            raise ValueError("No versions provided")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        args.sequence_data_dir = maybe_download_sequence_from_s3(
            input_dir=args.sequence_data_dir,
            s3_prefix=args.sequence_data_s3_prefix,
            tasks=tasks,
            source_version=args.source_version,
        )

        args.audit_dir = maybe_download_audit_from_s3(
            audit_dir=args.audit_dir,
            audit_s3_prefix=args.audit_s3_prefix,
        )

        all_summaries: list[dict[str, Any]] = []
        for task in tasks:
            all_summaries.extend(build_one_task(args, task, versions, args.audit_dir))

        summary_df = pd.DataFrame(all_summaries)
        summary_path = args.output_dir / "all_compression_version_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        if not args.skip_s3_upload:
            upload_directory_to_s3(args.output_dir, args.output_s3_prefix)

        if clearml_task is not None:
            clearml_task.upload_artifact("all_compression_version_summary", summary_df)
            zip_path = zip_dir(args.output_dir, args.output_dir.parent / f"{args.output_dir.name}.zip")
            clearml_task.upload_artifact("sequence_compression_datasets_zip", artifact_object=str(zip_path))

        print("=" * 100)
        print("DONE")
        print(f"Local output: {args.output_dir.resolve()}")
        if not args.skip_s3_upload and args.output_s3_prefix:
            print(f"S3/MinIO output: {args.output_s3_prefix.rstrip('/')}")
        print("Summary:")
        print(summary_df[["task", "version", "n_examples", "mean_seq_len", "p90_seq_len", "mean_events_removed_vs_raw", "n_chronic_codes_selected"]])

    except Exception:
        print("ERROR: build failed", file=sys.stderr)
        traceback.print_exc()
        if clearml_task is not None:
            try:
                clearml_task.get_logger().report_text("ERROR: build failed\n" + traceback.format_exc())
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
