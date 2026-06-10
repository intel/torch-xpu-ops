#!/usr/bin/env python3
"""
Compare PyTorch Dynamo Benchmark test results (target vs baseline).

Supports comparing two directories of CSV results, or generating a report
from a single directory (treating the missing side as empty).

Usage:
    python compare-e2e.py -t results/target/ -b results/baseline/ -o comparison.xlsx
    python compare-e2e.py -t results/target/ -b results/baseline/ -o out.csv -m report.md
    python compare-e2e.py -t results/target/ -o comparison.xlsx
    python compare-e2e.py -t target/ -b baseline/ -o out.xlsx --suite huggingface --mode inference
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from glob import glob
from html import escape as html_escape
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
KNOWN_SUITES = {"huggingface", "timm_models", "torchbench", "pt2e"}
KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16", "int8"}
KNOWN_MODES = {"inference", "training"}
DEFAULT_PERF_THRESHOLD = 0.1  # 10% change = regression / improvement

# PT2E accuracy thresholds (max allowed top1 drop as fraction)
PT2E_ACC_THRESHOLDS = {
    "float32": 0.0,   # 0% tolerance
    "int8": 0.05,     # 5% tolerance
}

MERGE_KEYS = ["suite", "data_type", "mode", "model"]

ACC_CSV_COLS = ["dev", "name", "batch_size", "accuracy"]
PERF_CSV_COLS = ["dev", "name", "batch_size", "speedup", "abs_latency"]

ACC_OUTPUT_COLS = [
    "suite", "data_type", "mode", "model",
    "batch_size_target", "accuracy_target",
    "batch_size_baseline", "accuracy_baseline",
    "comparison",
]
PERF_OUTPUT_COLS = [
    "suite", "data_type", "mode", "model",
    "batch_size_target", "inductor_target", "eager_target",
    "batch_size_baseline", "inductor_baseline", "eager_baseline",
    "inductor_ratio", "eager_ratio", "comparison",
]

SUMMARY_LEVELS = [
    ("Overall", []),
    ("By Suite", ["suite"]),
    ("By Suite+DataType+Mode", ["suite", "data_type", "mode"]),
]


# ── File Discovery ────────────────────────────────────────────────────
def find_result_files(root_dir: str) -> list[str]:
    """Recursively find *performance.csv and *accuracy.csv files."""
    files = []
    for suffix in ("*performance.csv", "*accuracy.csv"):
        files.extend(glob(os.path.join(root_dir, "**", suffix), recursive=True))
    return files


# ── Filename Parsing ──────────────────────────────────────────────────
def parse_filename(filename: str) -> tuple[str, str, str, str]:
    """
    Extract (suite, data_type, mode, result_type) from a CSV filename.

    Supported patterns:
        inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
        inductor-<suite>-<data_type>-<mode>-xpu-<result_type>.csv

    Raises ValueError on unrecognised filenames.
    """
    if not filename.endswith(".csv"):
        raise ValueError(f"Not a CSV file: {filename}")

    # ── Pattern 1: dash-separated ──
    if "inductor-" in filename:
        base = filename.split("inductor-results-")[-1][:-4]
        parts = base.split("-")
        if len(parts) < 5:
            raise ValueError(f"Too few dash-separated parts in: {filename}")
        suite, data_type, mode, _, result_type = parts[:5]
        # Handle underscore data types like amp_bf16 encoded as amp_bf16
        if result_type not in ("accuracy", "performance"):
            raise ValueError(f"Unknown result type '{result_type}' in {filename}")
        return suite, data_type, mode, result_type

    # ── Pattern 2: underscore-separated ──
    base = filename[:-4]
    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor': {filename}")
    rest = base[len("inductor_"):]

    def _match_prefix(rest: str, candidates: set[str]) -> tuple[str | None, str]:
        """Match longest known prefix followed by '_', return (match, remainder)."""
        for c in sorted(candidates, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c, rest[len(c) + 1:]
        return None, rest

    suite, rest = _match_prefix(rest, KNOWN_SUITES)
    if suite is None:
        raise ValueError(f"Unknown suite in {filename}")

    data_type, rest = _match_prefix(rest, KNOWN_DATA_TYPES)
    data_type = data_type or ""

    mode, rest = _match_prefix(rest, KNOWN_MODES)
    if mode is None:
        raise ValueError(f"Could not find mode (inference/training) in {filename}")

    # rest should be xpu_[optional_extra_]<result_type>
    parts = rest.split("_")
    if len(parts) < 2 or parts[0] != "xpu":
        raise ValueError(f"Expected '_xpu_<result_type>' after mode in {filename}")
    result_type = parts[-1]
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Unknown result type '{result_type}' in {filename}")

    return suite, data_type, mode, result_type


# ── Record Deduplication ──────────────────────────────────────────────
def _best_accuracy_record(records: list[dict]) -> dict:
    """Pick best accuracy record: prefer 'pass' over 'fail' over others."""
    for predicate in (
        lambda r: "pass" in str(r.get("accuracy", "")),
        lambda r: "fail" in str(r.get("accuracy", "")),
    ):
        matches = [r for r in records if predicate(r)]
        if matches:
            return matches[0]
    return records[0]


def _best_performance_record(records: list[dict]) -> dict:
    """
    Pick best performance record.
    Priority: both positive > one positive > zero > NaN/negative.
    Among equally-ranked records, prefer smaller inductor latency.
    """
    def _sort_key(r):
        ind = r.get("inductor")
        eag = r.get("eager")
        ind_ok = pd.notna(ind) and ind > 0
        eag_ok = pd.notna(eag) and eag > 0
        if ind_ok and eag_ok:
            return (0, ind, eag)
        if ind_ok or eag_ok:
            return (1, ind if ind_ok else float("inf"),
                    eag if eag_ok else float("inf"))
        if (pd.notna(ind) and ind == 0) or (pd.notna(eag) and eag == 0):
            return (2, 0, 0)
        return (3, float("inf"), float("inf"))

    return min(records, key=_sort_key)


# ── CSV Loading ───────────────────────────────────────────────────────
def _read_csv_safe(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV with graceful fallback for older pandas versions."""
    try:
        return pd.read_csv(
            path, usecols=usecols, on_bad_lines="skip",
            engine="c", encoding="utf-8",
        )
    except TypeError:
        # Older pandas without on_bad_lines
        return pd.read_csv(
            path, usecols=usecols, error_bad_lines=False,
            warn_bad_lines=True, engine="python", encoding="utf-8",
        )


def load_results(file_list: list[str], result_type: str) -> list[dict]:
    """
    Load CSV files of *result_type* ('accuracy' or 'performance'),
    deduplicate by (suite, data_type, mode, model), and return a list
    of record dicts.
    """
    usecols = ACC_CSV_COLS if result_type == "accuracy" else PERF_CSV_COLS
    raw: dict[tuple, list[dict]] = {}

    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = parse_filename(os.path.basename(fpath))
        except ValueError as exc:
            log.debug("Skipping %s: %s", fpath, exc)
            continue
        if res_type != result_type:
            continue

        is_pt2e = suite == "pt2e"
        try:
            if is_pt2e:
                df = _read_csv_safe(fpath)
            else:
                df = _read_csv_safe(fpath, usecols)
        except Exception as exc:
            log.warning("Failed to read %s: %s", fpath, exc)
            continue

        for _, row in df.iterrows():
            dev = row.get("dev")
            if pd.isna(dev) or str(dev).strip() not in ("cpu", "xpu", "cuda"):
                continue

            # For PT2E int8 performance, append quantization type to data_type
            rec_data_type = data_type
            if is_pt2e and data_type == "int8" and result_type == "performance":
                quant = str(row.get("quantization", "")).strip()
                if quant:
                    rec_data_type = f"int8_{quant}"

            key = (suite, rec_data_type, mode, row["name"])
            rec = {
                "suite": suite, "data_type": rec_data_type,
                "mode": mode, "model": row["name"],
                "batch_size": row["batch_size"],
            }

            if result_type == "accuracy":
                if is_pt2e:
                    # PT2E accuracy: preserve top1/top5 as raw numeric values
                    top1 = pd.to_numeric(row.get("top1"), errors="coerce")
                    top5 = pd.to_numeric(row.get("top5"), errors="coerce")
                    rec["top1"] = top1 if pd.notna(top1) else np.nan
                    rec["top5"] = top5 if pd.notna(top5) else np.nan
                    rec["accuracy"] = f"pass_top1={top1:.3f}" if pd.notna(top1) else "fail_to_run"
                else:
                    rec["accuracy"] = row["accuracy"]
            else:
                if is_pt2e:
                    # PT2E performance: preserve throughput and quantization
                    throughput = pd.to_numeric(row.get("throughput"), errors="coerce")
                    quant = str(row.get("quantization", "")).strip()
                    rec["throughput"] = throughput if pd.notna(throughput) and throughput > 0 else np.nan
                    rec["quantization"] = quant
                    if pd.isna(throughput) or throughput <= 0:
                        rec.update(eager=np.nan, inductor=np.nan, speedup=np.nan)
                    else:
                        rec.update(eager=np.nan, inductor=throughput, speedup=np.nan)
                else:
                    speedup = pd.to_numeric(row.get("speedup"), errors="coerce")
                    abs_lat = pd.to_numeric(row.get("abs_latency"), errors="coerce")
                    if pd.isna(speedup) or pd.isna(abs_lat):
                        log.debug(
                            "Missing speedup/abs_latency for %s in %s",
                            row.get("name"), fpath,
                        )
                        rec.update(eager=np.nan, inductor=np.nan, speedup=np.nan)
                    else:
                        rec.update(
                            eager=speedup * abs_lat,
                            inductor=abs_lat,
                            speedup=speedup,
                        )

            raw.setdefault(key, []).append(rec)

    # Deduplicate
    picker = _best_accuracy_record if result_type == "accuracy" else _best_performance_record
    return [picker(recs) for recs in raw.values()]


# ── Comparison Helpers ────────────────────────────────────────────────
def _is_acc_pass(val) -> bool:
    return pd.notna(val) and "pass" in str(val)


def _is_acc_fail(val) -> bool:
    return pd.notna(val) and "pass" not in str(val) and str(val).strip() != ""


def _extract_pt2e_top1(val) -> float:
    """Extract numeric top1 value from PT2E accuracy string like 'pass_top1=56.556'."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    if "top1=" in s:
        try:
            return float(s.split("top1=")[1])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def _classify_pt2e_accuracy(row: pd.Series) -> str:
    """Classify PT2E accuracy by comparing top1 values with dtype-specific thresholds."""
    tgt, bsl = row.get("accuracy_target"), row.get("accuracy_baseline")
    tgt_top1 = _extract_pt2e_top1(tgt)
    bsl_top1 = _extract_pt2e_top1(bsl)
    data_type = str(row.get("data_type", "float32"))

    # Determine threshold based on dtype (int8 variants all use int8 threshold)
    dt_key = "int8" if "int8" in data_type else data_type
    threshold = PT2E_ACC_THRESHOLDS.get(dt_key, 0.0)

    tgt_valid = pd.notna(tgt_top1) and tgt_top1 > 0
    bsl_valid = pd.notna(bsl_top1) and bsl_top1 > 0

    if tgt_valid and bsl_valid:
        # Compare: if target drops more than threshold below baseline, it's a regression
        if bsl_top1 > 0 and (bsl_top1 - tgt_top1) / bsl_top1 > threshold:
            return "new_failed"
        if bsl_top1 > 0 and (tgt_top1 - bsl_top1) / bsl_top1 > threshold:
            return "new_passed"
        return "no_changed"
    if tgt_valid and not bsl_valid:
        return "new_case"
    if bsl_valid and not tgt_valid:
        return "not_run"
    return "no_changed"


def _is_positive(val) -> bool:
    return pd.notna(val) and isinstance(val, (int, float)) and val > 0


def _geomean(series: pd.Series) -> float:
    """Geometric mean of positive values; NaN if none."""
    vals = series.dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return np.nan
    return float(np.exp(np.log(vals).mean()))


# ── Accuracy Merge ────────────────────────────────────────────────────
def _classify_accuracy(row: pd.Series) -> str:
    """
    Classify an accuracy comparison row.

    Labels:
        no_changed       – same status in both
        new_passed       – was failing in baseline, now passing (improvement)
        new_failed       – was passing in baseline, now failing (regression)
        new_case         – not in baseline, now passing in target
        new_case_failed  – not in baseline, failing in target
        not_run          – was passing in baseline, absent from target
    """
    tgt, bsl = row.get("accuracy_target"), row.get("accuracy_baseline")
    tgt_pass, bsl_pass = _is_acc_pass(tgt), _is_acc_pass(bsl)
    tgt_fail, bsl_fail = _is_acc_fail(tgt), _is_acc_fail(bsl)

    if tgt_pass and bsl_pass:
        return "no_changed"
    if tgt_pass and bsl_fail:
        return "new_passed"       # was failing, now passes
    if tgt_pass and not bsl_fail:
        return "new_case"         # target passes, baseline absent
    if bsl_pass and tgt_fail:
        return "new_failed"       # REGRESSION
    if bsl_pass and not tgt_fail:
        return "not_run"          # baseline passes, target absent
    if tgt_fail and not bsl_fail and not bsl_pass:
        return "new_case_failed"  # new test that fails, baseline absent
    return "no_changed"


def merge_accuracy(
    target_records: list[dict], baseline_records: list[dict],
) -> pd.DataFrame:
    """Outer-join target and baseline accuracy records, classify each model."""
    tgt = pd.DataFrame(target_records)
    bsl = pd.DataFrame(baseline_records)

    if tgt.empty and bsl.empty:
        return pd.DataFrame(columns=ACC_OUTPUT_COLS)

    if tgt.empty:
        tgt = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "accuracy"])
    if bsl.empty:
        bsl = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "accuracy"])

    merged = pd.merge(
        tgt, bsl, on=MERGE_KEYS, how="outer", suffixes=("_target", "_baseline"),
    )
    for col in ("batch_size_target", "batch_size_baseline"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Use PT2E-specific classifier for pt2e suite (numeric top1 comparison)
    is_pt2e = merged["suite"] == "pt2e" if "suite" in merged.columns else pd.Series(False, index=merged.index)
    merged["comparison"] = ""
    if is_pt2e.any():
        merged.loc[is_pt2e, "comparison"] = merged[is_pt2e].apply(_classify_pt2e_accuracy, axis=1)
    if (~is_pt2e).any():
        merged.loc[~is_pt2e, "comparison"] = merged[~is_pt2e].apply(_classify_accuracy, axis=1)

    for c in ACC_OUTPUT_COLS:
        if c not in merged.columns:
            merged[c] = None
    return merged[ACC_OUTPUT_COLS].sort_values(MERGE_KEYS).reset_index(drop=True)


# ── PT2E Accuracy Merge (pivoted by dtype) ────────────────────────────
PT2E_ACC_OUTPUT_COLS = [
    "suite", "mode", "model", "category",
    "fp32_target", "int8_target", "int8/fp32_target",
    "fp32_baseline", "int8_baseline", "int8/fp32_baseline",
    "fp32_comparison", "int8_comparison",
]


def _pt2e_acc_compare(tgt: float, bsl: float, threshold: float) -> str:
    """Compare two top1/top5 values with a relative threshold."""
    tgt_valid = pd.notna(tgt) and tgt > 0
    bsl_valid = pd.notna(bsl) and bsl > 0
    if tgt_valid and bsl_valid:
        if bsl > 0 and (bsl - tgt) / bsl > threshold:
            return "new_failed"
        if bsl > 0 and (tgt - bsl) / bsl > threshold:
            return "new_passed"
        return "no_changed"
    if tgt_valid and not bsl_valid:
        return "new_case"
    if bsl_valid and not tgt_valid:
        return "not_run"
    return "no_changed"


def merge_pt2e_accuracy(
    target_records: list[dict], baseline_records: list[dict],
) -> pd.DataFrame:
    """Merge PT2E accuracy records into a pivoted format by dtype (fp32/int8)."""
    if not target_records and not baseline_records:
        return pd.DataFrame(columns=PT2E_ACC_OUTPUT_COLS)

    tgt = pd.DataFrame(target_records) if target_records else pd.DataFrame()
    bsl = pd.DataFrame(baseline_records) if baseline_records else pd.DataFrame()

    # Build lookup: (mode, model, data_type) -> {top1, top5}
    def _build_map(df: pd.DataFrame) -> dict:
        m: dict[tuple, dict] = {}
        if df.empty:
            return m
        for _, row in df.iterrows():
            key = (row["mode"], row["model"], row.get("data_type", ""))
            m[key] = {"top1": row.get("top1", np.nan), "top5": row.get("top5", np.nan)}
        return m

    tgt_map = _build_map(tgt)
    bsl_map = _build_map(bsl)

    # Collect all (mode, model) pairs
    all_keys: set[tuple[str, str]] = set()
    for mode, model, _ in list(tgt_map.keys()) + list(bsl_map.keys()):
        all_keys.add((mode, model))

    rows = []
    for mode, model in sorted(all_keys):
        # For each metric (top1, top5)
        for category in ("top1", "top5"):
            fp32_tgt = tgt_map.get((mode, model, "float32"), {}).get(category, np.nan)
            int8_tgt = tgt_map.get((mode, model, "int8"), {}).get(category, np.nan)
            fp32_bsl = bsl_map.get((mode, model, "float32"), {}).get(category, np.nan)
            int8_bsl = bsl_map.get((mode, model, "int8"), {}).get(category, np.nan)

            # Skip rows where all values are NaN
            if all(pd.isna(v) for v in [fp32_tgt, int8_tgt, fp32_bsl, int8_bsl]):
                continue

            # Compute int8/fp32 ratios
            int8_fp32_tgt = (int8_tgt / fp32_tgt) if (
                pd.notna(int8_tgt) and pd.notna(fp32_tgt) and fp32_tgt > 0
            ) else np.nan
            int8_fp32_bsl = (int8_bsl / fp32_bsl) if (
                pd.notna(int8_bsl) and pd.notna(fp32_bsl) and fp32_bsl > 0
            ) else np.nan

            # Per-dtype comparison
            fp32_thresh = PT2E_ACC_THRESHOLDS.get("float32", 0.0)
            int8_thresh = PT2E_ACC_THRESHOLDS.get("int8", 0.05)
            fp32_comp = _pt2e_acc_compare(fp32_tgt, fp32_bsl, fp32_thresh)
            int8_comp = _pt2e_acc_compare(int8_tgt, int8_bsl, int8_thresh)

            rows.append({
                "suite": "pt2e", "mode": mode, "model": model,
                "category": category,
                "fp32_target": round(fp32_tgt, 3) if pd.notna(fp32_tgt) else np.nan,
                "int8_target": round(int8_tgt, 3) if pd.notna(int8_tgt) else np.nan,
                "int8/fp32_target": round(int8_fp32_tgt, 4) if pd.notna(int8_fp32_tgt) else np.nan,
                "fp32_baseline": round(fp32_bsl, 3) if pd.notna(fp32_bsl) else np.nan,
                "int8_baseline": round(int8_bsl, 3) if pd.notna(int8_bsl) else np.nan,
                "int8/fp32_baseline": round(int8_fp32_bsl, 4) if pd.notna(int8_fp32_bsl) else np.nan,
                "fp32_comparison": fp32_comp,
                "int8_comparison": int8_comp,
            })

    return pd.DataFrame(rows, columns=PT2E_ACC_OUTPUT_COLS)


# ── PT2E Performance Merge (pivoted by quantization) ──────────────────
PT2E_PERF_OUTPUT_COLS = [
    "suite", "mode", "model",
    "fp32_target", "symm_target", "asymm_target",
    "symm/fp32_target", "asymm/fp32_target",
    "fp32_baseline", "symm_baseline", "asymm_baseline",
    "symm/fp32_baseline", "asymm/fp32_baseline",
    "comparison",
]


def merge_pt2e_performance(
    target_records: list[dict], baseline_records: list[dict],
    threshold: float,
) -> pd.DataFrame:
    """Merge PT2E performance records into a pivoted format by quantization."""
    if not target_records and not baseline_records:
        return pd.DataFrame(columns=PT2E_PERF_OUTPUT_COLS)

    tgt = pd.DataFrame(target_records) if target_records else pd.DataFrame()
    bsl = pd.DataFrame(baseline_records) if baseline_records else pd.DataFrame()

    # Build lookup: (mode, model) -> {fp32: throughput, symm: throughput, asymm: throughput}
    def _build_map(df: pd.DataFrame) -> dict:
        m: dict[tuple, dict] = {}
        if df.empty:
            return m
        for _, row in df.iterrows():
            key = (row["mode"], row["model"])
            dt = str(row.get("data_type", ""))
            quant = str(row.get("quantization", "")).strip()
            throughput = row.get("throughput", np.nan)

            if dt == "float32":
                slot = "fp32"
            elif quant in ("symm", "symmetric"):
                slot = "symm"
            elif quant in ("asymm", "asymmetric"):
                slot = "asymm"
            else:
                slot = "fp32" if "float32" in dt else "symm"

            m.setdefault(key, {})[slot] = throughput
        return m

    tgt_map = _build_map(tgt)
    bsl_map = _build_map(bsl)

    all_keys: set[tuple[str, str]] = set()
    for k in list(tgt_map.keys()) + list(bsl_map.keys()):
        all_keys.add(k)

    rows = []
    for mode, model in sorted(all_keys):
        tgt_vals = tgt_map.get((mode, model), {})
        bsl_vals = bsl_map.get((mode, model), {})

        fp32_t = tgt_vals.get("fp32", np.nan)
        symm_t = tgt_vals.get("symm", np.nan)
        asymm_t = tgt_vals.get("asymm", np.nan)
        fp32_b = bsl_vals.get("fp32", np.nan)
        symm_b = bsl_vals.get("symm", np.nan)
        asymm_b = bsl_vals.get("asymm", np.nan)

        # Compute ratios (throughput: higher is better, so ratio = target/baseline)
        def _ratio(val, fp32):
            if pd.notna(val) and pd.notna(fp32) and fp32 > 0:
                return round(val / fp32, 4)
            return np.nan

        symm_fp32_t = _ratio(symm_t, fp32_t)
        asymm_fp32_t = _ratio(asymm_t, fp32_t)
        symm_fp32_b = _ratio(symm_b, fp32_b)
        asymm_fp32_b = _ratio(asymm_b, fp32_b)

        # Overall comparison: check if any throughput regressed
        comp = "no_changed"
        for t_val, b_val in [(fp32_t, fp32_b), (symm_t, symm_b), (asymm_t, asymm_b)]:
            t_ok = pd.notna(t_val) and t_val > 0
            b_ok = pd.notna(b_val) and b_val > 0
            if t_ok and b_ok:
                ratio = t_val / b_val
                if ratio < 1 - threshold:
                    comp = "new_dropped"
                    break
                if ratio > 1 + threshold:
                    comp = "new_improved"
            elif t_ok and not b_ok:
                comp = "new_passed"
            elif b_ok and not t_ok:
                comp = "new_failed"
                break

        rows.append({
            "suite": "pt2e", "mode": mode, "model": model,
            "fp32_target": round(fp32_t, 4) if pd.notna(fp32_t) else np.nan,
            "symm_target": round(symm_t, 4) if pd.notna(symm_t) else np.nan,
            "asymm_target": round(asymm_t, 4) if pd.notna(asymm_t) else np.nan,
            "symm/fp32_target": symm_fp32_t,
            "asymm/fp32_target": asymm_fp32_t,
            "fp32_baseline": round(fp32_b, 4) if pd.notna(fp32_b) else np.nan,
            "symm_baseline": round(symm_b, 4) if pd.notna(symm_b) else np.nan,
            "asymm_baseline": round(asymm_b, 4) if pd.notna(asymm_b) else np.nan,
            "symm/fp32_baseline": symm_fp32_b,
            "asymm/fp32_baseline": asymm_fp32_b,
            "comparison": comp,
        })

    return pd.DataFrame(rows, columns=PT2E_PERF_OUTPUT_COLS)


# ── Performance Merge ─────────────────────────────────────────────────
def _classify_performance(row: pd.Series, threshold: float) -> str:
    """
    Classify a performance comparison row.

    Labels:
        no_changed    – within threshold or both invalid
        new_dropped   – significantly slower (regression)
        new_improved  – significantly faster (improvement)
        new_passed    – target valid, baseline invalid (fix)
        new_failed    – baseline valid, target invalid (regression)
    """
    ind_tgt = row.get("inductor_target")
    ind_bsl = row.get("inductor_baseline")
    tgt_ok, bsl_ok = _is_positive(ind_tgt), _is_positive(ind_bsl)

    if tgt_ok and bsl_ok:
        ind_r = row.get("inductor_ratio")
        eag_r = row.get("eager_ratio")
        if (pd.notna(ind_r) and ind_r < 1 - threshold) or \
           (pd.notna(eag_r) and eag_r < 1 - threshold):
            return "new_dropped"
        if (pd.notna(ind_r) and ind_r > 1 + threshold) or \
           (pd.notna(eag_r) and eag_r > 1 + threshold):
            return "new_improved"
        return "no_changed"
    if tgt_ok:
        return "new_passed"
    if bsl_ok:
        return "new_failed"
    return "no_changed"


def merge_performance(
    target_records: list[dict], baseline_records: list[dict],
    threshold: float,
) -> pd.DataFrame:
    """Outer-join target and baseline performance records, compute ratios, classify."""
    tgt = pd.DataFrame(target_records)
    bsl = pd.DataFrame(baseline_records)

    if tgt.empty and bsl.empty:
        return pd.DataFrame(columns=PERF_OUTPUT_COLS)

    if tgt.empty:
        tgt = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "eager", "inductor", "speedup"])
    if bsl.empty:
        bsl = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "eager", "inductor", "speedup"])

    merged = pd.merge(
        tgt, bsl, on=MERGE_KEYS, how="outer", suffixes=("_target", "_baseline"),
    )
    for col in ("batch_size_target", "batch_size_baseline"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Ratios: baseline / target  (>1 means target is faster)
    for metric in ("inductor", "eager"):
        tgt_col = f"{metric}_target"
        bsl_col = f"{metric}_baseline"
        ratio_col = f"{metric}_ratio"
        if tgt_col in merged.columns and bsl_col in merged.columns:
            mask = merged[tgt_col].notna() & (merged[tgt_col] > 0)
            merged[ratio_col] = np.where(
                mask,
                merged[bsl_col].astype(float) / merged[tgt_col].astype(float),
                np.nan,
            )

    for col in ("inductor_target", "inductor_baseline", "eager_target", "eager_baseline"):
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    for col in ("inductor_ratio", "eager_ratio"):
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    merged["comparison"] = merged.apply(
        lambda r: _classify_performance(r, threshold), axis=1,
    )

    for c in PERF_OUTPUT_COLS:
        if c not in merged.columns:
            merged[c] = None
    return merged[PERF_OUTPUT_COLS].sort_values(MERGE_KEYS).reset_index(drop=True)


# ── Summary Generation ────────────────────────────────────────────────
def _group_metrics(group: pd.DataFrame, is_perf: bool) -> pd.Series:
    """Compute summary metrics for a group of rows."""
    comp = group["comparison"]
    if is_perf:
        tgt_passed = group["inductor_target"].apply(_is_positive).sum()
        bsl_passed = group["inductor_baseline"].apply(_is_positive).sum()
    else:
        tgt_passed = group["accuracy_target"].apply(_is_acc_pass).sum()
        bsl_passed = group["accuracy_baseline"].apply(_is_acc_pass).sum()

    result = {
        "target_passed": tgt_passed,
        "baseline_passed": bsl_passed,
        "total": len(group),
        "new_fail": int((comp == "new_failed").sum()),
        "new_pass": int((comp == "new_passed").sum()),
        "new_drop": int((comp == "new_dropped").sum()) if is_perf else 0,
        "new_improve": int((comp == "new_improved").sum()) if is_perf else 0,
    }
    if is_perf:
        result["ind_ratio"] = _geomean(
            group["inductor_ratio"] if "inductor_ratio" in group.columns
            else pd.Series(dtype=float)
        )
        result["eag_ratio"] = _geomean(
            group["eager_ratio"] if "eager_ratio" in group.columns
            else pd.Series(dtype=float)
        )
    if not is_perf:
        result["not_run"] = int((comp == "not_run").sum())
    return pd.Series(result)


def generate_summary(
    acc_merged: pd.DataFrame, perf_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Generate summary at all grouping levels and combine."""
    frames = []
    for level_name, group_cols in SUMMARY_LEVELS:
        for df, label, is_perf in [
            (acc_merged, "Accuracy", False),
            (perf_merged, "Performance", True),
        ]:
            if df.empty:
                continue
            if group_cols:
                grp = (
                    df.groupby(group_cols, dropna=False)
                    .apply(lambda g: _group_metrics(g, is_perf), include_groups=False)
                    .reset_index()
                )
                grp["Category"] = grp[group_cols].astype(str).agg("_".join, axis=1)
            else:
                grp = _group_metrics(df, is_perf).to_frame().T
                grp["Category"] = "Overall"
            grp["Level"] = level_name
            grp["Type"] = label
            frames.append(grp)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Pass rates
    combined["target_passrate"] = (
        combined["target_passed"] / combined["total"] * 100
    ).round(2)
    combined["baseline_passrate"] = (
        combined["baseline_passed"] / combined["total"] * 100
    ).round(2)

    # Integer columns
    for c in ("target_passed", "baseline_passed", "total",
              "new_fail", "new_drop", "new_pass", "new_improve"):
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Int64")
    for c in ("ind_ratio", "eag_ratio"):
        if c in combined.columns:
            combined[c] = combined[c].round(3)

    # Sort by level priority, then type, then category
    level_order = {name: i for i, (name, _) in enumerate(SUMMARY_LEVELS)}
    combined["_sort"] = combined["Level"].map(level_order)
    combined.sort_values(["_sort", "Type", "Category"], inplace=True)

    cols = [
        "Level", "Type", "Category",
        "target_passed", "baseline_passed", "total",
        "target_passrate", "baseline_passrate",
        "new_fail", "new_drop", "new_pass", "new_improve",
        "ind_ratio", "eag_ratio",
    ]
    for c in cols:
        if c not in combined.columns:
            combined[c] = np.nan
    return combined[cols].reset_index(drop=True)


# ── Markdown Helpers ──────────────────────────────────────────────────
def _fmt_ratio(val: Any, threshold: float) -> str:
    """Format ratio with visual indicator when outside threshold."""
    if pd.isna(val) or val == "":
        return ""
    try:
        num = float(val)
        if num < 1 - threshold:
            return f"**{num:.3f}** 🔴"
        if num > 1 + threshold:
            return f"**{num:.3f}** 🟢"
        return f"{num:.3f}"
    except (TypeError, ValueError):
        return str(val)


def _html_table(
    df: pd.DataFrame, columns: list[str], cond_col: str,
    fail_vals: set[str], pass_vals: set[str],
) -> str:
    """Render an HTML table with conditional row background colours."""
    lines = ["<table>", "<thead><tr>"]
    for c in columns:
        lines.append(f"  <th>{html_escape(c)}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    for _, row in df.iterrows():
        val = row.get(cond_col, "")
        if val in fail_vals:
            style = ' style="background-color: #f8d7da;"'
        elif val in pass_vals:
            style = ' style="background-color: #d4edda;"'
        else:
            style = ""
        lines.append(f"<tr{style}>")
        for c in columns:
            cell = html_escape(str(row.get(c, "")) if pd.notna(row.get(c)) else "")
            lines.append(f"  <td>{cell}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a plain Markdown table."""
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        cells = [
            str(row.get(c, "")) if pd.notna(row.get(c)) else ""
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Markdown Report ───────────────────────────────────────────────────
FAIL_LABELS = {"new_failed", "new_dropped"}
PASS_LABELS = {"new_passed", "new_improved"}


def write_markdown(
    summary: pd.DataFrame,
    acc_df: pd.DataFrame, perf_df: pd.DataFrame,
    threshold: float, filename: str,
) -> None:
    """Write a single Markdown report: summary -> failures -> improvements."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Dynamo Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ── Overall summary table ──
        f.write("## Summary\n\n")
        overall = summary[summary["Category"] == "Overall"] if not summary.empty else pd.DataFrame()
        if not overall.empty:
            f.write(
                "| Type | Total | Target Pass | Baseline Pass "
                "| Target % | Baseline % "
                "| ❌ Fail | 📉 Drop | ✅ Pass | 📈 Improve "
                "| Ind Ratio | Eag Ratio |\n"
            )
            f.write(
                "|------|-------|-------------|---------------"
                "|----------|------------"
                "|---------|---------|---------|------------"
                "|-----------|----------|\n"
            )
            for _, row in overall.iterrows():
                typ = row["Type"]
                is_perf = typ == "Performance"
                nd = row.get("new_drop", 0) if is_perf else "/"
                ni = row.get("new_improve", 0) if is_perf else "/"
                ir = _fmt_ratio(row.get("ind_ratio"), threshold) if is_perf else "/"
                er = _fmt_ratio(row.get("eag_ratio"), threshold) if is_perf else "/"
                f.write(
                    f"| {typ} | {row['total']} "
                    f"| {row['target_passed']} | {row['baseline_passed']} "
                    f"| {row['target_passrate']}% | {row['baseline_passrate']}% "
                    f"| {row['new_fail']} | {nd} | {row['new_pass']} | {ni} "
                    f"| {ir} | {er} |\n"
                )
            f.write("\n")
        else:
            f.write("No summary data available.\n\n")

        # ── Suite-level overview (collapsible) ──
        by_suite = summary[summary["Level"] == "By Suite"] if not summary.empty else pd.DataFrame()
        if not by_suite.empty:
            f.write("<details>\n<summary><b>Overview by Suite</b></summary>\n\n")
            f.write(
                "| Type | Suite | Total | Target | Baseline "
                "| ❌ Fail | 📉 Drop | ✅ Pass | 📈 Improve "
                "| Ind Ratio | Eag Ratio |\n"
            )
            f.write(
                "|------|-------|-------|--------|----------"
                "|---------|---------|---------|------------"
                "|-----------|----------|\n"
            )
            for _, row in by_suite.iterrows():
                typ = row["Type"]
                is_perf = typ == "Performance"
                nd = row.get("new_drop", 0) if is_perf else "/"
                ni = row.get("new_improve", 0) if is_perf else "/"
                ir = _fmt_ratio(row.get("ind_ratio"), threshold) if is_perf else "/"
                er = _fmt_ratio(row.get("eag_ratio"), threshold) if is_perf else "/"
                f.write(
                    f"| {typ} | {row['Category']} | {row['total']} "
                    f"| {row['target_passed']} | {row['baseline_passed']} "
                    f"| {row['new_fail']} | {nd} | {row['new_pass']} | {ni} "
                    f"| {ir} | {er} |\n"
                )
            f.write("\n</details>\n\n")

        # ── NEW FAILURES & REGRESSIONS (shown first, most important) ──
        acc_fails = (
            acc_df[acc_df["comparison"] == "new_failed"]
            if not acc_df.empty else pd.DataFrame()
        )
        perf_dropped = (
            perf_df[perf_df["comparison"] == "new_dropped"]
            if not perf_df.empty else pd.DataFrame()
        )
        perf_failed = (
            perf_df[perf_df["comparison"] == "new_failed"]
            if not perf_df.empty else pd.DataFrame()
        )
        n_issues = len(acc_fails) + len(perf_dropped) + len(perf_failed)

        f.write(f"## ❌ New Failures & Regressions ({n_issues} models)\n\n")

        if n_issues == 0:
            f.write("None! 🎉\n\n")
        else:
            if not acc_fails.empty:
                f.write(f"### Accuracy Failures ({len(acc_fails)} models)\n\n")
                cols = [c for c in [
                    "suite", "data_type", "mode", "model",
                    "accuracy_target", "accuracy_baseline", "comparison",
                ] if c in acc_fails.columns]
                f.write(_html_table(acc_fails, cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            perf_cols = [c for c in [
                "suite", "data_type", "mode", "model",
                "inductor_target", "eager_target",
                "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison",
            ] if c in perf_df.columns] if not perf_df.empty else []

            if not perf_dropped.empty:
                f.write(
                    f"### Performance Regressions "
                    f"({len(perf_dropped)} models, ratio < {(1-threshold)*100:.0f}%)\n\n"
                )
                f.write(_html_table(perf_dropped, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            if not perf_failed.empty:
                f.write(f"### Performance Failures ({len(perf_failed)} models)\n\n")
                f.write(_html_table(perf_failed, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

        # ── NEW PASSES & IMPROVEMENTS (shown second) ──
        acc_passes = (
            acc_df[acc_df["comparison"] == "new_passed"]
            if not acc_df.empty else pd.DataFrame()
        )
        perf_improved = (
            perf_df[perf_df["comparison"] == "new_improved"]
            if not perf_df.empty else pd.DataFrame()
        )
        perf_new_pass = (
            perf_df[perf_df["comparison"] == "new_passed"]
            if not perf_df.empty else pd.DataFrame()
        )
        n_good = len(acc_passes) + len(perf_improved) + len(perf_new_pass)

        f.write(f"## ✅ New Passes & Improvements ({n_good} models)\n\n")

        if n_good == 0:
            f.write("None.\n\n")
        else:
            if not acc_passes.empty:
                f.write(f"### Accuracy New Passes ({len(acc_passes)} models)\n\n")
                cols = [c for c in [
                    "suite", "data_type", "mode", "model",
                    "accuracy_target", "accuracy_baseline", "comparison",
                ] if c in acc_passes.columns]
                f.write(_html_table(acc_passes, cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            perf_cols = [c for c in [
                "suite", "data_type", "mode", "model",
                "inductor_target", "eager_target",
                "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison",
            ] if c in perf_df.columns] if not perf_df.empty else []

            if not perf_improved.empty:
                f.write(
                    f"### Performance Improvements "
                    f"({len(perf_improved)} models, ratio > {(1+threshold)*100:.0f}%)\n\n"
                )
                f.write(_html_table(perf_improved, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            if not perf_new_pass.empty:
                f.write(f"### Performance New Passes ({len(perf_new_pass)} models)\n\n")
                f.write(_html_table(perf_new_pass, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

        # ── Not-run models (collapsible) ──
        acc_missing = (
            acc_df[acc_df["comparison"] == "not_run"]
            if not acc_df.empty else pd.DataFrame()
        )
        if not acc_missing.empty:
            f.write(
                f"<details>\n"
                f"<summary><b>⚠️ Not Run in Target ({len(acc_missing)} models)</b></summary>\n\n"
            )
            cols = [c for c in [
                "suite", "data_type", "mode", "model", "accuracy_baseline",
            ] if c in acc_missing.columns]
            f.write(_md_table(acc_missing, cols))
            f.write("\n\n</details>\n\n")

    log.info("Markdown report written to %s", filename)


# ── Console Output ────────────────────────────────────────────────────
def print_report(
    target_acc_n: int, target_perf_n: int,
    baseline_acc_n: int, baseline_perf_n: int,
    acc_merged: pd.DataFrame, perf_merged: pd.DataFrame,
    output_file: str, title: str = "DYNAMO BENCHMARK COMPARISON",
) -> None:
    """Print a structured summary to stdout."""
    W = 64
    sep = "=" * W
    thin = "-" * W

    print(f"\n{sep}")
    print(f"{title:^{W}}")
    print(sep)

    print(f"  {'Target records:':<28} acc={target_acc_n:<6} perf={target_perf_n}")
    print(f"  {'Baseline records:':<28} acc={baseline_acc_n:<6} perf={baseline_perf_n}")
    print(thin)

    if not acc_merged.empty:
        comp = acc_merged["comparison"]
        total = len(acc_merged)
        tgt_pass = int(acc_merged["accuracy_target"].apply(_is_acc_pass).sum())
        bsl_pass = int(acc_merged["accuracy_baseline"].apply(_is_acc_pass).sum())

        print(f"  ACCURACY ({total} models)")
        if total:
            print(f"    Target pass rate:    {tgt_pass}/{total} ({tgt_pass/total*100:.1f}%)")
            print(f"    Baseline pass rate:  {bsl_pass}/{total} ({bsl_pass/total*100:.1f}%)")

        for label, emoji in [
            ("new_failed", "❌ New failures"),
            ("new_passed", "✅ New passes"),
            ("not_run", "⚠️  Not run"),
        ]:
            count = int((comp == label).sum())
            if count:
                print(f"    {emoji + ':':<24} {count}")
        print(thin)

    if not perf_merged.empty:
        comp = perf_merged["comparison"]
        total = len(perf_merged)
        tgt_pass = int(perf_merged["inductor_target"].apply(_is_positive).sum())
        bsl_pass = int(perf_merged["inductor_baseline"].apply(_is_positive).sum())

        print(f"  PERFORMANCE ({total} models)")
        if total:
            print(f"    Target pass rate:    {tgt_pass}/{total} ({tgt_pass/total*100:.1f}%)")
            print(f"    Baseline pass rate:  {bsl_pass}/{total} ({bsl_pass/total*100:.1f}%)")

        for label, emoji in [
            ("new_failed", "❌ New failures"),
            ("new_dropped", "📉 Dropped Cases"),
            ("new_passed", "✅ New passes"),
            ("new_improved", "📈 Improved Cases"),
        ]:
            count = int((comp == label).sum())
            if count:
                print(f"    {emoji + ':':<24} {count}")

        ind_gm = _geomean(
            perf_merged["inductor_ratio"] if "inductor_ratio" in perf_merged.columns
            else pd.Series(dtype=float)
        )
        eag_gm = _geomean(
            perf_merged["eager_ratio"] if "eager_ratio" in perf_merged.columns
            else pd.Series(dtype=float)
        )
        if pd.notna(ind_gm):
            print(f"    Inductor ratio (geomean): {ind_gm:.3f}")
        if pd.notna(eag_gm):
            print(f"    Eager ratio (geomean):    {eag_gm:.3f}")
        print(thin)

    print(f"  Output: {output_file}")
    print(f"{sep}\n")


# ── File Output (Excel / CSV) ────────────────────────────────────────
def write_excel(
    summary: pd.DataFrame, acc: pd.DataFrame, perf: pd.DataFrame, path: str,
) -> None:
    placeholder = pd.DataFrame({"Info": ["No data available"]})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        (summary if not summary.empty else placeholder).to_excel(
            w, sheet_name="Summary", index=False,
        )
        (acc if not acc.empty else placeholder).to_excel(
            w, sheet_name="Accuracy Details", index=False,
        )
        (perf if not perf.empty else placeholder).to_excel(
            w, sheet_name="Performance Details", index=False,
        )
    log.info("Excel written to %s", path)


def write_csv(
    summary: pd.DataFrame, acc: pd.DataFrame, perf: pd.DataFrame, base: str,
) -> None:
    for df, suffix in [
        (summary, "_summary"),
        (acc, "_accuracy"),
        (perf, "_performance"),
    ]:
        if not df.empty:
            path = f"{base}{suffix}.csv"
            df.to_csv(path, index=False, na_rep="")
            log.info("Written %s", path)


# ── Main ──────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch Dynamo Benchmark results (target vs baseline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -t target_dir/ -b baseline_dir/ -o comparison.xlsx
  %(prog)s -t target_dir/ -o comparison.xlsx
  %(prog)s -t target/ -b baseline/ -o out.csv -m report.md
  %(prog)s -t target/ -b baseline/ -o out.xlsx --threshold 0.15
  %(prog)s -t target/ -b baseline/ -o out.xlsx --suite huggingface torchbench
  %(prog)s -t target/ -b baseline/ -o out.xlsx --mode inference --data-type bfloat16
""",
    )
    parser.add_argument("-t", "--target-dir",
                        help="Directory with target (new) CSV results")
    parser.add_argument("-b", "--baseline-dir",
                        help="Directory with baseline (reference) CSV results")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file (.xlsx or .csv)")
    parser.add_argument("-m", "--markdown",
                        help="Markdown report filename (e.g. report.md)")
    parser.add_argument("--threshold", type=float,
                        default=DEFAULT_PERF_THRESHOLD,
                        help=f"Performance change threshold (default: {DEFAULT_PERF_THRESHOLD})")
    parser.add_argument("--suite", nargs="*", choices=sorted(KNOWN_SUITES),
                        help="Filter to specific suites")
    parser.add_argument("--mode", nargs="*", choices=sorted(KNOWN_MODES),
                        help="Filter to specific modes")
    parser.add_argument("--data-type", nargs="*", choices=sorted(KNOWN_DATA_TYPES),
                        help="Filter to specific data types")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.target_dir and not args.baseline_dir:
        parser.error("At least one of -t/--target-dir or -b/--baseline-dir is required.")

    out_base, out_ext = os.path.splitext(args.output)
    if out_ext not in (".xlsx", ".csv"):
        parser.error("Output file must end with .xlsx or .csv")

    # ── Load data ──
    def _load_dir(dir_path: str | None, label: str) -> tuple[list[dict], list[dict]]:
        if not dir_path:
            log.info("No %s directory provided; treating as empty.", label)
            return [], []
        if not os.path.isdir(dir_path):
            log.error("%s directory does not exist: %s", label.capitalize(), dir_path)
            sys.exit(1)
        files = find_result_files(dir_path)
        log.info("Found %d CSV files in %s directory.", len(files), label)
        return load_results(files, "accuracy"), load_results(files, "performance")

    target_acc, target_perf = _load_dir(args.target_dir, "target")
    baseline_acc, baseline_perf = _load_dir(args.baseline_dir, "baseline")

    # ── Optional filters ──
    def _apply_filters(records: list[dict]) -> list[dict]:
        out = records
        if args.suite:
            out = [r for r in out if r["suite"] in args.suite]
        if args.mode:
            out = [r for r in out if r["mode"] in args.mode]
        if args.data_type:
            out = [r for r in out if r["data_type"] in args.data_type]
        return out

    target_acc = _apply_filters(target_acc)
    target_perf = _apply_filters(target_perf)
    baseline_acc = _apply_filters(baseline_acc)
    baseline_perf = _apply_filters(baseline_perf)

    # ── Merge & compare ──
    # Separate pt2e from regular suites
    def _split_pt2e(records: list[dict]) -> tuple[list[dict], list[dict]]:
        regular = [r for r in records if r["suite"] != "pt2e"]
        pt2e = [r for r in records if r["suite"] == "pt2e"]
        return regular, pt2e

    target_acc_reg, target_acc_pt2e = _split_pt2e(target_acc)
    target_perf_reg, target_perf_pt2e = _split_pt2e(target_perf)
    baseline_acc_reg, baseline_acc_pt2e = _split_pt2e(baseline_acc)
    baseline_perf_reg, baseline_perf_pt2e = _split_pt2e(baseline_perf)

    acc_merged = merge_accuracy(target_acc_reg, baseline_acc_reg)
    perf_merged = merge_performance(target_perf_reg, baseline_perf_reg, args.threshold)

    acc_pt2e = merge_pt2e_accuracy(target_acc_pt2e, baseline_acc_pt2e)
    perf_pt2e = merge_pt2e_performance(target_perf_pt2e, baseline_perf_pt2e, args.threshold)

    # ── Summary ──
    summary = generate_summary(acc_merged, perf_merged)

    # ── Write outputs ──
    if out_ext == ".xlsx":
        write_excel(summary, acc_merged, perf_merged, args.output)
    else:
        write_csv(summary, acc_merged, perf_merged, out_base)

    # Write pt2e outputs separately if present
    has_pt2e = not acc_pt2e.empty or not perf_pt2e.empty
    if has_pt2e:
        pt2e_base = f"{out_base}_pt2e"
        if not acc_pt2e.empty:
            acc_pt2e.to_csv(f"{pt2e_base}_accuracy.csv", index=False, na_rep="")
            log.info("Written %s", f"{pt2e_base}_accuracy.csv")
        if not perf_pt2e.empty:
            perf_pt2e.to_csv(f"{pt2e_base}_performance.csv", index=False, na_rep="")
            log.info("Written %s", f"{pt2e_base}_performance.csv")

    if args.markdown:
        md_file = args.markdown if args.markdown.endswith(".md") else args.markdown + ".md"
        write_markdown(summary, acc_merged, perf_merged, args.threshold, md_file)

    # ── Console report ──
    print_report(
        len(target_acc_reg), len(target_perf_reg),
        len(baseline_acc_reg), len(baseline_perf_reg),
        acc_merged, perf_merged,
        args.output,
    )
    if has_pt2e:
        print("\n" + "=" * 60)
        print(" PT2E BENCHMARK COMPARISON")
        print("=" * 60)
        if not acc_pt2e.empty:
            print(f"\n  Accuracy: {len(acc_pt2e)} rows")
            for col in ("fp32_comparison", "int8_comparison"):
                if col in acc_pt2e.columns:
                    counts = acc_pt2e[col].value_counts()
                    print(f"    {col}: {dict(counts)}")
        if not perf_pt2e.empty:
            print(f"\n  Performance: {len(perf_pt2e)} rows")
            if "comparison" in perf_pt2e.columns:
                counts = perf_pt2e["comparison"].value_counts()
                print(f"    comparison: {dict(counts)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
