#!/usr/bin/env python3
"""
Comparison tool for PyTorch Dynamo Benchmark test results (target vs baseline).
Supports comparing two directories, or generating a report from a single
directory (treating missing side as empty).
Usage:
    python compare-e2e.py -t "results/target/" -b "results/baseline/" --output comparison.xlsx
    python compare-e2e.py -t "results/target/" -b "results/baseline/" --output comparison.csv --markdown
    python compare-e2e.py -t "results/target/" --output comparison.xlsx
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from typing import Any
from html import escape as html_escape

# ----------------------------------------------------------------------
# Constants – easily adjustable
# ----------------------------------------------------------------------
KNOWN_SUITES = {"huggingface", "timm_models", "torchbench"}
KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
KNOWN_MODES = {"inference", "training"}
DEFAULT_PERFORMANCE_THRESHOLD = 0.1  # 10% change considered regression/improvement

# Grouping levels for summary (level name, group columns, sort priority)
SUMMARY_LEVELS = [
    ("Overall", [], 0),
    ("By Suite", ["suite"], 1),
    ("By Suite+DataType+Mode", ["suite", "data_type", "mode"], 2)
]

# ----------------------------------------------------------------------
# File discovery and parsing
# ----------------------------------------------------------------------
def find_result_files(root_dir: str) -> list[str]:
    """Recursively find all *performance.csv and *accuracy.csv files."""
    perf_files = glob(os.path.join(root_dir, "**", "*performance.csv"), recursive=True)
    acc_files = glob(os.path.join(root_dir, "**", "*accuracy.csv"), recursive=True)
    return perf_files + acc_files


def _parse_filename_components(filename: str) -> tuple[str, str, str, str]:
    """
    Extract suite, data_type, mode, result_type from a filename like
    inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
    """
    if not filename.endswith(".csv"):
        raise ValueError("Not a CSV file")

    if "inductor-results-" in filename:
        base = filename.split("inductor-results-")[-1][:-4]
        parts = base.split('-')
        if len(parts) >= 5:
            suite, data_type, mode, _, result_type = parts[:5]
            return suite, data_type, mode, result_type
        raise ValueError(f"inductor-results filename has too few parts: {filename}")

    base = filename[:-4]
    if not base.startswith("inductor"):
        raise ValueError(f"Filename does not start with 'inductor': {filename}")
    rest = base[len("inductor_"):]

    # Identify suite (longest match first)
    suite = None
    for s in sorted(KNOWN_SUITES, key=len, reverse=True):
        if rest.startswith(s + "_"):
            suite = s
            rest = rest[len(s) + 1:]
            break
    if suite is None:
        raise ValueError(f"Unknown suite in {filename}")

    # Identify mode
    parts = rest.split('_')
    mode_index = None
    for i, part in enumerate(parts):
        if part in KNOWN_MODES:
            mode_index = i
            break
    if mode_index is None:
        raise ValueError(f"Could not find mode (inference/training) in {filename}")
    mode = parts[mode_index]

    # Remaining part before mode is data_type
    data_type = "_".join(parts[:mode_index])
    if data_type not in KNOWN_DATA_TYPES:
        print(f"Warning: Unknown data_type '{data_type}' in {filename}")

    # After mode must be 'xpu' and then result type
    if mode_index + 1 >= len(parts) or parts[mode_index + 1] != "xpu":
        raise ValueError(f"Missing 'xpu' after mode in {filename}")
    if mode_index + 2 >= len(parts):
        raise ValueError(f"Missing result type in {filename}")
    result_type = parts[mode_index + 2]
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Result type not recognized in {filename}")

    return suite, data_type, mode, result_type


def _merge_accuracy_records(records: list[dict]) -> dict:
    """
    Merge multiple accuracy records for the same key.
    Prefer 'pass' over 'fail', otherwise take the first.
    """
    pass_recs = [r for r in records if 'pass' in str(r['accuracy'])]
    if pass_recs:
        return pass_recs[0]
    fail_recs = [r for r in records if 'fail' in str(r['accuracy'])]
    if fail_recs:
        return fail_recs[0]
    return records[0]


def _merge_performance_records(records: list[dict]) -> dict | None:
    """
    Merge multiple performance records for the same key.
    Prefer records with positive inductor/eager values, then choose the one with
    smallest (inductor, eager) as a tie‑breaker. If none, return the largest values.
    Returns None only if the list is empty (should not happen).
    """
    positive = [r for r in records if pd.notna(r['inductor']) and r['inductor'] > 0
                and pd.notna(r['eager']) and r['eager'] > 0]
    if positive:
        return min(positive, key=lambda r: (r['inductor'], r['eager']))
    if records:
        return max(records, key=lambda r: (r['inductor'] if pd.notna(r['inductor']) else -1,
                                            r['eager'] if pd.notna(r['eager']) else -1))
    return None


def load_results(file_list: list[str], result_type_filter: str) -> list[dict]:
    """
    Load all CSV files of a given result type, parse them, and merge duplicates.
    Returns a list of records (each a dict).
    """
    raw_by_key: dict[tuple, list[dict]] = {}

    # Define required columns based on type
    if result_type_filter == "accuracy":
        usecols = ['dev', 'name', 'batch_size', 'accuracy']
    else:  # performance
        usecols = ['dev', 'name', 'batch_size', 'speedup', 'abs_latency']

    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = _parse_filename_components(os.path.basename(fpath))
        except ValueError as e:
            print(f"Skipping {fpath}: {e}")
            continue
        if res_type != result_type_filter:
            continue

        try:
            # Try reading with modern pandas (skip bad lines)
            df = pd.read_csv(
                fpath,
                usecols=usecols,
                on_bad_lines='skip',
                engine='c',
                encoding='utf-8'
            )
        except Exception as e:
            # Fallback for older pandas or if engine='c' fails
            print(f"Warning: {fpath} - {e}, trying fallback parser...")
            try:
                df = pd.read_csv(
                    fpath,
                    usecols=usecols,
                    error_bad_lines=False,   # deprecated but works in older versions
                    warn_bad_lines=True,
                    engine='python',         # more robust for irregular quoting
                    encoding='utf-8'
                )
                print(f"Loaded {fpath} with fallback ({len(df)} rows)")
            except Exception as e2:
                print(f"Failed to read {fpath}: {e2}")
                continue

        for _, row in df.iterrows():
            dev_val = row.get("dev")
            if pd.isna(dev_val) or str(dev_val).strip() not in ['cpu', 'xpu', 'cuda']:
                continue

            if result_type_filter == "accuracy":
                record = {
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "accuracy": row["accuracy"]
                }
            else:  # performance
                speedup = row.get("speedup")
                abs_latency = row.get("abs_latency")

                if pd.isna(speedup) or pd.isna(abs_latency):
                    print(f"Warning: Missing speedup/abs_latency for {suite}/{data_type}/{mode}/{row.get('name')} in {fpath}, including with NaN.")
                    inductor = np.nan
                    eager = np.nan
                else:
                    eager = speedup * abs_latency
                    inductor = abs_latency

                record = {
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "eager": eager,
                    "inductor": inductor,
                    "speedup": speedup
                }

            key = (suite, data_type, mode, row["name"])
            raw_by_key.setdefault(key, []).append(record)

    # Merge duplicate keys
    merged = []
    for rec_list in raw_by_key.values():
        if result_type_filter == "accuracy":
            merged.append(_merge_accuracy_records(rec_list))
        else:
            m = _merge_performance_records(rec_list)
            if m is not None:
                merged.append(m)

    return merged


# ----------------------------------------------------------------------
# Merging target and baseline
# ----------------------------------------------------------------------
def merge_accuracy(target_records: list[dict], baseline_records: list[dict]) -> pd.DataFrame:
    """Merge target and baseline accuracy DataFrames and compute comparison column."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    # If both are empty, return empty DataFrame
    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    # If target is empty, all baseline entries become new_failed
    if target_df.empty:
        # Add target columns with None
        baseline_df["batch_size_target"] = pd.NA
        baseline_df["accuracy_target"] = pd.NA
        baseline_df["comparison_acc"] = "new_failed"
        # Rename baseline columns to match output schema
        baseline_df.rename(columns={
            "batch_size": "batch_size_baseline",
            "accuracy": "accuracy_baseline"
        }, inplace=True)
        # Ensure all expected columns exist
        for col in ["suite", "data_type", "mode", "model", "batch_size_target", "accuracy_target",
                    "batch_size_baseline", "accuracy_baseline", "comparison_acc"]:
            if col not in baseline_df.columns:
                baseline_df[col] = None
        # Select and order columns
        result = baseline_df[["suite", "data_type", "mode", "model",
                              "batch_size_target", "accuracy_target",
                              "batch_size_baseline", "accuracy_baseline",
                              "comparison_acc"]].copy()
        result["batch_size_target"] = pd.to_numeric(result["batch_size_target"], errors='coerce').astype('Int64')
        result["batch_size_baseline"] = pd.to_numeric(result["batch_size_baseline"], errors='coerce').astype('Int64')
        return result.sort_values(by=["suite", "data_type", "mode", "model"])

    # If baseline is empty, all target entries become new_passed
    if baseline_df.empty:
        target_df["batch_size_baseline"] = pd.NA
        target_df["accuracy_baseline"] = pd.NA
        target_df["comparison_acc"] = "new_passed"
        target_df.rename(columns={
            "batch_size": "batch_size_target",
            "accuracy": "accuracy_target"
        }, inplace=True)
        for col in ["suite", "data_type", "mode", "model", "batch_size_target", "accuracy_target",
                    "batch_size_baseline", "accuracy_baseline", "comparison_acc"]:
            if col not in target_df.columns:
                target_df[col] = None
        result = target_df[["suite", "data_type", "mode", "model",
                            "batch_size_target", "accuracy_target",
                            "batch_size_baseline", "accuracy_baseline",
                            "comparison_acc"]].copy()
        result["batch_size_target"] = pd.to_numeric(result["batch_size_target"], errors='coerce').astype('Int64')
        result["batch_size_baseline"] = pd.to_numeric(result["batch_size_baseline"], errors='coerce').astype('Int64')
        return result.sort_values(by=["suite", "data_type", "mode", "model"])

    # Both non-empty: perform merge as before
    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    # Convert batch_size to nullable integer
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    def _is_acc_pass(val):
        return pd.notna(val) and 'pass' in str(val)

    def _is_acc_fail(val):
        return pd.notna(val) and 'fail' in str(val)

    def compare_acc(row):
        tgt = row.get("accuracy_target")
        bsl = row.get("accuracy_baseline")
        tgt_pass = _is_acc_pass(tgt)
        bsl_pass = _is_acc_pass(bsl)

        # Both passed — no change
        if tgt_pass and bsl_pass:
            return "no_changed"
        # 1. Target passed, baseline didn't
        if tgt_pass:
            if _is_acc_fail(bsl):
                return "new_passed"
            return "new_case"
        # 2. Baseline passed, target didn't
        if bsl_pass:
            if _is_acc_fail(tgt):
                return "new_failed"
            return "not_run"
        # 3. Everything else
        return "no_changed"

    merged["comparison_acc"] = merged.apply(compare_acc, axis=1)

    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "accuracy_target",
            "batch_size_baseline", "accuracy_baseline",
            "comparison_acc"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])


def merge_performance(target_records: list[dict], baseline_records: list[dict],
                      threshold: float) -> pd.DataFrame:
    """Merge target and baseline performance DataFrames, compute ratios and comparison."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    # If target is empty, all baseline entries become new_failed
    if target_df.empty:
        baseline_df["batch_size_target"] = pd.NA
        baseline_df["inductor_target"] = pd.NA
        baseline_df["eager_target"] = pd.NA
        baseline_df["inductor_ratio"] = pd.NA
        baseline_df["eager_ratio"] = pd.NA
        baseline_df["comparison_perf"] = "new_failed"
        baseline_df.rename(columns={
            "batch_size": "batch_size_baseline",
            "inductor": "inductor_baseline",
            "eager": "eager_baseline"
        }, inplace=True)
        # Ensure all expected columns exist
        cols = ["suite", "data_type", "mode", "model",
                "batch_size_target", "inductor_target", "eager_target",
                "batch_size_baseline", "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison_perf"]
        for c in cols:
            if c not in baseline_df.columns:
                baseline_df[c] = None
        result = baseline_df[cols].copy()
        # Convert batch sizes to nullable integers
        for col in ["batch_size_target", "batch_size_baseline"]:
            result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')
        return result.sort_values(by=["suite", "data_type", "mode", "model"])

    # If baseline is empty, all target entries become new_passed
    if baseline_df.empty:
        target_df["batch_size_baseline"] = pd.NA
        target_df["inductor_baseline"] = pd.NA
        target_df["eager_baseline"] = pd.NA
        target_df["inductor_ratio"] = pd.NA
        target_df["eager_ratio"] = pd.NA
        target_df["comparison_perf"] = "new_passed"
        target_df.rename(columns={
            "batch_size": "batch_size_target",
            "inductor": "inductor_target",
            "eager": "eager_target"
        }, inplace=True)
        cols = ["suite", "data_type", "mode", "model",
                "batch_size_target", "inductor_target", "eager_target",
                "batch_size_baseline", "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison_perf"]
        for c in cols:
            if c not in target_df.columns:
                target_df[c] = None
        result = target_df[cols].copy()
        for col in ["batch_size_target", "batch_size_baseline"]:
            result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')
        return result.sort_values(by=["suite", "data_type", "mode", "model"])

    # Both non-empty: perform merge as before
    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    # inductor_ratio = baseline / target (if both positive)
    mask = merged["inductor_target"].notna() & (merged["inductor_target"] > 0)
    merged.loc[mask, "inductor_ratio"] = (
        merged.loc[mask, "inductor_baseline"].astype(float) /
        merged.loc[mask, "inductor_target"].astype(float)
    )

    mask = merged["eager_target"].notna() & (merged["eager_target"] > 0)
    merged.loc[mask, "eager_ratio"] = (
        merged.loc[mask, "eager_baseline"].astype(float) /
        merged.loc[mask, "eager_target"].astype(float)
    )

    # Round for readability
    for col in ["inductor_target", "inductor_baseline", "eager_target", "eager_baseline"]:
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    for col in ["inductor_ratio", "eager_ratio"]:
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    def _is_positive_number(val):
        return pd.notna(val) and isinstance(val, (int, float)) and val > 0

    def compare_perf(row):
        ind_tgt = row.get("inductor_target")
        ind_bsl = row.get("inductor_baseline")
        ind_ratio = row.get("inductor_ratio")
        eag_ratio = row.get("eager_ratio")

        tgt_ok = _is_positive_number(ind_tgt)
        bsl_ok = _is_positive_number(ind_bsl)

        # 1. Both are valid positive numbers — compare ratios
        if tgt_ok and bsl_ok:
            ind_drop = pd.notna(ind_ratio) and ind_ratio < 1 - threshold
            ind_improve = pd.notna(ind_ratio) and ind_ratio > 1 + threshold
            eag_drop = pd.notna(eag_ratio) and eag_ratio < 1 - threshold
            eag_improve = pd.notna(eag_ratio) and eag_ratio > 1 + threshold
            if ind_drop or eag_drop:
                return "new_dropped"
            if ind_improve or eag_improve:
                return "new_improved"
        # 2. Target valid, baseline not — new_passed
        if tgt_ok and not bsl_ok:
            return "new_passed"
        # 3. Baseline valid, target not — new_failed
        if bsl_ok and not tgt_ok:
            return "new_failed"
        # 4. Neither valid
        return "no_changed"

    merged["comparison_perf"] = merged.apply(compare_perf, axis=1)

    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "inductor_target", "eager_target",
            "batch_size_baseline", "inductor_baseline", "eager_baseline",
            "inductor_ratio", "eager_ratio", "comparison_perf"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])


# ----------------------------------------------------------------------
# Summary generation
# ----------------------------------------------------------------------
def _accuracy_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute accuracy summary metrics for a group."""
    def is_acc_pass(val):
        return pd.notna(val) and str(val) != "" and 'pass' in str(val)
    return pd.Series({
        'target_passed': group['accuracy_target'].apply(is_acc_pass).sum(),
        'baseline_passed': group['accuracy_baseline'].apply(is_acc_pass).sum(),
        'total': len(group),
        'new_failed': (group['comparison_acc'] == 'new_failed').sum(),
        'new_dropped': 0,
        'new_passed': (group['comparison_acc'] == 'new_passed').sum(),
        'new_improved': 0,
    })


def _performance_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute performance summary metrics for a group."""
    def is_perf_pass(val):
        return pd.notna(val) and val > 0

    def geomean(series):
        vals = series.dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            return np.nan
        return np.exp(np.log(vals).mean())

    return pd.Series({
        'target_passed': group['inductor_target'].apply(is_perf_pass).sum(),
        'baseline_passed': group['inductor_baseline'].apply(is_perf_pass).sum(),
        'total': len(group),
        'new_failed': (group['comparison_perf'] == 'new_failed').sum(),
        'new_dropped': (group['comparison_perf'] == 'new_dropped').sum(),
        'new_passed': (group['comparison_perf'] == 'new_passed').sum(),
        'new_improved': (group['comparison_perf'] == 'new_improved').sum(),
        'inductor_ratio_geomean': geomean(group['inductor_ratio']),
        'eager_ratio_geomean': geomean(group['eager_ratio']),
    })


def _compute_group_summary(acc_merged: pd.DataFrame, perf_merged: pd.DataFrame,
                           group_cols: list[str], level_name: str) -> pd.DataFrame:
    """Compute summary for one grouping level (Overall, By Suite, etc.)."""
    summaries = []
    if not acc_merged.empty:
        if not group_cols:
            acc_sum = acc_merged.assign(_dummy='Overall').groupby('_dummy').apply(_accuracy_metrics).reset_index(drop=True)
            acc_sum['Category'] = 'Overall'
        else:
            acc_sum = acc_merged.groupby(group_cols).apply(_accuracy_metrics).reset_index()
            acc_sum['Category'] = acc_sum[group_cols].astype(str).agg('_'.join, axis=1)
        acc_sum['Type'] = 'Accuracy'
        acc_sum['Level'] = level_name
        summaries.append(acc_sum)

    if not perf_merged.empty:
        if not group_cols:
            perf_sum = perf_merged.assign(_dummy='Overall').groupby('_dummy').apply(_performance_metrics).reset_index(drop=True)
            perf_sum['Category'] = 'Overall'
        else:
            perf_sum = perf_merged.groupby(group_cols).apply(_performance_metrics).reset_index()
            perf_sum['Category'] = perf_sum[group_cols].astype(str).agg('_'.join, axis=1)
        perf_sum['Type'] = 'Performance'
        perf_sum['Level'] = level_name
        summaries.append(perf_sum)

    if not summaries:
        return pd.DataFrame()

    combined = pd.concat(summaries, ignore_index=True, sort=False)
    combined['target passrate'] = combined['target_passed'] / combined['total']
    combined['baseline passrate'] = combined['baseline_passed'] / combined['total']
    combined.rename(columns={
        'target_passed': 'target passed',
        'baseline_passed': 'baseline passed',
        'new_failed': 'new_fail',
        'new_dropped': 'new_drop',
        'new_passed': 'new_pass',
        'new_improved': 'new_improve',
        'inductor_ratio_geomean': 'ind_ratio',
        'eager_ratio_geomean': 'eag_ratio'
    }, inplace=True)

    # Ensure all expected columns exist
    cols = ['Level', 'Type', 'Category', 'target passed', 'baseline passed', 'total',
            'target passrate', 'baseline passrate', 'new_fail', 'new_drop', 'new_pass',
            'new_improve', 'ind_ratio', 'eag_ratio']
    for col in cols:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[cols]


def generate_all_summaries(acc_merged: pd.DataFrame, perf_merged: pd.DataFrame) -> pd.DataFrame:
    """Generate all summary levels and combine into one DataFrame."""
    all_summaries = []
    for level_name, group_cols, priority in SUMMARY_LEVELS:
        df = _compute_group_summary(acc_merged, perf_merged, group_cols, level_name)
        if not df.empty:
            df['SortPriority'] = priority
            all_summaries.append(df)

    if not all_summaries:
        return pd.DataFrame()

    final = pd.concat(all_summaries, ignore_index=True, sort=False)
    # Convert counts to nullable integers
    for col in ['target passed', 'baseline passed', 'total', 'new_fail', 'new_drop', 'new_pass', 'new_improve']:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors='coerce').astype('Int64')
    # Convert passrates to percentages
    for col in ['target passrate', 'baseline passrate']:
        if col in final.columns:
            final[col] = (final[col] * 100).round(2)
    # Round ratios
    for col in ['ind_ratio', 'eag_ratio']:
        if col in final.columns:
            final[col] = final[col].round(3)

    final.sort_values(['SortPriority', 'Type', 'Category'], inplace=True)
    final.drop(columns=['Level', 'SortPriority'], errors='ignore', inplace=True)
    return final.reset_index(drop=True)


# ----------------------------------------------------------------------
# Markdown output
# ----------------------------------------------------------------------
def _fmt_ratio(val: Any, threshold: float) -> str:
    """Format ratio with emoji if below/above threshold."""
    if pd.notna(val):
        try:
            num = float(val)
            if num < 1 - threshold:
                return f"{val} 🔴"
            if num > 1 + threshold:
                return f"{val} 🟢"
        except (TypeError, ValueError):
            pass
    return str(val) if pd.notna(val) else ""


def write_summary_markdown(combined_summary: pd.DataFrame, threshold: float, filename: str):
    """Write a Markdown file with the Overall Summary table (split new_fail/drop and new_pass/improve)."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n\n# Dynamo Benchmark Test Results - Summary\n\n")

        overall = combined_summary[combined_summary['Category'] == 'Overall']
        if overall.empty:
            f.write("No overall summary data available.\n")
            return

        f.write("| Category | Total | Tgt / Bsl Pass | Tgt / Bsl Pass Rate | ❌ New Fail | 📉 New Drop | ✅ New Pass | 📈 New Improve | IND Ratio | EAG Ratio |\n")
        f.write("|----------|-------|----------------|---------------------|-------------|--------------|-------------|-----------------|-----------|-----------|\n")
        for _, row in overall.iterrows():
            type_label = row['Type']
            tgt_ps = row.get('target passed', '')
            bsl_ps = row.get('baseline passed', '')
            total = row.get('total', '')
            if type_label == 'Accuracy':
                # For accuracy, new_drop and new_improve are not applicable -> show "/"
                new_fail = row.get('new_fail')
                new_pass = row.get('new_pass')
                new_drop = new_improve = ind_ratio = eag_ratio = "/"
            else:
                new_fail = row.get('new_fail')
                new_pass = row.get('new_pass')
                new_drop = row.get('new_drop')
                new_improve = row.get('new_improve')
                ind_ratio = _fmt_ratio(row.get('ind_ratio'), threshold)
                eag_ratio = _fmt_ratio(row.get('eag_ratio'), threshold)
            tgt_pass = row.get('target passrate', '')
            bsl_pass = row.get('baseline passrate', '')
            f.write(f"| {type_label} | {total} | {tgt_ps} / {bsl_ps} | {tgt_pass}% / {bsl_pass}% | {new_fail} | {new_drop} | {new_pass} | {new_improve} | {ind_ratio} | {eag_ratio} |\n")
        f.write("\n")


def _write_html_table(rows: pd.DataFrame, columns: list[str], condition_col: str,
                      fail_color: str, pass_color: str, file_handle):
    """Write an HTML table with row background colors based on condition_col."""
    file_handle.write('<table>\n')
    file_handle.write('<thead>\n<tr>')
    for col in columns:
        file_handle.write(f'<th>{col}</th>')
    file_handle.write('</tr>\n</thead>\n')
    file_handle.write('<tbody>\n')
    for _, row in rows.iterrows():
        val = row.get(condition_col, '')
        bg_color = ''
        if val in ['new_failed', 'new_dropped']:
            bg_color = f' style="background-color: {fail_color};"'
        elif val in ['new_passed', 'new_improved']:
            bg_color = f' style="background-color: {pass_color};"'
        file_handle.write(f'<tr{bg_color}>')
        for col in columns:
            cell = html_escape(str(row.get(col, '')))
            file_handle.write(f'<td>{cell}</td>')
        file_handle.write('</tr>\n')
    file_handle.write('</tbody>\n')
    file_handle.write('</table>\n\n')


def write_details_markdown(acc_df: pd.DataFrame, perf_df: pd.DataFrame,
                           threshold: float, filename: str):
    """Write a Markdown file with suite overview and tables for new failures/improvements."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n\n# Dynamo Benchmark Test Results - Details\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ----- Overview by Suite -----
        f.write("## 📊 Overview by Suite\n\n")

        # Collect all suites from both dataframes
        suites = set()
        if not acc_df.empty:
            suites.update(acc_df['suite'].dropna().unique())
        if not perf_df.empty:
            suites.update(perf_df['suite'].dropna().unique())

        if not suites:
            f.write("No suite information available.\n\n")
        else:
            suite_rows = []
            for suite in sorted(suites):
                # Accuracy metrics for this suite
                if not acc_df.empty:
                    acc_sub = acc_df[acc_df['suite'] == suite]
                    acc_total = len(acc_sub)
                    acc_fail = (acc_sub['comparison_acc'] == 'new_failed').sum()
                    acc_pass = (acc_sub['comparison_acc'] == 'new_passed').sum()
                    # Absolute pass rate (target)
                    if 'accuracy_target' in acc_sub.columns:
                        acc_pass_rate = (
                            acc_sub['accuracy_target'].astype(str).str.contains('pass', na=False).sum() / acc_total * 100
                            if acc_total > 0 else 0
                        )
                        acc_pass_rate = f"{acc_pass_rate:.1f}%"
                    else:
                        acc_pass_rate = ""
                else:
                    acc_total = acc_fail = acc_pass = 0
                    acc_pass_rate = ""

                # Performance metrics for this suite
                if not perf_df.empty:
                    perf_sub = perf_df[perf_df['suite'] == suite]
                    perf_total = len(perf_sub)
                    perf_fail = (perf_sub['comparison_perf'] == 'new_failed').sum()
                    perf_drop = (perf_sub['comparison_perf'] == 'new_dropped').sum()
                    perf_pass = (perf_sub['comparison_perf'] == 'new_passed').sum()
                    perf_improve = (perf_sub['comparison_perf'] == 'new_improved').sum()
                    # Geometric means
                    ind_ratio = ""
                    if 'inductor_ratio' in perf_sub.columns:
                        vals = perf_sub['inductor_ratio'].dropna()
                        vals = vals[vals > 0]
                        if not vals.empty:
                            ind_ratio = f"{np.exp(np.log(vals).mean()):.3f}"
                    eag_ratio = ""
                    if 'eager_ratio' in perf_sub.columns:
                        vals = perf_sub['eager_ratio'].dropna()
                        vals = vals[vals > 0]
                        if not vals.empty:
                            eag_ratio = f"{np.exp(np.log(vals).mean()):.3f}"
                else:
                    perf_total = perf_fail = perf_drop = perf_pass = perf_improve = 0
                    ind_ratio = eag_ratio = ""

                suite_rows.append({
                    'suite': suite,
                    'acc_total': acc_total,
                    'acc_fail': acc_fail,
                    'acc_pass': acc_pass,
                    'acc_pass_rate': acc_pass_rate,
                    'perf_total': perf_total,
                    'perf_fail': perf_fail,
                    'perf_drop': perf_drop,
                    'perf_pass': perf_pass,
                    'perf_improve': perf_improve,
                    'ind_ratio': ind_ratio,
                    'eag_ratio': eag_ratio,
                })

            f.write("| Suite | Acc Total | ❌ Acc Fail | ✅ Acc Pass | Acc Pass Rate | Perf Total | ❌ Perf Fail | 📉 Perf Drop | ✅ Perf Pass | 📈 Perf Improve | Ind Ratio | Eag Ratio |\n")
            f.write("|-------|-----------|-------------|--------------|---------------|------------|--------------|---------------|---------------|-----------------|-----------|-----------|\n")
            for s in suite_rows:
                # Convert all values to strings to avoid TypeError
                row = [
                    str(s['suite']),
                    str(s['acc_total']),
                    str(s['acc_fail']),
                    str(s['acc_pass']),
                    str(s['acc_pass_rate']),
                    str(s['perf_total']),
                    str(s['perf_fail']),
                    str(s['perf_drop']),
                    str(s['perf_pass']),
                    str(s['perf_improve']),
                    _fmt_ratio(s['ind_ratio'], threshold),
                    _fmt_ratio(s['eag_ratio'], threshold),
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

        # ----- New Failures & Regressions -----
        f.write("## ❌ New Failures & Regressions\n\n")

        if not acc_df.empty:
            acc_fail = acc_df[acc_df['comparison_acc'] == 'new_failed']
            if not acc_fail.empty:
                f.write("### 🧪 Accuracy Failures\n\n")
                cols = ['suite', 'data_type', 'mode', 'model', 'batch_size_target', 'accuracy_target',
                        'batch_size_baseline', 'accuracy_baseline', 'comparison_acc']
                available = [c for c in cols if c in acc_fail.columns]
                _write_html_table(acc_fail, available, 'comparison_acc', "#f8d7da", "#d4edda", f)

        if not perf_df.empty:
            perf_regress = perf_df[perf_df['comparison_perf'].isin(['new_dropped', 'new_failed'])]
            if not perf_regress.empty:
                f.write(f"### ⏱️ Performance Regressions (ratio < {((1 - threshold) * 100):.0f}%)\n\n")
                cols = ['suite', 'data_type', 'mode', 'model', 'inductor_target', 'eager_target',
                        'inductor_baseline', 'eager_baseline', 'inductor_ratio', 'eager_ratio', 'comparison_perf']
                available = [c for c in cols if c in perf_regress.columns]
                _write_html_table(perf_regress, available, 'comparison_perf', "#f8d7da", "#d4edda", f)

        # ----- New Passes & Improvements -----
        f.write("## ✅ New Passes & Improvements\n\n")

        if not acc_df.empty:
            acc_pass = acc_df[acc_df['comparison_acc'] == 'new_passed']
            if not acc_pass.empty:
                f.write("### 🧪 Accuracy New Passes\n\n")
                cols = ['suite', 'data_type', 'mode', 'model', 'batch_size_target', 'accuracy_target',
                        'batch_size_baseline', 'accuracy_baseline', 'comparison_acc']
                available = [c for c in cols if c in acc_pass.columns]
                _write_html_table(acc_pass, available, 'comparison_acc', "#f8d7da", "#d4edda", f)

        if not perf_df.empty:
            perf_impr = perf_df[perf_df['comparison_perf'].isin(['new_improved', 'new_passed'])]
            if not perf_impr.empty:
                f.write(f"### ⏱️ Performance Improvements (ratio > {(1 + threshold) * 100:.0f}%)\n\n")
                cols = ['suite', 'data_type', 'mode', 'model', 'inductor_target', 'eager_target',
                        'inductor_baseline', 'eager_baseline', 'inductor_ratio', 'eager_ratio', 'comparison_perf']
                available = [c for c in cols if c in perf_impr.columns]
                _write_html_table(perf_impr, available, 'comparison_perf', "#f8d7da", "#d4edda", f)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch inductor test results with baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -t target_dir -b baseline_dir -o comparison.xlsx
  %(prog)s -t target_dir -o comparison.xlsx   # only target, baseline treated as empty
  %(prog)s -b baseline_dir -o comparison.csv  # only baseline, target treated as empty
  %(prog)s -t target_dir -b baseline_dir -o comparison.xlsx -m report --threshold 0.15
        """
    )
    parser.add_argument("-t", "--target_dir", help="Directory containing target (test) result CSV files (searched recursively)")
    parser.add_argument("-b", "--baseline_dir", help="Directory containing baseline (reference) result CSV files (searched recursively)")
    parser.add_argument("-o", "--output", required=True, help="Output file name (without extension). Use .xlsx for Excel, .csv for CSV files.")
    parser.add_argument("-m", "--markdown", help="Base name for Markdown reports (e.g., report -> report.summary.md, report.details.md)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PERFORMANCE_THRESHOLD,
                        help=f"Performance change threshold (default: {DEFAULT_PERFORMANCE_THRESHOLD})")
    args = parser.parse_args()

    # At least one of target_dir or baseline_dir must be provided
    if not args.target_dir and not args.baseline_dir:
        print("Error: At least one of -t/--target_dir or -b/--baseline_dir must be provided.")
        return 1

    out_base, out_ext = os.path.splitext(args.output)
    if out_ext not in ('.xlsx', '.csv'):
        print("Output file must end with .xlsx or .csv")
        return 1

    # Load target data if directory provided and exists
    target_acc = []
    target_perf = []
    if args.target_dir:
        if not os.path.isdir(args.target_dir):
            print(f"Error: Target directory '{args.target_dir}' does not exist.")
            return 1
        target_files = find_result_files(args.target_dir)
        print(f"Found {len(target_files)} CSV files in target directory.")
        target_acc = load_results(target_files, "accuracy")
        target_perf = load_results(target_files, "performance")
    else:
        print("No target directory provided. Treating target as empty.")

    # Load baseline data if directory provided and exists
    baseline_acc = []
    baseline_perf = []
    if args.baseline_dir:
        if not os.path.isdir(args.baseline_dir):
            print(f"Error: Baseline directory '{args.baseline_dir}' does not exist.")
            return 1
        baseline_files = find_result_files(args.baseline_dir)
        print(f"Found {len(baseline_files)} CSV files in baseline directory.")
        baseline_acc = load_results(baseline_files, "accuracy")
        baseline_perf = load_results(baseline_files, "performance")
    else:
        print("No baseline directory provided. Treating baseline as empty.")

    acc_merged = merge_accuracy(target_acc, baseline_acc)
    perf_merged = merge_performance(target_perf, baseline_perf, args.threshold)

    failed_acc = acc_merged.query("comparison_acc == 'new_failed'").copy()
    failed_perf = perf_merged.query("comparison_perf == 'new_failed'").copy()
    passed_acc = acc_merged.query("comparison_acc == 'new_passed'").copy()
    passed_perf = perf_merged.query("comparison_perf == 'new_passed'").copy()
    dropped_cases = perf_merged.query("comparison_perf == 'new_dropped'").copy()
    improved_cases = perf_merged.query("comparison_perf == 'new_improved'").copy()

    failed_cases = pd.concat([failed_acc, failed_perf], ignore_index=True)
    passed_cases = pd.concat([passed_acc, passed_perf], ignore_index=True)

    combined_summary = generate_all_summaries(acc_merged, perf_merged)

    # Generate markdown files if requested
    if args.markdown:
        md_base = os.path.splitext(args.markdown)[0]  # remove any extension
        summary_file = md_base + ".summary.md"
        details_file = md_base + ".details.md"
        write_summary_markdown(combined_summary, args.threshold, summary_file)
        write_details_markdown(acc_merged, perf_merged, args.threshold, details_file)
        print(f"Markdown summary written to {summary_file}")
        print(f"Markdown details written to {details_file}")

    # Write output files (Excel or CSV)
    if out_ext == '.xlsx':
        summary_file = args.output
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            # Summary sheet
            if not combined_summary.empty:
                combined_summary.to_excel(writer, sheet_name="Summary", index=False)
            else:
                pd.DataFrame({"Info": ["No summary data available"]}).to_excel(writer, sheet_name="Summary", index=False)

            # Accuracy details sheet
            if not acc_merged.empty:
                acc_merged.to_excel(writer, sheet_name="Accuracy Details", index=False)
            else:
                pd.DataFrame({"Info": ["No accuracy data available"]}).to_excel(writer, sheet_name="Accuracy Details", index=False)

            # Performance details sheet
            if not perf_merged.empty:
                perf_merged.to_excel(writer, sheet_name="Performance Details", index=False)
            else:
                pd.DataFrame({"Info": ["No performance data available"]}).to_excel(writer, sheet_name="Performance Details", index=False)

        print(f"Excel written to {args.output} (sheets: Summary, Accuracy Details, Performance Details)")

    else:  # .csv
        summary_file = out_base + "_summary.csv"
        if not combined_summary.empty:
            combined_summary.to_csv(summary_file, index=False, na_rep='')
        else:
            print("No summary data to write.")

        # Accuracy details CSV
        acc_file = out_base + "_accuracy_details.csv"
        if not acc_merged.empty:
            acc_merged.to_csv(acc_file, index=False, na_rep='')
            print(f"Accuracy details written to {acc_file}")
        else:
            print("No accuracy data to write.")

        # Performance details CSV
        perf_file = out_base + "_performance_details.csv"
        if not perf_merged.empty:
            perf_merged.to_csv(perf_file, index=False, na_rep='')
            print(f"Performance details written to {perf_file}")
        else:
            print("No performance data to write.")

    print("\n" + "=" * 60)
    print(" " * 20 + "E2E SUMMARY")
    print("=" * 60)
    label_width = 30
    print(f"{'🎯 Target accuracy:':<{label_width}} {len(target_acc):>6}")
    print(f"{'⚡ Target performance:':<{label_width}} {len(target_perf):>6}")
    print("-" * 60)
    print(f"{'📊 Baseline accuracy:':<{label_width}} {len(baseline_acc):>6}")
    print(f"{'⏱️ Baseline performance:':<{label_width}} {len(baseline_perf):>6}")
    print("-" * 60)
    print(f"{'❌ Failed cases:':<{label_width}} {len(failed_cases):>6}")
    print(f"{'✅ Passed cases:':<{label_width}} {len(passed_cases):>6}")
    print(f"{'🗑️ Dropped cases:':<{label_width}} {len(dropped_cases):>6}")
    print(f"{'📈 Improved cases:':<{label_width}} {len(improved_cases):>6}")
    print("-" * 60)
    print(f"{'📁 Output file:':<{label_width}} {summary_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
