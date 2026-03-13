#!/usr/bin/env python3
"""
Enhanced comparison tool for PyTorch inductor test results (target vs baseline).
Recursively finds all *_performance.csv and *_accuracy.csv files,
validates known suites, data types, and modes,
merges data by suite, data_type, mode, model,
and writes comparison to Excel (two sheets: Summary, Details) or CSV (two files).

All missing cells are filled with empty strings.
If performance files are missing, performance columns are omitted.
If accuracy files are missing, accuracy columns are omitted.

New features:
- Shortened column headers in output files for brevity.
- Optional `--markdown` argument to generate a GitHub‑flavored Markdown summary report.
"""

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

# ----------------------------------------------------------------------
# Constants – known valid values
# ----------------------------------------------------------------------
KNOWN_SUITES = {"huggingface", "timm_models", "torchbench"}
KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
KNOWN_MODES = {"inference", "training"}
PERFORMANCE_THRESHOLD = 0.1

# Grouping levels for summary (name, group_cols, sort_priority)
SUMMARY_LEVELS = [
    ("Overall", [], 0),
    ("By Suite", ["suite"], 1),
    ("By Suite+DataType+Mode", ["suite", "data_type", "mode"], 2)
]

# ----------------------------------------------------------------------
# Column renaming mapping for output (shortened headers)
# ----------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    # Common
    'data_type': 'dtype',
    # Accuracy details
    'batch_size_target': 'bs_tgt',
    'batch_size_baseline': 'bs_bsl',
    'accuracy_target': 'acc_tgt',
    'accuracy_baseline': 'acc_bsl',
    'comparison_acc': 'cmp_acc',
    # Performance details
    'inductor_target': 'ind_tgt',
    'inductor_baseline': 'ind_bsl',
    'eager_target': 'eag_tgt',
    'eager_baseline': 'eag_bsl',
    'inductor_ratio': 'ind_ratio',
    'eager_ratio': 'eag_ratio',
    'comparison_perf': 'cmp_perf',
    'comparison': 'cmp',
    # Summary specific (with spaces)
    'target passed': 'tgt_ps',
    'baseline passed': 'bsl_ps',
    'total': 'total',
    'target passrate': 'tgt_pass%',
    'baseline passrate': 'bsl_pass%',
    'New failed': 'new_fail',
    'New passed': 'new_pass',
    'inductor ratio': 'ind_ratio',      # from summary
    'eager ratio': 'eag_ratio',          # from summary
    # Grouping columns (kept as-is, but we can map if desired)
    # 'suite' -> 'suite' (unchanged)
    # 'mode' -> 'mode' (unchanged)
    # 'model' -> 'model' (unchanged)
    # 'Category' -> 'Category' (unchanged)
    # 'Type' -> 'Type' (unchanged)
}

# ----------------------------------------------------------------------
# File discovery and parsing (unchanged)
# ----------------------------------------------------------------------
def find_result_files(root_dir):
    """
    Recursively find all files ending with _performance.csv or _accuracy.csv.
    Returns a list of absolute file paths.
    """
    perf_files = glob(os.path.join(root_dir, "**", "*_performance.csv"), recursive=True)
    acc_files = glob(os.path.join(root_dir, "**", "*_accuracy.csv"), recursive=True)
    return perf_files + acc_files


def parse_filename(filepath):
    """
    Extract suite, data_type, mode, result_type from a filename.
    Expected format: inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
    Example: inductor_timm_models_bfloat16_training_xpu_accuracy.csv
    Returns (suite, data_type, mode, result_type) or raises ValueError.
    """
    basename = os.path.basename(filepath)
    if not basename.endswith(".csv"):
        raise ValueError("Not a CSV file")

    base = basename[:-4]  # remove .csv

    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor_': {basename}")
    rest = base[len("inductor_"):]

    # Identify suite (check longer first)
    suite = None
    for s in sorted(KNOWN_SUITES, key=len, reverse=True):
        if rest.startswith(s + "_"):
            suite = s
            rest = rest[len(s) + 1:]
            break
    if suite is None:
        raise ValueError(f"Unknown suite in {basename}")

    # Locate mode (inference/training)
    parts = rest.split('_')
    mode_index = None
    for i, part in enumerate(parts):
        if part in KNOWN_MODES:
            mode_index = i
            break
    if mode_index is None:
        raise ValueError(f"Could not find mode (inference/training) in {basename}")
    mode = parts[mode_index]

    # Data type is everything before mode
    data_type = "_".join(parts[:mode_index])
    if data_type not in KNOWN_DATA_TYPES:
        print(f"Warning: Unknown data_type '{data_type}' in {basename}")

    # After mode, expect "xpu" then result_type
    if mode_index + 1 >= len(parts) or parts[mode_index + 1] != "xpu":
        raise ValueError(f"Missing 'xpu' after mode in {basename}")
    if mode_index + 2 >= len(parts):
        raise ValueError(f"Missing result type in {basename}")
    result_type = parts[mode_index + 2]
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Result type not recognized in {basename}")

    return suite, data_type, mode, result_type


def load_results(file_list, result_type_filter):
    """
    Load all files of a given result_type (accuracy or performance),
    merge duplicates by (suite, data_type, mode, model) according to rules,
    and return a list of merged records.
    """
    # Helper merge functions
    def merge_accuracy(records):
        # records: list of dicts with 'accuracy', 'batch_size'
        pass_recs = [r for r in records if 'pass' in str(r['accuracy'])]
        if pass_recs:
            return pass_recs[0]
        fail_recs = [r for r in records if 'fail' in str(r['accuracy'])]
        if fail_recs:
            return fail_recs[0]
        return records[0]  # fallback

    def merge_performance(records):
        # records: list of dicts with 'inductor', 'eager', 'batch_size'
        # 1. Prefer positive records (valid latencies)
        positive = [r for r in records if r['inductor'] > 0 and r['eager'] > 0]
        if positive:
            # Choose record with smallest inductor (best performance), tie‑break by smallest eager
            return min(positive, key=lambda r: (r['inductor'], r['eager']))

        # 2. No positive records – keep the one with largest inductor (closest to zero)
        if records:
            # Among non‑positive, the largest inductor is the least negative (or zero)
            return max(records, key=lambda r: (r['inductor'], r['eager']))

        return None  # should never happen because records is non‑empty when called

    # Temporary dictionary to collect all raw records by key
    raw_by_key = {}

    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = parse_filename(fpath)
        except ValueError as e:
            print(f"Skipping {fpath}: {e}")
            continue

        if res_type != result_type_filter:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        for _, row in df.iterrows():
            if row["dev"].strip() not in ['cpu', 'xpu', 'cuda']:
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
                    print(f"Warning: Missing speedup/abs_latency for {suite}/{data_type}/{mode}/{row.get('name')} in {fpath}")
                    continue
                eager = speedup * abs_latency
                inductor = abs_latency
                record = {
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "inductor": inductor,
                    "eager": eager
                }

            key = (suite, data_type, mode, row["name"])
            raw_by_key.setdefault(key, []).append(record)

    # Merge duplicates per key
    merged = []
    for key, rec_list in raw_by_key.items():
        if result_type_filter == "accuracy":
            merged.append(merge_accuracy(rec_list))
        else:  # performance
            m = merge_performance(rec_list)
            if m is not None:
                merged.append(m)

    return merged


# ----------------------------------------------------------------------
# Merging functions (unchanged except for minor improvements)
# ----------------------------------------------------------------------
def merge_accuracy(target_records, baseline_records):
    """Merge accuracy records and add comparison column."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    # Convert batch_size to nullable integer
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    # Comparison logic
    def compare_acc(row):
        if pd.isna(row.get("accuracy_target")) and pd.isna(row.get("accuracy_baseline")):
            return ""
        elif pd.isna(row.get("accuracy_target")):
            return "new_failed"
        elif pd.isna(row.get("accuracy_baseline")):
            return "new_passed"
        elif 'pass' not in row["accuracy_target"] and 'pass' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'fail_accuracy' not in row["accuracy_target"] and 'fail_accuracy' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'pass' in row["accuracy_target"] and 'pass' not in row["accuracy_baseline"]:
            return "new_passed"
        elif 'fail_accuracy' in row["accuracy_target"] and 'fail_accuracy' not in row["accuracy_baseline"]:
            return "new_passed"
        else:
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


def merge_performance(target_records, baseline_records):
    """Merge performance records and compute ratios."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    # Compute inductor_ratio
    mask = merged["inductor_target"].notna() & (merged["inductor_target"].astype(float) > 0)
    merged.loc[mask, "inductor_ratio"] = (
        merged.loc[mask, "inductor_baseline"].astype(float) /
        merged.loc[mask, "inductor_target"].astype(float)
    )
    # Compute eager_ratio
    mask = merged["eager_target"].notna() & (merged["eager_target"].astype(float) > 0)
    merged.loc[mask, "eager_ratio"] = (
        merged.loc[mask, "eager_baseline"].astype(float) /
        merged.loc[mask, "eager_target"].astype(float)
    )

    # Round
    for col in ["inductor_target", "inductor_baseline", "eager_target", "eager_baseline"]:
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    for col in ["inductor_ratio", "eager_ratio"]:
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    def compare_perf(row):
        if pd.isna(row.get("inductor_target")) and pd.isna(row.get("inductor_baseline")):
            return ""
        elif row["inductor_ratio"] < 0 and row["inductor_baseline"] < 0:
            return ""
        elif pd.isna(row.get("inductor_target")) or row["inductor_ratio"] < 0:
            return "new_failed"
        elif pd.isna(row.get("inductor_baseline")) or row["inductor_baseline"] < 0:
            return "new_passed"
        elif row["inductor_ratio"] < 1 - PERFORMANCE_THRESHOLD or row["eager_ratio"] < 1 - PERFORMANCE_THRESHOLD:
            return "new_dropped"
        elif row["inductor_ratio"] > 1 + PERFORMANCE_THRESHOLD or row["eager_ratio"] > 1 + PERFORMANCE_THRESHOLD:
            return "new_improved"
        else:
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


def combine_results(acc_merged, perf_merged):
    """
    Combine accuracy and performance merged dataframes into one wide-format dataframe.
    If one type is missing, return only the available data (with renamed batch size columns).
    """
    acc_renamed = None
    perf_renamed = None

    if not acc_merged.empty:
        acc_renamed = acc_merged.rename(columns={
            "batch_size_target": "batch_size_accuracy_target",
            "batch_size_baseline": "batch_size_accuracy_baseline"
        })

    if not perf_merged.empty:
        perf_renamed = perf_merged.rename(columns={
            "batch_size_target": "batch_size_performance_target",
            "batch_size_baseline": "batch_size_performance_baseline"
        })

    if acc_renamed is not None and perf_renamed is not None:
        merge_keys = ["suite", "data_type", "mode", "model"]
        combined = pd.merge(acc_renamed, perf_renamed, on=merge_keys, how="outer")

        def compare_result(row):
            if pd.isna(row.get("comparison_acc")) and pd.isna(row.get("comparison_perf")):
                return ""
            elif pd.isna(row.get("comparison_acc")):
                return row.get("comparison_perf")
            elif pd.isna(row.get("comparison_perf")):
                return row.get("comparison_acc")
            elif ('new_dropped' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                  'new_failed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_failed"
            elif ('new_improved' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                  'new_passed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_improved"
            else:
                return "no_changed"

        combined["comparison"] = combined.apply(compare_result, axis=1)
        return combined.sort_values(by=merge_keys)

    elif acc_renamed is not None:
        return acc_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    elif perf_renamed is not None:
        return perf_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    else:
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Summary generation (enhanced with multiple grouping levels)
# ----------------------------------------------------------------------
def accuracy_metrics(group):
    """Compute accuracy metrics for a grouped DataFrame."""
    def is_acc_pass(val):
        return pd.notna(val) and str(val) != "" and 'pass' in str(val)

    return pd.Series({
        'target_passed': group['accuracy_target'].apply(is_acc_pass).sum(),
        'baseline_passed': group['accuracy_baseline'].apply(is_acc_pass).sum(),
        'total': len(group),
        'new_failed': (group['comparison_acc'] == 'new_failed').sum(),
        'new_passed': (group['comparison_acc'] == 'new_passed').sum(),
    })


def performance_metrics(group):
    """Compute performance metrics for a grouped DataFrame."""
    def is_perf_pass(val):
        return pd.notna(val) and str(val) != "" and int(val) > 0

    def geomean(series):
        vals = series.replace("", np.nan).replace(0, np.nan).dropna()
        if len(vals) == 0:
            return np.nan
        return np.exp(np.log(vals).mean())

    return pd.Series({
        'target_passed': group['inductor_target'].apply(is_perf_pass).sum(),
        'baseline_passed': group['inductor_baseline'].apply(is_perf_pass).sum(),
        'total': len(group),
        'new_failed': ((group['comparison_perf'] == 'new_failed') | (group['comparison_perf'] == 'new_dropped')).sum(),
        'new_passed': ((group['comparison_perf'] == 'new_passed') | (group['comparison_perf'] == 'new_improved')).sum(),
        'inductor_ratio_geomean': geomean(group['inductor_ratio']),
        'eager_ratio_geomean': geomean(group['eager_ratio']),
    })


def compute_group_summary(acc_merged, perf_merged, group_cols, level_name):
    """
    Generate a summary DataFrame for a given grouping level.
    Returns a DataFrame with columns:
        Level (internal), Type, Category, target passed, baseline passed, total,
        target passrate, baseline passrate, New failed, New passed,
        inductor ratio, eager ratio
    """
    summaries = []

    # Accuracy summary
    if not acc_merged.empty:
        if not group_cols:
            # Overall: create a dummy column
            acc_sum = acc_merged.assign(_dummy='Overall').groupby('_dummy').apply(accuracy_metrics).reset_index(drop=True)
            acc_sum['Category'] = 'Overall'
        else:
            acc_sum = acc_merged.groupby(group_cols).apply(accuracy_metrics).reset_index()
            # Create Category by concatenating group values
            acc_sum['Category'] = acc_sum[group_cols].astype(str).agg('_'.join, axis=1)
        acc_sum['Type'] = 'Accuracy'
        acc_sum['Level'] = level_name
        summaries.append(acc_sum)

    # Performance summary
    if not perf_merged.empty:
        if not group_cols:
            perf_sum = perf_merged.assign(_dummy='Overall').groupby('_dummy').apply(performance_metrics).reset_index(drop=True)
            perf_sum['Category'] = 'Overall'
        else:
            perf_sum = perf_merged.groupby(group_cols).apply(performance_metrics).reset_index()
            perf_sum['Category'] = perf_sum[group_cols].astype(str).agg('_'.join, axis=1)
        perf_sum['Type'] = 'Performance'
        perf_sum['Level'] = level_name
        summaries.append(perf_sum)

    if not summaries:
        return pd.DataFrame()

    # Combine accuracy and performance rows
    combined = pd.concat(summaries, ignore_index=True, sort=False)

    # Compute pass rates
    combined['target passrate'] = combined['target_passed'] / combined['total']
    combined['baseline passrate'] = combined['baseline_passed'] / combined['total']

    # Rename columns for output
    combined.rename(columns={
        'target_passed': 'target passed',
        'baseline_passed': 'baseline passed',
        'new_failed': 'New failed',
        'new_passed': 'New passed',
        'inductor_ratio_geomean': 'inductor ratio',
        'eager_ratio_geomean': 'eager ratio'
    }, inplace=True)

    # Reorder columns (keep Level for now)
    cols = ['Level', 'Type', 'Category', 'target passed', 'baseline passed', 'total',
            'target passrate', 'baseline passrate', 'New failed', 'New passed',
            'inductor ratio', 'eager ratio']
    for col in cols:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[cols]


def generate_all_summaries(acc_merged, perf_merged):
    """
    Generate summary dataframes for all predefined grouping levels.
    Returns a single concatenated DataFrame with hierarchical ordering and formatted numbers.
    """
    all_summaries = []
    for level_name, group_cols, priority in SUMMARY_LEVELS:
        df = compute_group_summary(acc_merged, perf_merged, group_cols, level_name)
        if not df.empty:
            df['SortPriority'] = priority
            all_summaries.append(df)

    if not all_summaries:
        return pd.DataFrame()

    final = pd.concat(all_summaries, ignore_index=True, sort=False)

    # Convert counts to nullable integers
    for col in ['target passed', 'baseline passed', 'total', 'New failed', 'New passed']:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors='coerce').astype('Int64')

    # Convert pass rates to percentages (multiply by 100) and round to 2 decimals
    for col in ['target passrate', 'baseline passrate']:
        if col in final.columns:
            final[col] = (final[col] * 100).round(2)

    # Ratios already rounded to 3 decimals in merge_performance, but ensure
    for col in ['inductor ratio', 'eager ratio']:
        if col in final.columns:
            final[col] = final[col].round(3)

    # Sort by priority, then Type, then Category
    final.sort_values(['SortPriority', 'Type', 'Category'], inplace=True)

    # Drop internal columns
    final.drop(columns=['Level', 'SortPriority'], inplace=True, errors='ignore')

    return final.reset_index(drop=True)


# ----------------------------------------------------------------------
# Markdown summary generation (new)
# ----------------------------------------------------------------------
def generate_markdown_summary(combined_summary, details, markdown_file):
    """
    Write a GitHub‑flavored Markdown report with:
    - Emojis outside tables for section headings.
    - Plain Markdown tables for per‑suite and overall summaries, using emoji indicators.
    - HTML tables with row background colors for new failures/passes (for easy scanning).
    """
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(f"# Inductor Test Results Comparison: Target vs Baseline\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ----- Quick overview per suite (plain Markdown, with emoji indicators) -----
        if not details.empty and 'suite' in details.columns:
            f.write("## 📊 Overview by Suite\n\n")
            suite_rows = []
            for suite in details['suite'].dropna().unique():
                suite_df = details[details['suite'] == suite]
                # Accuracy counts
                acc_fail = 0
                if 'cmp_acc' in suite_df.columns:
                    acc_fail = (suite_df['cmp_acc'] == 'new_failed').sum()
                acc_pass = 0
                if 'cmp_acc' in suite_df.columns:
                    acc_pass = (suite_df['cmp_acc'] == 'new_passed').sum()
                # Performance counts
                perf_fail = 0
                perf_drop = 0
                perf_pass = 0
                perf_improve = 0
                if 'cmp_perf' in suite_df.columns:
                    perf_fail = (suite_df['cmp_perf'] == 'new_failed').sum()
                    perf_drop = (suite_df['cmp_perf'] == 'new_dropped').sum()
                    perf_pass = (suite_df['cmp_perf'] == 'new_passed').sum()
                    perf_improve = (suite_df['cmp_perf'] == 'new_improved').sum()
                suite_rows.append({
                    'suite': suite,
                    'acc_fail': acc_fail,
                    'acc_pass': acc_pass,
                    'perf_fail': perf_fail,
                    'perf_drop': perf_drop,
                    'perf_pass': perf_pass,
                    'perf_improve': perf_improve
                })

            if suite_rows:
                # Helper to format a number with emoji if >0
                def fmt_count(val, good=False):
                    if val > 0:
                        emoji = "🟢" if good else "🔴"
                        return f"{val} {emoji}"
                    return str(val)

                f.write("| Suite | 🧪 Acc Fail | 🧪 Acc Pass | ⏱️ Perf Fail | ⏱️ Perf Drop | ⏱️ Perf Pass | ⏱️ Perf Improve |\n")
                f.write("|-------|-------------|-------------|---------------|---------------|---------------|-----------------|\n")
                for s in suite_rows:
                    row = [
                        s['suite'],
                        fmt_count(s['acc_fail'], good=False),
                        fmt_count(s['acc_pass'], good=True),
                        fmt_count(s['perf_fail'], good=False),
                        fmt_count(s['perf_drop'], good=False),
                        fmt_count(s['perf_pass'], good=True),
                        fmt_count(s['perf_improve'], good=True)
                    ]
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")
            else:
                f.write("No suite information available.\n\n")
        else:
            f.write("## 📊 Overview by Suite\n\nNo suite information available.\n\n")

        # ----- Overall summary (plain Markdown, with emoji indicators for thresholds) -----
        f.write("## 📈 Overall Summary\n\n")
        overall = combined_summary[combined_summary['Category'] == 'Overall']
        if not overall.empty:
            # Helper to add emoji for new_fail/new_pass
            def fmt_new(val, is_fail=True):
                if pd.notna(val) and val > 0:
                    emoji = "🔴" if is_fail else "🟢"
                    return f"{val} {emoji}"
                return str(val) if pd.notna(val) else ""

            # Helper to add emoji for ratios based on thresholds
            def fmt_ratio(val):
                if pd.notna(val):
                    if val < 0.95:
                        return f"{val} 🔴"
                    elif val > 1.05:
                        return f"{val} 🟢"
                return str(val) if pd.notna(val) else ""

            f.write("| Type | tgt_ps | bsl_ps | total | new_fail | new_pass | tgt_pass% | bsl_pass% | ind_ratio | eag_ratio |\n")
            f.write("|------|--------|--------|-------|----------|----------|-----------|-----------|-----------|-----------|\n")
            for _, row in overall.iterrows():
                type_label = "Accuracy" if row['Type'] == 'Accuracy' else "Performance"
                tgt_ps = row.get('tgt_ps', '')
                bsl_ps = row.get('bsl_ps', '')
                total = row.get('total', '')
                new_fail = fmt_new(row.get('new_fail'), is_fail=True)
                new_pass = fmt_new(row.get('new_pass'), is_fail=False)
                tgt_pass = row.get('tgt_pass%', '')
                bsl_pass = row.get('bsl_pass%', '')
                ind_ratio = fmt_ratio(row.get('ind_ratio'))
                eag_ratio = fmt_ratio(row.get('eag_ratio'))
                f.write(f"| {type_label} | {tgt_ps} | {bsl_ps} | {total} | {new_fail} | {new_pass} | {tgt_pass} | {bsl_pass} | {ind_ratio} | {eag_ratio} |\n")
            f.write("\n")
        else:
            f.write("No overall summary data available.\n\n")

        if details.empty:
            f.write("No detailed data available.\n")
            return

        # ----- Helper to generate an HTML table with row background colors -----
        def write_html_table(rows, columns, condition_column, fail_color="#f8d7da", pass_color="#d4edda"):
            """
            Write an HTML table with rows colored based on the value in condition_column.
            - rows: DataFrame subset
            - columns: list of column names to display
            - condition_column: column used to determine color (e.g., 'cmp_acc')
            - fail_color: background color for "new_failed"/"new_dropped"
            - pass_color: background color for "new_passed"/"new_improved"
            """
            f.write('<table>\n')
            f.write('<thead><tr>')
            for col in columns:
                f.write(f'<th>{col}</th>')
            f.write('</tr></thead>\n')
            f.write('<tbody>\n')
            for _, row in rows.iterrows():
                val = row.get(condition_column, '')
                bg_color = ''
                if val in ['new_failed', 'new_dropped']:
                    bg_color = f' style="background-color: {fail_color};"'
                elif val in ['new_passed', 'new_improved']:
                    bg_color = f' style="background-color: {pass_color};"'
                f.write(f'<tr{bg_color}>')
                for col in columns:
                    cell = str(row.get(col, ''))
                    f.write(f'<td>{cell}</td>')
                f.write('</tr>\n')
            f.write('</tbody>\n')
            f.write('</table>\n\n')

        # ----- New Failures -----
        f.write("## ❌ New Failures & Regressions\n\n")

        # Accuracy failures
        acc_fail = details[details['cmp_acc'] == 'new_failed']
        if not acc_fail.empty:
            f.write("### 🧪 Accuracy Failures\n\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'bs_acc_tgt', 'acc_tgt', 'bs_acc_bsl', 'acc_bsl', 'cmp_acc']
            available = [c for c in cols if c in acc_fail.columns]
            write_html_table(acc_fail, available, 'cmp_acc', fail_color="#f8d7da")
        else:
            f.write("✅ No new accuracy failures.\n\n")

        # Performance regressions (new_failed + new_dropped)
        perf_regress = details[details['cmp_perf'].isin(['new_dropped', 'new_failed'])]
        if not perf_regress.empty:
            f.write("### ⏱️ Performance Regressions (ratio < {:.0f}%)\n\n".format((1 - PERFORMANCE_THRESHOLD) * 100))
            cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
            available = [c for c in cols if c in perf_regress.columns]
            write_html_table(perf_regress, available, 'cmp_perf', fail_color="#f8d7da")
        else:
            f.write("✅ No performance regressions.\n\n")

        # ----- New Passes / Improvements -----
        f.write("## ✅ New Passes & Improvements\n\n")

        # Accuracy new passes
        acc_pass = details[details['cmp_acc'] == 'new_passed']
        if not acc_pass.empty:
            f.write("### 🧪 Accuracy New Passes\n\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'bs_acc_tgt', 'acc_tgt', 'bs_acc_bsl', 'acc_bsl', 'cmp_acc']
            available = [c for c in cols if c in acc_pass.columns]
            write_html_table(acc_pass, available, 'cmp_acc', pass_color="#d4edda")
        else:
            f.write("ℹ️ No new accuracy passes.\n\n")

        # Performance improvements (new_improved + new_passed)
        perf_impr = details[details['cmp_perf'].isin(['new_improved', 'new_passed'])]
        if not perf_impr.empty:
            f.write("### ⏱️ Performance Improvements (ratio > {:.0f}%)\n\n".format((1 + PERFORMANCE_THRESHOLD) * 100))
            cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
            available = [c for c in cols if c in perf_impr.columns]
            write_html_table(perf_impr, available, 'cmp_perf', pass_color="#d4edda")
        else:
            f.write("ℹ️ No performance improvements.\n\n")

        # ----- Suggestions -----
        f.write("## 💡 Suggestions\n\n")
        suggestions = []
        if not acc_fail.empty:
            suggestions.append("❌ Investigate the new accuracy failures.")
        if not perf_regress.empty:
            suggestions.append("📉 Review performance regressions; they may indicate a real slowdown.")
        if acc_pass.empty and perf_impr.empty:
            suggestions.append("ℹ️ No new passes or improvements detected.")
        else:
            if not acc_pass.empty:
                suggestions.append("✅ New accuracy passes are good; ensure they are not due to test changes.")
            if not perf_impr.empty:
                suggestions.append("🚀 Performance improvements are encouraging; verify they are consistent.")
        if suggestions:
            for s in suggestions:
                f.write(f"- {s}\n")
        else:
            f.write("- All metrics are stable. No action required.\n")


# ----------------------------------------------------------------------
# Main (updated with column renaming and markdown option)
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch inductor test results with baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target_dir --baseline_dir --comparison.xlsx               # Creates Excel with Summary & Details sheets
  %(prog)s --target_dir --baseline_dir --comparison.csv                # Creates summary.csv and details.csv
  %(prog)s --target_dir --baseline_dir --comparison.xlsx --markdown report.md   # Also generates a Markdown summary
        """
    )
    parser.add_argument("-t", "--target_dir", help="Directory containing target (test) result CSV files (searched recursively)")
    parser.add_argument("-b", "--baseline_dir", help="Directory containing baseline (reference) result CSV files (searched recursively)")
    parser.add_argument("-o", "--output", help="Output file name (without extension). Use .xlsx for Excel, .csv for CSV files.")
    parser.add_argument("-m", "--markdown", help="Generate a Markdown summary report and save to this file (e.g., report.md)")
    args = parser.parse_args()

    # Determine output base name and format
    out_base, out_ext = os.path.splitext(args.output)
    if out_ext not in ('.xlsx', '.csv'):
        print("Output file must end with .xlsx or .csv")
        return 1

    # Find files
    target_files = find_result_files(args.target_dir)
    baseline_files = find_result_files(args.baseline_dir)

    print(f"Found {len(target_files)} CSV files in target directory.")
    print(f"Found {len(baseline_files)} CSV files in baseline directory.")

    # Load records
    target_acc = load_results(target_files, "accuracy")
    target_perf = load_results(target_files, "performance")
    baseline_acc = load_results(baseline_files, "accuracy")
    baseline_perf = load_results(baseline_files, "performance")

    print(f"Target accuracy records: {len(target_acc)}")
    print(f"Target performance records: {len(target_perf)}")
    print(f"Baseline accuracy records: {len(baseline_acc)}")
    print(f"Baseline performance records: {len(baseline_perf)}")

    # Merge data
    acc_merged = merge_accuracy(target_acc, baseline_acc)
    perf_merged = merge_performance(target_perf, baseline_perf)

    # Generate combined summary (with multiple levels)
    combined_summary = generate_all_summaries(acc_merged, perf_merged)

    # Generate detailed combined data
    details = combine_results(acc_merged, perf_merged)

    # Apply column renaming for output (short headers)
    # Only rename columns that exist in each dataframe
    combined_summary.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in combined_summary.columns}, inplace=True)
    details.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in details.columns}, inplace=True)

    # Generate markdown if requested
    if args.markdown:
        generate_markdown_summary(combined_summary, details, args.markdown)
        print(f"Markdown summary written to {args.markdown}")

    if out_ext == '.xlsx':
        # Excel: two sheets
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            # Summary sheet
            if not combined_summary.empty:
                combined_summary.to_excel(writer, sheet_name="Summary", index=False)
            else:
                pd.DataFrame({"Info": ["No summary data available"]}).to_excel(writer, sheet_name="Summary", index=False)

            # Details sheet
            if not details.empty:
                details.to_excel(writer, sheet_name="Details", index=False)
            else:
                pd.DataFrame({"Info": ["No detailed data available"]}).to_excel(writer, sheet_name="Details", index=False)

        print(f"Excel written to {args.output} (sheets: Summary, Details)")

    else:  # .csv
        # Write two CSV files: summary.csv and details.csv
        summary_file = out_base + "_summary.csv"
        details_file = out_base + "_details.csv"

        if not combined_summary.empty:
            combined_summary.to_csv(summary_file, index=False, na_rep='')
            print(f"Summary written to {summary_file}")
        else:
            print("No summary data to write.")

        if not details.empty:
            details.to_csv(details_file, index=False, na_rep='')
            print(f"Details written to {details_file}")
        else:
            print("No detailed data to write.")

    return 0


if __name__ == "__main__":
    exit(main())
