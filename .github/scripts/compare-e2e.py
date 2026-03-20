#!/usr/bin/env python3
"""
Comparison tool for PyTorch inductor test results (target vs baseline).
"""

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

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

# Column renaming for output (shortened headers)
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
    'new_fail': 'new_fail',
    'new_drop': 'new_drop',
    'new_pass': 'new_pass',
    'new_improve': 'new_improve',
    'ind_ratio': 'ind_ratio',
    'eag_ratio': 'eag_ratio',
}

# ----------------------------------------------------------------------
# File discovery and parsing
# ----------------------------------------------------------------------
def find_result_files(root_dir: str) -> List[str]:
    """Recursively find all *_performance.csv and *_accuracy.csv files."""
    perf_files = glob(os.path.join(root_dir, "**", "*_performance.csv"), recursive=True)
    acc_files = glob(os.path.join(root_dir, "**", "*_accuracy.csv"), recursive=True)
    return perf_files + acc_files


def _parse_filename_components(filename: str) -> Tuple[str, str, str, str]:
    """
    Extract suite, data_type, mode, result_type from a filename like
    inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
    """
    if not filename.endswith(".csv"):
        raise ValueError("Not a CSV file")
    base = filename[:-4]
    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor_': {filename}")
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


def _merge_accuracy_records(records: List[Dict]) -> Dict:
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


def _merge_performance_records(records: List[Dict]) -> Optional[Dict]:
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


def load_results(file_list: List[str], result_type_filter: str) -> List[Dict]:
    """
    Load all CSV files of a given result type, parse them, and merge duplicates.
    Returns a list of records (each a dict).
    """
    raw_by_key: Dict[Tuple, List[Dict]] = {}

    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = _parse_filename_components(os.path.basename(fpath))
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

                # Include row even if speedup/abs_latency are missing (set to NaN)
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
                    "inductor": inductor,
                    "eager": eager
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
def merge_accuracy(target_records: List[Dict], baseline_records: List[Dict]) -> pd.DataFrame:
    """Merge target and baseline accuracy DataFrames and compute comparison column."""
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

    def compare_acc(row):
        tgt = row.get("accuracy_target")
        bsl = row.get("accuracy_baseline")
        if pd.isna(tgt) and pd.isna(bsl):
            return ""
        if pd.isna(tgt):
            return "new_failed"
        if pd.isna(bsl):
            return "new_passed"
        # Both present: check pass/fail status
        tgt_pass = 'pass' in str(tgt)
        bsl_pass = 'pass' in str(bsl)
        if not tgt_pass and bsl_pass:
            return "new_failed"
        if tgt_pass and not bsl_pass:
            return "new_passed"
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


def merge_performance(target_records: List[Dict], baseline_records: List[Dict],
                      threshold: float) -> pd.DataFrame:
    """Merge target and baseline performance DataFrames, compute ratios and comparison."""
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

    def compare_perf(row):
        # If both missing -> empty
        if pd.isna(row.get("inductor_target")) and pd.isna(row.get("inductor_baseline")):
            return ""
        # If one side missing -> treat as new fail/pass
        if pd.isna(row.get("inductor_target")):
            return "new_passed"
        if pd.isna(row.get("inductor_baseline")):
            return "new_failed"
        # If either ratio negative (unphysical) -> treat as fail/pass based on availability
        if row["inductor_ratio"] < 0 or row["eager_ratio"] < 0:
            return "new_failed" if pd.notna(row["inductor_target"]) else "new_passed"
        # Check thresholds
        if row["inductor_ratio"] < 1 - threshold or row["eager_ratio"] < 1 - threshold:
            return "new_dropped"
        if row["inductor_ratio"] > 1 + threshold or row["eager_ratio"] > 1 + threshold:
            return "new_improved"
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


def combine_results(acc_merged: pd.DataFrame, perf_merged: pd.DataFrame) -> pd.DataFrame:
    """Combine accuracy and performance merged DataFrames into one details table."""
    # Rename batch_size columns to avoid conflict
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
            acc = row.get("comparison_acc")
            perf = row.get("comparison_perf")
            if pd.isna(acc) and pd.isna(perf):
                return ""
            if pd.isna(acc):
                return perf
            if pd.isna(perf):
                return acc
            # Prioritise fail/drop over improve/pass
            if any(x in [acc, perf] for x in ("new_failed", "new_dropped")):
                return "new_failed"
            if any(x in [acc, perf] for x in ("new_passed", "new_improved")):
                return "new_improved"
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
                           group_cols: List[str], level_name: str) -> pd.DataFrame:
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
def _fmt_count(val: Any, good: bool = False) -> str:
    """Format a count with emoji if >0."""
    if pd.notna(val) and val > 0:
        emoji = "🟢" if good else "🔴"
        return f"{val} {emoji}"
    return str(val) if pd.notna(val) else ""


def _fmt_ratio(val: Any, threshold: float) -> str:
    """Format ratio with emoji if below/above threshold."""
    if pd.notna(val):
        try:
            num = float(val)
            if num < 1 - threshold:
                return f"{val} 🔴"
            if num > 1 + threshold:
                return f"{val} 🟢"
        except:
            pass
    return str(val) if pd.notna(val) else ""


def write_summary_markdown(combined_summary: pd.DataFrame, threshold: float, filename: str):
    """Write a Markdown file with the Overall Summary table (split new_fail/drop and new_pass/improve)."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"\n\n# Dynamo Benchmark Test Results - Summary\n\n")

        overall = combined_summary[combined_summary['Category'] == 'Overall']
        if overall.empty:
            f.write("No overall summary data available.\n")
            return

        f.write("| Type | tgt_ps | bsl_ps | total | new_fail | new_drop | new_pass | new_improve | tgt_pass% | bsl_pass% | ind_ratio | eag_ratio |\n")
        f.write("|------|--------|--------|-------|----------|----------|----------|-------------|-----------|-----------|-----------|-----------|\n")
        for _, row in overall.iterrows():
            type_label = row['Type']
            tgt_ps = row.get('tgt_ps', '')
            bsl_ps = row.get('bsl_ps', '')
            total = row.get('total', '')
            if type_label == 'Accuracy':
                # For accuracy, new_drop and new_improve are not applicable -> show "/"
                new_fail = _fmt_count(row.get('new_fail'), good=False)
                new_pass = _fmt_count(row.get('new_pass'), good=True)
                new_drop = new_improve = ind_ratio = eag_ratio = "/"
            else:
                new_fail = _fmt_count(row.get('new_fail'), good=False)
                new_pass = _fmt_count(row.get('new_pass'), good=True)
                new_drop = _fmt_count(row.get('new_drop'), good=False)
                new_improve = _fmt_count(row.get('new_improve'), good=True)
                ind_ratio = _fmt_ratio(row.get('ind_ratio'), threshold)
                eag_ratio = _fmt_ratio(row.get('eag_ratio'), threshold)
            tgt_pass = row.get('tgt_pass%', '')
            bsl_pass = row.get('bsl_pass%', '')
            f.write(f"| {type_label} | {tgt_ps} | {bsl_ps} | {total} | {new_fail} | {new_drop} | {new_pass} | {new_improve} | {tgt_pass} | {bsl_pass} | {ind_ratio} | {eag_ratio} |\n")
        f.write("\n")


def _write_html_table(rows: pd.DataFrame, columns: List[str], condition_col: str,
                      fail_color: str, pass_color: str, file_handle):
    """Write an HTML table with row background colors based on condition_col."""
    file_handle.write('<table>\n')
    file_handle.write('<thead><tr>')
    for col in columns:
        file_handle.write(f'<th>{col}</th>')
    file_handle.write('</tr></thead>\n')
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
            cell = str(row.get(col, ''))
            file_handle.write(f'<td>{cell}</td>')
        file_handle.write('</tr>\n')
    file_handle.write('</tbody>\n')
    file_handle.write('</table>\n\n')


def write_details_markdown(details_df: pd.DataFrame, threshold: float, filename: str):
    """Write a Markdown file with suite overview and tables for new failures/improvements."""
    if details_df.empty:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Detailed Report\n\nNo detailed data available.\n")
        return

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"\n\n# Dynamo Benchmark Test Results - Details\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ----- Overview by Suite -----
        if 'suite' in details_df.columns:
            f.write("## 📊 Overview by Suite\n\n")
            suite_rows = []
            for suite in details_df['suite'].dropna().unique():
                suite_df = details_df[details_df['suite'] == suite]

                # Accuracy totals & counts
                if 'cmp_acc' in suite_df.columns:
                    acc_total = suite_df['model'].notna().sum()
                    acc_fail = (suite_df['cmp_acc'] == 'new_failed').sum()
                    acc_pass = (suite_df['cmp_acc'] == 'new_passed').sum()
                else:
                    acc_total = acc_fail = acc_pass = 0

                # Performance totals & counts
                if 'cmp_perf' in suite_df.columns:
                    perf_total = suite_df['model'].notna().sum()
                    perf_fail = (suite_df['cmp_perf'] == 'new_failed').sum()
                    perf_drop = (suite_df['cmp_perf'] == 'new_dropped').sum()
                    perf_pass = (suite_df['cmp_perf'] == 'new_passed').sum()
                    perf_improve = (suite_df['cmp_perf'] == 'new_improved').sum()
                else:
                    perf_total = perf_fail = perf_drop = perf_pass = perf_improve = 0

                # Accuracy pass rate (absolute, not change)
                acc_pass_rate = ""
                if 'acc_tgt' in suite_df.columns:
                    acc_rows = suite_df['acc_tgt'].dropna()
                    if len(acc_rows) > 0:
                        pass_count = acc_rows.astype(str).str.contains('pass', na=False).sum()
                        acc_pass_rate = f"{(pass_count / acc_total * 100):.1f}%"

                # Performance ratios (geometric mean)
                def geomean(series):
                    vals = series.dropna()
                    vals = vals[vals > 0]
                    if len(vals) == 0:
                        return None
                    return np.exp(np.log(vals).mean())

                ind_ratio = ""
                if 'ind_ratio' in suite_df.columns:
                    gm = geomean(suite_df['ind_ratio'])
                    if gm is not None:
                        ind_ratio = f"{gm:.3f}"
                eag_ratio = ""
                if 'eag_ratio' in suite_df.columns:
                    gm = geomean(suite_df['eag_ratio'])
                    if gm is not None:
                        eag_ratio = f"{gm:.3f}"

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

            if suite_rows:
                # Helper to format a number with emoji
                def fmt_count(val, good=False):
                    if val > 0:
                        emoji = "🟢" if good else "🔴"
                        return f"{val} {emoji}"
                    return str(val)

                f.write("| Suite | 🧪 Acc Total | 🧪 Acc Fail | 🧪 Acc Pass | 🧪 Acc Pass Rate | ⏱️ Perf Total | ⏱️ Perf Fail | ⏱️ Perf Drop | ⏱️ Perf Pass | ⏱️ Perf Improve | ⏱️ Ind Ratio | ⏱️ Eag Ratio |\n")
                f.write("|-------|--------------|-------------|--------------|------------------|---------------|--------------|---------------|---------------|-----------------|--------------|--------------|\n")
                for s in suite_rows:
                    row = [
                        s['suite'],
                        str(s['acc_total']),
                        fmt_count(s['acc_fail'], good=False),
                        fmt_count(s['acc_pass'], good=True),
                        s['acc_pass_rate'],
                        str(s['perf_total']),
                        fmt_count(s['perf_fail'], good=False),
                        fmt_count(s['perf_drop'], good=False),
                        fmt_count(s['perf_pass'], good=True),
                        fmt_count(s['perf_improve'], good=True),
                        _fmt_ratio(s['ind_ratio'], threshold),
                        _fmt_ratio(s['eag_ratio'], threshold),
                    ]
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")
            else:
                f.write("No suite information available.\n\n")
        else:
            f.write("## 📊 Overview by Suite\n\nNo suite information available.\n\n")

        # ----- New Failures & Regressions -----
        # Only attempt if columns exist
        if 'cmp_acc' in details_df.columns:
            acc_fail = details_df[details_df['cmp_acc'] == 'new_failed']
        else:
            acc_fail = pd.DataFrame()

        if 'cmp_perf' in details_df.columns:
            perf_regress = details_df[details_df['cmp_perf'].isin(['new_dropped', 'new_failed'])]
        else:
            perf_regress = pd.DataFrame()

        if not acc_fail.empty or not perf_regress.empty:
            f.write("## ❌ New Failures & Regressions\n\n")

            if not acc_fail.empty:
                f.write("### 🧪 Accuracy Failures\n\n")
                cols = ['suite', 'dtype', 'mode', 'model', 'bs_tgt', 'acc_tgt', 'bs_bsl', 'acc_bsl', 'cmp_acc']
                available = [c for c in cols if c in acc_fail.columns]
                _write_html_table(acc_fail, available, 'cmp_acc', "#f8d7da", "#d4edda", f)

            if not perf_regress.empty:
                f.write("### ⏱️ Performance Regressions (ratio < {:.0f}%)\n\n".format((1 - threshold) * 100))
                cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
                available = [c for c in cols if c in perf_regress.columns]
                _write_html_table(perf_regress, available, 'cmp_perf', "#f8d7da", "#d4edda", f)

        # ----- New Passes & Improvements -----
        if 'cmp_acc' in details_df.columns:
            acc_pass = details_df[details_df['cmp_acc'] == 'new_passed']
        else:
            acc_pass = pd.DataFrame()

        if 'cmp_perf' in details_df.columns:
            perf_impr = details_df[details_df['cmp_perf'].isin(['new_improved', 'new_passed'])]
        else:
            perf_impr = pd.DataFrame()

        if not acc_pass.empty or not perf_impr.empty:
            f.write("## ✅ New Passes & Improvements\n\n")

            if not acc_pass.empty:
                f.write("### 🧪 Accuracy New Passes\n\n")
                cols = ['suite', 'dtype', 'mode', 'model', 'bs_tgt', 'acc_tgt', 'bs_bsl', 'acc_bsl', 'cmp_acc']
                available = [c for c in cols if c in acc_pass.columns]
                _write_html_table(acc_pass, available, 'cmp_acc', "#f8d7da", "#d4edda", f)

            if not perf_impr.empty:
                f.write("### ⏱️ Performance Improvements (ratio > {:.0f}%)\n\n".format((1 + threshold) * 100))
                cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
                available = [c for c in cols if c in perf_impr.columns]
                _write_html_table(perf_impr, available, 'cmp_perf', "#f8d7da", "#d4edda", f)


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
  %(prog)s -t target_dir -b baseline_dir -o comparison.csv
  %(prog)s -t target_dir -b baseline_dir -o comparison.xlsx -m report --threshold 0.15
        """
    )
    parser.add_argument("-t", "--target_dir", required=True, help="Directory containing target (test) result CSV files (searched recursively)")
    parser.add_argument("-b", "--baseline_dir", required=True, help="Directory containing baseline (reference) result CSV files (searched recursively)")
    parser.add_argument("-o", "--output", required=True, help="Output file name (without extension). Use .xlsx for Excel, .csv for CSV files.")
    parser.add_argument("-m", "--markdown", help="Base name for Markdown reports (e.g., report -> report.summary.md, report.details.md)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PERFORMANCE_THRESHOLD,
                        help=f"Performance change threshold (default: {DEFAULT_PERFORMANCE_THRESHOLD})")
    args = parser.parse_args()

    out_base, out_ext = os.path.splitext(args.output)
    if out_ext not in ('.xlsx', '.csv'):
        print("Output file must end with .xlsx or .csv")
        return 1

    target_files = find_result_files(args.target_dir)
    baseline_files = find_result_files(args.baseline_dir)

    print(f"Found {len(target_files)} CSV files in target directory.")
    print(f"Found {len(baseline_files)} CSV files in baseline directory.")

    target_acc = load_results(target_files, "accuracy")
    target_perf = load_results(target_files, "performance")
    baseline_acc = load_results(baseline_files, "accuracy")
    baseline_perf = load_results(baseline_files, "performance")

    print(f"Target accuracy records: {len(target_acc)}")
    print(f"Target performance records: {len(target_perf)}")
    print(f"Baseline accuracy records: {len(baseline_acc)}")
    print(f"Baseline performance records: {len(baseline_perf)}")

    acc_merged = merge_accuracy(target_acc, baseline_acc)
    perf_merged = merge_performance(target_perf, baseline_perf, args.threshold)

    combined_summary = generate_all_summaries(acc_merged, perf_merged)
    details = combine_results(acc_merged, perf_merged)

    # Apply column renaming for output
    combined_summary.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in combined_summary.columns}, inplace=True)
    details.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in details.columns}, inplace=True)

    # Generate markdown files if requested
    if args.markdown:
        md_base = os.path.splitext(args.markdown)[0]  # remove any extension
        summary_file = md_base + ".summary.md"
        details_file = md_base + ".details.md"
        write_summary_markdown(combined_summary, args.threshold, summary_file)
        write_details_markdown(details, args.threshold, details_file)
        print(f"Markdown summary written to {summary_file}")
        print(f"Markdown details written to {details_file}")

    if out_ext == '.xlsx':
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            if not combined_summary.empty:
                combined_summary.to_excel(writer, sheet_name="Summary", index=False)
            else:
                pd.DataFrame({"Info": ["No summary data available"]}).to_excel(writer, sheet_name="Summary", index=False)
            if not details.empty:
                details.to_excel(writer, sheet_name="Details", index=False)
            else:
                pd.DataFrame({"Info": ["No detailed data available"]}).to_excel(writer, sheet_name="Details", index=False)
        print(f"Excel written to {args.output} (sheets: Summary, Details)")

    else:  # .csv
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
