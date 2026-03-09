#!/usr/bin/env python3
"""
Compare PyTorch Inductor test results (target) against a baseline.
Reads CSV files from two directories, computes comparisons, and writes an Excel report.

Usage:
    python compare_inductor_results.py baseline_dir target_dir output.xlsx
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Regex to parse filenames like:
#   inductor_huggingface_float32_inference_xpu_accuracy.csv
#   inductor_timm_models_amp_bf16_training_xpu_performance.csv
FILENAME_PATTERN = re.compile(
    r"^inductor_(?P<suite>.+)_(?P<dtype>amp_bf16|amp_fp16|bfloat16|float16|float32)_(?P<mode>inference|training)_xpu_(?P<scenario>accuracy|performance)\.csv$"
)

# Mapping of dtype strings to readable names (optional)
DTYPE_MAP = {
    'float32': 'float32',
    'float16': 'float16',
    'bfloat16': 'bfloat16',
    'amp_bf16': 'amp_bf16',
    'amp_fp16': 'amp_fp16',
}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_filename(filename):
    print(filename)
    for fname in filename:
        match = FILENAME_PATTERN.match(fname)
        if not match:
            print(f"Failed to parse: {fname}")
            continue
        suite = match.group('suite')
        dtype = match.group('dtype')
        mode = match.group('mode')
        scenario = match.group('scenario')
    
    return suite, dtype, mode, scenario


def read_csv(filepath, scenario):
    """Read CSV and return DataFrame with appropriate columns."""
    df = pd.read_csv(filepath)
    # Ensure required columns exist
    required = ['name', 'batch_size']
    if scenario == 'accuracy':
        required.append('accuracy')
    else:  # performance
        required.extend(['speedup', 'abs_latency'])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File {filepath} missing columns: {missing}")
    return df


def compute_performance_metrics(df):
    """Add 'eager' and 'inductor' columns to performance DataFrame."""
    df = df.copy()
    df['eager'] = df['speedup'] * df['abs_latency']
    df['inductor'] = df['abs_latency']
    return df


def safe_geomean(series):
    """Compute geometric mean, ignoring NaNs and non-positive values."""
    series = series.dropna()
    series = series[series > 0]
    if len(series) == 0:
        return np.nan
    return np.exp(np.log(series).mean())


# ----------------------------------------------------------------------
# Main comparison logic
# ----------------------------------------------------------------------

def collect_files(directory, recursive=False, case_sensitive=False, include_hidden=False):
    """
    Scan directory for CSV files.
    Returns dict: (suite, dtype, mode, scenario) -> Path object.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search.
    recursive : bool
        If True, search subdirectories recursively.
    case_sensitive : bool
        If False, match .csv, .CSV, .Csv, etc.
    include_hidden : bool
        If True, include files whose names start with a dot.
    """
    files = {}
    dir_path = Path(directory).expanduser().resolve()  # resolve to absolute path
    if not dir_path.is_dir():
        print(f"Directory {dir_path} does not exist or is not a directory.")
        return files

    # Choose the glob method based on recursion
    if recursive:
        iterator = dir_path.rglob('*')
    else:
        iterator = dir_path.glob('*')

    for p in iterator:
        # Skip directories
        if not p.is_file():
            continue
        # Skip hidden files unless requested
        if not include_hidden and p.name.startswith('.'):
            continue
        # Check extension with optional case sensitivity
        if case_sensitive:
            match = p.suffix == '.csv'
        else:
            match = p.suffix.lower() == '.csv'
        if not match:
            continue

        parsed = parse_filename(p.name)
        if parsed:
            suite, dtype, mode, scenario = parsed
            key = (suite, dtype, mode, scenario)
            if key in files:
                print(f"Duplicate key {key} for files {files[key]} and {p}")
            files[key] = p

    return files


def load_and_merge(baseline_dir, target_dir):
    """
    Load all CSV files from both directories, align by keys,
    and return combined data for accuracy and performance.
    Returns two dicts:
        accuracy_data: (suite, dtype, mode) -> (baseline_df, target_df)
        performance_data: (suite, dtype, mode) -> (baseline_df, target_df)
    """
    baseline_files = collect_files(baseline_dir, recursive=True, case_sensitive=False)
    target_files = collect_files(target_dir, recursive=True, case_sensitive=False)

    all_keys = set(baseline_files.keys()) | set(target_files.keys())
    
    print(target_files)

    accuracy_data = {}
    performance_data = {}

    for key in all_keys:
        suite, dtype, mode, scenario = key
        base_key = (suite, dtype, mode, scenario)
        target_key = (suite, dtype, mode, scenario)

        base_path = baseline_files.get(base_key)
        target_path = target_files.get(target_key)

        # Skip if either file is missing (we could still compare if one side missing?)
        if not base_path or not target_path:
            print(f"Warning: Missing file for {key} – skipping")
            continue

        try:
            base_df = read_csv(base_path, scenario)
            target_df = read_csv(target_path, scenario)
        except Exception as e:
            print(f"Error reading {key}: {e}")
            continue

        # Store in appropriate dict
        if scenario == 'accuracy':
            accuracy_data[(suite, dtype, mode)] = (base_df, target_df)
        else:  # performance
            # Add computed columns
            base_df = compute_performance_metrics(base_df)
            target_df = compute_performance_metrics(target_df)
            performance_data[(suite, dtype, mode)] = (base_df, target_df)

    return accuracy_data, performance_data


def compare_accuracy(base_df, target_df, suite, dtype, mode):
    """
    Merge baseline and target accuracy data on (name, batch_size).
    Returns DataFrame with columns for detailed Accuracy sheet.
    """
    merged = pd.merge(
        base_df[['name', 'batch_size', 'accuracy']],
        target_df[['name', 'batch_size', 'accuracy']],
        on=['name', 'batch_size'],
        how='outer',
        suffixes=('_baseline', '_target')
    )
    # Add suite, dtype, mode
    merged.insert(0, 'suite', suite)
    merged.insert(1, 'data_type', dtype)
    merged.insert(2, 'mode', mode)

    # Rename columns to match spec
    merged.rename(columns={
        'batch_size_baseline': 'baseline batch_size',
        'accuracy_baseline': 'baseline accuracy',
        'batch_size_target': 'target batch_size',
        'accuracy_target': 'target accuracy',
    }, inplace=True)

    # Create comparison string
    def comp_str(row):
        base = str(row.get('baseline accuracy', '')).lower()
        targ = str(row.get('target accuracy', '')).lower()
        if pd.isna(base) or base == 'nan':
            base = 'missing'
        if pd.isna(targ) or targ == 'nan':
            targ = 'missing'
        return f"{base} -> {targ}"
    merged['comparison'] = merged.apply(comp_str, axis=1)

    # Reorder columns
    cols = ['suite', 'data_type', 'mode', 'name',
            'target batch_size', 'target accuracy',
            'baseline batch_size', 'baseline accuracy',
            'comparison']
    return merged[cols]


def compare_performance(base_df, target_df, suite, dtype, mode):
    """
    Merge baseline and target performance data on (name, batch_size).
    Returns DataFrame with columns for detailed Performance sheet.
    """
    # Select necessary columns
    base_cols = ['name', 'batch_size', 'inductor', 'eager']
    target_cols = ['name', 'batch_size', 'inductor', 'eager']

    merged = pd.merge(
        base_df[base_cols],
        target_df[target_cols],
        on=['name', 'batch_size'],
        how='outer',
        suffixes=('_baseline', '_target')
    )
    merged.insert(0, 'suite', suite)
    merged.insert(1, 'data_type', dtype)
    merged.insert(2, 'mode', mode)

    # Compute ratios (baseline / target)
    merged['inductor_ratio'] = merged['inductor_baseline'] / merged['inductor_target']
    merged['eager_ratio'] = merged['eager_baseline'] / merged['eager_target']

    # Rename columns to match spec
    merged.rename(columns={
        'inductor_target': 'target inductor',
        'eager_target': 'target eager',
        'inductor_baseline': 'baseline inductor',
        'eager_baseline': 'baseline eager',
        'inductor_ratio': 'target baseline/inductor inductor',
        'eager_ratio': 'target baseline/inductor eager',
    }, inplace=True)

    # Keep required columns
    cols = ['suite', 'data_type', 'mode', 'name',
            'target inductor', 'target eager',
            'baseline inductor', 'baseline eager',
            'target baseline/inductor inductor',
            'target baseline/inductor eager']
    return merged[cols]


def build_comparison_sheets(accuracy_data, performance_data):
    """
    Generate DataFrames for Accuracy Comparison and Performance Comparison sheets.
    Also compute status columns for conditional formatting.
    """
    # Accuracy comparison: add status column based on change
    acc_comp_rows = []
    for (suite, dtype, mode), (base_df, target_df) in accuracy_data.items():
        merged = compare_accuracy(base_df, target_df, suite, dtype, mode)
        # Determine status
        def acc_status(row):
            base = str(row['baseline accuracy']).lower()
            target = str(row['target accuracy']).lower()
            if 'pass' in target and 'pass' not in base:
                return 'new passed'
            elif 'pass' in base and 'pass' not in target:
                return 'new failed'
            else:
                return 'no change'
        merged['status'] = merged.apply(acc_status, axis=1)
        acc_comp_rows.append(merged)
    acc_comp_df = pd.concat(acc_comp_rows, ignore_index=True) if acc_comp_rows else pd.DataFrame()

    # Performance comparison: add status column based on ratios
    perf_comp_rows = []
    for (suite, dtype, mode), (base_df, target_df) in performance_data.items():
        merged = compare_performance(base_df, target_df, suite, dtype, mode)
        # Determine status for inductor ratio (can also do for eager, but spec seems to use inductor)
        def perf_status(row):
            ratio = row['target baseline/inductor inductor']
            if ratio > 1.1:
                return 'improvement'
            elif ratio < 0.9:
                return 'dropped'
            else:
                return 'no change'
        merged['status'] = merged.apply(perf_status, axis=1)
        perf_comp_rows.append(merged)
    perf_comp_df = pd.concat(perf_comp_rows, ignore_index=True) if perf_comp_rows else pd.DataFrame()

    return acc_comp_df, perf_comp_df


def build_summary(acc_comp_df, perf_comp_df):
    """
    Generate summary DataFrame with pass rates, new passed/failed, geomean, etc.
    Groups by (suite, data_type, mode), by suite, and overall.
    """
    # Prepare accuracy data for aggregation
    acc_data = acc_comp_df.copy() if not acc_comp_df.empty else pd.DataFrame()
    if not acc_data.empty:
        acc_data['is_pass'] = acc_data['target accuracy'].astype(str).str.contains('pass', case=False, na=False)
        acc_data['is_new_passed'] = (acc_data['status'] == 'new passed')
        acc_data['is_new_failed'] = (acc_data['status'] == 'new failed')

    # Prepare performance data for aggregation
    perf_data = perf_comp_df.copy() if not perf_comp_df.empty else pd.DataFrame()
    if not perf_data.empty:
        perf_data['inductor_ratio'] = perf_data['target baseline/inductor inductor']
        perf_data['is_improved'] = (perf_data['status'] == 'improvement')
        perf_data['is_dropped'] = (perf_data['status'] == 'dropped')
        # pass rate for performance = not dropped
        perf_data['is_perf_pass'] = ~perf_data['is_dropped']

    # Define grouping levels
    groupings = []

    # 1. By (suite, data_type, mode)
    if not acc_data.empty or not perf_data.empty:
        groupings.append(['suite', 'data_type', 'mode'])

    # 2. By suite
    if not acc_data.empty or not perf_data.empty:
        groupings.append(['suite'])

    # 3. Overall (empty list)
    groupings.append([])

    summary_rows = []
    for group_cols in groupings:
        # Initialize aggregators
        if acc_data.empty:
            acc_agg = pd.DataFrame()
        else:
            acc_agg = acc_data.groupby(group_cols).agg(
                total_models=('name', 'count'),
                pass_count=('is_pass', 'sum'),
                new_passed=('is_new_passed', 'sum'),
                new_failed=('is_new_failed', 'sum')
            ).reset_index()
            acc_agg['accuracy_pass_rate'] = acc_agg['pass_count'] / acc_agg['total_models'] * 100

        if perf_data.empty:
            perf_agg = pd.DataFrame()
        else:
            perf_agg = perf_data.groupby(group_cols).agg(
                total_perf_models=('name', 'count'),
                perf_pass_count=('is_perf_pass', 'sum'),
                improved=('is_improved', 'sum'),
                dropped=('is_dropped', 'sum'),
                inductor_ratio_list=('inductor_ratio', lambda x: list(x))
            ).reset_index()
            perf_agg['performance_pass_rate'] = perf_agg['perf_pass_count'] / perf_agg['total_perf_models'] * 100
            # Compute geomean of inductor ratios
            perf_agg['performance_geomean'] = perf_agg['inductor_ratio_list'].apply(
                lambda ratios: safe_geomean(pd.Series(ratios))
            )
            perf_agg.drop(columns=['inductor_ratio_list'], inplace=True)

        # Merge accuracy and performance aggregates
        if not acc_agg.empty and not perf_agg.empty:
            merged = pd.merge(acc_agg, perf_agg, on=group_cols if group_cols else [], how='outer')
        elif not acc_agg.empty:
            merged = acc_agg
            # Add performance columns as NaN
            for col in ['total_perf_models', 'perf_pass_count', 'improved', 'dropped', 'performance_pass_rate', 'performance_geomean']:
                merged[col] = np.nan
        elif not perf_agg.empty:
            merged = perf_agg
            # Add accuracy columns as NaN
            for col in ['total_models', 'pass_count', 'new_passed', 'new_failed', 'accuracy_pass_rate']:
                merged[col] = np.nan
        else:
            continue

        # Fill missing group columns for overall
        if not group_cols:
            merged.insert(0, 'suite', 'overall')
            merged.insert(1, 'data_type', '')
            merged.insert(2, 'mode', '')

        # Select final columns
        final_cols = []
        if 'suite' in merged.columns:
            final_cols.append('suite')
        if 'data_type' in merged.columns:
            final_cols.append('data_type')
        if 'mode' in merged.columns:
            final_cols.append('mode')
        final_cols.extend([
            'accuracy_pass_rate', 'new_passed', 'new_failed',
            'performance_pass_rate', 'performance_geomean', 'improved', 'dropped'
        ])
        # Ensure all columns exist
        for col in final_cols:
            if col not in merged.columns:
                merged[col] = np.nan
        summary_rows.append(merged[final_cols])

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    return summary_df


def write_excel_report(output_path, acc_detail_df, perf_detail_df, acc_comp_df, perf_comp_df, summary_df):
    """
    Write all DataFrames to an Excel file with formatting.
    """
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Write each sheet
        acc_detail_df.to_excel(writer, sheet_name='Accuracy', index=False)
        perf_detail_df.to_excel(writer, sheet_name='Performance', index=False)
        acc_comp_df.to_excel(writer, sheet_name='Acc Comparison', index=False)
        perf_comp_df.to_excel(writer, sheet_name='Performance Comparison', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Get worksheet objects
        acc_comp_sheet = writer.sheets['Acc Comparison']
        perf_comp_sheet = writer.sheets['Performance Comparison']

        # Conditional formatting
        # Accuracy Comparison: highlight status column
        if 'status' in acc_comp_df.columns:
            # Find column index for status (column after last data column)
            status_col_idx = acc_comp_df.columns.get_loc('status')
            # Apply green for 'new passed', red for 'new failed'
            green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

            # Get the range for the status column (excluding header)
            start_row = 2  # 1-indexed, header row 1, data starts at row 2
            end_row = len(acc_comp_df) + 1
            col_letter = chr(65 + status_col_idx)  # A=65
            cell_range = f'{col_letter}{start_row}:{col_letter}{end_row}'

            acc_comp_sheet.conditional_format(cell_range, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'new passed',
                'format': green_format
            })
            acc_comp_sheet.conditional_format(cell_range, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'new failed',
                'format': red_format
            })

        # Performance Comparison: highlight based on ratio columns? But spec says based on status.
        if 'status' in perf_comp_df.columns:
            status_col_idx = perf_comp_df.columns.get_loc('status')
            start_row = 2
            end_row = len(perf_comp_df) + 1
            col_letter = chr(65 + status_col_idx)
            cell_range = f'{col_letter}{start_row}:{col_letter}{end_row}'

            green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

            perf_comp_sheet.conditional_format(cell_range, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'improvement',
                'format': green_format
            })
            perf_comp_sheet.conditional_format(cell_range, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'dropped',
                'format': red_format
            })

        # Optional: format numbers in Performance sheet
        perf_sheet = writer.sheets['Performance']
        perf_sheet.set_column('E:H', None, workbook.add_format({'num_format': '0.000000'}))
        perf_sheet.set_column('I:J', None, workbook.add_format({'num_format': '0.00'}))

        # Summary formatting
        summary_sheet = writer.sheets['Summary']
        summary_sheet.set_column('A:A', 15)  # suite
        summary_sheet.set_column('B:B', 12)  # data_type
        summary_sheet.set_column('C:C', 10)  # mode
        summary_sheet.set_column('D:D', 12, workbook.add_format({'num_format': '0.00%'}))  # accuracy_pass_rate
        summary_sheet.set_column('E:E', 10)  # new_passed
        summary_sheet.set_column('F:F', 10)  # new_failed
        summary_sheet.set_column('G:G', 12, workbook.add_format({'num_format': '0.00%'}))  # performance_pass_rate
        summary_sheet.set_column('H:H', 12, workbook.add_format({'num_format': '0.00'}))   # performance_geomean
        summary_sheet.set_column('I:I', 8)   # improved
        summary_sheet.set_column('J:J', 8)   # dropped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch Inductor test results.')
    parser.add_argument('baseline_dir', help='Directory containing baseline CSV files')
    parser.add_argument('target_dir', help='Directory containing target CSV files')
    parser.add_argument('output', help='Output Excel file (should end with .xlsx)')
    args = parser.parse_args()

    # Load and merge data
    print("Loading and merging data...")
    accuracy_data, performance_data = load_and_merge(args.baseline_dir, args.target_dir)

    if not accuracy_data and not performance_data:
        print("No matching files found. Exiting.")
        return

    # Build detailed sheets
    print("Building detailed accuracy sheet...")
    acc_detail_rows = []
    for (suite, dtype, mode), (base_df, target_df) in accuracy_data.items():
        acc_detail_rows.append(compare_accuracy(base_df, target_df, suite, dtype, mode))
    acc_detail_df = pd.concat(acc_detail_rows, ignore_index=True) if acc_detail_rows else pd.DataFrame()

    print("Building detailed performance sheet...")
    perf_detail_rows = []
    for (suite, dtype, mode), (base_df, target_df) in performance_data.items():
        perf_detail_rows.append(compare_performance(base_df, target_df, suite, dtype, mode))
    perf_detail_df = pd.concat(perf_detail_rows, ignore_index=True) if perf_detail_rows else pd.DataFrame()

    # Build comparison sheets
    print("Building comparison sheets...")
    acc_comp_df, perf_comp_df = build_comparison_sheets(accuracy_data, performance_data)

    # Build summary
    print("Building summary...")
    summary_df = build_summary(acc_comp_df, perf_comp_df)

    # Write Excel
    print(f"Writing report to {args.output}...")
    write_excel_report(args.output, acc_detail_df, perf_detail_df, acc_comp_df, perf_comp_df, summary_df)
    print("Done.")


if __name__ == '__main__':
    main()
    