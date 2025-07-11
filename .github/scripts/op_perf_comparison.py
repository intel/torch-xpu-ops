"""
To compare the op perf diff
# usage
python op_perf_comparison.py --xpu_file /path/to/xpu/performance/result/dir/forward.csv --baseline_file /path/to/baselineence/dir/baseline.csv

"""

import pandas as pd
import argparse
import os
from ast import literal_eval
from tabulate import tabulate

def preprocess_row(row):
    processed = {}
    for col, val in row.items():
        if pd.isna(val):
            processed[col] = "NULL"
        else:
            try:
                processed[col] = literal_eval(str(val))
            except (ValueError, SyntaxError):
                processed[col] = val
    return processed

def display_row(record):
    formatted = {}
    for key, value in record.items():
        if isinstance(value, (list, tuple, dict)):
            formatted[key] = str(value)
        elif value == "NULL":
            formatted[key] = "NULL"
        else:
            formatted[key] = value
    return formatted

def write_to_github_summary(content):
    github_step_summary = os.getenv('GITHUB_STEP_SUMMARY')
    if github_step_summary:
        with open(github_step_summary, 'a') as f:
            f.write(content + "\n")

def display_comparison(results, threshold):
    if results.empty:
        print(f"\n No outlier exceeding ({threshold:.0%})")
        write_to_github_summary(f"## No outlier exceeding ({threshold:.0%})")
        return

    regression = results[results['change'] == '↓']
    improvement = results[results['change'] == '↑']

    if not regression.empty:
        print("\n🔴 Regression:")
        display_records = []
        for _, row in regression.iterrows():
            record = display_row(row)
            display_records.append({
                **{k: v for k, v in record.items() if k not in ['time_xpu_file', 'time_baseline_file', 'difference', 'change']},
                'Current Time(us)': record['time_xpu_file'],
                'Baseline Time(us)': record['time_baseline_file'],
                'Difference': record['difference']
            })

        print(tabulate(
            display_records,
            headers="keys",
            tablefmt='grid',
            showindex=False,
            floatfmt=".2f"
        ))

    if not improvement.empty:
        print("\n🟢 Improvement:")
        display_records = []
        for _, row in improvement.iterrows():
            record = display_row(row)
            display_records.append({
                **{k: v for k, v in record.items() if k not in ['time_xpu_file', 'time_baseline_file', 'difference', 'change']},
                'Current Time(us)': record['time_xpu_file'],
                'Baseline Time(us)': record['time_baseline_file'],
                'Difference': record['difference']
            })

        print(tabulate(
            display_records,
            headers="keys",
            tablefmt='grid',
            showindex=False,
            floatfmt=".2f"
        ))
    # Print Summary on Github Action Summary
    summary_output = "## Performance Comparison Results\n"
    if not regression.empty:
        summary_output += "\n### 🔴 Regression\n"
        display_records = []
        for _, row in regression.iterrows():
            record = display_row(row)
            display_records.append({
                **{k: v for k, v in record.items() if k not in ['time_xpu_file', 'time_baseline_file', 'difference', 'change']},
                'Current Time(us)': record['time_xpu_file'],
                'Baseline Time(us)': record['time_baseline_file'],
                'Difference': record['difference']
            })

        summary_output += tabulate(
            display_records,
            headers="keys",
            tablefmt='github',
            showindex=False,
            floatfmt=".2f"
        ) + "\n"

    if not improvement.empty:
        summary_output += "\n### 🟢 Improvement\n"
        display_records = []
        for _, row in improvement.iterrows():
            record = display_row(row)
            display_records.append({
                **{k: v for k, v in record.items() if k not in ['time_xpu_file', 'time_baseline_file', 'difference', 'change']},
                'Current Time(us)': record['time_xpu_file'],
                'Baseline Time(us)': record['time_baseline_file'],
                'Difference': record['difference']
            })

        summary_output += tabulate(
            display_records,
            headers="keys",
            tablefmt='github',
            showindex=False,
            floatfmt=".2f"
        ) + "\n"

    write_to_github_summary(summary_output)

def compare_op_time_values(xpu_file, baseline_file, threshold=0.05, output_file=None):
    df_xpu = pd.read_csv(xpu_file, sep=';')
    df_baseline = pd.read_csv(baseline_file, sep=';')

    records_xpu = [preprocess_row(row) for _, row in df_xpu.iterrows()]
    records_baseline = [preprocess_row(row) for _, row in df_baseline.iterrows()]

    dict_xpu = {
        tuple((k, str(v)) for k, v in record.items() if k != 'time(us)'):
        record['time(us)']
        for record in records_xpu
    }
    dict_baseline = {
        tuple((k, str(v)) for k, v in record.items() if k != 'time(us)'):
        record['time(us)']
        for record in records_baseline
    }
    common_keys = set(dict_xpu.keys()) & set(dict_baseline.keys())
    results = []

    for key in common_keys:
        time_xpu = dict_xpu[key]
        time_baseline = dict_baseline[key]

        # Skip comparison if time_xpu is 0
        if time_xpu == 0:
            continue

        diff = (time_baseline - time_xpu) / time_xpu
        # Compare Time, Lower is better
        if abs(diff) > threshold:
            record = dict(key)
            print(record)
            record.update({
                'time_xpu_file': time_xpu,
                'time_baseline_file': time_baseline,
                'difference': f"{diff:.2%}",
                'change': "↑" if diff > 0 else "↓"
            })
            results.append(record)

    result_df = pd.DataFrame(results) if results else pd.DataFrame()
    display_comparison(result_df, threshold)


def main():
    parser = argparse.ArgumentParser(description='Compare time values between two CSV files')
    parser.add_argument('-x', '--xpu_file', required=True, help='XPU OP performance result csv files dir')
    parser.add_argument('-b', '--baseline_file', required=True, help="XPU OP baseline result csv files dir")
    parser.add_argument('-t', '--threshold', type=float, default=0.10,
                       help='Threshold for time difference (default: 0.10 for 10%)')
    args = parser.parse_args()

    print(f" Compared file: {args.xpu_file} 和 {args.baseline_file}")
    print(f" Threshold: {args.threshold:.0%}")
    write_to_github_summary("## Performance Comparison Set")
    write_to_github_summary(f"- Threshold: {args.threshold:.0%}")

    compare_op_time_values(
        xpu_file=args.xpu_file,
        baseline_file=args.baseline_file,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
