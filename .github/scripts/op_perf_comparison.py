"""
To compare the op perf diff
# usage
python op_perf_comparison.py --xpu_file /path/to/xpu/performance/result/dir/forward.csv --baseline_file /path/to/baselineence/dir/baseline.csv
--profile-only: Only compare record['time(us)']
--e2e-only: Only compare record['E2E total time(us)']
Default: Compare both record['time(us)'] and record['E2E total time(us)'] in same table
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
        if isinstance(value, list | tuple | dict):
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

def format_parameters(record):
    params = []
    for key, value in record.items():
        if key not in ['case_name', 'op_name', 'datatype',
                       'profile_time_xpu', 'profile_time_base', 'profile_diff', 'profile_change',
                       'e2e_time_xpu', 'e2e_time_base', 'e2e_diff', 'e2e_change',
                       'E2E total time(us)', 'E2E forward time(us)']:
            if value != "NULL":
                params.append(f"{key}: {value}")
    return "<br>".join(params)

def display_comparison(results, threshold, xpu_file, compare_both):
    if 'forward' in xpu_file.lower():
        direction = "Forward"
    elif 'backward' in xpu_file.lower():
        direction = "Backward"
    else:
        direction = "Operation"

    if results.empty:
        print(f"\n {direction} No outlier exceeding ({threshold:.0%})")
        write_to_github_summary(f"## {direction} No outlier exceeding ({threshold:.0%})")
        return

    # Prepare display records - always include both metrics when available
    display_records = []
    for _, row in results.iterrows():
        record = display_row(row)
        display_record = {
            'Case Name': record['case_name'],
            'Op Name': record['op_name'],
            'Datatype': record['datatype'],
            'Parameters': format_parameters(record)
        }

        # Always try to include profile time if it exists in the data
        if 'profile_time_xpu' in record or 'profile_time_base' in record:
            display_record.update({
                'Profile Current(us)': record.get('profile_time_xpu', 'N/A'),
                'Profile Baseline(us)': record.get('profile_time_base', 'N/A'),
                'Profile Diff': record.get('profile_diff', 'N/A'),
                'Profile Change': record.get('profile_change', '')
            })

        # Always try to include E2E time if it exists in the data
        if 'e2e_time_xpu' in record or 'e2e_time_base' in record:
            display_record.update({
                'E2E Current(us)': record.get('e2e_time_xpu', 'N/A'),
                'E2E Baseline(us)': record.get('e2e_time_base', 'N/A'),
                'E2E Diff': record.get('e2e_diff', 'N/A'),
                'E2E Change': record.get('e2e_change', '')
            })

        display_records.append(display_record)

    # Classify records based on changes
    regression_records = []
    improvement_records = []

    for record in results.to_dict('records'):
        has_profile_change = 'profile_change' in record and record['profile_change'] in ('↑', '↓')
        has_e2e_change = 'e2e_change' in record and record['e2e_change'] in ('↑', '↓')

        # If either metric shows regression, count as regression
        if (has_profile_change and record['profile_change'] == '↓') or \
           (has_e2e_change and record['e2e_change'] == '↓'):
            regression_records.append(record)
        # If either metric shows improvement, count as improvement
        elif (has_profile_change and record['profile_change'] == '↑') or \
             (has_e2e_change and record['e2e_change'] == '↑'):
            improvement_records.append(record)

    # Print results
    if regression_records:
        print("\n🔴 Regression:")
        regression_display = [r for r in display_records
                            if r['Case Name'] in [x['case_name'] for x in regression_records]]
        print(tabulate(
            regression_display,
            headers="keys",
            tablefmt='grid',
            showindex=False,
            floatfmt=".2f"
        ))

    if improvement_records:
        print("\n🟢 Improvement:")
        improvement_display = [r for r in display_records
                             if r['Case Name'] in [x['case_name'] for x in improvement_records]]
        print(tabulate(
            improvement_display,
            headers="keys",
            tablefmt='grid',
            showindex=False,
            floatfmt=".2f"
        ))

    # Generate GitHub summary
    summary_output = f"## {direction} Performance Comparison Results\n"

    if regression_records:
        summary_output += "\n### 🔴 Regression\n"
        summary_output += tabulate(
            [r for r in display_records if r['Case Name'] in [x['case_name'] for x in regression_records]],
            headers="keys",
            tablefmt='github',
            showindex=False,
            floatfmt=".2f"
        ) + "\n"

    if improvement_records:
        summary_output += "\n### 🟢 Improvement\n"
        summary_output += tabulate(
            [r for r in display_records if r['Case Name'] in [x['case_name'] for x in improvement_records]],
            headers="keys",
            tablefmt='github',
            showindex=False,
            floatfmt=".2f"
        ) + "\n"

    write_to_github_summary(summary_output)

def compare_time_values(xpu_file, baseline_file, threshold=0.05, profile_only=False, e2e_only=False):
    def prepare_df(df):
        df.columns = df.columns.str.strip()
        if 'time(us)' not in df.columns:
            df['time(us)'] = float('nan')
        if 'E2E total time(us)' not in df.columns:
            df['E2E total time(us)'] = float('nan')
        return df

    df_xpu = prepare_df(pd.read_csv(xpu_file, sep=';'))
    df_baseline = prepare_df(pd.read_csv(baseline_file, sep=';'))

    for col in ['time(us)', 'E2E total time(us)']:
        df_xpu[col] = pd.to_numeric(df_xpu[col], errors='coerce')
        df_baseline[col] = pd.to_numeric(df_baseline[col], errors='coerce')

    records_xpu = [preprocess_row(row) for _, row in df_xpu.iterrows()]
    records_baseline = [preprocess_row(row) for _, row in df_baseline.iterrows()]

    data_dict = {
        'xpu': {'profile': {}, 'e2e': {}},
        'baseline': {'profile': {}, 'e2e': {}}
    }

    for record, source in [(records_xpu, 'xpu'), (records_baseline, 'baseline')]:
        for r in record:
            key = tuple((k, str(v)) for k, v in r.items()
                   if k not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)'])

            for time_type in ['profile', 'e2e']:
                col = 'time(us)' if time_type == 'profile' else 'E2E total time(us)'
                if col in r:
                    try:
                        time_val = float(r[col])
                        if not pd.isna(time_val):
                            data_dict[source][time_type][key] = time_val
                    except (ValueError, TypeError):
                        continue

    results = []
    compare_both = not profile_only and not e2e_only
    all_keys = set().union(*[set(data_dict[s][t].keys())
                          for s in data_dict for t in data_dict[s]])

    for key in all_keys:
        record = dict(key)
        should_include = False

        if not e2e_only and key in data_dict['xpu']['profile'] and key in data_dict['baseline']['profile']:
            xpu_time = data_dict['xpu']['profile'][key]
            base_time = data_dict['baseline']['profile'][key]

            if xpu_time != 0 and base_time != 0:
                try:
                    diff = (base_time - xpu_time) / xpu_time
                    record.update({
                        'profile_time_xpu': xpu_time,
                        'profile_time_base': base_time,
                        'profile_diff': f"{diff:.2%}",
                        'profile_change': "↑" if diff > threshold else "↓" if diff < -threshold else ""
                    })
                    if abs(diff) > threshold:
                        should_include = True
                except (TypeError, ValueError):
                    pass

        if not profile_only and key in data_dict['xpu']['e2e'] and key in data_dict['baseline']['e2e']:
            xpu_time = data_dict['xpu']['e2e'][key]
            base_time = data_dict['baseline']['e2e'][key]

            if xpu_time != 0 and base_time != 0:
                try:
                    diff = (base_time - xpu_time) / xpu_time
                    record.update({
                        'e2e_time_xpu': xpu_time,
                        'e2e_time_base': base_time,
                        'e2e_diff': f"{diff:.2%}",
                        'e2e_change': "↑" if diff > threshold else "↓" if diff < -threshold else ""
                    })
                    if abs(diff) > threshold:
                        should_include = True
                except (TypeError, ValueError):
                    pass

        if compare_both:
            if should_include:
                results.append(record)
        else:
            if ((profile_only and 'profile_change' in record and record['profile_change']) or
                (e2e_only and 'e2e_change' in record and record['e2e_change'])):
                results.append(record)

    result_df = pd.DataFrame(results) if results else pd.DataFrame()
    display_comparison(result_df, threshold, xpu_file, compare_both)

def main():
    parser = argparse.ArgumentParser(description='Compare time values between two CSV files')
    parser.add_argument('-x', '--xpu_file', required=True, help='XPU OP performance result csv files dir')
    parser.add_argument('-b', '--baseline_file', required=True, help="XPU OP baseline result csv files dir")
    parser.add_argument('-t', '--threshold', type=float, default=0.10,
                       help='Threshold for time difference (default: 0.10 for 10%)')
    parser.add_argument('--profile-only', action='store_true',
                       help='Only compare profile time (time(us))')
    parser.add_argument('--e2e-only', action='store_true',
                       help='Only compare E2E time (E2E total time(us))')
    args = parser.parse_args()

    if args.profile_only and args.e2e_only:
        raise ValueError("Cannot specify both --profile-only and --e2e-only")

    print(f" Compared file: {args.xpu_file} and {args.baseline_file}")
    print(f" Threshold: {args.threshold:.0%}")
    if args.profile_only:
        print(" Comparing only profile time (time(us))")
    elif args.e2e_only:
        print(" Comparing only E2E time (E2E total time(us))")
    else:
        print(" Comparing both profile time and E2E time in same table")

    write_to_github_summary("## Performance Comparison Set")
    write_to_github_summary(f"- Threshold: {args.threshold:.0%}")
    if args.profile_only:
        write_to_github_summary("- Comparing only profile time (time(us))")
    elif args.e2e_only:
        write_to_github_summary("- Comparing only E2E time (E2E total time(us))")
    else:
        write_to_github_summary("- Comparing both profile time and E2E time in same table")

    compare_time_values(
        xpu_file=args.xpu_file,
        baseline_file=args.baseline_file,
        threshold=args.threshold,
        profile_only=args.profile_only,
        e2e_only=args.e2e_only
    )

if __name__ == "__main__":
    main()
