"""
To update the op perf baseline, use the better performance value
# usage
python op_calculate_best_perf.py --xpu /path/to/xpu/performance/result/dir/forward.csv --baseline /path/to/baseline/dir/new_baseline.csv -r

"""

import csv
import argparse
from pathlib import Path

updated_rows = []
added_cases = []
updated_cases = []
removed_cases = []

def update_baseline(xpu_file, baseline_file, remove_missing=False):
    with open(xpu_file) as f:
        xpu_reader = csv.DictReader(f, delimiter=';')
        xpu_rows = list(xpu_reader)
        xpu_fieldnames = xpu_reader.fieldnames  # Keep original field order
        fieldnames = [f for f in xpu_fieldnames if f not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']]
        xpu_data = {make_key(row, fieldnames): (float(row['time(us)']), row) for row in xpu_rows}

    with open(baseline_file) as f:
        baseline_reader = csv.DictReader(f, delimiter=';')
        baseline_rows = list(baseline_reader)
        baseline_fieldnames = baseline_reader.fieldnames

    # To add new parameter of new ops into baseline file
    all_fieldnames = xpu_fieldnames + [f for f in baseline_fieldnames if f not in xpu_fieldnames]
    fieldnames = [f for f in all_fieldnames if f not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']]

    baseline_keys = {make_key(row, fieldnames) for row in baseline_rows}
    xpu_keys = set(xpu_data.keys())

    # Resolve existing cases
    for row in baseline_rows:
        key = make_key(row, fieldnames)
        if key in xpu_data:
            xpu_time, xpu_row = xpu_data[key]
            baseline_time = float(row['time(us)'])

            if xpu_time < baseline_time:
                updated_row = {}
                for field in all_fieldnames:
                    updated_row[field] = xpu_row.get(field, row.get(field, ''))
                updated_row['time(us)'] = str(xpu_time)
                if 'E2E total time(us)' in row:
                    updated_row['E2E total time(us)'] = row['E2E total time(us)']
                updated_cases.append((key, baseline_time, xpu_time, updated_row))
                updated_rows.append(updated_row)
            else:
                ordered_row = {}
                for field in all_fieldnames:
                    ordered_row[field] = row.get(field, '')
                updated_rows.append(ordered_row)
        elif not remove_missing:
            ordered_row = {}
            for field in all_fieldnames:
                ordered_row[field] = row.get(field, '')
            updated_rows.append(ordered_row)

    # Add new cases
    for key in xpu_keys - baseline_keys:
        xpu_time, xpu_row = xpu_data[key]
        new_row = {}
        for field in all_fieldnames:
            new_row[field] = xpu_row.get(field, '')
        new_row['time(us)'] = str(xpu_time)
        updated_rows.append(new_row)
        added_cases.append((key, xpu_time, new_row))

    # Resolve removed cases
    if remove_missing:
        for key in baseline_keys - xpu_keys:
            removed_case = next(row for row in baseline_rows if make_key(row, fieldnames) == key)
            removed_cases.append((key, float(removed_case['time(us)']), removed_case))

    if added_cases:
        print(f"\nAdded {len(added_cases)} new case(s):")
        for key, time, row in added_cases:
            print(f"\n[New Case] {format_case(key)}")
            print(f"Time: {time} us")
            print("Parameters:")
            for k, v in row.items():
                if k not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if updated_cases:
        print(f"\nUpdated {len(updated_cases)} case(s):")
        for key, old_time, new_time, row in updated_cases:
            print(f"\n[Updated] {format_case(key)}")
            print(f"Time: {old_time} us → {new_time} us")
            print("Parameters:")
            for k, v in row.items():
                if k not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if remove_missing and removed_cases:
        print(f"\nRemoved {len(removed_cases)} case(s):")
        for key, time, row in removed_cases:
            print(f"\n[Removed] {format_case(key)}")
            print(f"Time: {time} us")
            print("Parameters:")
            for k, v in row.items():
                if k not in ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if not (added_cases or updated_cases or (remove_missing and removed_cases)):
        print("\nNo changes detected between files.")

    backup_file = baseline_file.replace('.csv', '_backup.csv')
    Path(baseline_file).rename(backup_file)

    with open(baseline_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(updated_rows)

    print("\n" + "-" * 80)
    print(f"Update complete! Total cases in new baseline: {len(updated_rows)}")
    print(f"Updated baseline saved to {baseline_file}")
    print(f"Original backup created at {backup_file}")

def make_key(row, fieldnames):
    return tuple(str(row.get(field, '')) for field in fieldnames)

def format_case(key):
    return f"{key[0]} | {key[1]} | {key[2]} (shape: {key[3]})"

def main():
    parser = argparse.ArgumentParser(description='Compare and synchronize operation performance data')
    parser.add_argument('-x', '--xpu', required=True, help='Path to xpu_op_summary.csv')
    parser.add_argument('-b', '--baseline', required=True, help='Path to baseline_op_summary.csv')
    parser.add_argument('-r', '--remove-missing', action='store_true',
                      help='Remove cases not present in xpu file')

    args = parser.parse_args()

    if not Path(args.xpu).exists():
        print(f"Error: XPU file not found at {args.xpu}")
        return
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found at {args.baseline}")
        return

    update_baseline(args.xpu, args.baseline, args.remove_missing)


if __name__ == "__main__":
    main()
