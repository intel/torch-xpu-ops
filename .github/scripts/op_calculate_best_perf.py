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

def safe_float_convert(value):
    try:
        return float(value) if value.strip() else None
    except (ValueError, AttributeError):
        return None

def update_baseline(xpu_file, baseline_file, remove_missing=False):
    with open(xpu_file) as f:
        xpu_reader = csv.DictReader(f, delimiter=';')
        xpu_rows = list(xpu_reader)
        xpu_fieldnames = xpu_reader.fieldnames
        time_fields = ['time(us)', 'E2E total time(us)', 'E2E forward time(us)']
        fieldnames = [f for f in xpu_fieldnames if f not in time_fields]
        xpu_data = {}
        for row in xpu_rows:
            key = make_key(row, fieldnames)
            time_values = {}
            if 'time(us)' in row:
                time_val = safe_float_convert(row['time(us)'])
                if time_val is not None:
                    time_values['time(us)'] = time_val
            if 'E2E total time(us)' in row:
                e2e_val = safe_float_convert(row['E2E total time(us)'])
                if e2e_val is not None:
                    time_values['E2E total time(us)'] = e2e_val
            xpu_data[key] = (time_values, row)

    with open(baseline_file) as f:
        baseline_reader = csv.DictReader(f, delimiter=';')
        baseline_rows = list(baseline_reader)
        baseline_fieldnames = baseline_reader.fieldnames

    # To add new parameter of new ops into baseline file
    all_fieldnames = list(set(xpu_fieldnames + baseline_fieldnames))
    # Maintain original order as much as possible
    ordered_fieldnames = []
    for f in xpu_fieldnames:
        if f in all_fieldnames and f not in ordered_fieldnames:
            ordered_fieldnames.append(f)
    for f in baseline_fieldnames:
        if f in all_fieldnames and f not in ordered_fieldnames:
            ordered_fieldnames.append(f)

    baseline_keys = {make_key(row, fieldnames) for row in baseline_rows}
    xpu_keys = set(xpu_data.keys())

    # Resolve existing cases
    for row in baseline_rows:
        key = make_key(row, fieldnames)
        if key in xpu_data:
            xpu_times, xpu_row = xpu_data[key]
            updated_row = {}

            # Copy all fields from baseline first
            for field in ordered_fieldnames:
                updated_row[field] = row.get(field, '')

            # Update with xpu values where they exist
            for field in ordered_fieldnames:
                if field in xpu_row and xpu_row[field]:
                    updated_row[field] = xpu_row[field]

            # Handle time fields
            updated = False
            if 'time(us)' in xpu_times and 'time(us)' in row:
                baseline_time = safe_float_convert(row['time(us)'])
                if baseline_time is not None:
                    xpu_time = xpu_times['time(us)']
                    if xpu_time < baseline_time:
                        updated_row['time(us)'] = str(xpu_time)
                        updated = True

            if 'E2E total time(us)' in xpu_times and 'E2E total time(us)' in row:
                baseline_e2e = safe_float_convert(row['E2E total time(us)'])
                if baseline_e2e is not None:
                    xpu_e2e = xpu_times['E2E total time(us)']
                    if xpu_e2e < baseline_e2e:
                        updated_row['E2E total time(us)'] = str(xpu_e2e)
                        updated = True

            if updated:
                updated_cases.append((key, row, updated_row))
            updated_rows.append(updated_row)
        elif not remove_missing:
            ordered_row = {}
            for field in ordered_fieldnames:
                ordered_row[field] = row.get(field, '')
            updated_rows.append(ordered_row)

    # Add new cases
    for key in xpu_keys - baseline_keys:
        xpu_times, xpu_row = xpu_data[key]
        new_row = {}
        for field in ordered_fieldnames:
            new_row[field] = xpu_row.get(field, '')

        if 'time(us)' in xpu_times:
            new_row['time(us)'] = str(xpu_times['time(us)'])
        if 'E2E total time(us)' in xpu_times:
            new_row['E2E total time(us)'] = str(xpu_times['E2E total time(us)'])

        updated_rows.append(new_row)
        added_cases.append((key, xpu_times, new_row))

    # Resolve removed cases
    if remove_missing:
        for key in baseline_keys - xpu_keys:
            removed_case = next(row for row in baseline_rows if make_key(row, fieldnames) == key)
            removed_cases.append((key, removed_case))

    if added_cases:
        print(f"\nAdded {len(added_cases)} new case(s):")
        for key, times, row in added_cases:
            print(f"\n[New Case] {format_case(key)}")
            if 'time(us)' in times:
                print(f"Time: {times['time(us)']} us")
            if 'E2E total time(us)' in times:
                print(f"E2E Time: {times['E2E total time(us)']} us")
            print("Parameters:")
            for k, v in row.items():
                if k not in time_fields:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if updated_cases:
        print(f"\nUpdated {len(updated_cases)} case(s):")
        for key, old_row, new_row in updated_cases:
            print(f"\n[Updated] {format_case(key)}")
            if 'time(us)' in old_row and 'time(us)' in new_row:
                old_time = safe_float_convert(old_row['time(us)'])
                new_time = safe_float_convert(new_row['time(us)'])
                if old_time is not None and new_time is not None and old_time != new_time:
                    print(f"Time: {old_time} us → {new_time} us")

            if 'E2E total time(us)' in old_row and 'E2E total time(us)' in new_row:
                old_e2e = safe_float_convert(old_row['E2E total time(us)'])
                new_e2e = safe_float_convert(new_row['E2E total time(us)'])
                if old_e2e is not None and new_e2e is not None and old_e2e != new_e2e:
                    print(f"E2E Time: {old_e2e} us → {new_e2e} us")

            print("Parameters:")
            for k, v in new_row.items():
                if k not in time_fields:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if remove_missing and removed_cases:
        print(f"\nRemoved {len(removed_cases)} case(s):")
        for key, row in removed_cases:
            print(f"\n[Removed] {format_case(key)}")
            if 'time(us)' in row:
                time_val = safe_float_convert(row['time(us)'])
                if time_val is not None:
                    print(f"Time: {time_val} us")
            if 'E2E total time(us)' in row:
                e2e_val = safe_float_convert(row['E2E total time(us)'])
                if e2e_val is not None:
                    print(f"E2E Time: {e2e_val} us")
            print("Parameters:")
            for k, v in row.items():
                if k not in time_fields:
                    print(f"  {k}: {v}")
        print("-" * 60)

    if not (added_cases or updated_cases or (remove_missing and removed_cases)):
        print("\nNo changes detected between files.")

    backup_file = baseline_file.replace('.csv', '_backup.csv')
    Path(baseline_file).rename(backup_file)

    with open(baseline_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames, delimiter=';')
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
