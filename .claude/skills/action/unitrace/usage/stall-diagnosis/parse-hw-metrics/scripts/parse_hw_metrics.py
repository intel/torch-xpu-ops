#!/usr/bin/env python3
"""Parse unitrace hardware-counter metric CSV output.

Handles the comma-in-kernel-name ambiguity by parsing columns from
right-to-left: since all columns except the first (Kernel) are numeric,
we split from the right using the known column count from the header.

Usage:
    python parse_hw_metrics.py <logfile> [options]

Examples:
    # Show table of a specific metric
    python parse_hw_metrics.py metrics.log --metric "GPU_MEMORY_BYTE_READ[bytes]"

    # Summary statistics per kernel, skipping warmup
    python parse_hw_metrics.py metrics.log --metric "XVE_STALL_SBID[%]" --summary

    # Export all parsed data as CSV
    python parse_hw_metrics.py metrics.log --format csv > parsed.csv
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def find_metrics_sections(lines):
    """Find all '=== Device #N Metrics ===' sections in the log.

    Returns list of (device_id, header_line_idx, data_start_idx).
    """
    sections = []
    for i, line in enumerate(lines):
        m = re.match(r"^=== Device #(\d+) Metrics ===$", line.strip())
        if m:
            device_id = int(m.group(1))
            # Header is next non-empty line
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    sections.append((device_id, j, j + 1))
                    break
    return sections


def parse_section(lines, header_idx, data_start_idx):
    """Parse a single metrics section.

    Returns (header_fields, rows) where each row is a dict mapping
    column name -> value string.

    Key insight: parse from right-to-left. The header tells us
    there are N columns. All columns except the first (Kernel) are
    numeric and never contain commas. So for each data row:
      - Split by ','
      - Take the last N-1 fields as metric values
      - Join the remaining leftmost fields as the kernel name
    """
    header_line = lines[header_idx].strip()
    header_fields = [h.strip() for h in header_line.split(",")]
    num_columns = len(header_fields)

    rows = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()

        # Stop at next section marker or end
        if not line:
            continue
        if line.startswith("=== Device #"):
            break

        fields = line.split(",")

        # Must have at least as many fields as header columns
        if len(fields) < num_columns:
            continue

        # --- Right-to-left parsing ---
        # Last (num_columns - 1) fields are the numeric metric values
        metric_values = fields[-(num_columns - 1):]
        # Everything before that is the kernel name (may contain commas)
        kernel_parts = fields[: len(fields) - (num_columns - 1)]
        kernel_name = ",".join(kernel_parts).strip().strip('"')

        row = {"Kernel": kernel_name}
        for col_name, value in zip(header_fields[1:], metric_values):
            row[col_name] = value.strip()
        rows.append(row)

    return header_fields, rows


def parse_log(log_path):
    """Parse a unitrace log file. Returns list of (device_id, header, rows)."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sections = find_metrics_sections(lines)
    if not sections:
        print(f"ERROR: No '=== Device #N Metrics ===' section found in {log_path}",
              file=sys.stderr)
        sys.exit(1)

    results = []
    for device_id, header_idx, data_start_idx in sections:
        header, rows = parse_section(lines, header_idx, data_start_idx)
        results.append((device_id, header, rows))

    return results


def filter_rows(rows, kernel_pattern=None, skip_first=0):
    """Filter rows by kernel pattern and skip warmup instances."""
    if kernel_pattern:
        rows = [r for r in rows if kernel_pattern in r["Kernel"]]

    if skip_first > 0:
        # Group by kernel, skip first N instances of each
        from collections import defaultdict
        groups = defaultdict(list)
        for row in rows:
            groups[row["Kernel"]].append(row)
        filtered = []
        for kernel, kernel_rows in groups.items():
            filtered.extend(kernel_rows[skip_first:])
        rows = filtered

    return rows


def format_table(header_fields, rows, metric=None):
    """Format rows as an aligned text table."""
    if metric:
        # Show only Kernel + selected metric
        if metric not in header_fields:
            # Try partial match
            matches = [h for h in header_fields if metric.lower() in h.lower()]
            if matches:
                metric = matches[0]
            else:
                print(f"ERROR: Metric '{metric}' not found in header.", file=sys.stderr)
                print(f"Available: {', '.join(header_fields[1:])}", file=sys.stderr)
                sys.exit(1)
        display_cols = ["Kernel", metric]
    else:
        display_cols = header_fields

    # Compute column widths
    col_widths = {}
    for col in display_cols:
        values = [row.get(col, "") for row in rows]
        col_widths[col] = max(len(col), max((len(v) for v in values), default=0))

    # Print header
    header_str = "  ".join(col.ljust(col_widths[col]) for col in display_cols)
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in rows:
        line = "  ".join(
            row.get(col, "").ljust(col_widths[col]) for col in display_cols
        )
        print(line)


def format_csv(header_fields, rows, metric=None):
    """Format rows as CSV."""
    if metric:
        if metric not in header_fields:
            matches = [h for h in header_fields if metric.lower() in h.lower()]
            if matches:
                metric = matches[0]
            else:
                print(f"ERROR: Metric '{metric}' not found.", file=sys.stderr)
                sys.exit(1)
        display_cols = ["Kernel", metric]
    else:
        display_cols = header_fields

    print(",".join(display_cols))
    for row in rows:
        values = []
        for col in display_cols:
            v = row.get(col, "")
            # Quote kernel name if it contains commas
            if col == "Kernel" and "," in v:
                v = f'"{v}"'
            values.append(v)
        print(",".join(values))


def format_json(header_fields, rows, metric=None):
    """Format rows as JSON."""
    if metric:
        if metric not in header_fields:
            matches = [h for h in header_fields if metric.lower() in h.lower()]
            if matches:
                metric = matches[0]
            else:
                print(f"ERROR: Metric '{metric}' not found.", file=sys.stderr)
                sys.exit(1)
        output = []
        for row in rows:
            output.append({"Kernel": row["Kernel"], metric: row.get(metric, "")})
    else:
        output = rows

    print(json.dumps(output, indent=2))


def show_summary(rows, metric):
    """Show per-kernel statistics for a specific metric."""
    from collections import defaultdict

    if not metric:
        print("ERROR: --metric is required with --summary", file=sys.stderr)
        sys.exit(1)

    # Find exact or partial metric match from rows
    if rows:
        available = [k for k in rows[0].keys() if k != "Kernel"]
        if metric not in available:
            matches = [h for h in available if metric.lower() in h.lower()]
            if matches:
                metric = matches[0]
            else:
                print(f"ERROR: Metric '{metric}' not found.", file=sys.stderr)
                print(f"Available: {', '.join(available)}", file=sys.stderr)
                sys.exit(1)

    groups = defaultdict(list)
    for row in rows:
        try:
            val = float(row[metric])
            groups[row["Kernel"]].append(val)
        except (ValueError, KeyError):
            pass

    if not groups:
        print("No numeric data found for the specified metric.", file=sys.stderr)
        sys.exit(1)

    print(f"Metric: {metric}")
    print(f"{'Kernel':<60} {'Count':>6} {'Median':>12} {'Mean':>12} {'Min':>12} {'Max':>12}")
    print("-" * 120)

    for kernel, values in sorted(groups.items()):
        n = len(values)
        med = statistics.median(values)
        mean = statistics.mean(values)
        mn = min(values)
        mx = max(values)
        # Truncate long kernel names
        k_display = kernel if len(kernel) <= 58 else kernel[:55] + "..."
        print(f"{k_display:<60} {n:>6} {med:>12.4f} {mean:>12.4f} {mn:>12.4f} {mx:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse unitrace hardware-counter metric CSV output. "
                    "Handles commas in kernel names by parsing right-to-left."
    )
    parser.add_argument("logfile", help="Path to unitrace log file")
    parser.add_argument("--metric", "-m", help="Extract a specific metric column")
    parser.add_argument("--kernel", "-k", help="Filter by kernel name substring")
    parser.add_argument(
        "--skip-first", "-s", type=int, default=1,
        help="Skip first N instances per kernel (warmup). Default: 1"
    )
    parser.add_argument(
        "--format", "-f", choices=["table", "csv", "json"], default="table",
        help="Output format. Default: table"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show per-kernel summary statistics (median, mean, min, max)"
    )
    parser.add_argument(
        "--device", "-d", type=int, default=None,
        help="Select specific device ID (default: first device)"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Do not skip any instances (override --skip-first)"
    )

    args = parser.parse_args()

    if args.no_skip:
        args.skip_first = 0

    # Parse log
    results = parse_log(args.logfile)

    # Select device
    if args.device is not None:
        selected = [(d, h, r) for d, h, r in results if d == args.device]
        if not selected:
            available = [d for d, _, _ in results]
            print(f"ERROR: Device #{args.device} not found. Available: {available}",
                  file=sys.stderr)
            sys.exit(1)
        results = selected

    # Use first matching device
    device_id, header_fields, rows = results[0]

    if len(results) > 1 and args.device is None:
        print(f"Note: Multiple devices found. Using Device #{device_id}. "
              f"Use --device to select.", file=sys.stderr)

    # Apply filters
    rows = filter_rows(rows, kernel_pattern=args.kernel, skip_first=args.skip_first)

    if not rows:
        print("No data rows after filtering.", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.summary:
        show_summary(rows, args.metric)
    elif args.format == "table":
        format_table(header_fields, rows, metric=args.metric)
    elif args.format == "csv":
        format_csv(header_fields, rows, metric=args.metric)
    elif args.format == "json":
        format_json(header_fields, rows, metric=args.metric)


if __name__ == "__main__":
    main()
