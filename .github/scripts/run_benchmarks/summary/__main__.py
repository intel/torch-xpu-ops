"""Entry point for `python -m run_benchmarks.compare`.

Compare PyTorch Dynamo Benchmark test results (target vs baseline).

Supports comparing two directories of CSV results, or generating a report
from a single directory (treating the missing side as empty).
"""

import argparse
import logging
import os
import sys

from .constants import (
    DEFAULT_PERF_THRESHOLD,
    KNOWN_DATA_TYPES,
    KNOWN_MODES,
    KNOWN_SUITES,
)
from .loader import find_result_files, load_results
from .merge import (
    generate_summary,
    merge_accuracy,
    merge_performance,
    merge_pt2e_accuracy,
    merge_pt2e_performance,
)
from .report import print_report, write_csv, write_excel, write_markdown

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m run_benchmarks.compare",
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
