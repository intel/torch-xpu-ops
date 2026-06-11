"""Entry point for `python -m run_benchmarks.summary`.

Compare PyTorch Dynamo Benchmark test results (target vs baseline).

Supports comparing two directories of CSV results, or generating a report
from a single directory (treating the missing side as empty).
"""

import argparse
import logging
import os
import sys

import pandas as pd

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
    _geomean,
)
from .report import print_report, write_csv, write_excel, write_markdown

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m run_benchmarks.summary",
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
        write_excel(summary, acc_merged, perf_merged, args.output,
                    pt2e_acc=acc_pt2e, pt2e_perf=perf_pt2e)
    else:
        write_csv(summary, acc_merged, perf_merged, out_base)

    # Write pt2e CSV outputs separately (xlsx already includes pt2e sheets)
    has_pt2e = not acc_pt2e.empty or not perf_pt2e.empty
    if has_pt2e and out_ext == ".csv":
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

        has_baseline = bool(args.baseline_dir)

        def _passrate(df: pd.DataFrame, col: str) -> str:
            if col not in df.columns:
                return "n/a"
            vals = pd.to_numeric(df[col], errors="coerce")
            total = len(vals)
            passed = int((vals > 0).sum())
            pct = (100.0 * passed / total) if total else 0.0
            return f"{passed}/{total} ({pct:.1f}%)"

        def _geo(df: pd.DataFrame, col: str) -> str:
            if col not in df.columns:
                return "n/a"
            g = _geomean(pd.to_numeric(df[col], errors="coerce"))
            return f"{g:.4f}" if pd.notna(g) else "n/a"

        def _geo_ratio(df: pd.DataFrame, dtype: str) -> str:
            tcol, bcol = f"{dtype}_target", f"{dtype}_baseline"
            if tcol not in df.columns or bcol not in df.columns:
                return "n/a"
            tgt = pd.to_numeric(df[tcol], errors="coerce")
            bsl = pd.to_numeric(df[bcol], errors="coerce")
            ratio = tgt / bsl
            ratio = ratio[(bsl > 0) & (tgt > 0)]
            g = _geomean(ratio)
            return f"{g:.4f}" if pd.notna(g) else "n/a"

        if not acc_pt2e.empty:
            print(f"\n  Accuracy: {len(acc_pt2e)} rows")
            print(f"    passrate target:   {_passrate(acc_pt2e, 'int8_target')}")
            if has_baseline:
                print(f"    passrate baseline: {_passrate(acc_pt2e, 'int8_baseline')}")
        if not perf_pt2e.empty:
            print(f"\n  Performance: {len(perf_pt2e)} rows")
            print(f"    passrate target:   {_passrate(perf_pt2e, 'symm_target')}")
            if has_baseline:
                print(f"    passrate baseline: {_passrate(perf_pt2e, 'symm_baseline')}")
                print("    geomean (target/baseline): "
                      f"fp32={_geo_ratio(perf_pt2e, 'fp32')}, "
                      f"symm={_geo_ratio(perf_pt2e, 'symm')}, "
                      f"asymm={_geo_ratio(perf_pt2e, 'asymm')}")
            else:
                print("    geomean target: "
                      f"fp32={_geo(perf_pt2e, 'fp32_target')}, "
                      f"symm={_geo(perf_pt2e, 'symm_target')}, "
                      f"asymm={_geo(perf_pt2e, 'asymm_target')}")
        print()

    # ── Regression check (failures the target itself has) ──
    def _by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty or "comparison" not in df.columns:
            return df.iloc[0:0]
        return df[df["comparison"] == label]

    def _acc_target_fail(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "accuracy_target" not in df.columns:
            return df.iloc[0:0]
        tgt = df["accuracy_target"].astype(str)
        tgt_fail = df["accuracy_target"].notna() & ~tgt.str.contains("pass") & (tgt.str.strip() != "")
        # ignore cases where baseline also failed (not a regression)
        if "accuracy_baseline" in df.columns:
            bsl = df["accuracy_baseline"].astype(str)
            bsl_fail = df["accuracy_baseline"].notna() & ~bsl.str.contains("pass") & (bsl.str.strip() != "")
            return df[tgt_fail & ~bsl_fail]
        return df[tgt_fail]

    def _num_target_fail(df: pd.DataFrame, col: str, bsl_col: str | None = None) -> pd.DataFrame:
        if df.empty or col not in df.columns:
            return df.iloc[0:0]
        vals = pd.to_numeric(df[col], errors="coerce")
        # present but not positive (ignore target-null cases)
        tgt_fail = vals.notna() & ~(vals > 0)
        # ignore cases where baseline also failed (not a regression)
        if bsl_col and bsl_col in df.columns:
            bvals = pd.to_numeric(df[bsl_col], errors="coerce")
            bsl_fail = bvals.notna() & ~(bvals > 0)
            return df[tgt_fail & ~bsl_fail]
        return df[tgt_fail]

    def _model_id(row: pd.Series) -> str:
        parts = [str(row.get(c)) for c in ("suite", "data_type", "mode", "model")
                 if c in row.index and pd.notna(row.get(c))]
        return ",".join(parts)

    def _fmt_num(v) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if pd.isna(f):
            return "n/a"
        return f"{f:.4f}".rstrip("0").rstrip(".")

    def _ratio(num, den) -> str:
        try:
            n, d = float(num), float(den)
        except (TypeError, ValueError):
            return "n/a"
        if pd.isna(n) or pd.isna(d) or d == 0:
            return "n/a"
        return f"{n / d:.3f}"

    def _detail(kind: str, row: pd.Series) -> str:
        if kind == "acc":
            return f"accuracy={row.get('accuracy_target')}"
        if kind == "perf_fail":
            return (f"target={_fmt_num(row.get('inductor_target'))}, "
                    f"baseline={_fmt_num(row.get('inductor_baseline'))}")
        if kind == "perf_drop":
            return (f"ratio={_fmt_num(row.get('inductor_ratio'))}  "
                    f"(target={_fmt_num(row.get('inductor_target'))}, "
                    f"baseline={_fmt_num(row.get('inductor_baseline'))})")
        if kind == "pt2e_acc":
            return (f"int8={_fmt_num(row.get('int8_target'))}, "
                    f"baseline={_fmt_num(row.get('int8_baseline'))}")
        if kind == "pt2e_perf_fail":
            return (f"symm_target={_fmt_num(row.get('symm_target'))}, "
                    f"baseline={_fmt_num(row.get('symm_baseline'))}")
        if kind == "pt2e_perf_drop":
            return (f"ratio={_ratio(row.get('symm_target'), row.get('symm_baseline'))}  "
                    f"(symm_target={_fmt_num(row.get('symm_target'))}, "
                    f"baseline={_fmt_num(row.get('symm_baseline'))})")
        return ""

    regressions = [
        ("New accuracy failures", _acc_target_fail(acc_merged), "acc"),
        ("New performance failures", _by_label(perf_merged, "new_failed"), "perf_fail"),
        ("New performance drops", _by_label(perf_merged, "new_dropped"), "perf_drop"),
        ("New pt2e accuracy failures", _num_target_fail(acc_pt2e, "int8_target", "int8_baseline"), "pt2e_acc"),
        ("New pt2e performance failures", _by_label(perf_pt2e, "new_failed"), "pt2e_perf_fail"),
        ("New pt2e performance drops", _by_label(perf_pt2e, "new_dropped"), "pt2e_perf_drop"),
    ]

    total_regressions = sum(len(df) for _, df, _ in regressions)
    if total_regressions:
        # Compute alignment width across all reported models
        id_width = max(
            (len(_model_id(row)) for _, df, _ in regressions for _, row in df.iterrows()),
            default=0,
        )
        bar = "=" * 72
        print("\n" + bar)
        print(" REGRESSIONS DETECTED")
        print(bar)
        for title, df, kind in regressions:
            if df.empty:
                continue
            print(f"\n  {title}: {len(df)}")
            print(f"    {'Cases':<{id_width}}  Details")
            print(f"    {'-' * id_width}  {'-' * 6}")
            for _, row in df.iterrows():
                print(f"    {_model_id(row):<{id_width}}  {_detail(kind, row)}")
        print(f"\n  Total regressions: {total_regressions}")
        print(bar)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
