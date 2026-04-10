#!/usr/bin/env python3
"""
Unit test and end-to-end comparison for GitHub Actions.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from collections.abc import Callable

import pandas as pd


# Enums for status and change types
class TestStatus(Enum):
    PASSED = "passed"
    XFAIL = "xfail"
    FAILED = "failed"
    ERROR = "error"

    @classmethod
    def is_passed(cls, status: str) -> bool:
        return status in {cls.PASSED.value, cls.XFAIL.value}

    @classmethod
    def is_failed(cls, status: str) -> bool:
        return status in {cls.FAILED.value, cls.ERROR.value}


class AccuracyChange(Enum):
    NEW_PASSED = "new_passed"
    NEW_FAILED = "new_failed"


class PerformanceChange(Enum):
    NEW_PASSED = "new_passed"
    NEW_FAILED = "new_failed"
    NEW_IMPROVED = "new_improved"
    NEW_DROPPED = "new_dropped"


# Configuration dataclasses
@dataclass
class CSVConfig:
    """Configuration for processing a comparison CSV."""
    filename: str
    condition: Callable[[pd.DataFrame], pd.Series]
    display_columns: list[str] | None
    message: str
    is_failure: bool


@dataclass
class ComparisonContext:
    """Holds all runtime configuration and state."""
    results_dir: Path
    check_categories: set[str]
    has_regression: bool = False
    reporter: Optional["ComparisonReporter"] = None

    def set_regression(self) -> None:
        self.has_regression = True

    def should_fail(self, category: str) -> bool:
        return category in self.check_categories


# Color and output helpers
def _colorize(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    colors = {
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "yellow": "\033[0;33m",
        "bold": "\033[1m",
    }
    return f"{colors.get(color, '')}{text}\033[0m"


class ComparisonReporter:
    """Collects and prints messages, and updates the regression flag."""

    def __init__(self, title: str, ctx: ComparisonContext):
        self.title = title
        self.ctx = ctx
        self.passed_messages: list[str] = []
        self.failed_messages: list[str] = []

    def add(self, count: int, message: str, is_failure: bool,
            rows: pd.DataFrame, columns: list[str] | None = None) -> None:
        if count == 0:
            return

        target = self.failed_messages if is_failure else self.passed_messages
        icon = "❌" if is_failure else "✅"
        color = "red" if is_failure else "green"
        target.append(_colorize(color, f"{icon} {message}: {count}"))

        if rows.empty:
            return

        cols = columns if columns is not None else list(rows.columns)
        target.append(f"  {','.join(cols)}")
        for _, row in rows.iterrows():
            values = [str(row.get(col, "?")) for col in cols]
            target.append(f"  {','.join(values)}")

        if is_failure:
            self.ctx.set_regression()

    def print_summary(self) -> None:
        if not self.passed_messages and not self.failed_messages:
            print(_colorize("yellow", "No comparison results to report."))
            return

        print(_colorize("bold", f"\n{'=' * 60}\n{self.title}\n{'=' * 60}"))

        if self.passed_messages:
            print("\n".join(self.passed_messages))

        if self.failed_messages:
            print("\n".join(self.failed_messages))

        print(_colorize("bold", f"\n{'=' * 60}\n"))


# CSV utilities
def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read CSV as strings; return None if file missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
        return None


def _filter_dataframe(df: pd.DataFrame | None,
                      condition: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        mask = condition(df)
        return df.loc[mask].copy()
    except Exception as e:
        print(f"Warning: Filtering failed: {e}", file=sys.stderr)
        return pd.DataFrame()


def _process_csv_configs(reporter: ComparisonReporter, configs: list[CSVConfig]) -> None:
    for cfg in configs:
        df = _read_csv(Path(cfg.filename))
        filtered = _filter_dataframe(df, cfg.condition)
        reporter.add(len(filtered), cfg.message, cfg.is_failure, filtered, cfg.display_columns)


# Unit test comparison
def run_ut_comparison(ctx: ComparisonContext, reporter: ComparisonReporter) -> None:
    print("::group::Unit Test Comparison")

    result = subprocess.run(
        [
            "python", ".github/scripts/compare-ut.py",
            "--input", str(ctx.results_dir),
            "--output", "ut_comparison.csv",
            "--check-changes",
            "--markdown",
            "--markdown-output", "ut_comparison.md"
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        reporter.add(1, "Unit test comparison script failed", True, pd.DataFrame(), None)
        return

    configs = [
        CSVConfig(
            filename="ut_comparison_new_passes.csv",
            condition=lambda df: df["status_baseline"].apply(TestStatus.is_failed)
                                 & df["status_target"].apply(TestStatus.is_passed),
            display_columns=["testfile_target", "classname_target", "name_target",
                             "status_baseline", "status_target", "message_baseline"],
            message="New passed unit tests (compared with baseline)",
            is_failure=ctx.should_fail("ut_new_pass"),
        ),
        CSVConfig(
            filename="ut_comparison_new_failures.csv",
            condition=lambda df: df["status_baseline"].apply(TestStatus.is_passed)
                                 & df["status_target"].apply(TestStatus.is_failed),
            display_columns=["testfile_baseline", "classname_baseline", "name_baseline",
                             "status_baseline", "status_target", "message_target"],
            message="New failed unit tests (compared with baseline)",
            is_failure=ctx.should_fail("ut_new_fail"),
        ),
        CSVConfig(
            filename="ut_comparison_tracked_passes.csv",
            condition=lambda df: df["status_target"].apply(TestStatus.is_passed),
            display_columns=None,
            message="Tracked failed tests that now pass (Update the corresponding issues)",
            is_failure=ctx.should_fail("ut_tracked_pass"),
        ),
        CSVConfig(
            filename="ut_comparison_untracked_failures.csv",
            condition=lambda df: df["status_target"].apply(TestStatus.is_failed),
            display_columns=None,
            message="Untracked failed unit tests (New failed cases)",
            is_failure=ctx.should_fail("ut_untracked_fail"),
        ),
    ]
    _process_csv_configs(reporter, configs)


# Accuracy and performance details processing
def _process_accuracy_details(reporter: ComparisonReporter, path: Path,
                              ctx: ComparisonContext) -> None:
    df = _read_csv(path)
    if df is None:
        return

    new_passes = _filter_dataframe(
        df, lambda d: d["comparison_acc"] == AccuracyChange.NEW_PASSED.value
    )
    reporter.add(
        len(new_passes), "New accuracy passes (compared with baseline)",
        ctx.should_fail("acc_new_pass"),
        new_passes,
        ["suite", "data_type", "mode", "model", "accuracy_baseline",
         "accuracy_target", "comparison_acc"]
    )

    new_failures = _filter_dataframe(
        df, lambda d: d["comparison_acc"] == AccuracyChange.NEW_FAILED.value
    )
    reporter.add(
        len(new_failures), "New accuracy failures (compared with baseline)",
        ctx.should_fail("acc_new_fail"),
        new_failures,
        ["suite", "data_type", "mode", "model", "accuracy_baseline",
         "accuracy_target", "comparison_acc"]
    )


def _process_performance_details(reporter: ComparisonReporter, path: Path,
                                 ctx: ComparisonContext) -> None:
    df = _read_csv(path)
    if df is None:
        return

    columns = ["suite", "data_type", "mode", "model", "eager_baseline", "inductor_baseline",
               "eager_target", "inductor_target", "eager_ratio", "inductor_ratio", "comparison_perf"]

    for change_type, msg, category in [
        (PerformanceChange.NEW_PASSED, "New performance passes (compared with baseline)", "perf_new_pass"),
        (PerformanceChange.NEW_FAILED, "New performance failures (compared with baseline)", "perf_new_fail"),
        (PerformanceChange.NEW_IMPROVED, "Improved performance cases (compared with baseline)", "perf_improved"),
        (PerformanceChange.NEW_DROPPED, "Dropped performance cases (compared with baseline)", "perf_dropped"),
    ]:
        filtered = _filter_dataframe(
            df, lambda d, ct=change_type: d["comparison_perf"] == ct.value
        )
        reporter.add(len(filtered), msg, ctx.should_fail(category), filtered, columns)


# Legacy accuracy summary script (restored)
def _run_accuracy_summary_script(reporter: ComparisonReporter, ctx: ComparisonContext) -> None:
    """Run the legacy e2e_summary.sh script and parse its output."""
    script = Path(".github/scripts/e2e_summary.sh")
    if not script.is_file() or not os.access(script, os.X_OK):
        reporter.add(1, "e2e_summary.sh not found or not executable", True, pd.DataFrame(), None)
        return

    target_dir = ctx.results_dir / "target"
    baseline_dir = ctx.results_dir / "baseline"
    subprocess.run(["bash", str(script), str(target_dir), str(baseline_dir)], capture_output=False)

    acc_file = Path("/tmp/tmp-acc-result.txt")
    if not acc_file.is_file():
        return

    acc_failed = 0
    try:
        with open(acc_file, encoding="utf-8") as f:
            for line in f:
                # Expect lines like "some description X" where X is number of failures
                parts = line.strip().split()
                if parts:
                    try:
                        acc_failed += int(parts[-1])
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Warning: Could not read {acc_file}: {e}", file=sys.stderr)

    if acc_failed > 0:
        reporter.add(acc_failed, "Accuracy regressions (compared with saved reference)",
                     ctx.should_fail("acc_untracked_fail"), pd.DataFrame(), None)

        # Also capture any other tmp-*.txt files for additional context
        for tmp_file in Path("/tmp").glob("tmp-*.txt"):
            if tmp_file == acc_file:
                continue
            try:
                with open(tmp_file, encoding="utf-8") as f:
                    for line in f:
                        if any(key in line for key in ("Real failed", "to passed", "Warning timeout", "Summary for")):
                            if re.search(r":\s*[1-9]", line):
                                reporter.failed_messages.append(f"  {line.strip()}")
            except Exception:
                pass


# End-to-end comparison
def run_e2e_comparison(ctx: ComparisonContext, reporter: ComparisonReporter) -> None:
    print("::group::E2E Comparison")

    target_dir = ctx.results_dir / "target"
    baseline_dir = ctx.results_dir / "baseline"

    result = subprocess.run(
        [
            "python", ".github/scripts/compare-e2e.py",
            "-t", str(target_dir),
            "-b", str(baseline_dir),
            "-o", "e2e_comparison.csv",
            "-m", "e2e_comparison.md"
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        reporter.add(1, "E2E comparison script failed", True, pd.DataFrame(), None)
        return

    # Process accuracy details CSV
    acc_files = list(Path(".").glob("e2e_comparison_accuracy_details.csv"))
    if acc_files:
        _process_accuracy_details(reporter, acc_files[0], ctx)

    # Process performance details CSV
    perf_file = Path("e2e_comparison_performance_details.csv")
    if perf_file.is_file():
        _process_performance_details(reporter, perf_file, ctx)

    # Run the legacy summary script for additional accuracy issue checks
    _run_accuracy_summary_script(reporter, ctx)


# Environment and summary aggregation
def prepare_environment(ctx: ComparisonContext) -> None:
    print("::group::Environment setup")
    component = os.environ.get("COMPONENT", "unknown")
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    baseline = os.environ.get("BASELINE_RUN_ID", "none")
    print(f"Comparing results for {component}...")
    print(f"Target:   {run_id}")
    print(f"Baseline: {baseline}")

    if not ctx.results_dir.is_dir():
        print(_colorize("red", f"Error: Results directory '{ctx.results_dir}' does not exist!"))
        sys.exit(1)

    for d in ctx.results_dir.glob("*/*"):
        print(d)

    target_dir = ctx.results_dir / "target"
    baseline_dir = ctx.results_dir / "baseline"
    if not target_dir.is_dir() or not baseline_dir.is_dir():
        print(_colorize("red",
                        f"Error: Missing 'target' and/or 'baseline' subdirectory under {ctx.results_dir}"))
        sys.exit(1)


def aggregate_summaries() -> None:
    print("::group::Aggregate summaries to GITHUB_STEP_SUMMARY")
    summary_files = ["ut_comparison.md", "e2e_comparison.md"]
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary:
        print("GITHUB_STEP_SUMMARY not set, skipping aggregation")
        return

    with open(step_summary, "a", encoding="utf-8") as out_f:
        out_f.write("\n## Comparison Results\n\n")
        for file in summary_files:
            path = Path(file)
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                out_f.write(content)
                out_f.write("\n\n")


def final_status(ctx: ComparisonContext, reporter: ComparisonReporter) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    artifact_msg = (
        f"More details in artifacts: https://github.com/{repo}/actions/runs/{run_id}#artifacts"
        if repo and run_id else "More details in output summary *.csv or *.xlsx"
    )

    reporter.print_summary()

    if ctx.has_regression:
        print(_colorize("red", f"\n❌ Comparisons failed. See details above.\n\n{artifact_msg}"))
        sys.exit(1)
    else:
        print(_colorize("green", f"\n✅ All comparisons passed.\n\n{artifact_msg}"))
        sys.exit(0)


# Main entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare test results between two runs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing 'target' and 'baseline' subdirectories (default: ./results)"
    )
    parser.add_argument(
        "--check",
        nargs='+',
        default=["ut_untracked_fail", "acc_untracked_fail"],
        choices=[
            # new is for comparison, tracked or not is only for target result check
            "ut_new_pass", "ut_new_fail", "ut_tracked_pass", "ut_untracked_fail",
            "acc_new_pass", "acc_new_fail", "acc_untracked_fail",
            "perf_new_pass", "perf_new_fail", "perf_improved", "perf_dropped",
        ],
        help="Categories that should cause the script to exit with failure (regression)"
    )
    args = parser.parse_args()
    print(args)

    ctx = ComparisonContext(
        results_dir=Path(args.results_dir).resolve(),
        check_categories=set(args.check)
    )
    ctx.reporter = ComparisonReporter(
        title=f"Test Comparison Report - {os.environ.get('COMPONENT', 'unknown')}",
        ctx=ctx
    )

    prepare_environment(ctx)
    run_ut_comparison(ctx, ctx.reporter)
    run_e2e_comparison(ctx, ctx.reporter)
    aggregate_summaries()
    final_status(ctx, ctx.reporter)


if __name__ == "__main__":
    main()
