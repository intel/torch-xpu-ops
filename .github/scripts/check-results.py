#!/usr/bin/env python3
"""
Unit test and end-to-end comparison for GitHub Actions.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Callable

import pandas as pd


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


@dataclass
class CSVConfig:
    """Configuration for processing a comparison CSV."""
    filename: str
    condition: Callable[[pd.DataFrame], pd.Series]
    display_columns: list[str] | None
    message: str
    is_failure: bool   # whether rows in this category indicate a regression


_has_regression = False

def _set_regression() -> None:
    global _has_regression
    _has_regression = True

def has_regression() -> bool:
    return _has_regression


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


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read CSV as strings; return None if file missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
        return None


def _filter_dataframe(df: pd.DataFrame | None, condition: Callable) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return df.loc[condition(df)].copy()
    except Exception as e:
        print(f"Warning: Filtering failed: {e}", file=sys.stderr)
        return pd.DataFrame()


class ComparisonReporter:
    """Collects and prints messages, and updates the global regression flag."""

    def __init__(self, title: str):
        self.title = title
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

        cols = columns if columns is not None else rows.columns.tolist()
        target.append(f"  {','.join(cols)}")
        for _, row in rows.iterrows():
            values = [str(row.get(col, "?")) for col in cols]
            target.append(f"  {','.join(values)}")

        if is_failure:
            _set_regression()

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


def _process_csv_configs(reporter: ComparisonReporter, configs: list[CSVConfig]) -> None:
    for cfg in configs:
        df = _read_csv(Path(cfg.filename))
        filtered = _filter_dataframe(df, cfg.condition)
        reporter.add(len(filtered), cfg.message, cfg.is_failure, filtered, cfg.display_columns)


def run_ut_comparison(reporter: ComparisonReporter) -> None:
    print("::group::Unit Test Comparison")

    result = subprocess.run(
        [
            "python", ".github/scripts/compare-ut.py",
            "--input", "results/",
            "--output", "ut_comparison.csv",
            "--check-changes",
            "--markdown",
            "--markdown-output", "ut_comparison.md"
        ],
        capture_output=False
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
            message="New passed unit tests (compared to baseline)",
            is_failure=False,
        ),
        CSVConfig(
            filename="ut_comparison_new_failures.csv",
            condition=lambda df: df["status_baseline"].apply(TestStatus.is_passed)
                                 & df["status_target"].apply(TestStatus.is_failed),
            display_columns=["testfile_baseline", "classname_baseline", "name_baseline",
                             "status_baseline", "status_target", "message_target"],
            message="New failing unit tests (compared to baseline)",
            is_failure=False,
        ),
        CSVConfig(
            filename="ut_comparison_tracked_passes.csv",
            condition=lambda df: df["status_target"].apply(TestStatus.is_passed),
            display_columns=None,
            message="Previously failing tests that now pass (tracked)",
            is_failure=False,
        ),
        CSVConfig(
            filename="ut_comparison_untracked_failures.csv",
            condition=lambda df: df["status_target"].apply(TestStatus.is_failed),
            display_columns=None,
            message="Untracked failing unit tests (regression)",
            is_failure=True,
        ),
    ]
    _process_csv_configs(reporter, configs)


def _process_accuracy_details(reporter: ComparisonReporter, path: Path,
                              failure_map: dict[str, bool] | None = None) -> None:
    if failure_map is None:
        failure_map = {AccuracyChange.NEW_FAILED.value: False}

    df = _read_csv(path)
    if df is None:
        return

    new_passes = _filter_dataframe(df, lambda d: d["comparison_acc"] == AccuracyChange.NEW_PASSED.value)
    reporter.add(len(new_passes), "New accuracy passes",
                 failure_map.get(AccuracyChange.NEW_PASSED.value, False),
                 new_passes,
                 ["suite", "data_type", "mode", "model", "accuracy_baseline",
                  "accuracy_target", "comparison_acc"])

    new_failures = _filter_dataframe(df, lambda d: d["comparison_acc"] == AccuracyChange.NEW_FAILED.value)
    reporter.add(len(new_failures), "New accuracy failures",
                 failure_map.get(AccuracyChange.NEW_FAILED.value, False),
                 new_failures,
                 ["suite", "data_type", "mode", "model", "accuracy_baseline",
                  "accuracy_target", "comparison_acc"])


def _process_performance_details(reporter: ComparisonReporter, path: Path,
                                 failure_map: dict[str, bool] | None = None) -> None:
    if failure_map is None:
        failure_map = {
            PerformanceChange.NEW_FAILED.value: False,
            PerformanceChange.NEW_DROPPED.value: False,
        }

    df = _read_csv(path)
    if df is None:
        return

    columns = ["suite", "data_type", "mode", "model", "eager_baseline", "inductor_baseline",
               "eager_target", "inductor_target", "eager_ratio", "inductor_ratio", "comparison_perf"]

    for change_type, msg in [
        (PerformanceChange.NEW_PASSED, "New performance passes"),
        (PerformanceChange.NEW_FAILED, "New performance failures"),
        (PerformanceChange.NEW_IMPROVED, "Improved performance cases"),
        (PerformanceChange.NEW_DROPPED, "Dropped performance cases"),
    ]:
        filtered = _filter_dataframe(df, lambda d, ct=change_type: d["comparison_perf"] == ct.value)
        is_failure = failure_map.get(change_type.value, False)
        reporter.add(len(filtered), msg, is_failure, filtered, columns)


def _run_accuracy_summary_script(reporter: ComparisonReporter) -> None:
    script = Path(".github/scripts/e2e_summary.sh")
    if not script.is_file() or not os.access(script, os.X_OK):
        reporter.add(1, "e2e_summary.sh not found or not executable", True, pd.DataFrame(), None)
        return

    subprocess.run(["bash", str(script), "results/target", "results/baseline"], capture_output=False)

    acc_file = Path("/tmp/tmp-acc-result.txt")
    if not acc_file.is_file():
        return

    acc_failed = 0
    try:
        with open(acc_file, encoding="utf-8") as f:
            for line in f:
                acc_failed += int(line.split(' ')[-1])
    except Exception as e:
        print(f"Warning: Could not read {acc_file}: {e}", file=sys.stderr)

    if acc_failed > 0:
        reporter.add(acc_failed, "Accuracy regressions from legacy summary script", True, pd.DataFrame(), None)
        _set_regression()
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


def run_e2e_comparison(reporter: ComparisonReporter,
                       accuracy_failure_map: dict[str, bool] | None = None,
                       performance_failure_map: dict[str, bool] | None = None) -> None:
    print("::group::E2E Comparison")

    result = subprocess.run(
        [
            "python", ".github/scripts/compare-e2e.py",
            "-t", "results/target",
            "-b", "results/baseline",
            "-o", "e2e_comparison.csv",
            "-m", "e2e_comparison.md"
        ],
        capture_output=False
    )
    if result.returncode != 0:
        reporter.add(1, "E2E comparison script failed", True, pd.DataFrame(), None)
        return

    acc_files = list(Path(".").glob("e2e_comparison_accuracy_details.csv"))
    if acc_files:
        _process_accuracy_details(reporter, acc_files[0], accuracy_failure_map)

    perf_file = Path("e2e_comparison_performance_details.csv")
    if perf_file.is_file():
        _process_performance_details(reporter, perf_file, performance_failure_map)

    _run_accuracy_summary_script(reporter)


def prepare_environment() -> None:
    print("::group::Environment setup")
    component = os.environ.get("COMPONENT", "unknown")
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    baseline = os.environ.get("BASELINE_RUN_ID", "none")
    print(f"Comparing results for {component}...")
    print(f"Target:   {run_id}")
    print(f"Baseline: {baseline}")

    results_dir = Path("./results")
    if not results_dir.is_dir():
        print(_colorize("red", "Error: No results directory found after test!"))
        sys.exit(1)

    for d in results_dir.glob("*/*"):
        print(d)

    target_dir = results_dir / "target"
    baseline_dir = results_dir / "baseline"
    if not target_dir.is_dir() and not baseline_dir.is_dir():
        print(_colorize("red", "Error: Missing 'target' and/or 'baseline' subdirectory under results/"))
        sys.exit(1)


def aggregate_summaries() -> None:
    print("::group::Aggregate summaries to GITHUB_STEP_SUMMARY")
    summary_files = ["ut_comparison.md", "e2e_comparison.md"]
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary:
        print("GITHUB_STEP_SUMMARY not set, skipping aggregation")
        return

    with open(step_summary, "a", encoding="utf-8") as out_f:
        for file in summary_files:
            path = Path(file)
            if path.is_file():
                out_f.write(path.read_text(encoding="utf-8"))


def final_status(reporter: ComparisonReporter) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    artifact_msg = (
        f"More details in artifacts: https://github.com/{repo}/actions/runs/{run_id}#artifacts"
        if repo and run_id else "More details in output summary *.csv or *.xlsx"
    )

    reporter.print_summary()

    if has_regression():
        print(_colorize("red", f"\n❌ Comparisons failed. See details above.\n\n{artifact_msg}"))
        sys.exit(1)
    else:
        print(_colorize("green", f"\n✅ All comparisons passed.\n\n{artifact_msg}"))
        sys.exit(0)


def main() -> None:
    prepare_environment()
    component = os.environ.get("COMPONENT", "unknown")
    reporter = ComparisonReporter(title=f"Test Comparison Report - {component}")

    run_ut_comparison(reporter)
    run_e2e_comparison(reporter)
    aggregate_summaries()
    final_status(reporter)


if __name__ == "__main__":
    main()
