#!/usr/bin/env python3
"""
Accuracy Check Script

Compares inductor accuracy test results against a reference CSV and reports
regressions, new passes, new models, and expected failures.

Usage:
    python check_expected.py --results-dir ./1 --expected expected_accuracy_lts.csv
    python check_expected.py --results-dir ./1 --expected expected_accuracy_lts.csv --update
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Constants
SUITES = ("huggingface", "timm_models", "torchbench")
DTYPES = ("float32", "bfloat16", "float16", "amp_bf16", "amp_fp16")
MODES = ("inference", "training")

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Models skipped from pass-rate calculation (known environmental issues)
SKIPPED_MODEL_PATTERNS = (
    "detectron2",
    "torchrec_dlrm",
    "stable_diffusion_text_encoder",
    "stable_diffusion_unet",
)

ACCURACY_CSV_PATTERN = re.compile(
    r"(inductor[-_]results[-_]|inductor[-_])?"
    r"(?P<suite>huggingface|timm_models|torchbench)[-_]"
    r"(?P<dtype>float32|bfloat16|float16|amp_bf16|amp_fp16)[-_]"
    r"(?P<mode>inference|training)[-_]"
    r"xpu[-_]accuracy\.csv$"
)


# Data structures
@dataclass
class ModelResult:
    """Comparison result for a single model in a specific suite/dtype/mode."""

    name: str
    actual: str  # from test results ("N/A" if not run)
    expected: str  # from reference CSV ("N/A" if new model)
    dtype: str = ""
    mode: str = ""
    category: str = ""


@dataclass
class CheckSummary:
    """Aggregated results for one suite/dtype/mode combination."""

    suite: str
    dtype: str
    mode: str
    passed: list[ModelResult] = field(default_factory=list)
    real_failed: list[ModelResult] = field(default_factory=list)
    expected_failed: list[ModelResult] = field(default_factory=list)
    new_pass: list[ModelResult] = field(default_factory=list)
    new: list[ModelResult] = field(default_factory=list)
    timeout: list[ModelResult] = field(default_factory=list)
    lost: list[ModelResult] = field(default_factory=list)
    skipped: list[ModelResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(
            len(lst)
            for lst in (
                self.passed,
                self.real_failed,
                self.expected_failed,
                self.new_pass,
                self.new,
                self.timeout,
                self.lost,
                self.skipped,
            )
        )

    @property
    def pass_rate(self) -> float:
        return len(self.passed) / self.total * 100 if self.total else 0.0

    @property
    def has_regressions(self) -> bool:
        return len(self.real_failed) > 0


# Loading helpers
def load_expected(csv_path: Path) -> pd.DataFrame:
    """Load the expected accuracy CSV (one row per suite/model, merged modes)."""
    df = pd.read_csv(csv_path, comment="#")
    df = df.set_index(["suite", "name"])
    return df

def load_verified_file(verified_path: Path) -> list[tuple[str, str, str, str, str]]:
    """Load a verified file (space/tab/comma separated) with verified results.

    Expected columns: scenario, suite, dtype, mode, name, accuracy
    Returns list of (suite, dtype, mode, name, accuracy) tuples.
    """
    entries: list[tuple[str, str, str, str, str]] = []
    with open(verified_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r'[\t,\s]+', line)
            if lineno == 1 and parts[0].lower() in ("scenario", "suite"):
                continue  # skip header
            if len(parts) < 6:
                print(f"  WARN: skipping line {lineno} (too few fields): {line}")
                continue
            _, suite, dtype, mode, name, accuracy = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            entries.append((suite, dtype, mode, name, accuracy))
    return entries


def discover_result_csvs(results_dir: Path) -> list[Path]:
    """Recursively find all inductor accuracy CSVs under results_dir."""
    return sorted(
        p
        for p in results_dir.rglob("*accuracy.csv")
        if ACCURACY_CSV_PATTERN.search(p.name)
    )


def parse_result_csv(csv_path: Path) -> Optional[tuple[str, str, str, pd.DataFrame]]:
    """Parse one test result CSV, returning (suite, dtype, mode, dataframe)."""
    match = ACCURACY_CSV_PATTERN.search(csv_path.name)
    if not match:
        return None
    suite, dtype, mode = match.group("suite"), match.group("dtype"), match.group("mode")
    df = pd.read_csv(csv_path)
    return suite, dtype, mode, df


def load_all_results(
    results_dir: Path, verified_path: Path | None = None
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Load all accuracy CSVs into a dict keyed by (suite, dtype, mode).

    If *verified_path* is provided, override test results with verified actuals.
    """
    results: dict[tuple[str, str, str], pd.DataFrame] = {}
    for csv_path in discover_result_csvs(results_dir):
        parsed = parse_result_csv(csv_path)
        if parsed:
            suite, dtype, mode, df = parsed
            results[(suite, dtype, mode)] = df

    if verified_path:
        entries = load_verified_file(verified_path)
        for suite, dtype, mode, name, accuracy in entries:
            key = (suite, dtype, mode)
            if key not in results:
                results[key] = pd.DataFrame({"name": [name], "accuracy": [accuracy]})
                continue
            df = results[key]
            if name in df["name"].values:
                df.loc[df["name"] == name, "accuracy"] = accuracy
            else:
                results[key] = pd.concat(
                    [df, pd.DataFrame({"name": [name], "accuracy": [accuracy]})],
                    ignore_index=True,
                )
        print(f"  Applied {len(entries)} verified override(s) to test results")

    return results


# Comparison logic
def is_skipped(model_name: str) -> bool:
    """Check if model should be skipped from comparison."""
    return any(pattern in model_name for pattern in SKIPPED_MODEL_PATTERNS)


def categorize(actual: str, expected: str) -> str:
    """Determine the category for a model based on actual vs expected accuracy."""
    if actual == "N/A":
        return "lost"
    if "pass" in actual:
        if expected == "N/A":
            return "new"
        if "pass" not in expected:
            return "new_pass"
        return "passed"
    if "timeout" in actual:
        if expected == "N/A":
            return "new"
        return "timeout"
    # Failure cases
    if expected == "N/A":
        return "new"
    if "pass" in expected:
        return "real_failed"
    if actual != expected:
        return "expected_failed"
    return "expected_failed"


def compare_one(
    suite: str,
    dtype: str,
    mode: str,
    test_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> CheckSummary:
    """Compare test results against expected for one suite/dtype/mode."""
    summary = CheckSummary(suite=suite, dtype=dtype, mode=mode)
    col = f"{mode}_{dtype}"

    # Build lookup from test results: model_name -> accuracy
    test_map: dict[str, str] = {}
    if "name" in test_df.columns and "accuracy" in test_df.columns:
        for _, row in test_df.iterrows():
            test_map[row["name"]] = str(row["accuracy"])

    # Gather all model names from both expected and actual
    expected_models: set[str] = set()
    if (suite,) == (suite,):  # always true; filter expected_df by suite
        try:
            suite_df = expected_df.loc[suite]
            expected_models = set(suite_df.index)
        except KeyError:
            pass

    all_models = expected_models | set(test_map.keys())

    for model_name in sorted(all_models):
        actual = test_map.get(model_name, "N/A")

        # Get expected value
        try:
            expected_val = str(expected_df.loc[(suite, model_name), col])
            if expected_val == "nan":
                expected_val = "N/A"
        except KeyError:
            expected_val = "N/A"

        result = ModelResult(name=model_name, actual=actual, expected=expected_val, dtype=dtype, mode=mode)

        if is_skipped(model_name):
            result.category = "skipped"
            summary.skipped.append(result)
            continue

        result.category = categorize(actual, expected_val)
        getattr(summary, result.category).append(result)

    return summary


# Reporting
def aggregate_by_suite(summaries: list[CheckSummary]) -> list[CheckSummary]:
    """Merge summaries with the same suite into one combined CheckSummary."""
    from collections import OrderedDict

    grouped: OrderedDict[str, CheckSummary] = OrderedDict()
    for s in summaries:
        if s.suite not in grouped:
            grouped[s.suite] = CheckSummary(suite=s.suite, dtype="", mode="")
        g = grouped[s.suite]
        g.passed.extend(s.passed)
        g.real_failed.extend(s.real_failed)
        g.expected_failed.extend(s.expected_failed)
        g.new_pass.extend(s.new_pass)
        g.new.extend(s.new)
        g.timeout.extend(s.timeout)
        g.lost.extend(s.lost)
        g.skipped.extend(s.skipped)
    return list(grouped.values())


def _pass_bar(rate: float, width: int = 30) -> str:
    """Render a visual pass-rate bar."""
    filled = int(rate / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if rate >= 80 else YELLOW if rate >= 60 else RED
    return f"{color}{bar}{RESET} {rate:.1f}%"


def _print_grouped_results(results: list[ModelResult], color: str, suite: str, fmt: str) -> None:
    """Print each result as suite,dtype,mode,name,accuracy."""
    for r in sorted(results, key=lambda r: (r.mode, r.dtype, r.name)):
        detail = fmt.format(r=r)
        print(f"     {color}{suite},{r.dtype},{r.mode},{r.name},{detail}{RESET}")


def print_summary(summary: CheckSummary) -> None:
    """Print a formatted summary for one suite."""
    print(f"\n{'=' * 72}")
    print(f"  {BOLD}{summary.suite}{RESET}")
    print(f"{'=' * 72}")
    print(f"  Pass rate:  {_pass_bar(summary.pass_rate)}")
    print(f"  {'─' * 68}")

    # Table header
    cols = ["Total", "Pass", "Fail", "ExpFail", "Timeout", "New", "Fixed", "Lost", "Skip"]
    vals = [
        summary.total,
        len(summary.passed),
        len(summary.real_failed),
        len(summary.expected_failed),
        len(summary.timeout),
        len(summary.new),
        len(summary.new_pass),
        len(summary.lost),
        len(summary.skipped),
    ]
    colors = [None, GREEN, RED, None, None, CYAN, GREEN, YELLOW, None]

    print(f"  {BOLD}{'  '.join(f'{c:>6}' for c in cols)}{RESET}")
    print(f"  {'─' * 68}")
    cells = []
    for val, color in zip(vals, colors):
        cell = f"{val:>6}"
        if color and val > 0:
            cell = f"{color}{cell}{RESET}"
        cells.append(cell)
    print(f"  {'  '.join(cells)}")

    sections = [
        (summary.real_failed, RED, "** REGRESSIONS", "{r.actual}"),
        (summary.new_pass, GREEN, "++ NEW PASSES", "{r.actual}"),
        (summary.new, CYAN, "?? NEW MODELS", "{r.actual}"),
    ]
    for items, color, title, fmt in sections:
        if items:
            print(f"\n  {color}{BOLD}{title} ({len(items)}):{RESET}")
            _print_grouped_results(items, color, suite=summary.suite, fmt=fmt)


def print_overall(summaries: list[CheckSummary]) -> None:
    """Print overall statistics across all checks."""
    total_regressions = sum(len(s.real_failed) for s in summaries)
    total_new_passes = sum(len(s.new_pass) for s in summaries)
    total_cases = sum(s.total for s in summaries)
    total_passed = sum(len(s.passed) for s in summaries)
    overall_rate = total_passed / total_cases * 100 if total_cases else 0.0

    print(f"\n{'=' * 72}")
    print(f"  {BOLD}OVERALL SUMMARY{RESET}  ({len(summaries)} checks)")
    print(f"{'=' * 72}")
    print(f"  Pass rate:  {_pass_bar(overall_rate)}")
    print(f"  {'─' * 68}")

    # Per-suite breakdown table
    suite_summaries = aggregate_by_suite(summaries)
    header = f"  {BOLD}{'Suite':<15} {'Total':>6} {'Pass':>6} {'Regress':>7} {'ExpFail':>7} {'New':>5} {'Lost':>5} {'Rate':>8}{RESET}"
    print(header)
    print(f"  {'─' * 68}")
    for s in suite_summaries:
        rate = s.pass_rate
        rate_color = GREEN if rate >= 80 else YELLOW if rate >= 60 else RED
        regress_str = f"{RED}{len(s.real_failed):>7}{RESET}" if s.real_failed else f"{len(s.real_failed):>7}"
        print(
            f"  {s.suite:<15} {s.total:>6} {GREEN}{len(s.passed):>6}{RESET} "
            f"{regress_str} {len(s.expected_failed):>7} "
            f"{len(s.new):>5} {len(s.lost):>5} {rate_color}{rate:>7.1f}%{RESET}"
        )
    print(f"  {'─' * 68}")
    print(
        f"  {BOLD}{'TOTAL':<15}{RESET} {total_cases:>6} {GREEN}{total_passed:>6}{RESET} "
        f"{RED}{total_regressions:>7}{RESET} "
        f"{sum(len(s.expected_failed) for s in summaries):>7} "
        f"{sum(len(s.new) for s in summaries):>5} "
        f"{sum(len(s.lost) for s in summaries):>5} "
        f"{rate_color}{overall_rate:>7.1f}%{RESET}"
    )

    if total_regressions:
        print(f"\n  {RED}{BOLD}*** REGRESSIONS DETECTED - SEE DETAILS ABOVE ***{RESET}")


# Update logic
UPDATE_CATEGORIES = ("real_failed", "new", "new_pass", "expected_failed")
DEFAULT_UPDATE_CATEGORIES = ("new", "new_pass", "expected_failed")


def apply_updates(
    expected_df: pd.DataFrame, summaries: list[CheckSummary],
    categories: tuple[str, ...] = DEFAULT_UPDATE_CATEGORIES,
) -> pd.DataFrame:
    """Apply new/changed results back to the expected dataframe."""
    df = expected_df.copy()
    for summary in summaries:
        col = f"{summary.mode}_{summary.dtype}"
        results_to_update: list[ModelResult] = []
        for cat in categories:
            results_to_update.extend(getattr(summary, cat))
        for result in results_to_update:
            if result.actual == "N/A":
                continue
            idx = (summary.suite, result.name)
            if idx in df.index:
                df.loc[idx, col] = result.actual
            else:
                new_row = pd.DataFrame(
                    {c: ["N/A"] for c in df.columns}, index=pd.MultiIndex.from_tuples([idx], names=df.index.names)
                )
                new_row[col] = result.actual
                df = pd.concat([df, new_row])
    return df


# Main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check inductor accuracy results against expected baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing test result CSVs (searched recursively).",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Path to expected accuracy CSV. Default: expected_accuracy_lts.csv next to this script.",
    )
    parser.add_argument(
        "--update",
        nargs="*",
        default=None,
        metavar="CATEGORY",
        help="Update expected CSV. Optional categories: real_failed, new, new_pass, expected_failed. "
             "Default (no args): new, new_pass, expected_failed (excludes real_failed).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regressions are found.",
    )
    parser.add_argument(
        "--verified",
        type=Path,
        default=None,
        help="Path to a file with verified results (space/tab/comma separated) "
             "to override test results before comparison. "
             "Columns: scenario suite dtype mode name accuracy",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve expected CSV path
    expected_path = args.expected or (Path(__file__).parent / "expected_accuracy_lts.csv")
    if not expected_path.exists():
        print(f"ERROR: Expected CSV not found: {expected_path}", file=sys.stderr)
        return 1

    # Load data
    expected_df = load_expected(expected_path)

    verified_path = None
    if args.verified:
        if not args.verified.exists():
            print(f"ERROR: Verified file not found: {args.verified}", file=sys.stderr)
            return 1
        verified_path = args.verified

    results = load_all_results(args.results_dir, verified_path)

    if not results:
        print(f"ERROR: No accuracy CSVs found under {args.results_dir}", file=sys.stderr)
        return 1

    print(f"Loaded {len(results)} result file(s) from {args.results_dir}")
    print(f"Expected baseline: {expected_path} ({len(expected_df)} models)")

    # Run comparisons
    summaries: list[CheckSummary] = []
    for (suite, dtype, mode), test_df in sorted(results.items()):
        summary = compare_one(suite, dtype, mode, test_df, expected_df)
        summaries.append(summary)

    # Print per-suite aggregated summaries
    for suite_summary in aggregate_by_suite(summaries):
        print_summary(suite_summary)

    # Overall
    print_overall(summaries)

    # Update expected CSV if requested
    if args.update is not None:
        if args.update:
            categories = tuple(args.update)
            invalid = set(categories) - set(UPDATE_CATEGORIES)
            if invalid:
                print(f"ERROR: Invalid update categories: {invalid}", file=sys.stderr)
                return 1
        else:
            categories = DEFAULT_UPDATE_CATEGORIES
        updated_df = apply_updates(expected_df, summaries, categories)
        updated_df = updated_df.sort_index()
        updated_df.to_csv(expected_path)
        print(f"\nUpdated ({', '.join(categories)}): {expected_path}")

    # Exit code
    if args.fail_on_regression and any(s.has_regressions for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
