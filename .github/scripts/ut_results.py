#!/usr/bin/env python3
"""UT results for torch-xpu-ops CI.

    report   junit/log reports    -> per-category case logs and a results table
    check    a run's UT artifacts -> new and known failures per job, and a verdict
    summary  the checks of a run  -> one markdown report

The UT job runs `report`; the check job of each UT scope runs `check` and fails
on a new failure; the report job runs `summary` once for the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = "intel/torch-xpu-ops"
# Cases that pass despite being skipped, so a newly passing one is visible.
NEW_PASSED_ISSUE = 2333
SCRIPT_DIR = Path(__file__).resolve().parent

OP_UT_EXPECTED = {"linux": 287444, "windows": 105263}

# A category losing more than 5% of its cases is a broken run, not a fixed one.
EXPECTED_CASES = {
    "op_extended": 5352,
    "op_regression": 353,
    "op_regression_dev1": 1,
    "test_xpu": 69,
    "upstream_default": 83126,
    "upstream_inductor": 33999,
    "upstream_distributed": 3323,
}

# Cases that pass at random; a pass of one says nothing about a fix.
IGNORE_TESTS = (
    "test_parity__foreach_div_fastpath_inplace_xpu_complex128",
    "test_parity__foreach_div_fastpath_outplace_xpu_complex128",
    "test_parity__foreach_addcdiv_fastpath_inplace_xpu_complex128",
    "test_parity__foreach_addcdiv_fastpath_outplace_xpu_complex128",
    "test_python_ref__refs_log2_xpu_complex128",
    "_jiterator_",
)

ERROR_TYPES = (
    "RuntimeError",
    "ValueError",
    "TypeError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "ImportError",
    "AssertionError",
    "Exception",
    "OSError",
    "Failed",
    "TimeoutError",
    "asyncio.TimeoutError",
    "FileNotFoundError",
    "PermissionError",
    "NotImplementedError",
)

MAIN_CATEGORIES = (
    "op_regression",
    "op_regression_dev1",
    "op_extended",
    "op_ut",
    "test_xpu",
    "upstream_default",
    "upstream_inductor",
    "upstream_distributed",
)

# A UT job name maps to the categories its reports carry.
JOB_CATEGORIES = {"basic": ("op_regression", "op_extended", "op_regression_dev1")}


# --------------------------------------------------------------------------- #
# Shell-ish helpers
# --------------------------------------------------------------------------- #


def warn(message: str) -> None:
    print(f"::warning::{message}", flush=True)


def read_lines(path: Path) -> list[str]:
    """Non-empty lines, free of the trailing whitespace a CRLF log leaves."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def gh(*args: str, text_input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, input=text_input, check=False
    )


def expected_cases(category: str, platform: str) -> int | None:
    if category == "op_ut":
        return OP_UT_EXPECTED[platform]
    return EXPECTED_CASES.get(category)


def resolve_platform() -> str:
    raw = os.environ.get("TEST_PLATFORM", "linux")
    platform = raw.lower()
    if platform not in ("linux", "windows"):
        print(f"⚠️  Unsupported TEST_PLATFORM='{raw}', defaulting to 'linux'", file=sys.stderr)
        return "linux"
    return platform


def issue_platform(platform: str, runner: str) -> str:
    """Label for an issue comment, from the runner so distributed is not Linux."""
    if platform == "windows":
        return "Windows"
    runner = runner.lower()
    if "bmg" in runner:
        return "BMG"
    if "distributed" in runner:
        return "distributed"
    return "Linux"


def categories_of(ut_job: str) -> tuple[str, ...]:
    if ut_job in JOB_CATEGORIES:
        return JOB_CATEGORIES[ut_job]
    return tuple(part for part in ut_job.split(",") if part)


# --------------------------------------------------------------------------- #
# report - junit and log reports to per-category case logs
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    category: str
    classname: str
    name: str
    status: str
    message: str = ""
    source: str = "XML"

    @property
    def line(self) -> str:
        return f"{self.category},{self.classname},{self.name}"


@dataclass
class SuiteSummary:
    category: str
    ut: str
    tests: int
    passed: int
    skipped: int
    failures: int
    errors: int
    source: str

    @property
    def counts(self) -> tuple[int, int, int, int, int]:
        return self.tests, self.passed, self.skipped, self.failures, self.errors

    @property
    def row(self) -> list[str]:
        return [self.category, self.ut, *(str(c) for c in self.counts), self.source]


SUMMARY_HEADER = [
    "Category",
    "UT",
    "Test cases",
    "Passed",
    "Skipped",
    "Failures",
    "Errors",
    "Source",
]
COUNT_KEYS = ("Test cases", "Passed", "Skipped", "Failures", "Errors")
CATEGORY_HEADER = ["Category", *COUNT_KEYS]
FAILURE_HEADER = ["Category", "Class name", "Test name", "Status", "Message", "Source"]


def determine_category(ut: str) -> str:
    exact = {"op_regression", "op_regression_dev1", "op_extended", "test_xpu"}
    if ut in exact:
        return ut
    for marker in ("op_ut", "upstream_distributed", "upstream_inductor",
                   "upstream_default", "xpu_profiling"):
        if marker in ut:
            return marker
    return "unknown"


def normalize(value: str | None) -> str:
    return " ".join(str(value or "").split())


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [md_row(header), md_row(["---"] * len(header)), *(md_row(r) for r in rows)]


def md_details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body, "", "</details>"]


def plural(count: int, noun: str, many: str = "") -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {many or noun + 's'}"


def case_message(result: ET.Element) -> str:
    """The exception line(s) of a failure, not the traceback around them."""
    full_text = result.text or result.get("message") or ""
    messages = []
    for line in full_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for error_type in ERROR_TYPES:
            if stripped.startswith(error_type + ": "):
                messages.append(f"{error_type}: {stripped[len(error_type) + 2:]}")
                break
            if f"{error_type}:" in stripped and "Traceback" not in stripped:
                messages.append(f"{error_type}: {stripped.split(f'{error_type}:')[-1].strip()}")
                break
    if messages:
        return " ; ".join(messages)
    message = result.get("message") or ""
    return message.splitlines()[0] if message else ""


def iter_testsuites(path: Path):
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        yield root
    else:
        yield from root.iter("testsuite")


def suite_counts(suite: ET.Element) -> tuple[int, int, int, int]:
    def attr(name: str) -> int:
        try:
            return int(suite.get(name, 0) or 0)
        except ValueError:
            return 0

    return attr("tests"), attr("skipped"), attr("failures"), attr("errors")


class Report:
    """Every case of a UT job, and the per-suite counts behind them."""

    def __init__(self) -> None:
        self.cases: list[Case] = []
        self.summaries: list[SuiteSummary] = []

    @property
    def failures(self) -> list[Case]:
        return [c for c in self.cases if c.status not in ("passed", "skipped")]

    def totals(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(COUNT_KEYS, 0))
        for s in self.summaries:
            for key, value in zip(COUNT_KEYS, s.counts):
                totals[s.category][key] += value
        return totals

    def add(self, path: Path) -> None:
        try:
            if path.suffix == ".xml":
                self._add_xml(path)
            elif path.suffix == ".log":
                self._add_log(path)
            else:
                print(f"Skipping unknown file type: {path}", file=sys.stderr)
        except Exception as exc:  # a broken report must not lose the others
            print(f"Error processing {path}: {exc}", file=sys.stderr)

    def _add_xml(self, path: Path) -> None:
        ut = path.stem
        category = determine_category(path.name.split(".")[0])
        for suite in iter_testsuites(path):
            tests, skipped, failures, errors = suite_counts(suite)
            self.summaries.append(SuiteSummary(
                category=category, ut=ut, tests=tests,
                passed=tests - skipped - failures - errors,
                skipped=skipped, failures=failures, errors=errors, source="XML",
            ))
            for case in suite.iter("testcase"):
                result = next(
                    (c for c in case if c.tag in ("failure", "error", "skipped")), None
                )
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}.get(
                    result.tag if result is not None else "", "passed"
                )
                self.cases.append(Case(
                    category=category,
                    classname=normalize(case.get("classname")),
                    name=normalize(case.get("name")),
                    status=status,
                    message=case_message(result) if status in ("failed", "error") else "",
                ))

    def _add_log(self, path: Path) -> None:
        """Unittest console output, for the suites that produce no junit xml."""
        content = path.read_text(encoding="utf-8", errors="replace")
        ut = path.stem
        category = determine_category(ut)

        tests = skipped = 0
        if m := re.search(r"Ran (\d+) tests in [\d.]+s", content):
            tests = int(m.group(1))
        if m := re.search(r"skipped[ =](\d+)", content, re.IGNORECASE):
            skipped = int(m.group(1))
        elif m := re.search(r"skipped (\d+) cases?", content, re.IGNORECASE):
            skipped = int(m.group(1))

        seen: set[str] = set()
        error_pattern = r"(" + "|".join(ERROR_TYPES) + r"):.*?(?=\n\S|\n\n|\n=+\n|\Z)"
        for block in re.findall(r"(FAIL:.*?)(?:\n\n|\n=+\n|\Z)", content, re.DOTALL):
            m = re.match(r"FAIL: (\w+) \(__mp_main__\.(\w+)\)", block)
            if not m or m.group(1) in seen:
                continue
            seen.add(m.group(1))
            messages = [hit.group(0).strip()
                        for hit in re.finditer(error_pattern, block, re.DOTALL)]
            if not messages and "Traceback" in block:
                messages = ["Unknown error (see traceback)"]
            self.cases.append(Case(
                category=category, classname=ut, name=f"{m.group(2)}:{m.group(1)}",
                status="failed", message=" ".join(messages), source="Log",
            ))

        failures = len(seen)
        self.summaries.append(SuiteSummary(
            category=category, ut=ut, tests=tests, passed=tests - failures - skipped,
            skipped=skipped, failures=failures, errors=0, source="Log",
        ))

    def write_case_logs(self, out_dir: Path) -> None:
        """One file per category per status, each line `category,class,test`.

        all_cases_* carries the whole roster: passed and failures together
        cannot express "skipped", which the baseline comparison needs to tell a
        newly enabled case from a brand-new one.
        """
        groups: dict[tuple[str, str], list[Case]] = defaultdict(list)
        for case in self.cases:
            groups["all_cases", case.category].append(case)
            if case.status == "passed":
                groups["passed", case.category].append(case)
            elif case.status != "skipped":
                groups["failures", case.category].append(case)
        for (kind, category), cases in groups.items():
            write_lines(out_dir / f"{kind}_{category}.log", [c.line for c in cases])

        for category, totals in self.totals().items():
            if totals["Test cases"] == 0:
                continue
            write_lines(out_dir / f"category_{category}.log", [
                f"Category: {category}",
                *(f"{key}: {value}" for key, value in totals.items()),
            ])

    def failure_rows(self, cases: list[Case] | None = None) -> list[list[str]]:
        return [[c.category, c.classname, c.name, c.status, c.message, c.source]
                for c in (self.failures if cases is None else cases)]

    def write_failure_csv(self, path: Path) -> None:
        """Headerless, pipe delimited: the summary greps it row by row."""
        write_lines(path, [md_row(row) for row in self.failure_rows()])

    def render(self, new_lines: set[str] | None = None) -> str:
        """Without `new_lines` no failure can be told from another, so all show."""
        blocks = []
        if self.failures:
            blocks.append("\n".join(["### Test Failures", "",
                                     *self.failures_md(new_lines)]))
        blocks.append("\n".join(["### Results Summary", "", *self.summary_md()]))
        return "\n\n".join(blocks)

    def failures_md(self, new_lines: set[str] | None = None) -> list[str]:
        if new_lines is None:
            return md_table(FAILURE_HEADER, self.failure_rows())
        new = [c for c in self.failures if c.line in new_lines]
        known = [c for c in self.failures if c.line not in new_lines]
        lines = md_table(FAILURE_HEADER, self.failure_rows(new)) if new \
            else ["✅ No new failures."]
        if known:
            lines += ["", *md_details(plural(len(known), "known failure"),
                                      md_table(FAILURE_HEADER,
                                               self.failure_rows(known)))]
        return lines

    def summary_md(self) -> list[str]:
        totals = self.totals()
        rows = [[category, *(str(counts[key]) for key in COUNT_KEYS)]
                for category, counts in sorted(totals.items())]
        grand = ["**Total**"] + [str(sum(c[key] for c in totals.values()))
                                 for key in COUNT_KEYS]
        lines = md_table(CATEGORY_HEADER, [grand])
        if rows:
            lines += ["", *md_details(plural(len(rows), "category", "categories"),
                                      md_table(CATEGORY_HEADER, rows))]
        if self.summaries:
            lines += ["", *md_details(plural(len(self.summaries), "report"),
                                      md_table(SUMMARY_HEADER,
                                               [s.row for s in self.summaries]))]
        return lines


def build_report(ut_name: str, inputs: list[Path], out_dir: Path) -> Report:
    report = Report()
    for path in inputs:
        report.add(path)
    # skipped_ut is a list of cases expected to fail; its failures are not news.
    if ut_name != "skipped_ut":
        report.write_failure_csv(out_dir / "ut_failure_list.csv")
    report.write_case_logs(out_dir)
    return report


# --------------------------------------------------------------------------- #
# check - one category against the known-failure list
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    category: str
    passed: bool
    new_failures: list[str] = field(default_factory=list)
    counts_normal: bool | None = None
    total: int = 0
    new_passes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "✅ passed" if self.passed else "❌ failed"


@dataclass
class Checker:
    """Reads the case logs of one working directory, writes its verdict there."""

    work_dir: Path
    pytorch: str = "nightly_wheel"
    platform: str = "linux"
    runner: str = ""
    health_file: Path | None = None

    @classmethod
    def from_env(cls, work_dir: Path, pytorch: str) -> Checker:
        health = os.environ.get("UT_RUN_HEALTH_FILE", "")
        return cls(
            work_dir=work_dir,
            pytorch=pytorch,
            platform=resolve_platform(),
            runner=os.environ.get("TEST_RUNNER", ""),
            health_file=Path(health) if health else None,
        )

    # -- inputs ------------------------------------------------------------- #

    def path(self, name: str) -> Path:
        return self.work_dir / name

    @property
    def known_issues(self) -> list[str]:
        return read_lines(self.path("Known_issue.log"))

    # -- building blocks ---------------------------------------------------- #

    def record_health(self, category: str, expected: int, actual: int,
                      healthy: bool, reason: str) -> None:
        """One JSON line per category, read by the evidence collector.

        A category that never ran gets no line, which is how "absent" stays
        distinguishable from "truncated".
        """
        if self.health_file is None:
            return
        self.health_file.parent.mkdir(parents=True, exist_ok=True)
        with self.health_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "category": category,
                "expected": expected,
                "actual": actual,
                "ratio": round(actual / expected, 4) if expected else 0,
                "healthy": healthy,
                "reason": reason,
            }) + "\n")

    def total_cases(self, category: str) -> int:
        for line in read_lines(self.path(f"category_{category}.log")):
            if m := re.match(r"Test cases: (\d+)", line):
                return int(m.group(1))
        return 0

    def check_case_counts(self, category: str) -> bool | None:
        """None when there is no count to read, which is not a verdict."""
        log = self.path(f"category_{category}.log")
        if not log.is_file():
            print(f"❌ File not found: {log}", file=sys.stderr)
            return None
        all_pass = True
        current = ""
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if m := re.match(r"^Category: (\S+)", line):
                current = m.group(1)
                continue
            m = re.search(r"Test cases: (\d+)", line)
            if not (m and current):
                continue
            name, current = current, ""
            actual = int(m.group(1))
            expected = expected_cases(name, self.platform)
            if name == "op_ut":
                print(f"op_ut expected value for {self.platform} platform: {expected}")
            if not expected:
                continue
            threshold = round(expected * 0.95)
            healthy = actual >= threshold
            print(f"📈 {name}:")
            print(f"   Expected: {expected}, Actual: {actual}")
            print(f"   Threshold: {threshold}, "
                  f"Reduction: {(actual / expected - 1) * 100:.2f}%")
            print("   Status: ✅ Normal" if healthy
                  else "   Status: ❌ Abnormal (>5% reduction)")
            print("-" * 40)
            all_pass = all_pass and healthy
            self.record_health(name, expected, actual, healthy,
                               "" if healthy else "below 95% threshold")
        return all_pass

    def report_filtered(self, category: str, failures: list[str]) -> None:
        """The known failures this run hit, which are not news."""
        known = self.known_issues
        hits = [(number, line) for number, line in enumerate(failures, 1)
                if any(pattern in line for pattern in known)]
        write_lines(self.path(f"failures_{category}_removed.log"),
                    [f"{n}:{line}" for n, line in hits])
        if hits:
            print(f"⏩ Skipping {len(hits)} known issues:")
            for number, line in hits:
                print(f"   Line {number:3d}: {line}")
        else:
            print("✅ No skipped cases")

    def report_passing_known_issues(self, passed: list[str]) -> list[str]:
        known = set(self.known_issues)
        now_passing = sorted({line for line in passed if line in known})
        print("\n📊 New Passing Known Issues:")
        if now_passing:
            print("\n".join(now_passing))
            print(f"✅ {len(now_passing)} known issues are now passing!")
        else:
            print("ℹ️  No known issues are now passing")
        if now_passing and self.may_write_issues():
            self.mark_passed_issues(now_passing)
        return now_passing

    def new_failures(self, category: str, failures: list[str]) -> list[str]:
        known = set(self.known_issues)
        new = [line for line in failures if line not in known]
        write_lines(self.path(f"failures_{category}_filtered.log"), new)
        print("\n📊 New Failed Cases Summary:")
        if new:
            print("❌ New failures found:\n")
            print("\n".join(new))
        else:
            print("✅ No new failed cases")
        return new

    def may_write_issues(self) -> bool:
        """Only a scheduled run built from source may rewrite an issue.

        An on-demand run tests an arbitrary commit, so what passes in it says
        nothing about the issue.
        """
        return (os.environ.get("GITHUB_EVENT_NAME") == "schedule"
                and self.pytorch != "nightly_wheel")

    # -- GitHub ------------------------------------------------------------- #

    def mark_passed_issues(self, passed: list[str]) -> None:
        """Strike through the cases of an issue that this run passed."""
        issues_log = self.path("issues.log")
        if not issues_log.is_file():
            print(f"❌ Missing {issues_log}", file=sys.stderr)
            return
        if shutil.which("gh") is None:
            print("❌ GitHub CLI required", file=sys.stderr)
            return
        random_issues = set(gh(
            "issue", "list", "--repo", REPO, "--state", "open", "--label", "random",
            "--json", "number", "--jq", ".[].number",
        ).stdout.split())

        passed_set = set(passed)
        per_issue: dict[str, set[str]] = defaultdict(set)
        issue_id = ""
        in_cases = False
        for raw in issues_log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                in_cases = False
                continue
            if line.startswith("Cases:"):
                in_cases = True
                continue
            if m := re.search(r"Issue #(\d+)", line):
                issue_id = m.group(1)
                continue
            if in_cases and issue_id and issue_id not in random_issues and line in passed_set:
                per_issue[issue_id].add(line)
        print(f"✅ Done! Found {len(per_issue)} issues with passed items")

        run_url = ""
        if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
            run_url = (f" [nightly testing](https://github.com/"
                       f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                       f"{os.environ['GITHUB_RUN_ID']})")
        label = issue_platform(self.platform, self.runner)
        for issue_id, cases in per_issue.items():
            listed = " ".join(sorted(cases))
            print(f"📝 Processing issue #{issue_id} with cases: {listed}")
            body = gh("--repo", REPO, "issue", "view", issue_id, "--json", "body",
                      "-q", ".body")
            if body.returncode != 0:
                warn(f"could not read issue #{issue_id}: {body.stderr.strip()}")
                continue
            struck = "\n".join(
                f"~~{line}~~" if line.strip() in cases else line
                for line in body.stdout.splitlines()
            )
            body_file = self.path(f"issue-body-{issue_id}.txt")
            body_file.write_text(struck, encoding="utf-8")
            gh("--repo", REPO, "issue", "edit", issue_id, "--body-file", str(body_file))
            gh("--repo", REPO, "issue", "comment", issue_id, "--body",
               f"✅ {listed} Passed in{run_url or ' nightly testing'} on {label}")
            body_file.unlink(missing_ok=True)
            print(f"✅ Updated issue #{issue_id}")

    # -- per-category runners ----------------------------------------------- #

    def run(self, category: str) -> CheckResult:
        if category in MAIN_CATEGORIES:
            return self.run_main(category)
        if category == "xpu_distributed":
            return self.run_distributed(category)
        if category == "skipped_ut":
            return self.run_skipped()
        if category == "xpu_profiling":
            return self.run_profiling()
        print(f"❌ Unknown test suite: {category}", file=sys.stderr)
        print("💡 Available: " + ", ".join([*MAIN_CATEGORIES, "xpu_distributed",
                                            "skipped_ut", "xpu_profiling"]),
              file=sys.stderr)
        return CheckResult(category=category, passed=True)

    def banner(self, category: str) -> None:
        print("=" * 73)
        print(f"Running tests for: {category}")
        print("=" * 73)

    def run_main(self, category: str) -> CheckResult:
        self.banner(category)
        failures = read_lines(self.path(f"failures_{category}.log"))
        passed = read_lines(self.path(f"passed_{category}.log"))

        print("📋 Failed Cases:")
        print("\n".join(failures) if failures else "✅ No failed cases")

        print("\n📊 Test Case Counts:")
        counts_normal = self.check_case_counts(category)

        print("\n🔍 Filtered Cases:")
        self.report_filtered(category, failures)

        print("\n✅ Passing Known Issues:")
        new_passes = self.report_passing_known_issues(passed)

        print("\nChecking New Failures:")
        new = self.new_failures(category, failures) if failures else []

        return self.finish(category, new, len(passed), counts_normal,
                           total=self.total_cases(category) or len(passed) + len(failures),
                           new_passes=new_passes)

    def run_distributed(self, category: str) -> CheckResult:
        """The distributed suite prints a pytest console log, not a junit xml."""
        self.banner(category)
        failures = self.scan_console(category, ("FAILED", "ERROR"))
        passed = self.scan_console(category, ("PASSED",))
        write_lines(self.path(f"{category}_failed.log"), failures)
        write_lines(self.path(f"{category}_passed.log"), passed)

        print("📋 Failed Cases:")
        print("\n".join(failures))

        print("\n🔍 Filtered Cases:")
        self.report_filtered(category, failures)
        new_passes = self.report_passing_known_issues(passed)
        new = self.new_failures(category, failures)
        return self.finish(category, new, len(passed), None,
                           total=len(passed) + len(failures), new_passes=new_passes)

    def scan_console(self, category: str, markers: tuple[str, ...]) -> list[str]:
        """Sorted, unique `file::class::case` node ids of one outcome."""
        found: set[str] = set()
        for log in sorted(self.work_dir.glob(f"{category}_test*.log")):
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if not any(marker in line for marker in markers):
                    continue
                found.update(word for word in line.split()
                             if re.search(r"::.*::", word))
        return sorted(found)

    def run_skipped(self) -> CheckResult:
        """Cases skipped on purpose that pass anyway: a review item, not a gate."""
        print("🔍 Checking for newly passed tests in skipped UT...")
        test_log = self.path("skipped_ut_with_skip_test.log")
        if not test_log.is_file():
            print(f"❌ Test log not found: {test_log}", file=sys.stderr)
            return CheckResult(category="skipped_ut", passed=True)

        body = gh("--repo", REPO, "issue", "view", str(NEW_PASSED_ISSUE),
                  "--json", "body", "-q", ".body")
        known = sorted({word for line in body.stdout.splitlines()
                        for word in line.split() if re.search(r"::.*::", word)}) \
            if body.returncode == 0 else []
        print("✅ Fetched known tests from GitHub" if known
              else "⚠️  Using empty known tests file")

        now_passing = sorted({
            word
            for line in test_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if "PASSED" in line
            for word in line.split() if re.search(r"::.*::", word)
        })
        write_lines(self.path("known-passed-issue.cases"), known)
        write_lines(self.path("new-passed-issue.cases"), now_passing)

        new = [case for case in now_passing
               if case not in set(known)
               and not any(ignored in case for ignored in IGNORE_TESTS)]
        write_lines(self.path("new-passed-this-run.cases"), new)
        if new:
            print(f"❌ {len(new)} NEW PASSING TESTS:")
            print("\n".join(new))
            print("Please review these tests!")
        else:
            print("✅ No new passing tests found")

        if self.may_write_issues() and (known or now_passing):
            gh("--repo", REPO, "issue", "edit", str(NEW_PASSED_ISSUE), "--body-file",
               str(self.path("new-passed-issue.cases")))
            print(f"✅ Successfully updated issue #{NEW_PASSED_ISSUE}")
        return CheckResult(category="skipped_ut", passed=True)

    def run_profiling(self) -> CheckResult:
        """Profiling has no case roster, only pass/fail counts per report."""
        self.banner("xpu_profiling")
        reports = sorted(set(self.work_dir.rglob("*.xml"))
                         | set(self.work_dir.parent.glob("*.xml")))
        if not reports:
            print("❌ TEST FAILED: xpu_profiling - no pytest junit xml report found")
            return CheckResult(category="xpu_profiling", passed=False)

        print("📋 Test Reports:")
        total_tests = total_failures = total_errors = 0
        report_count = 0
        for path in reports:
            try:
                suite = next(iter_testsuites(path))
            except (ET.ParseError, StopIteration):
                print(f"   ⚠️  {path}: not a pytest report, skipped")
                continue
            tests, _, failures, errors = suite_counts(suite)
            report_count += 1
            total_tests += tests
            total_failures += failures
            total_errors += errors
            if failures or errors:
                print(f"   ❌ {path}: {tests} tests, {failures} failures, {errors} errors")
                for case in suite.iter("testcase"):
                    if any(child.tag in ("failure", "error") for child in case):
                        print(f"      - {case.get('name', '')}")
            else:
                print(f"   ✅ {path}: {tests} tests")

        passed = read_lines(self.path("passed_xpu_profiling.log"))
        failed = read_lines(self.path("failures_xpu_profiling.log"))
        new_passes: list[str] = []
        if passed and self.known_issues:
            print("\n✅ Passing Known Issues:")
            new_passes = self.report_passing_known_issues(passed)
        if failed and self.known_issues:
            print("\nChecking New Failures:")
            self.new_failures("xpu_profiling", failed)

        print("\n📈 Final Summary:")
        print(f"   Reports: {report_count}, Tests: {total_tests}, "
              f"Failed: {total_failures}, Errors: {total_errors}")
        ok = total_failures == 0 and total_errors == 0 and total_tests > 0
        print(f"{'✅ TEST PASSED' if ok else '❌ TEST FAILED'}: xpu_profiling")
        return CheckResult(category="xpu_profiling", passed=ok, total=total_tests,
                           new_passes=new_passes)

    def finish(self, category: str, new: list[str], passed_count: int,
               counts_normal: bool | None, total: int = 0,
               new_passes: list[str] | None = None) -> CheckResult:
        """A category passes when it has new failures, no passes, no shrink."""
        print("\n📈 Final Summary:")
        print(f"   Failed: {len(new)}, Passed: {passed_count}, "
              f"All counts normal: {'false' if counts_normal is False else 'true'}")
        ok = not new and passed_count > 0 and counts_normal is not False
        print(f"{'✅ TEST PASSED' if ok else '❌ TEST FAILED'}: {category}")
        return CheckResult(category=category, passed=ok, new_failures=new,
                           counts_normal=counts_normal, total=total,
                           new_passes=new_passes or [])


# --------------------------------------------------------------------------- #
# summary - every UT job of a run
# --------------------------------------------------------------------------- #


@dataclass
class JobMeta:
    ut_job: str
    runner: str
    category: str

    @classmethod
    def read(cls, path: Path) -> JobMeta | None:
        """Artifact content is test output, so it is read as data, never run."""
        values = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip()] = value.strip()
        ut_job = values.get("UT_NAME", "")
        return cls(ut_job, values.get("UT_RUNNER", ""), values.get("UT_CATEGORY", "")) \
            if ut_job else None


class KnownIssues:
    """The known-failure list of a run, per runner and category.

    The build-start snapshot keeps issues closed mid-run; the live fetch picks
    up issues opened after it.
    """

    def __init__(self, snapshot_dir: Path, work_dir: Path) -> None:
        self.snapshot = snapshot_dir
        self.work = work_dir
        self.static = self._fetch_static()
        self._open: dict[tuple[str, str], str] = {}

    def _snapshot_text(self, name: str) -> str:
        path = self.snapshot / name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    def _fetch(self, *args: str) -> str:
        proc = subprocess.run(
            ["bash", str(SCRIPT_DIR / "ut_known_issues.sh"), *args],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            warn(f"ut_known_issues.sh {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def _fetch_static(self) -> str:
        return self._snapshot_text("known_issue_static.log") + self._fetch("static")

    def open_issues(self, category: str, runner: str) -> str:
        key = (category, runner)
        if key not in self._open:
            text = self._snapshot_text("issues_snapshot.log")
            if "bmg" in runner:
                text += self._snapshot_text("issues_snapshot_bmg.log")
            if category == "dpclang":
                text += self._snapshot_text("issues_snapshot_dpclang.log")
            self._open[key] = text + self._fetch("open", category, runner)
        return self._open[key]

    @staticmethod
    def cases_of(issues_text: str, category: str) -> list[str]:
        """The `Cases:` block of every issue that names this category."""
        lines = []
        in_cases = False
        for raw in issues_text.splitlines():
            line = raw.rstrip()
            if not re.search(r"[a-zA-Z0-9]", line):
                in_cases = False
            elif in_cases:
                first = line.split()[0]
                if re.search(category, first):
                    lines.append(line)
            if "Cases:" in line:
                in_cases = True
        return lines

    def write_for(self, category: str, meta: JobMeta, dest: Path) -> None:
        issues = self.open_issues(meta.category, meta.runner)
        (dest.parent / "issues.log").write_text(issues, encoding="utf-8")
        dest.write_text(
            self.static + "\n" + "\n".join(self.cases_of(issues, category)) + "\n",
            encoding="utf-8",
        )


ARTIFACT_SUFFIXES = {".log", ".csv", ".xml"}


def collect_job_dir(artifact: Path, job_dir: Path) -> None:
    """Flatten an artifact: every check reads its inputs from one directory."""
    job_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            target = job_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)


def enrich_failures(job_dir: Path, lines: list[str], what: str = "new") -> list[str]:
    """Give each failure the error message only the csv carries.

    Keyed on category too: one job's csv holds every category it ran, and the
    same class and test name can appear in two of them.
    """
    rows = read_lines(job_dir / "ut_failure_list.csv")
    index = {}
    for row in rows:
        fields = [f.strip() for f in row.strip("|").split(" | ")]
        if len(fields) >= 3:
            index.setdefault(tuple(fields[:3]), row)
    enriched = [index[key] for line in lines
                if (key := tuple(line.split(",", 2))) in index]
    if rows and len(enriched) != len(lines):
        warn(f"{job_dir.name} {what} failure count mismatch: "
             f"filtered={len(lines)}, csv={len(enriched)}")
    return enriched


# How much of a failure table the report carries; a step summary is capped at
# 1MB and a bad night has thousands of rows. The artifact keeps all of them.
REPORT_FAILURE_LIMIT = 100
SUMMARY_TABLE_HEADER = ["UT job", "Category", "Total cases", "Known failures", "New failures",
                       "New passes", "Result"]


def check_run(artifacts_dir: Path, snapshot_dir: Path, work_dir: Path,
              pytorch: str, platform: str = "linux") -> int:
    """Compare every UT job in `artifacts_dir` with the known-failure list.

    Writes, per UT job: `verdict.json`, the new and known failure tables, and
    `run_health.jsonl`. Fails when a category has a new failure or did not run
    healthily.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    known = KnownIssues(snapshot_dir, work_dir)
    failed = False
    jobs = 0

    for artifact in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        meta_file = next(iter(sorted(artifact.rglob("ut_meta.env"))), None)
        meta = JobMeta.read(meta_file) if meta_file else None
        if meta is None:
            warn(f"{artifact.name} names no UT job, skipped")
            continue
        jobs += 1
        job_dir = work_dir / meta.ut_job.replace(",", "_")
        collect_job_dir(artifact, job_dir)

        print(f"::group::{meta.ut_job} (runner: {meta.runner or 'unknown'})", flush=True)
        # The UT job writes these too; rebuilding them keeps the check working
        # on an artifact whose job died before it reported.
        reports = sorted(job_dir.glob("*.xml"))
        if reports:
            build_report(meta.ut_job, reports, job_dir)
        else:
            warn(f"{meta.ut_job} produced no junit report")

        checker = Checker(
            work_dir=job_dir, pytorch=pytorch, platform=platform, runner=meta.runner,
            health_file=job_dir / "run_health.jsonl",
        )
        verdict = {"ut_job": meta.ut_job, "runner": meta.runner, "categories": []}
        new_lines: list[str] = []
        known_lines: list[str] = []
        for category in categories_of(meta.ut_job):
            known.write_for(category, meta, job_dir / "Known_issue.log")
            result = checker.run(category)
            failed = failed or not result.passed
            failures = read_lines(job_dir / f"failures_{category}.log")
            was_known = [line for line in failures if line not in set(result.new_failures)]
            new_lines += result.new_failures
            known_lines += was_known
            verdict["categories"].append({
                "category": category,
                "total": result.total,
                "known": len(was_known),
                "new": len(result.new_failures),
                "new_passes": len(result.new_passes),
                "passed": result.passed,
            })
        print("::endgroup::", flush=True)

        write_lines(job_dir / "new_failure_list.txt", new_lines)
        write_lines(job_dir / "new_ut_failure_list.csv",
                    enrich_failures(job_dir, new_lines, "new"))
        write_lines(job_dir / "known_ut_failure_list.csv",
                    enrich_failures(job_dir, known_lines, "known"))
        (job_dir / "verdict.json").write_text(json.dumps(verdict, indent=2),
                                              encoding="utf-8")

    if jobs == 0:
        print("::error::no UT data artifact carried a ut_meta.env; nothing to check")
        return 1
    return 1 if failed else 0


def render_report(checks_dir: Path, out_file: Path) -> int:
    """One report for a run, from what every check job concluded."""
    rows: list[list[str]] = []
    new_rows: list[str] = []
    known_rows: list[str] = []
    for verdict_file in sorted(checks_dir.rglob("verdict.json")):
        verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
        for entry in verdict["categories"]:
            rows.append([
                verdict["ut_job"], entry["category"],
                str(entry.get("total", 0)),
                str(entry["known"]), str(entry["new"]),
                str(entry.get("new_passes", 0)),
                "✅ passed" if entry["passed"] else "❌ failed",
            ])
        new_rows += read_lines(verdict_file.parent / "new_ut_failure_list.csv")
        known_rows += read_lines(verdict_file.parent / "known_ut_failure_list.csv")

    if not rows:
        warn("no UT check artifact to report on")

    def fold(title: str, table_rows: list[str]) -> list[str]:
        shown = table_rows[:REPORT_FAILURE_LIMIT]
        body = md_table(FAILURE_HEADER, []) + shown
        if len(shown) < len(table_rows):
            body += ["", f"… and {len(table_rows) - len(shown)} more, in the artifact"]
        return ["", *md_details(f"{title} ({len(table_rows)})", body)]

    report = ["## UT Report", "", "### Summary", "",
              *md_table(SUMMARY_TABLE_HEADER, rows)]
    report += fold("New failures", new_rows)
    report += fold("Known failures", known_rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_lines(out_file, report)
    write_lines(out_file.parent / "new_failures.csv", new_rows)
    write_lines(out_file.parent / "known_failures.csv", known_rows)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="junit/log reports -> case logs")
    report.add_argument("-n", "--ut-name", default="", help="UT job name")
    report.add_argument("-i", "--input-files", nargs="+", required=True,
                        help="JUnit XML or console log files")
    report.add_argument("--dir", default=".", help="where the case logs are written")

    check = commands.add_parser("check", help="a run's UT jobs against known failures")
    check.add_argument("--artifacts", required=True, type=Path,
                       help="downloaded Inductor-XPU-UT-Data artifacts")
    check.add_argument("--work", required=True, type=Path)
    check.add_argument("--snapshot", default="", type=Path)
    check.add_argument("--pytorch", default="nightly_wheel")
    check.add_argument("--platform", default="linux", choices=("linux", "windows"))

    report_run = commands.add_parser("summary", help="one report from the checks")
    report_run.add_argument("--checks", required=True, type=Path,
                            help="downloaded UT-Check artifacts")
    report_run.add_argument("--out", required=True, type=Path,
                            help="markdown report to write")

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if args.command == "report":
        out_dir = Path(args.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        built = build_report(args.ut_name, [Path(p) for p in args.input_files], out_dir)
        print(built.render())
        return 0

    if args.command == "check":
        return check_run(args.artifacts, Path(args.snapshot or "."), args.work,
                         args.pytorch, args.platform)

    return render_report(args.checks, args.out)


if __name__ == "__main__":
    sys.exit(main())
