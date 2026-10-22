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
# The issue whose body is the static known-failure list (ut_known_issues.sh).
STATIC_ISSUE = 1624
KNOWN_ISSUE_IDS = "known_issue_ids.json"
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
    file: str = ""

    @property
    def line(self) -> str:
        return f"{self.category},{self.classname},{self.name}"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.category, self.file, self.classname, self.name


# Which outcome wins when one case ran more than once (reruns, retried jobs).
STATUS_PRIORITY = {"passed": 0, "failed": 1, "skipped": 2}


def status_rank(status: str) -> int:
    return STATUS_PRIORITY.get(status, len(STATUS_PRIORITY))


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


def determine_category(ut: str, ut_job: str = "") -> str:
    # The distributed job names its junit reports op_ut_*, which they are not.
    if ut_job == "xpu_distributed":
        return ut_job
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

    def __init__(self, ut_job: str = "") -> None:
        self.ut_job = ut_job
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
        category = determine_category(path.name.split(".")[0], self.ut_job)
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
                    file=normalize(case.get("file")),
                ))

    def _add_log(self, path: Path) -> None:
        """Unittest console output, for the suites that produce no junit xml."""
        content = path.read_text(encoding="utf-8", errors="replace")
        ut = path.stem
        category = determine_category(ut, self.ut_job)

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

    def dedupe(self) -> None:
        """One result per case, the best one; junit counts are recounted from it."""
        best: dict[tuple[str, str, str, str], Case] = {}
        for case in self.cases:
            kept = best.get(case.key)
            if kept is None or status_rank(case.status) < status_rank(kept.status):
                best[case.key] = case
        self.cases = list(best.values())

        xml_categories = sorted({s.category for s in self.summaries if s.source == "XML"})
        self.summaries = [s for s in self.summaries if s.source != "XML"]
        for category in xml_categories:
            counts = defaultdict(int)
            for case in self.cases:
                if case.category == category and case.source == "XML":
                    counts[case.status] += 1
            tests = sum(counts.values())
            self.summaries.append(SuiteSummary(
                category=category, ut="merged", tests=tests, passed=counts["passed"],
                skipped=counts["skipped"], failures=counts["failed"],
                errors=tests - counts["passed"] - counts["skipped"] - counts["failed"],
                source="XML",
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


def build_report(ut_name: str, inputs: list[Path], out_dir: Path,
                 dedupe: bool = False) -> Report:
    report = Report(ut_name)
    for path in inputs:
        report.add(path)
    if dedupe:
        report.dedupe()
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
    failures: list[str] = field(default_factory=list)

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

    @property
    def issue_ids(self) -> dict[str, list[str]]:
        path = self.path(KNOWN_ISSUE_IDS)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

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
                           new_passes=new_passes, failures=failures)

    def run_distributed(self, category: str) -> CheckResult:
        """The distributed suite prints a pytest console log, not a junit xml."""
        self.banner(category)
        passed = self.scan_console(category, ("PASSED",))
        passed_set = set(passed)
        failures = [line for line in self.scan_console(category, ("FAILED", "ERROR"))
                    if line not in passed_set]
        write_lines(self.path(f"{category}_failed.log"), failures)
        write_lines(self.path(f"{category}_passed.log"), passed)

        print("📋 Failed Cases:")
        print("\n".join(failures))

        print("\n🔍 Filtered Cases:")
        self.report_filtered(category, failures)
        new_passes = self.report_passing_known_issues(passed)
        new = self.new_failures(category, failures)
        return self.finish(category, new, len(passed), None,
                           total=self.total_cases(category) or len(passed) + len(failures),
                           new_passes=new_passes, failures=failures)

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
        new: list[str] = []
        if passed:
            print("\n✅ Passing Known Issues:")
            new_passes = self.report_passing_known_issues(passed)
        if failed:
            print("\nChecking New Failures:")
            new = self.new_failures("xpu_profiling", failed)

        print("\n📈 Final Summary:")
        print(f"   Reports: {report_count}, Tests: {total_tests}, "
              f"Failed: {total_failures}, Errors: {total_errors}")
        ok = total_failures == 0 and total_errors == 0 and total_tests > 0
        print(f"{'✅ TEST PASSED' if ok else '❌ TEST FAILED'}: xpu_profiling")
        return CheckResult(category="xpu_profiling", passed=ok, total=total_tests,
                           new_failures=new, new_passes=new_passes, failures=failed)

    def finish(self, category: str, new: list[str], passed_count: int,
               counts_normal: bool | None, total: int = 0,
               new_passes: list[str] | None = None,
               failures: list[str] | None = None) -> CheckResult:
        """A category passes when it has new failures, no passes, no shrink."""
        print("\n📈 Final Summary:")
        print(f"   Failed: {len(new)}, Passed: {passed_count}, "
              f"All counts normal: {'false' if counts_normal is False else 'true'}")
        ok = not new and passed_count > 0 and counts_normal is not False
        print(f"{'✅ TEST PASSED' if ok else '❌ TEST FAILED'}: {category}")
        return CheckResult(category=category, passed=ok, new_failures=new,
                           counts_normal=counts_normal, total=total,
                           new_passes=new_passes or [], failures=failures or [])


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
        return [line for _, line in KnownIssues.issue_cases(issues_text, category)]

    @staticmethod
    def issue_cases(issues_text: str, category: str) -> list[tuple[str, str]]:
        """`cases_of`, each with the number of the issue that lists it."""
        pairs = []
        issue = ""
        in_cases = False
        for raw in issues_text.splitlines():
            line = raw.rstrip()
            if m := re.match(r"Issue #(\d+)", line):
                issue, in_cases = m.group(1), False
                continue
            if not re.search(r"[a-zA-Z0-9]", line):
                in_cases = False
            elif in_cases:
                first = line.split()[0]
                if re.search(category, first):
                    pairs.append((issue, line))
            if "Cases:" in line:
                in_cases = True
        return pairs

    def write_for(self, category: str, meta: JobMeta, dest: Path) -> None:
        issues = self.open_issues(meta.category, meta.runner)
        (dest.parent / "issues.log").write_text(issues, encoding="utf-8")
        pairs = self.issue_cases(issues, category)
        dest.write_text(
            self.static + "\n" + "\n".join(line for _, line in pairs) + "\n",
            encoding="utf-8",
        )
        ids: dict[str, list[str]] = defaultdict(list)
        for issue, line in [*((str(STATIC_ISSUE), raw) for raw in self.static.splitlines()),
                            *pairs]:
            if line.strip() and issue and issue not in ids[line.strip()]:
                ids[line.strip()].append(issue)
        (dest.parent / KNOWN_ISSUE_IDS).write_text(json.dumps(ids), encoding="utf-8")


ARTIFACT_SUFFIXES = {".log", ".csv", ".xml"}


def collect_job_dir(artifact: Path, job_dir: Path, tag: str = "") -> None:
    """Flatten an artifact: every check reads its inputs from one directory.

    A log another artifact of the job already brought is kept under `tag`, so
    console logs of every attempt are scanned; the junit xml is read in place.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            target = job_dir / path.name
            if target.exists() and tag and path.suffix == ".log":
                target = job_dir / f"{path.stem}.{tag}{path.suffix}"
            if not target.exists():
                shutil.copy2(path, target)


def case_fields(line: str) -> tuple[str, str]:
    """The class and test of either a `category,class,test` line or a node id."""
    if "::" in line:
        classname, _, name = line.rpartition("::")
        return classname, name
    fields = line.split(",", 2)
    return (fields[1], fields[2]) if len(fields) == 3 else ("", line)


def issue_links(ids: list[str]) -> str:
    return ", ".join(f"[#{i}](https://github.com/{REPO}/issues/{i})" for i in ids)


def node_id_parts(line: str) -> tuple[str, str]:
    """The class and test of a `path.py::Class::test` console node id."""
    parts = line.split("::")
    return (parts[-2] if len(parts) > 1 else ""), parts[-1]


def enrich_failures(job_dir: Path, items: list[tuple[str, str]],
                    what: str = "new") -> list[str]:
    """Give each failure the error message only the csv carries.

    Keyed on category too: one job's csv holds every category it ran, and the
    same class and test name can appear in two of them. A console node id
    names the same case as a junit row with a dotted module path, so it joins
    on the class and test alone.
    """
    rows = read_lines(job_dir / "ut_failure_list.csv")
    index: dict[tuple[str, ...], str] = {}
    by_case: dict[tuple[str, str], str] = {}
    for row in rows:
        fields = [f.strip() for f in row.strip("|").split(" | ")]
        if len(fields) >= 3:
            index.setdefault(tuple(fields[:3]), row)
            by_case.setdefault((fields[1].rpartition(".")[2], fields[2]), row)
    enriched = []
    missing = 0
    for category, line in items:
        row = index.get(tuple(line.split(",", 2)))
        if row is None and "::" in line:
            classname, name = node_id_parts(line)
            row = by_case.get((classname, name))
            if row is not None:
                fields = [f.strip() for f in row.strip("|").split(" | ")]
                row = md_row([category, *fields[1:]])
        if row is None:
            missing += 1
            classname, name = case_fields(line)
            row = md_row([category, classname, name, "failed", "", "Log"])
        enriched.append(row)
    if rows and missing:
        warn(f"{job_dir.name} {what} failures not in the junit csv: {missing} "
             f"of {len(items)}")
    return enriched


# How much of a failure table the report carries; a step summary is capped at
# 1MB and a bad night has thousands of rows. The artifact keeps all of them.
REPORT_FAILURE_LIMIT = 100
SUMMARY_TABLE_HEADER = ["OS", "Category", "Total cases", "Known failures",
                       "New failures", "New passes", "Result"]
PASS_HEADER = ["Category", "Class name", "Test name"]

# What a check publishes per UT job; the report reads the same names.
NEW_FAILURES_TXT = "new_failures.txt"
NEW_FAILURES_CSV = "new_failures.csv"
KNOWN_FAILURES_CSV = "known_failures.csv"
NEW_PASSES_CSV = "new_passes.csv"
CASE_LOG_KINDS = ("all_cases", "passed", "failures", "category")


def check_run(artifacts_dir: Path, snapshot_dir: Path, work_dir: Path, out_dir: Path,
              pytorch: str, platform: str = "linux") -> int:
    """Compare every UT job in `artifacts_dir` with the known-failure list.

    `work_dir` is scratch. `out_dir/<ut_job>/` gets what the check concluded:

        verdict.json  run_health.jsonl
        new_failures.txt  new_failures.csv  known_failures.csv  new_passes.csv
        cases/  per-category case logs and ut_failure_list.csv

    Fails when a category has a new failure or did not run healthily.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    known = KnownIssues(snapshot_dir, work_dir)
    failed = False
    jobs = 0

    # Every attempt of a UT job uploads its own artifact; they are checked as one.
    by_job: dict[str, list[tuple[Path, JobMeta]]] = defaultdict(list)
    for artifact in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        meta_file = next(iter(sorted(artifact.rglob("ut_meta.env"))), None)
        meta = JobMeta.read(meta_file) if meta_file else None
        if meta is None:
            warn(f"{artifact.name} names no UT job, skipped")
            continue
        by_job[meta.ut_job].append((artifact, meta))

    for ut_job, members in by_job.items():
        jobs += 1
        meta = members[0][1]
        job_dir = work_dir / ut_job.replace(",", "_")
        out_job = out_dir / ut_job.replace(",", "_")
        (out_job / "cases").mkdir(parents=True, exist_ok=True)
        reports: list[Path] = []
        for index, (artifact, _) in enumerate(members):
            collect_job_dir(artifact, job_dir, tag=str(index))
            reports += sorted(artifact.rglob("*.xml"))

        print(f"::group::{ut_job} ({plural(len(members), 'artifact')}, "
              f"runner: {meta.runner or 'unknown'})", flush=True)
        # The UT job writes these too; rebuilding them keeps the check working
        # on an artifact whose job died before it reported.
        if reports:
            build_report(ut_job, reports, job_dir, dedupe=True)
        else:
            warn(f"{ut_job} produced no junit report")

        checker = Checker(
            work_dir=job_dir, pytorch=pytorch, platform=platform, runner=meta.runner,
            health_file=out_job / "run_health.jsonl",
        )
        verdict = {"ut_job": ut_job, "runner": meta.runner, "os": platform,
                   "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT") or 0),
                   "categories": []}
        new_lines: list[tuple[str, str]] = []
        known_lines: list[tuple[str, str]] = []
        pass_lines: list[tuple[str, str]] = []
        links: dict[tuple[str, str], str] = {}
        for category in categories_of(meta.ut_job):
            known.write_for(category, meta, job_dir / "Known_issue.log")
            result = checker.run(category)
            failed = failed or not result.passed
            # new: failed, no known issue; known: failed, has one; new pass: passed, has one
            new_set = set(result.new_failures)
            was_known = [line for line in result.failures if line not in new_set]
            new_lines += [(category, line) for line in result.new_failures]
            known_lines += [(category, line) for line in was_known]
            pass_lines += [(category, line) for line in result.new_passes]
            ids = checker.issue_ids
            for line in [*was_known, *result.new_passes]:
                links[category, line] = issue_links(ids.get(line.strip(), []))
            verdict["categories"].append({
                "category": category,
                "total": result.total,
                "known": len(was_known),
                "new": len(result.new_failures),
                "new_passes": len(result.new_passes),
                "passed": result.passed,
            })
        print("::endgroup::", flush=True)

        write_lines(out_job / NEW_FAILURES_TXT, [line for _, line in new_lines])
        write_lines(out_job / NEW_FAILURES_CSV,
                    enrich_failures(job_dir, new_lines, "new"))
        write_lines(out_job / KNOWN_FAILURES_CSV, [
            f"{row} {links[item]} |" for item, row in
            zip(known_lines, enrich_failures(job_dir, known_lines, "known"))])
        write_lines(out_job / NEW_PASSES_CSV,
                    [md_row([category, *case_fields(line), links[category, line]])
                     for category, line in pass_lines])
        (out_job / "verdict.json").write_text(json.dumps(verdict, indent=2),
                                              encoding="utf-8")
        case_logs = [job_dir / f"{kind}_{category}.log" for category in
                     categories_of(ut_job) for kind in CASE_LOG_KINDS]
        for path in [*case_logs, job_dir / "ut_failure_list.csv"]:
            if path.is_file():
                shutil.copy2(path, out_job / "cases" / path.name)

    if jobs == 0:
        print("::error::no UT data artifact carried a ut_meta.env; nothing to check")
        return 1
    return 1 if failed else 0


def render_report(checks_dir: Path, out_dir: Path) -> int:
    """One report for a run, from what every check job concluded.

    Writes `out_dir/`: ut_report.md, new_failures.csv, known_failures.csv,
    new_passes.csv.
    """
    # A rerun check merged every attempt's cases, so the latest one is final.
    best: dict[tuple[str, str, str], tuple[tuple[int, int], list[str], Path]] = {}
    for verdict_file in sorted(checks_dir.rglob("verdict.json")):
        verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
        for entry in verdict["categories"]:
            result = entry.get("status") or ("passed" if entry["passed"] else "failed")
            mark = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(result, "❔")
            key = (verdict.get("os", "linux"), verdict["ut_job"], entry["category"])
            row = [
                key[0], key[2], str(entry.get("total", 0)),
                str(entry["known"]), str(entry["new"]),
                str(entry.get("new_passes", 0)),
                f"{mark} {result}",
            ]
            score = (-int(verdict.get("attempt") or 0), status_rank(result))
            if key not in best or score < best[key][0]:
                best[key] = (score, row, verdict_file.parent)

    rows: list[list[str]] = []
    new_rows: list[str] = []
    known_rows: list[str] = []
    pass_rows: list[str] = []
    for (_, _, category), (_, row, job_dir) in best.items():
        rows.append(row)
        for name, table in ((NEW_FAILURES_CSV, new_rows),
                            (KNOWN_FAILURES_CSV, known_rows),
                            (NEW_PASSES_CSV, pass_rows)):
            table += [line for line in read_lines(job_dir / name)
                      if line.strip("|").split(" | ")[0].strip() == category]

    if not rows:
        warn("no UT check artifact to report on")
    rows.sort(key=lambda row: row[0])  # group the OSes, keep each one's job order

    def fold(title: str, table_rows: list[str], header: list[str]) -> list[str]:
        shown = table_rows[:REPORT_FAILURE_LIMIT]
        if "Source" in header:
            # Source is last but one when an Issue column follows it.
            tail = len(header) - 1 - header.index("Source")
            header = [h for h in header if h != "Source"]
            shown = [" | ".join(parts[:1] + parts[2:]) + ("" if tail else " |")
                     for parts in (row.rsplit(" | ", tail + 1) for row in shown)]
        body = md_table(header, []) + shown
        if len(shown) < len(table_rows):
            body += ["", f"… and {len(table_rows) - len(shown)} more, in the artifact"]
        return ["", *md_details(f"{title} ({len(table_rows)})", body)]

    report = ["## UT Report", "", "### Summary", "",
              *md_table(SUMMARY_TABLE_HEADER, rows)]
    report += fold("New failures", new_rows, FAILURE_HEADER)
    report += fold("Known failures", known_rows, [*FAILURE_HEADER, "Issue"])
    report += fold("New passes", pass_rows, [*PASS_HEADER, "Issue"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(out_dir / "ut_report.md", report)
    write_lines(out_dir / NEW_FAILURES_CSV, new_rows)
    write_lines(out_dir / KNOWN_FAILURES_CSV, known_rows)
    write_lines(out_dir / NEW_PASSES_CSV, pass_rows)
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
    check.add_argument("--data", required=True, type=Path,
                       help="downloaded UT-Data artifacts, one dir each")
    check.add_argument("--known-issues", default="", type=Path,
                       help="downloaded UT-Known-Issues snapshot")
    check.add_argument("--work", required=True, type=Path, help="scratch dir")
    check.add_argument("--out", required=True, type=Path,
                       help="where the check output is written, one dir per UT job")
    check.add_argument("--pytorch", default="nightly_wheel")
    check.add_argument("--platform", default="linux", choices=("linux", "windows"))

    report_run = commands.add_parser("summary", help="one report from the checks")
    report_run.add_argument("--checks", required=True, type=Path,
                            help="downloaded UT-Check artifacts")
    report_run.add_argument("--out", required=True, type=Path,
                            help="where the report is written")

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if args.command == "report":
        out_dir = Path(args.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        built = build_report(args.ut_name, [Path(p) for p in args.input_files], out_dir)
        print(built.render())
        return 0

    if args.command == "check":
        return check_run(args.data, Path(args.known_issues or "."), args.work, args.out,
                         args.pytorch, args.platform)

    return render_report(args.checks, args.out)


if __name__ == "__main__":
    sys.exit(main())
