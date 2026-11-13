# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run unit test files one pytest process each and check their junit results.

A file that crashes, times out or collects no tests still leaves a junit xml,
<prefix>_with_<mode>.<module>.crash.xml, holding one failed case named after the
module, so a broken file is never silently missing from the reports.
Standard library only: the launcher itself never imports torch.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

NO_TESTS_COLLECTED = 5  # pytest.ExitCode.NO_TESTS_COLLECTED
STOP_GRACE_SECONDS = 300
MAX_SUMMARY_CASES = 300
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

STATUS_LABEL = {
    "passed": "PASS",
    "failed": "FAIL",
    "crash": "CRASH",
    "timeout": "TIMEOUT",
    "no_tests": "NO TESTS",
}


@dataclass
class CaseFailure:
    classname: str
    name: str
    status: str  # failed / error
    message: str

    @property
    def node(self) -> str:
        return f"{self.classname}::{self.name}" if self.classname else self.name


@dataclass
class FileResult:
    test_file: str
    returncode: int
    status: str  # a STATUS_LABEL key
    duration: float
    xml_files: list[Path] = field(default_factory=list)
    tests: int = 0
    passed: int = 0
    skipped: int = 0
    failures: list[CaseFailure] = field(default_factory=list)


def _k_term(pattern: str) -> str:
    # pytest -k has no "::", so Class::test means both names match
    names = [name for name in pattern.split("::") if name]
    return names[0] if len(names) == 1 else "(" + " and ".join(names) + ")"


def _as_groups(include) -> list[list[str]]:
    """A flat pattern list is one OR group; a list of lists is several AND-ed groups."""
    if not include:
        return []
    if all(isinstance(item, str) for item in include):
        return [list(include)]
    return [list(group) for group in include if group]


def k_expression(include=(), exclude=(), k: str | None = None) -> str:
    """pytest -k of: every include group (patterns OR-ed), no exclude pattern, and k."""
    parts = []
    for group in _as_groups(include):
        terms = [_k_term(p) for p in group]
        parts.append(terms[0] if len(terms) == 1 else "(" + " or ".join(terms) + ")")
    parts += [f"not {_k_term(p)}" for p in exclude]
    if k:
        parts.append(f"({k})")
    return " and ".join(parts)


def is_entire_file(cases) -> bool:
    return any(case.endswith((".py::", ".py")) for case in cases or ())


def merge_skip_dicts(base: dict, *overlays: dict) -> dict[str, list[str]]:
    """The files of base with the excluded cases of the overlays added.

    Overlay entries of files missing from base are ignored, and a whole-file
    overlay entry replaces the cases of that file.
    """
    merged = {key: list(cases or ()) for key, cases in base.items()}
    for overlay in overlays:
        for key, cases in overlay.items():
            if key not in merged or not cases:
                continue
            if is_entire_file(cases):
                merged[key] = list(cases)
            else:
                merged[key] += [case for case in cases if case not in merged[key]]
    return merged


def fmt_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes // 60}h{minutes % 60:02d}m{seconds:02d}s"


def _signal_group(proc: subprocess.Popen, sig) -> None:
    if os.name == "nt" or sig is None:
        proc.kill()
        return
    try:
        # The whole group, so xdist workers are stopped too
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _run(cmd: list[str], env: dict, timeout: float | None) -> tuple[int, bool]:
    """Return (returncode, timed_out)."""
    proc = subprocess.Popen(cmd, env=env, start_new_session=os.name != "nt")
    previous = {}
    if os.name != "nt":
        # pytest runs in its own session, so pass on the signals that stop us
        # (step timeout, job cancel, Ctrl-C) instead of leaving it orphaned
        def forward(signum, _frame):
            _signal_group(proc, signum)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, forward)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] exceeded {timeout}s, stopping: {' '.join(cmd)}", flush=True)
        # SIGINT first so pytest still writes the junit xml of the finished cases
        _signal_group(proc, signal.SIGINT)
        try:
            return proc.wait(timeout=STOP_GRACE_SECONDS), True
        except subprocess.TimeoutExpired:
            _signal_group(proc, getattr(signal, "SIGKILL", None))
            return proc.wait(), True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _write_marker_xml(path: Path, module: str, message: str, tag: str) -> None:
    errors, failures = (1, 0) if tag == "error" else (0, 1)
    # Same shape as a pytest collection failure: empty classname, module as name
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites><testsuite name="pytest" errors="{errors}" '
        f'failures="{failures}" skipped="0" tests="1">'
        f'<testcase classname="" name={quoteattr(module)} time="0">'
        f"<{tag} message={quoteattr(message)}>{escape(message)}</{tag}>"
        "</testcase></testsuite></testsuites>\n",
        encoding="utf-8",
    )


def _collect(xml_file: Path, result: FileResult) -> None:
    try:
        root = ET.parse(xml_file).getroot()
    except (ET.ParseError, OSError) as exc:
        result.failures.append(
            CaseFailure("", xml_file.name, "error", f"Unreadable junit xml: {exc}")
        )
        return
    for case in root.iter("testcase"):
        if not case.get("name"):
            continue
        result.tests += 1
        outcome = next(
            (c for c in case if c.tag in ("failure", "error", "skipped")), None
        )
        if outcome is None:
            result.passed += 1
        elif outcome.tag == "skipped":
            result.skipped += 1
        else:
            lines = (outcome.get("message") or outcome.text or "").strip().splitlines()
            result.failures.append(
                CaseFailure(
                    case.get("classname", ""),
                    case.get("name"),
                    "failed" if outcome.tag == "failure" else "error",
                    lines[0][:300] if lines else "",
                )
            )


def run_test_file(
    test_file: str,
    include=(),
    exclude=(),
    xml_prefix: str = "op_ut",
    xml_dir: str | os.PathLike = ".",
    timeout: float | None = None,
    k: str | None = None,
) -> FileResult:
    """Run one test file with pytest, cases selected as in k_expression."""
    module = test_file.removesuffix(".py").replace("\\", "/").replace("/", ".")
    mode = "exe" if include or k else "skip" if exclude else "all"
    xml_dir = Path(xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    # The UT report derives the result category from the xml prefix
    xml_file = xml_dir / f"{xml_prefix}_with_{mode}.{module}.xml"
    marker_file = xml_dir / f"{xml_prefix}_with_{mode}.{module}.crash.xml"
    for stale in (xml_file, marker_file):
        stale.unlink(missing_ok=True)

    cmd = [sys.executable, "-m", "pytest", f"--junit-xml={xml_file}", test_file]
    if expr := k_expression(include, exclude, k):
        cmd += ["-k", expr]
    env = {**os.environ, "PYTORCH_TEST_WITH_SLOW": "1"}

    start = time.monotonic()
    returncode, timed_out = _run(cmd, env, timeout)
    result = FileResult(
        test_file,
        returncode,
        "passed" if returncode == 0 else "failed",
        time.monotonic() - start,
    )
    if xml_file.exists():
        result.xml_files.append(xml_file)
        _collect(xml_file, result)

    marker = None
    if timed_out:
        marker = (
            "timeout",
            "error",
            f"Timeout: {test_file} exceeded {timeout}s and was killed",
        )
    elif returncode < 0:
        marker = (
            "crash",
            "error",
            f"Crash: {test_file} killed by signal {-returncode}",
        )
    elif not xml_file.exists():
        marker = (
            "crash",
            "error",
            f"Crash: {test_file} exited with {returncode} without a junit xml",
        )
    elif returncode == NO_TESTS_COLLECTED:
        marker = (
            "no_tests",
            "failure",
            f"No tests: {test_file} collected no test cases",
        )
    elif returncode and not result.failures:
        # e.g. a pytest usage or internal error, which leaves no failed case in the xml
        marker = (
            "crash",
            "error",
            f"Error: {test_file} exited with {returncode} without a failed case",
        )

    if marker:
        result.status, tag, message = marker
        print(f"[{STATUS_LABEL[result.status]}] {message}", flush=True)
        _write_marker_xml(marker_file, module, message, tag)
        result.returncode = returncode if returncode > 0 else 1
        result.xml_files.append(marker_file)
        _collect(marker_file, result)
    return result


def print_report(results: list[FileResult], elapsed: float) -> None:
    failed_files = [r for r in results if r.returncode]
    cases = [(r, f) for r in results for f in r.failures]
    print("=" * 30 + " UT result " + "=" * 30)
    for r in results:
        print(
            f"{STATUS_LABEL[r.status]:<8} {r.test_file}: {r.tests} tests, "
            f"{r.passed} passed, {r.skipped} skipped, {len(r.failures)} failed "
            f"({fmt_duration(r.duration)})"
        )
    if cases:
        print(f"---- {len(cases)} failed cases ----")
        for r, f in cases:
            print(f"{f.status.upper()} {r.test_file}::{f.node} - {f.message}")
    print(
        f"[SUMMARY] {len(results)} files, {len(results) - len(failed_files)} passed, "
        f"{len(failed_files)} failed, {len(cases)} failed cases "
        f"in {fmt_duration(elapsed)}",
        flush=True,
    )


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_report(results: list[FileResult], elapsed: float, title: str) -> str:
    failed_files = [r for r in results if r.returncode]
    cases = [(r, f) for r in results for f in r.failures]
    totals = {
        key: sum(getattr(r, key) for r in results)
        for key in ("tests", "passed", "skipped")
    }
    lines = [
        f"### {title}",
        "",
        f"{len(results)} files, {len(results) - len(failed_files)} passed, "
        f"{len(failed_files)} failed in {fmt_duration(elapsed)} | "
        f"{totals['tests']} cases: {totals['passed']} passed, "
        f"{totals['skipped']} skipped, {len(cases)} failed",
    ]
    if failed_files:
        lines += [
            "",
            "| File | Result | Tests | Passed | Skipped | Failed | Time |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| `{r.test_file}` | {STATUS_LABEL[r.status]} | {r.tests} | {r.passed} "
            f"| {r.skipped} | {len(r.failures)} | {fmt_duration(r.duration)} |"
            for r in failed_files
        ]
    if cases:
        lines += [
            "",
            f"<details><summary>{len(cases)} failed cases</summary>",
            "",
            "| File | Case | Status | Message |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{r.test_file}` | `{_md(f.node)}` | {f.status} | {_md(f.message)} |"
            for r, f in cases[:MAX_SUMMARY_CASES]
        ]
        if len(cases) > MAX_SUMMARY_CASES:
            lines.append(
                f"\n... {len(cases) - MAX_SUMMARY_CASES} more in the junit xml"
            )
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


CLI_EPILOG = """\
examples:
  {prog}                          every file of the skip list
  {prog} --test-cases skipped     only their excluded cases
  {prog} test_ops_xpu.py          one file, its excluded cases skipped
  {prog} test_ops_xpu.py --test-cases all -i test_add,test_sub -i float16 -e bfloat16
                                  (test_add or test_sub) and float16, not bfloat16
  {prog} test_ops_xpu.py -i TestCommonXPU::test_compare -i @cases.txt -k "not slow"

case scope of each file (--test-cases):
  selected  every case except the excluded ones; fully excluded files are not run
  skipped   only the excluded cases; fully excluded files run whole
  all       every case

case patterns, on top of the scope (pytest -k substring match):
  -i A,B    one include group, a case must match A or B; repeated -i groups
            must all match, so -i A,B -i C means (A or B) and C
  -e A,B    exclude cases matching A or B; repeatable
  -k EXPR   a raw pytest -k expression, AND-ed with the rest
  Class::name matches cases whose names contain both; @FILE reads patterns
  from FILE, one per line or comma-separated, # starts a comment.

Every run file leaves a junit xml in --junit-dir; a crash, timeout or no
collected tests is a failed case in <prefix>_with_<mode>.<module>.crash.xml.
The failed cases are printed at the end and, with --summary, appended as
markdown (default: $GITHUB_STEP_SUMMARY).
"""


def patterns(value: str) -> list[str]:
    """Comma-separated case patterns, or those of @FILE."""
    if value.startswith("@"):
        try:
            with open(value[1:], encoding="utf-8") as f:
                lines = [line.split("#", 1)[0] for line in f]
        except OSError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        items = [item for line in lines for item in line.split(",")]
    else:
        items = value.split(",")
    found = [item.strip() for item in items if item.strip()]
    if not found:
        raise argparse.ArgumentTypeError(f"no case pattern in {value!r}")
    return found


def parse_args(description: str, xml_prefix: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.epilog = CLI_EPILOG.format(prog=parser.prog)
    parser.add_argument(
        "files",
        nargs="*",
        help="test files to run (default: every file of the skip list)",
    )
    parser.add_argument(
        "--test-cases",
        choices=["selected", "skipped", "all"],
        default="selected",
        help="case scope of each file (default: selected)",
    )
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        type=patterns,
        metavar="A[,B...]|@FILE",
        help="include group: run cases matching any pattern; groups are AND-ed",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="extend",
        default=[],
        type=patterns,
        metavar="A[,B...]|@FILE",
        help="do not run cases matching any pattern (repeatable)",
    )
    parser.add_argument(
        "-k", dest="k", metavar="EXPR", help="raw pytest -k expression, AND-ed"
    )
    parser.add_argument(
        "--file-timeout",
        type=int,
        default=36000,
        metavar="SECONDS",
        help="timeout of each test file (default: 36000, 10h)",
    )
    parser.add_argument(
        "--junit-dir", default=".", help="junit xml output directory (default: .)"
    )
    parser.add_argument(
        "--xml-prefix",
        default=xml_prefix,
        help=f"junit xml name prefix, it sets the UT report category (default: {xml_prefix})",
    )
    parser.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        metavar="FILE",
        help="append a markdown result summary (default: $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="exit 0 even if tests fail, leaving the verdict to the junit xml check",
    )
    return parser.parse_args(argv)


def plan(cases, args: argparse.Namespace):
    """(include groups, exclude) of one file, or None to not run it."""
    cases = cases or []
    include, exclude = [list(g) for g in args.include], list(args.exclude)
    if args.test_cases == "selected":
        if is_entire_file(cases):
            return None
        exclude = cases + exclude
    elif args.test_cases == "skipped":
        if not cases:
            return None
        if not is_entire_file(cases):
            # -i narrows the excluded cases instead of replacing them
            include = [cases, *include]
    return include, exclude


def run_suite(skips: dict, args: argparse.Namespace, skip_files=()) -> int:
    """Run the files of skips (or args.files) one by one, report, return the exit code."""
    files = args.files or [key for key in skips if key not in skip_files]
    print(f"Platform: {os.name}, scope: {args.test_cases}, files: {len(files)}")

    results = []
    start = time.monotonic()
    for i, test_file in enumerate(files, 1):
        tag = f"[{i}/{len(files)}]"
        cases = skips.get(test_file) or skips.get(
            os.path.normpath(test_file).replace("\\", "/")
        )
        selection = plan(cases, args)
        if selection is None:
            print(f"{tag} [SKIP] {test_file}: no case in scope", flush=True)
            continue
        include, exclude = selection
        if IN_GITHUB_ACTIONS:
            print(f"::group::{test_file}", flush=True)
        print(f"{tag} [RUN] {test_file}", flush=True)
        if include:
            print(f"  include: {' AND '.join(map(str, include))}")
        if exclude:
            print(f"  exclude: {exclude}")
        if args.k:
            print(f"  -k: {args.k}")
        result = run_test_file(
            test_file,
            include,
            exclude,
            xml_prefix=args.xml_prefix,
            xml_dir=args.junit_dir,
            timeout=args.file_timeout,
            k=args.k,
        )
        results.append(result)
        if IN_GITHUB_ACTIONS:
            print("::endgroup::", flush=True)
        print(f"{tag} [{STATUS_LABEL[result.status]}] {test_file}", flush=True)

    elapsed = time.monotonic() - start
    print_report(results, elapsed)
    if args.summary:
        title = f"{args.xml_prefix} ({args.test_cases})"
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(markdown_report(results, elapsed, title))
    failed = any(r.returncode for r in results)
    return 1 if failed and not args.exit_zero else 0


def main(
    skips: dict, description: str, xml_prefix: str = "op_ut", skip_files=()
) -> int:
    """Command line entry of the run_*.py launchers."""
    return run_suite(skips, parse_args(description, xml_prefix), skip_files)
