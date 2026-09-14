#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Analyze UT results from CI: new failures, relevance to PR, new-test coverage.

Modes:
    --output FILE   Collect data and write JSON for downstream AI analysis.
    --deterministic Collect data and post a deterministic (no-LLM) report.

Usage:
    # Collect data for AI analysis (used by claude-code-action)
    python bot_ut_check.py --pr-number 123 --repo owner/repo --output /tmp/ut_data.json

    # Post deterministic fallback report
    python bot_ut_check.py --pr-number 123 --repo owner/repo --deterministic
"""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


def run(cmd, check=True):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def find_latest_run(repo, pr_number):
    """Find the latest 'pull' workflow run for this PR."""
    pr_json = run(f"gh pr view {pr_number} --repo {repo} --json headRefOid")
    head_sha = json.loads(pr_json)["headRefOid"]

    runs_json = run(
        f"gh run list --repo {repo} --workflow pull.yml "
        f"--commit {head_sha} --limit 1 --json databaseId,status"
    )
    runs = json.loads(runs_json)
    if not runs:
        return None, None
    return runs[0]["databaseId"], runs[0]["status"]


def download_artifacts(repo, run_id, download_dir):
    """Download UT-related artifacts from the workflow run.

    Returns (has_new_failures_artifact, new_failures_dir) where new_failures_dir
    is the directory that holds the authoritative New-UT-Failures artifact. We
    track it explicitly because the Inductor-XPU-UT-Data artifact also bundles a
    copy of new_ut_failure_list.csv; parsing both would double-count failures.
    """
    run(f"mkdir -p {download_dir}", check=False)

    new_failures_dir = os.path.join(download_dir, "_new_failures")
    result = subprocess.run(
        f"gh run download {run_id} --repo {repo} --dir {new_failures_dir} "
        f'--pattern "New-UT-Failures-*"',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    has_new_failures_artifact = result.returncode == 0 and any(
        Path(new_failures_dir).rglob("new_ut_failure_list.csv")
    )

    # Download UT data artifacts (for passed/skipped logs and category totals).
    subprocess.run(
        f"gh run download {run_id} --repo {repo} --dir {download_dir} "
        f'--pattern "Inductor-XPU-UT-Data-*"',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )

    return has_new_failures_artifact, new_failures_dir


def parse_new_failures(new_failures_dir):
    """Parse new_ut_failure_list.csv from the New-UT-Failures-* artifact.

    This artifact is the authoritative source for new failures. It is produced
    by the CI summary job which filters all failures against the known issues
    list. The file is a pipe-delimited markdown table:
        | Category | Class name | Test name | Status | Message | Source |

    Only the New-UT-Failures directory is scanned (not the whole download dir):
    the Inductor-XPU-UT-Data artifact bundles a duplicate copy of this file, and
    including it would count every failure twice. Results are de-duplicated by
    (category, class, test) as a further safety net against re-run attempts.
    """
    failures = []
    seen = set()
    for csv_file in Path(new_failures_dir).rglob("new_ut_failure_list.csv"):
        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "---" in line and "|" in line:
                    continue
                if not line.startswith("|"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]
                if len(parts) < 4:
                    continue
                if parts[0] == "Category":
                    continue
                key = (parts[0], parts[1], parts[2])
                if key in seen:
                    continue
                seen.add(key)
                failures.append(
                    {
                        "category": parts[0],
                        "class": parts[1],
                        "test": parts[2],
                        "status": parts[3],
                        "message": parts[4] if len(parts) > 4 else "",
                    }
                )
    return failures


def parse_passed_tests(download_dir):
    """Parse passed_*.log files to a set of 'full.class.path::test_name'."""
    passed = set()
    for log_file in Path(download_dir).rglob("passed_*.log"):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: category,class_name,test_name
                parts = line.split(",")
                if len(parts) >= 3:
                    passed.add(f"{parts[1]}::{parts[2]}")
    return passed


def parse_skipped_tests(download_dir):
    """Parse JUnit XML for skipped testcases -> set of 'classname::name'."""
    skipped = set()
    pattern = re.compile(
        r'<testcase\s+classname="([^"]+)"\s+name="([^"]+)"[^>]*>\s*<skipped'
    )
    for xml_file in Path(download_dir).rglob("*.xml"):
        try:
            text = xml_file.read_text(errors="ignore")
        except OSError:
            continue
        for classname, name in pattern.findall(text):
            skipped.add(f"{classname}::{name}")
    return skipped


def parse_totals(download_dir):
    """Parse category_*.log aggregate counts across all UT categories."""
    totals = {
        "test_cases": 0,
        "passed": 0,
        "skipped": 0,
        "failures": 0,
        "errors": 0,
    }
    key_map = {
        "Test cases": "test_cases",
        "Passed": "passed",
        "Skipped": "skipped",
        "Failures": "failures",
        "Errors": "errors",
    }
    found = False
    for log_file in Path(download_dir).rglob("category_*.log"):
        found = True
        for line in log_file.read_text(errors="ignore").splitlines():
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            label = label.strip()
            value = value.strip()
            if label in key_map and value.isdigit():
                totals[key_map[label]] += int(value)
    return totals if found else None


def get_pr_changed_files(repo, pr_number):
    """Get list of changed files in the PR, classified by category."""
    files_json = run(f"gh pr view {pr_number} --repo {repo} --json files")
    files = json.loads(files_json).get("files", [])

    classified = {
        "operator_source": [],
        "test_files": [],
        "skip_lists": [],
        "other": [],
    }
    for f in files:
        path = f["path"]
        if re.match(r"src/ATen/native/xpu/", path):
            classified["operator_source"].append(path)
        elif re.match(r"test/(xpu|regressions)/", path):
            if "skip_list" in path:
                classified["skip_lists"].append(path)
            else:
                classified["test_files"].append(path)
        else:
            classified["other"].append(path)

    return classified


def extract_new_test_names(repo, pr_number, test_files):
    """Extract new/modified test method names from the PR diff for test files."""
    if not test_files:
        return []

    diff = run(f"gh pr diff {pr_number} --repo {repo}")
    new_tests = []

    current_file = None
    current_class = None
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            current_class = None
        elif current_file in test_files:
            class_match = re.match(r"[+ ]class\s+(\w+)", line)
            if class_match:
                current_class = class_match.group(1)
            if line.startswith("+") and not line.startswith("+++"):
                test_match = re.match(r"\+\s+def\s+(test_\w+)", line)
                if test_match:
                    test_name = test_match.group(1)
                    if current_class:
                        new_tests.append(f"{current_class}::{test_name}")
                    else:
                        new_tests.append(test_name)

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for t in new_tests:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _module_of(class_path):
    """Extract the test module name from a full class path.

    'third_party...test.xpu.test_modules_xpu.TestModuleXPU' -> 'test_modules_xpu'
    """
    parts = class_path.split(".")
    for part in reversed(parts):
        if part.startswith("test_"):
            return part
    return parts[-2] if len(parts) >= 2 else class_path


def _changed_test_modules(changed_files):
    modules = set()
    for path in changed_files["test_files"]:
        stem = Path(path).stem  # e.g. test_foo_xpu
        modules.add(stem)
    return modules


def _op_stems(changed_files):
    """Derive operator name stems from changed operator source paths."""
    stems = set()
    for path in changed_files["operator_source"]:
        name = Path(path).stem  # Foo, FooKernels, FooKernel
        name = re.sub(r"Kernels?$", "", name)
        if name:
            stems.add(name.lower())
    return stems


def classify_relevance(failure, changed_test_modules, op_stems):
    """Deterministically classify a failure's relation to the PR changes.

    Returns 'Related', 'Possibly related', or 'Unrelated'. This is a heuristic
    baseline; the LLM analysis may refine it.
    """
    module = _module_of(failure["class"]).lower()
    if module in {m.lower() for m in changed_test_modules}:
        return "Related"
    for stem in op_stems:
        # Match op stem as a whole token inside the module name, e.g.
        # op 'conv' -> 'test_conv_xpu'. Guard against tiny stems.
        if len(stem) >= 3 and re.search(rf"(^|_){re.escape(stem)}(_|$)", module):
            return "Possibly related"
    return "Unrelated"


def _test_matches(new_test, result_keys):
    """Match a short 'Class::test' new test against full-path result keys.

    passed/skipped keys use the full class path
    ('a.b.TestFoo::test_bar'), while diff-derived new tests only carry the
    short class name ('TestFoo::test_bar'). Match by test name plus a class
    suffix so the two representations line up.
    """
    if "::" in new_test:
        ncls, ntest = new_test.split("::", 1)
    else:
        ncls, ntest = None, new_test
    for key in result_keys:
        fcls, _, ftest = key.partition("::")
        if ftest != ntest:
            continue
        if ncls is None or fcls == ncls or fcls.endswith("." + ncls):
            return True
    return False


def summarize_new_tests(new_tests, passed, skipped, failures):
    """Classify each new test as PASSED / FAILED / SKIPPED / NOT RUN."""
    failed_keys = {f"{f['class']}::{f['test']}" for f in failures}
    result = {"passed": [], "failed": [], "skipped": [], "not_run": []}
    for t in new_tests:
        if _test_matches(t, failed_keys):
            result["failed"].append(t)
        elif _test_matches(t, passed):
            result["passed"].append(t)
        elif _test_matches(t, skipped):
            result["skipped"].append(t)
        else:
            result["not_run"].append(t)
    return result


def compute_verdict(failures, new_summary):
    """Return (verdict, reason) for the safe-to-merge recommendation."""
    related = [f for f in failures if f.get("relevance") == "Related"]
    possibly = [f for f in failures if f.get("relevance") == "Possibly related"]

    if new_summary["failed"]:
        return (
            "Not safe to merge",
            f"{len(new_summary['failed'])} newly added test(s) failed.",
        )
    if related:
        return (
            "Not safe to merge",
            f"{len(related)} new failure(s) are in test modules this PR changes.",
        )
    if possibly:
        return (
            "Investigate before merging",
            f"{len(possibly)} new failure(s) may be related to this PR.",
        )
    if failures:
        return (
            "Likely safe to merge",
            f"{len(failures)} new failure(s), all unrelated to this PR's changes.",
        )
    if new_summary["not_run"] or new_summary["skipped"]:
        return (
            "Investigate before merging",
            "No new failures, but some newly added tests did not run.",
        )
    return ("Safe to merge", "No new failures detected.")


def _truncate_note(total, shown, noun):
    if total > shown:
        return (
            f"\n... and {total - shown} more {noun}. See CI logs for the full list.\n"
        )
    return ""


def build_report(data):
    """Build the unified markdown report shared by both output paths."""
    failures = data["failures"]
    new_summary = data["new_tests_summary"]
    new_tests = data["new_tests"]
    totals = data.get("totals")
    lines = [f"## UT Result Check: PR #{data['pr_number']}\n"]

    # New failures
    lines.append("### New Failures")
    if not failures:
        lines.append("No new failures detected.\n")
    else:
        lines.append(
            f"{len(failures)} new failure(s) detected (not in known issues).\n"
        )
        lines.append("| Test | Category | Status | Related to PR? |")
        lines.append("|------|----------|--------|----------------|")
        for f in failures[:20]:
            lines.append(
                f"| `{f['class']}::{f['test']}` | {f['category']} "
                f"| {f['status']} | {f.get('relevance', 'Unrelated')} |"
            )
        note = _truncate_note(len(failures), 20, "failure(s)")
        if note:
            lines.append(note)
        lines.append("")

    # New test coverage
    if new_tests:
        lines.append("### New Test Coverage")
        lines.append(
            f"This PR adds/modifies **{len(new_tests)}** test(s): "
            f"{len(new_summary['passed'])} passed, "
            f"{len(new_summary['failed'])} failed, "
            f"{len(new_summary['skipped'])} skipped, "
            f"{len(new_summary['not_run'])} not run.\n"
        )
        lines.append("| New/Modified Test | Status |")
        lines.append("|-------------------|--------|")
        status_of = {}
        for t in new_summary["failed"]:
            status_of[t] = "FAILED"
        for t in new_summary["passed"]:
            status_of[t] = "PASSED"
        for t in new_summary["skipped"]:
            status_of[t] = "SKIPPED"
        for t in new_summary["not_run"]:
            status_of[t] = "NOT RUN"
        for t in new_tests[:20]:
            lines.append(f"| `{t}` | {status_of.get(t, 'NOT RUN')} |")
        note = _truncate_note(len(new_tests), 20, "new test(s)")
        if note:
            lines.append(note)
        lines.append("")

    # PR changes context
    cf = data["changed_files"]
    change_bits = []
    for key, label in [
        ("operator_source", "operator source"),
        ("test_files", "test"),
        ("skip_lists", "skip list"),
    ]:
        if cf[key]:
            change_bits.append(f"{len(cf[key])} {label} file(s)")
    if change_bits:
        lines.append("### PR Changes")
        lines.append(", ".join(change_bits) + ".\n")

    # Verdict
    verdict, reason = data["verdict"], data["verdict_reason"]
    lines.append("### Recommendation")
    lines.append(f"**{verdict}.** {reason}")
    if totals:
        lines.append(
            f"\nRun totals: {totals['test_cases']} cases, {totals['passed']} passed, "
            f"{totals['skipped']} skipped, {totals['failures']} failures, "
            f"{totals['errors']} errors."
        )
    lines.append("")
    return "\n".join(lines)


def collect_data(repo, pr_number, run_id_arg):
    """Collect all UT data and return structured dict (or None if not ready)."""
    if run_id_arg:
        run_id = run_id_arg
        status = "completed"
    else:
        run_id, status = find_latest_run(repo, pr_number)
        if not run_id:
            run(
                f"gh pr comment {pr_number} --repo {repo} "
                f'--body "No CI workflow run found for this PR. '
                f'Please wait for CI to complete and try again."'
            )
            return None

    if status != "completed":
        run(
            f"gh pr comment {pr_number} --repo {repo} "
            f'--body "CI is still running (status: {status}). '
            f'Please wait for CI to complete and try again."'
        )
        return None

    download_dir = "/tmp/ut_artifacts"
    run(f"rm -rf {download_dir}", check=False)
    has_new_failures, new_failures_dir = download_artifacts(repo, run_id, download_dir)

    failures = parse_new_failures(new_failures_dir) if has_new_failures else []
    passed_tests = parse_passed_tests(download_dir)
    totals = parse_totals(download_dir)

    changed_files = get_pr_changed_files(repo, pr_number)
    new_tests = extract_new_test_names(repo, pr_number, changed_files["test_files"])
    skipped_tests = parse_skipped_tests(download_dir) if new_tests else set()

    changed_test_modules = _changed_test_modules(changed_files)
    op_stems = _op_stems(changed_files)
    for f in failures:
        f["relevance"] = classify_relevance(f, changed_test_modules, op_stems)

    new_summary = summarize_new_tests(new_tests, passed_tests, skipped_tests, failures)
    verdict, reason = compute_verdict(failures, new_summary)

    return {
        "pr_number": pr_number,
        "run_id": run_id,
        "failures": failures,
        "changed_files": changed_files,
        "new_tests": new_tests,
        "new_tests_summary": new_summary,
        "passed_tests_count": len(passed_tests),
        "totals": totals,
        "verdict": verdict,
        "verdict_reason": reason,
    }


def post_comment(repo, pr_number, body):
    with open("/tmp/ut_check_body.md", "w") as f:
        f.write(body)
    run(f"gh pr comment {pr_number} --repo {repo} --body-file /tmp/ut_check_body.md")


def main():
    parser = argparse.ArgumentParser(description="UT result analysis")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument(
        "--run-id", type=int, default=0, help="Workflow run ID (auto-detected if 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Write collected data as JSON to this file (for AI analysis)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Post a deterministic (no-LLM) report as a PR comment",
    )
    args = parser.parse_args()

    try:
        data = collect_data(args.repo, args.pr_number, args.run_id)
    except Exception:  # noqa: BLE001 - surface a clear diagnostic, never a bad report
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # Do NOT post a misleading report. Fail the step so the workflow's
        # error path can notify a maintainer instead of leaving a low-quality
        # comment on the PR.
        print("::error::UT check could not complete; see logs for details")
        sys.exit(1)

    if data is None:
        # A diagnostic comment (CI not ready) was already posted by collect_data.
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"UT data written to {args.output}")
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as f:
                f.write("has_data=true\n")
        if data["failures"]:
            print(f"::warning::{len(data['failures'])} new UT failure(s) detected")
        return

    # Deterministic report (default and --deterministic behave the same).
    report = build_report(data)
    post_comment(args.repo, args.pr_number, report)
    print(f"UT check report posted to PR #{args.pr_number}")
    if data["failures"]:
        print(f"::warning::{len(data['failures'])} new UT failure(s) detected")


if __name__ == "__main__":
    main()
