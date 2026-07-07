#!/usr/bin/env python3
# Copyright 2024-2025 Intel Corporation
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
    """Download UT-related artifacts from the workflow run."""
    artifacts_json = run(f"gh run view {run_id} --repo {repo} --json jobs")

    # Download all matching artifacts
    run(f"mkdir -p {download_dir}", check=False)

    # Try downloading new failures artifact
    result = subprocess.run(
        f"gh run download {run_id} --repo {repo} --dir {download_dir} "
        f'--pattern "New-UT-Failures-*"',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    has_new_failures_artifact = result.returncode == 0

    # Download UT data artifacts (for passed logs)
    subprocess.run(
        f"gh run download {run_id} --repo {repo} --dir {download_dir} "
        f'--pattern "Inductor-XPU-UT-Data-*"',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )

    return has_new_failures_artifact


def parse_new_failures(download_dir):
    """Parse new_ut_failure_list.csv files from downloaded artifacts."""
    failures = []
    for csv_file in Path(download_dir).rglob("new_ut_failure_list.csv"):
        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Category"):
                    continue
                # CSV format: Category | Class name | Test name | Status | Message | Source
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
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
    """Parse passed_*.log files to get set of passed test names."""
    passed = set()
    for log_file in Path(download_dir).rglob("passed_*.log"):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: category,class_name,test_name
                    parts = line.split(",")
                    if len(parts) >= 3:
                        passed.add(f"{parts[1]}::{parts[2]}")
    return passed


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

    # Match added lines that look like test method definitions
    current_file = None
    current_class = None
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            current_class = None
        elif current_file in test_files:
            # Track class context
            class_match = re.match(r"[+ ]class\s+(\w+)", line)
            if class_match:
                current_class = class_match.group(1)
            # Match added test methods
            if line.startswith("+") and not line.startswith("+++"):
                test_match = re.match(r"\+\s+def\s+(test_\w+)", line)
                if test_match:
                    test_name = test_match.group(1)
                    if current_class:
                        new_tests.append(f"{current_class}::{test_name}")
                    else:
                        new_tests.append(test_name)

    return new_tests


def build_deterministic_report(failures, changed_files, new_tests, passed_tests):
    """Build a report without LLM -- deterministic fallback."""
    lines = ["## UT Result Check\n"]

    # New failures section
    lines.append("### New Failures")
    if not failures:
        lines.append("No new failures detected.\n")
    else:
        lines.append(
            f"{len(failures)} new failure(s) detected (not in known issues).\n"
        )
        lines.append("| Test | Category | Status |")
        lines.append("|------|----------|--------|")
        for f in failures[:20]:
            lines.append(
                f"| `{f['class']}::{f['test']}` | {f['category']} | {f['status']} |"
            )
        if len(failures) > 20:
            lines.append(f"\n... and {len(failures) - 20} more\n")

    # Changed files summary
    lines.append("\n### PR Changes")
    if changed_files["operator_source"]:
        lines.append(
            f"- Operator source: {len(changed_files['operator_source'])} file(s)"
        )
    if changed_files["test_files"]:
        lines.append(f"- Test files: {len(changed_files['test_files'])} file(s)")
    if changed_files["skip_lists"]:
        lines.append(f"- Skip lists: {len(changed_files['skip_lists'])} file(s)")
    lines.append("")

    # New test coverage
    if new_tests:
        lines.append("### New Test Coverage")
        lines.append(
            "PR adds/modifies tests in: "
            + ", ".join(f"`{f}`" for f in changed_files["test_files"])
        )
        lines.append("")
        lines.append("| New/Modified Test | Found in Results? | Status |")
        lines.append("|-------------------|-------------------|--------|")
        for t in new_tests:
            if t in passed_tests:
                lines.append(f"| `{t}` | Yes | PASSED |")
            else:
                # Check if it failed
                failed_match = any(f"{f['class']}::{f['test']}" == t for f in failures)
                if failed_match:
                    lines.append(f"| `{t}` | Yes | FAILED |")
                else:
                    lines.append(f"| `{t}` | No | NOT RUN |")
        lines.append("")

    # Summary
    lines.append("### Summary")
    if not failures and all(t in passed_tests for t in new_tests):
        lines.append("All new tests passed and no new failures detected.")
    elif failures:
        lines.append(
            f"{len(failures)} new failure(s) detected. "
            "Please investigate before merging."
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

    # Download artifacts
    download_dir = "/tmp/ut_artifacts"
    run(f"rm -rf {download_dir}", check=False)
    has_new_failures = download_artifacts(repo, run_id, download_dir)

    # Parse results
    failures = parse_new_failures(download_dir) if has_new_failures else []
    passed_tests = parse_passed_tests(download_dir)

    # Get PR changes
    changed_files = get_pr_changed_files(repo, pr_number)
    new_tests = extract_new_test_names(repo, pr_number, changed_files["test_files"])

    passed_new = [t for t in new_tests if t in passed_tests]
    not_run = [
        t
        for t in new_tests
        if t not in passed_tests
        and not any(f"{f['class']}::{f['test']}" == t for f in failures)
    ]

    return {
        "pr_number": pr_number,
        "run_id": run_id,
        "failures": failures,
        "changed_files": changed_files,
        "new_tests": new_tests,
        "new_tests_passed": passed_new,
        "new_tests_not_run": not_run,
        "passed_tests_count": len(passed_tests),
    }


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

    data = collect_data(args.repo, args.pr_number, args.run_id)
    if data is None:
        return

    if args.output:
        # Write JSON for downstream AI analysis
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"UT data written to {args.output}")
        # Signal to the workflow that data is available
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as f:
                f.write("has_data=true\n")
        # Warn if there are new failures
        if data["failures"]:
            print(f"::warning::{len(data['failures'])} new UT failure(s) detected")
        return

    if args.deterministic:
        # Reconstruct passed_tests set for deterministic report
        passed_tests = set(data["new_tests_passed"])
        report = build_deterministic_report(
            data["failures"], data["changed_files"], data["new_tests"], passed_tests
        )
    else:
        # Default: deterministic report (no LLM fallback without --output)
        passed_tests = set(data["new_tests_passed"])
        report = build_deterministic_report(
            data["failures"], data["changed_files"], data["new_tests"], passed_tests
        )

    # Post report
    with open("/tmp/ut_check_body.md", "w") as f:
        f.write(report)
    run(
        f"gh pr comment {args.pr_number} --repo {args.repo} "
        f"--body-file /tmp/ut_check_body.md"
    )
    print(f"UT check report posted to PR #{args.pr_number}")

    if data["failures"]:
        print(f"::warning::{len(data['failures'])} new UT failure(s) detected")


if __name__ == "__main__":
    main()
