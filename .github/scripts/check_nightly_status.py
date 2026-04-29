#!/usr/bin/env python3
"""Check PyTorch nightly CI XPU test results.

Queries GitHub Actions API for recent XPU CI workflow runs.
Outputs a failure list if any tests failed, otherwise outputs ALL_PASS.

Supports filtering by trigger event type (schedule/push/pull_request),
and saves all collected run summaries for downstream bisect usage.
"""
import os
import sys
import json
import re
import argparse
import requests
from datetime import datetime, timedelta, timezone

# ============================================================================
# Global Configuration
# ============================================================================

# Read GitHub Token from environment variable.
# In GitHub Actions, this is auto-injected via secrets.GITHUB_TOKEN.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# GitHub API request headers: Bearer token auth + JSON format
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# Target repository: PyTorch main repo
PYTORCH_REPO = "pytorch/pytorch"
API_BASE = f"https://api.github.com/repos/{PYTORCH_REPO}"

# XPU CI workflow ID (fixed, found via GET /repos/pytorch/pytorch/actions/workflows
# where name="xpu", path=".github/workflows/xpu.yml")
XPU_WORKFLOW_ID = 79954307


# ============================================================================
# API Functions
# ============================================================================

def get_latest_xpu_runs(days=1, event=None):
    """Fetch XPU CI workflow runs from the last N days.

    Args:
        days: Number of days to look back (default: 1)
        event: Event type filter ('schedule', 'push', etc.)
               GitHub API supports one event value per request.

    Returns:
        list: Array of workflow_run objects containing id, head_sha,
              conclusion, event, created_at, etc.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"{API_BASE}/actions/workflows/{XPU_WORKFLOW_ID}/runs"
    params = {
        "per_page": 30,
        "status": "completed",
        "created": f">={since}",
    }
    if event:
        params["event"] = event

    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("workflow_runs", [])


def get_failed_jobs(run_id):
    """Get all failed jobs from a workflow run.

    A workflow run contains multiple jobs (e.g., different test shards).
    Only returns jobs with conclusion == "failure".

    Args:
        run_id: Workflow run ID

    Returns:
        list: List of failed job objects
    """
    url = f"{API_BASE}/actions/runs/{run_id}/jobs"
    params = {"per_page": 100, "filter": "latest"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    return [j for j in resp.json().get("jobs", []) if j["conclusion"] == "failure"]


def get_failed_test_cases(job_id):
    """Parse job logs to extract specific failed test case names.

    PyTorch CI logs contain consistently failing tests in this format:
      FAILED CONSISTENTLY: test/inductor/test_deterministic.py::DeterministicTest::test_mm_padding

    Uses regex to extract these test IDs.

    Args:
        job_id: Job ID

    Returns:
        list: Deduplicated list of failed test case IDs
    """
    url = f"{API_BASE}/actions/jobs/{job_id}/logs"
    try:
        resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=60)
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as e:
        print(f"    WARNING: Failed to download log for job {job_id}: {e.__class__.__name__}")
        return []
    if resp.status_code != 200:
        return []

    failed_tests = []
    for line in resp.text.split("\n"):
        m = re.search(r"FAILED CONSISTENTLY:\s+(.+)", line)
        if m:
            test_id = m.group(1).strip()
            if test_id not in failed_tests:
                failed_tests.append(test_id)
    return failed_tests


# ============================================================================
# Data Processing Functions
# ============================================================================

def parse_failure_info(job):
    """Extract key information from a failed job object.

    Returns:
        dict: Contains job_name, job_id, job_url, run_id, timestamps
    """
    return {
        "job_name": job["name"],
        "job_id": job["id"],
        "job_url": job["html_url"],
        "run_id": job["run_id"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }


def summarize_run(run):
    """Extract a compact summary from a workflow run object.

    Only keeps fields needed for bisect, avoids saving the full run object.
    """
    return {
        "run_id": run["id"],
        "event": run["event"],
        "head_branch": run["head_branch"],
        "head_sha": run["head_sha"],
        "conclusion": run["conclusion"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "html_url": run["html_url"],
    }


def process_runs(runs, parse_logs=False, label=""):
    """Process a set of workflow runs: collect failure info and run summaries.

    Args:
        runs: List of workflow run objects
        parse_logs: Whether to parse logs for specific test case names
                    (False = only know which job failed, not which test)
                    (True = get specific test case list, but slower)
        label: Prefix label for log output

    Returns:
        (all_failures, run_summaries):
          - all_failures: Detailed info for all failed jobs
          - run_summaries: Compact summaries for all runs
    """
    all_failures = []
    run_summaries = []

    for run in runs:
        run_summaries.append(summarize_run(run))
        commit_sha = run["head_sha"]
        event = run["event"]
        branch = run["head_branch"]
        print(f"\n  [{label}] Run: {run['name']} (commit: {commit_sha[:12]}, "
              f"event: {event}, branch: {branch}, conclusion: {run['conclusion']})")

        failed_jobs = get_failed_jobs(run["id"])
        if not failed_jobs:
            print("    All jobs passed [PASS]")
            continue

        for job in failed_jobs:
            info = parse_failure_info(job)
            info["commit_sha"] = commit_sha
            info["workflow_name"] = run["name"]
            info["event"] = event
            info["head_branch"] = branch

            if parse_logs:
                print(f"    Parsing log for: {info['job_name']}...")
                info["failed_tests"] = get_failed_test_cases(job["id"])
                for t in info["failed_tests"]:
                    print(f"      [FAIL] {t}")
            else:
                info["failed_tests"] = []

            all_failures.append(info)
            print(f"    FAIL: {info['job_name']}")
            print(f"      URL: {info['job_url']}")

    return all_failures, run_summaries


# ============================================================================
# Deep-scan: trace back to find first-fail run for EXISTING failures
# ============================================================================

def _test_file_from_id(test_id):
    """Extract test file path from a full test ID.

    e.g. 'test/inductor/test_flex_attention.py::TestClass::test_method'
         -> 'test/inductor/test_flex_attention.py'
    """
    return test_id.split("::")[0] if "::" in test_id else test_id


def deep_scan_existing_failures(existing_tests, already_checked_runs, event, days, max_lookback=14):
    """For EXISTING failures, look back through older runs to find the first-fail commit scope.

    Results are aggregated by test file. Each test file entry contains:
    - first_fail_sha / last_pass_sha for the file (earliest first_fail across all cases)
    - test_cases: list of individual test case IDs under this file

    Args:
        existing_tests: set of test IDs that are EXISTING (failed in both latest and previous runs)
        already_checked_runs: list of run objects already processed (to skip them)
        event: event type (e.g., 'schedule')
        days: initial lookback window in days
        max_lookback: max number of additional older runs to check

    Returns:
        dict: mapping test_file -> {
            "first_fail_sha": str, "first_fail_run_url": str,
            "last_pass_sha": str or None, "last_pass_run_url": str or None,
            "lookback_exhausted": bool,
            "test_cases": [str, ...]
        }
    """
    if not existing_tests:
        return {}

    remaining = set(existing_tests)
    # Initialize first_fail to the oldest already-checked failing run
    # so every result has a concrete first_fail even if resolved on the first deep-scan run
    oldest_checked = already_checked_runs[-1] if already_checked_runs else None
    init_ff_sha = oldest_checked["head_sha"] if oldest_checked else None
    init_ff_url = oldest_checked.get("html_url") if oldest_checked else None

    # Track per-test-case results first, then aggregate by file
    per_case = {t: {"first_fail_sha": init_ff_sha, "first_fail_run_url": init_ff_url,
                     "last_pass_sha": None, "last_pass_run_url": None,
                     "lookback_exhausted": False} for t in remaining}
    checked_ids = {r["id"] for r in already_checked_runs}

    # Fetch more runs with a wider time window
    wider_days = max(days * 2, 14)
    print(f"\n=== Deep-scan: checking up to {max_lookback} older [{event}] runs "
          f"(window: {wider_days} days) for {len(remaining)} EXISTING test(s) ===")

    all_runs = get_latest_xpu_runs(wider_days, event=event)
    older_runs = [r for r in all_runs
                  if r["id"] not in checked_ids and r["conclusion"] != "cancelled"]

    scanned = 0
    for run in older_runs:
        if not remaining or scanned >= max_lookback:
            break
        scanned += 1
        sha = run["head_sha"]
        print(f"  Deep-scan run {scanned}/{max_lookback}: "
              f"{run['created_at']}  {sha[:12]}  {run['conclusion']}")

        failed_jobs = get_failed_jobs(run["id"])
        run_failed_tests = set()
        for job in failed_jobs:
            for t in get_failed_test_cases(job["id"]):
                run_failed_tests.add(t)

        resolved = set()
        for test_id in remaining:
            if test_id not in run_failed_tests:
                # This run did NOT have the failure -> last pass found
                per_case[test_id]["last_pass_sha"] = sha
                per_case[test_id]["last_pass_run_url"] = run["html_url"]
                resolved.add(test_id)
            else:
                # Still failing in this older run -> update first_fail
                per_case[test_id]["first_fail_sha"] = sha
                per_case[test_id]["first_fail_run_url"] = run["html_url"]

        remaining -= resolved

    for test_id in remaining:
        per_case[test_id]["lookback_exhausted"] = True

    # Aggregate by test file
    from collections import defaultdict
    file_groups = defaultdict(list)
    for test_id in existing_tests:
        file_groups[_test_file_from_id(test_id)].append(test_id)

    result = {}
    # Build run_order: sha -> index (0=newest, higher=older) for deterministic comparison
    run_order = {}
    for idx, run in enumerate(older_runs):
        run_order[run["head_sha"]] = idx

    for test_file, cases in sorted(file_groups.items()):
        first_fail_sha = None
        first_fail_url = None
        first_fail_order = -1
        last_pass_sha = None
        last_pass_url = None
        last_pass_order = float("inf")
        exhausted = False
        for case in cases:
            c = per_case[case]
            # Pick the oldest (highest order index) first_fail among cases
            if c["first_fail_sha"]:
                order = run_order.get(c["first_fail_sha"], -1)
                if order > first_fail_order:
                    first_fail_sha = c["first_fail_sha"]
                    first_fail_url = c["first_fail_run_url"]
                    first_fail_order = order
            # Pick the newest (lowest order index) last_pass among cases
            if c["last_pass_sha"]:
                order = run_order.get(c["last_pass_sha"], float("inf"))
                if order < last_pass_order:
                    last_pass_sha = c["last_pass_sha"]
                    last_pass_url = c["last_pass_run_url"]
                    last_pass_order = order
            if c["lookback_exhausted"]:
                exhausted = True

        result[test_file] = {
            "first_fail_sha": first_fail_sha,
            "first_fail_run_url": first_fail_url,
            "last_pass_sha": last_pass_sha,
            "last_pass_run_url": last_pass_url,
            "lookback_exhausted": exhausted,
            "test_cases": sorted(cases),
        }
        status = "EXHAUSTED" if exhausted else "FOUND"
        print(f"    [{status}] {test_file} ({len(cases)} cases): "
              f"first_fail={first_fail_sha[:12] if first_fail_sha else 'N/A'}, "
              f"last_pass={last_pass_sha[:12] if last_pass_sha else 'N/A'}")

    print(f"  Deep-scan complete: {len(result)} test files, "
          f"{sum(1 for v in result.values() if not v['lookback_exhausted'])} resolved, "
          f"{sum(1 for v in result.values() if v['lookback_exhausted'])} exhausted")
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Check PyTorch XPU nightly CI status")

    parser.add_argument("--days", type=int, default=1, help="Look back N days")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--event", type=str, default="schedule",
                        help="Primary event type to check (default: schedule). "
                             "Push runs are always collected as bisect reference.")
    parser.add_argument("--num-runs", type=int, default=2,
                        help="Number of recent non-cancelled schedule runs to compare (default: 2). "
                             "Compares latest vs previous to identify NEW failures.")
    parser.add_argument("--parse-logs", action="store_true", default=False,
                        help="Parse job logs to extract specific test case names")
    parser.add_argument("--save-all-runs", type=str, default=None,
                        help="Save all collected run summaries (schedule + push) to this JSON file "
                             "for downstream bisect usage")
    parser.add_argument("--deep-scan", action="store_true", default=False,
                        help="For EXISTING failures, look back further to find the first "
                             "failing run and narrow the commit scope")
    parser.add_argument("--max-lookback", type=int, default=14,
                        help="Max number of older runs to check during deep-scan (default: 14)")

    args = parser.parse_args()

    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Fetch primary event runs (default: schedule)
    print(f"=== Checking XPU CI [{args.event}] runs from last {args.days} day(s) ===")
    all_primary_runs = get_latest_xpu_runs(args.days, event=args.event)
    print(f"Found {len(all_primary_runs)} [{args.event}] runs")

    # Filter out cancelled runs, take the most recent N
    valid_runs = [r for r in all_primary_runs if r["conclusion"] != "cancelled"]
    selected_runs = valid_runs[:args.num_runs]
    if not selected_runs:
        print("ERROR: No non-cancelled runs found in the time window.", file=sys.stderr)
        result = {"status": "NO_RUNS_FOUND", "event": args.event}
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        sys.exit(0)

    print(f"Selected {len(selected_runs)} non-cancelled run(s):")
    for r in selected_runs:
        print(f"  {r['created_at']}  {r['head_sha'][:12]}  {r['conclusion']}")

    # Process the latest run (full log parsing)
    latest_run = selected_runs[:1]
    primary_failures, primary_summaries = process_runs(
        latest_run, parse_logs=args.parse_logs, label=f"{args.event}-latest")

    # Process previous run(s) for comparison (to identify NEW failures)
    prev_runs = selected_runs[1:]
    prev_failures = []
    prev_summaries = []
    if prev_runs:
        print(f"\n=== Previous [{args.event}] run(s) for comparison ===")
        prev_failures, prev_summaries = process_runs(
            prev_runs, parse_logs=args.parse_logs, label=f"{args.event}-prev")
        primary_summaries.extend(prev_summaries)

    # Phase 2: Collect push runs as bisect reference
    push_summaries = []
    if args.event != "push":
        print("\n=== Collecting [push] runs as bisect reference ===")
        push_runs = get_latest_xpu_runs(args.days, event="push")
        print(f"Found {len(push_runs)} [push] runs")
        # Push runs: no log parsing needed, only PASS/FAIL status for commit scope
        _, push_summaries = process_runs(push_runs, parse_logs=False, label="push")

    # Phase 3: Compare latest vs previous, classify NEW / EXISTING / FIXED
    latest_tests = set()
    for f in primary_failures:
        for t in f.get("failed_tests", []):
            latest_tests.add(t)

    prev_tests = set()
    for f in prev_failures:
        for t in f.get("failed_tests", []):
            prev_tests.add(t)

    new_tests = sorted(latest_tests - prev_tests)
    existing_tests = sorted(latest_tests & prev_tests)
    fixed_tests = sorted(prev_tests - latest_tests)

    # Phase 3.5: Deep-scan EXISTING failures to find first-fail commit
    existing_scope = {}
    if args.deep_scan and existing_tests:
        existing_scope = deep_scan_existing_failures(
            existing_tests,
            already_checked_runs=selected_runs,
            event=args.event,
            days=args.days,
            max_lookback=args.max_lookback,
        )

    # Phase 4: Build output JSON
    if not primary_failures:
        print(f"\nALL_PASS (for [{args.event}] runs)")
        result = {
            "status": "ALL_PASS",
            "event": args.event,
            "failures": [],
            "unique_failed_tests": [],
            "new_failed_tests": [],
            "existing_failed_tests": [],
            "fixed_tests": [],
            "existing_failure_scope": {},
        }
    else:
        print(f"\n{'='*60}")
        print(f"RESULTS for [{args.event}] runs")
        print(f"{'='*60}")
        print(f"Total failed tests: {len(latest_tests)}")
        if prev_tests:
            print(f"  [NEW]      {len(new_tests)} (not in previous run)")
            print(f"  [EXISTING] {len(existing_tests)} (also failed in previous run)")
            print(f"  [FIXED]    {len(fixed_tests)} (failed before, now passing)")
            if new_tests:
                print("\n--- NEW Failures ---")
                for t in new_tests:
                    print(f"  [NEW]  {t}")
            if existing_tests:
                print("\n--- Existing Failures ---")
                for t in existing_tests:
                    print(f"  [OLD]  {t}")
            if fixed_tests:
                print("\n--- Fixed (was failing, now passing) ---")
                for t in fixed_tests:
                    print(f"  [FIX]  {t}")
        else:
            print("  (no previous run to compare)")
            for t in sorted(latest_tests):
                print(f"  [FAIL] {t}")

        prev_commit = prev_failures[0]["commit_sha"] if prev_failures else None
        result = {
            "status": "HAS_FAILURES",
            "event": args.event,
            "commit_sha": primary_failures[0]["commit_sha"],
            "prev_commit_sha": prev_commit,
            "failures": primary_failures,
            "unique_failed_tests": sorted(latest_tests),
            "new_failed_tests": new_tests,
            "existing_failed_tests": existing_tests,
            "fixed_tests": fixed_tests,
            "existing_failure_scope": existing_scope,
        }

    # Save primary results to JSON file
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nPrimary results saved to {args.output}")

    # Phase 5: Save all run summaries (for bisect and commit scope calculation)
    if args.save_all_runs:
        all_runs_data = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "schedule_runs": primary_summaries if args.event == "schedule" else [],
            "push_runs": push_summaries if args.event != "push" else primary_summaries,
        }
        with open(args.save_all_runs, "w") as f:
            json.dump(all_runs_data, f, indent=2)
        print(f"All run summaries saved to {args.save_all_runs}")

    # Phase 6: Set GitHub Actions output variables
    # Downstream steps can read via ${{ steps.check.outputs.has_failures }}
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"has_failures={'true' if primary_failures else 'false'}\n")
            if primary_failures:
                f.write(f"failure_count={len(primary_failures)}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
