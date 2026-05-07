#!/usr/bin/env python3
"""TEMPORARY: Polling loop that discovers and advances issues.
DELETE when migrating to GitHub Actions webhooks.
"""
from __future__ import annotations

import sys
import os
import time
import argparse

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.issue_fixing_agent import advance
from pytorch_agent.utils import git as gh
from pytorch_agent.utils.config import ISSUE_REPO, POLL_INTERVAL, ISSUE_LABEL, TERMINAL_STAGES, MAX_ADVANCE_LOOPS
from pytorch_agent.utils.body_templates import get_status
from pytorch_agent.utils.logger import log



def _get_active_issues() -> list[dict]:
    """Get all open issues from the issue repo with the tracking label."""
    return gh.get_issues(ISSUE_REPO, label=ISSUE_LABEL)


def run_cycle() -> None:
    """Run one discovery + advancement cycle."""
    issues = _get_active_issues()

    for issue in issues:
        number = issue["number"]
        body = issue.get("body", "") or ""
        status = get_status(body)

        # Skip terminal stages
        if status in TERMINAL_STAGES:
            continue

        try:
            advance(number)
        except Exception as e:
            log("ERROR", f"Failed to advance issue #{number}: {e}",
                issue=number, exc=e)

    log("INFO", f"Cycle complete: {len(issues)} issues checked")


def run_single(issue_number: int) -> None:
    """Run full pipeline for a single issue: loop advance until terminal or stuck."""
    log("INFO", f"Single-issue run for #{issue_number}", issue=issue_number)

    max_loops = MAX_ADVANCE_LOOPS
    for i in range(max_loops):
        detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
        body = detail.get("body", "") or ""
        prev_stage = get_status(body)

        if prev_stage in TERMINAL_STAGES:
            log("INFO", f"Issue #{issue_number} reached terminal stage: {prev_stage}",
                issue=issue_number)
            break

        log("INFO", f"Advance loop {i+1}: issue #{issue_number} at {prev_stage or 'DISCOVERED'}",
            issue=issue_number)
        try:
            advance(issue_number)
        except Exception as e:
            log("ERROR", f"advance() failed for #{issue_number} at {prev_stage}: {e}",
                issue=issue_number, exc=e)
            break

        # Check if stage moved
        detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
        body = detail.get("body", "") or ""
        new_stage = get_status(body)
        if new_stage == prev_stage:
            log("INFO", f"Issue #{issue_number} stage unchanged at {prev_stage}, "
                        "stopping (may need retry next run)",
                issue=issue_number)
            break
    else:
        log("WARN", f"Issue #{issue_number} hit max advance loops ({max_loops})",
            issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polling loop for pytorch-agent pipeline"
    )
    parser.add_argument("--once", action="store_true", help="Run single cycle")
    parser.add_argument("--issue", type=int, help="Run pipeline for a single issue")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help="Seconds between cycles")
    args = parser.parse_args()

    if args.issue:
        run_single(args.issue)
    elif args.once:
        run_cycle()
    else:
        while True:
            try:
                run_cycle()
            except Exception as e:
                log("ERROR", f"Cycle failed: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
