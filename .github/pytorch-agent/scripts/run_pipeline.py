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

from pytorch_agent.issue_discovery import discover_new_issues, process_issue
from pytorch_agent.issue_triaging_agent import triage_issue
from pytorch_agent.issue_fixing_agent import advance
from pytorch_agent.utils.state import get_all_tracked
from pytorch_agent.utils.config import POLL_INTERVAL
from pytorch_agent.utils.logger import log


TERMINAL_STAGES = {"DONE", "SKIPPED", "NEEDS_HUMAN"}


def run_cycle() -> None:
    """Run one discovery + advancement cycle."""
    # 1. Discover new issues
    new = discover_new_issues()
    for issue in new:
        tracked = process_issue(issue["number"])
        triage_issue(issue["number"])

    # 2. Advance each tracked issue (sequential — they share ~/pytorch workdir)
    active = [
        t for t in get_all_tracked()
        if t.stage not in TERMINAL_STAGES and not t.paused
    ]
    for t in active:
        try:
            # Loop advance until terminal or stuck
            from pytorch_agent.utils.state import load_tracked
            for _ in range(10):
                current = load_tracked(t.source_number)
                if current.stage in TERMINAL_STAGES:
                    break
                prev = current.stage
                advance(t.source_number)
                current = load_tracked(t.source_number)
                if current.stage == prev:
                    break
        except Exception as e:
            log("ERROR", f"Failed to advance issue #{t.source_number}: {e}",
                issue=t.source_number, exc=e)

    log("INFO", f"Cycle complete: {len(new)} new, {len(active)} active")



def run_single(issue_number: int) -> None:
    """Run full pipeline for a single issue: discover → triage → loop advance."""
    log("INFO", f"Single-issue run for #{issue_number}", issue=issue_number)
    tracked = process_issue(issue_number)
    if tracked.stage == "DISCOVERED":
        triage_issue(issue_number)

    # Re-load after triage may have changed stage
    from pytorch_agent.utils.state import load_tracked
    tracked = load_tracked(issue_number)

    if tracked.paused:
        log("INFO", f"Issue #{issue_number} is paused, skipping", issue=issue_number)
        return

    # Loop: keep advancing until terminal or stuck
    max_loops = 10  # safety guard against infinite loops
    for i in range(max_loops):
        tracked = load_tracked(issue_number)
        prev_stage = tracked.stage
        if prev_stage in TERMINAL_STAGES:
            log("INFO", f"Issue #{issue_number} reached terminal stage: {prev_stage}",
                issue=issue_number)
            break

        log("INFO", f"Advance loop {i+1}: issue #{issue_number} at {prev_stage}",
            issue=issue_number)
        try:
            advance(issue_number)
        except Exception as e:
            log("ERROR", f"advance() failed for #{issue_number} at {prev_stage}: {e}",
                issue=issue_number, exc=e)
            break

        # Check if stage actually moved — if not, don't loop forever
        tracked = load_tracked(issue_number)
        if tracked.stage == prev_stage:
            log("INFO", f"Issue #{issue_number} stage unchanged at {prev_stage}, "
                        "stopping loop (may need retry on next run)",
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
