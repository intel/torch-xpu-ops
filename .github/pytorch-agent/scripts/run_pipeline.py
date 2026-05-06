#!/usr/bin/env python3
"""Polling loop that discovers and advances issues.

Usage
-----
    python scripts/run_pipeline.py --once          # single cycle
    python scripts/run_pipeline.py --issue 3509    # single issue
    python scripts/run_pipeline.py                 # continuous loop

NOTE: This polling loop is temporary and will be replaced by
      GitHub Actions webhooks in a future PR.
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.config import POLL_INTERVAL  # noqa: E402 (after sys.path)
from pytorch_agent.utils.logger import log  # noqa: E402

# Stages where the pipeline should not call advance()
TERMINAL_STAGES = {"DONE", "SKIPPED", "NEEDS_HUMAN"}


def run_cycle() -> None:
    """Discover new issues and advance all active tracked issues by one step."""
    raise NotImplementedError


def run_single(issue_number: int) -> None:
    """Run the full pipeline for *issue_number*: triage → loop advance → terminal."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="pytorch-agent polling loop")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--issue", type=int, help="Run pipeline for one issue")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL)
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
