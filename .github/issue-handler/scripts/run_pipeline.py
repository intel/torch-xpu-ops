#!/usr/bin/env python3
"""Run one pipeline cycle: scan target issues and advance each one stage.

Usage:
  python scripts/run_pipeline.py --once                    # single cycle
  python scripts/run_pipeline.py --once --issues 191       # single issue
  python scripts/run_pipeline.py --once --issues 191 327   # multiple issues
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone

# Add parent dir so `issue_handler` is importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from issue_handler.orchestrator import advance
from issue_handler.utils import git as gh
from issue_handler.utils.config import ISSUE_REPO, ISSUE_LABEL, TERMINAL_STAGES
from issue_handler.utils.body_templates import get_status

# Import e2e report (scripts/ dir, add to path)
_scripts_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from e2e_report import collect_results, build_report, update_tracking_issue


def _get_open_agent_issues() -> list[int]:
    """Fetch all open issues with the ai_generated label."""
    issues = gh.get_issues(ISSUE_REPO, ISSUE_LABEL)
    return [i["number"] for i in issues]


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def run_cycle(issue_numbers: list[int] | None = None) -> dict[int, str]:
    """Run one advance cycle. Returns {issue_number: result_string}."""
    if issue_numbers:
        issues = issue_numbers
    else:
        issues = _get_open_agent_issues()

    if not issues:
        _log("No issues to process")
        return {}

    _log(f"Processing {len(issues)} issue(s): {issues}")
    results: dict[int, str] = {}

    for num in issues:
        try:
            # Check current stage
            detail = gh.get_issue_detail(ISSUE_REPO, num)
            body = detail.get("body", "") or ""
            stage = get_status(body)

            if stage in TERMINAL_STAGES:
                _log(f"  #{num}: at terminal stage {stage}, skipping")
                results[num] = f"skipped:{stage}"
                continue

            _log(f"  #{num}: stage={stage or 'NONE'}, advancing...")
            advance(num)

            # Re-read to report new stage
            detail_after = gh.get_issue_detail(ISSUE_REPO, num)
            new_stage = get_status(detail_after.get("body", "") or "")
            _log(f"  #{num}: advanced to {new_stage}")
            results[num] = f"ok:{stage or 'NONE'}->{new_stage}"

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"  #{num}: ERROR — {e}")
            _log(f"  {tb[-500:]}")
            results[num] = f"error:{e}"

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline cycle")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--issues", type=int, nargs="+", help="Specific issue numbers to process")
    args = parser.parse_args()

    if args.once:
        results = run_cycle(args.issues)
        _log(f"Cycle complete: {results}")

        # Generate and publish E2E report
        processed_issues = list(results.keys())
        if processed_issues:
            try:
                report_results = collect_results(ISSUE_REPO, processed_issues)
                report = build_report(report_results, repo=ISSUE_REPO)
                tracking_num = update_tracking_issue(ISSUE_REPO, report)
                _log(f"E2E report updated: #{tracking_num}")
            except Exception as e:
                _log(f"E2E report failed: {e}")
    else:
        print("Continuous mode not implemented. Use --once with cron.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
