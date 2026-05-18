#!/usr/bin/env python3
"""Run one pipeline cycle: scan target issues and advance each one stage.

Usage:
  python scripts/run_pipeline.py --once                                          # one stage advance, all issues
  python scripts/run_pipeline.py --once --issues 191                             # single issue, one stage
  python scripts/run_pipeline.py --once --issues 191 327                         # multiple issues, one stage each
  python scripts/run_pipeline.py --once --issues 191 --stages format triage      # run through triage
  python scripts/run_pipeline.py --once --issues 191 --stages format triage fix  # run all the way to fix

Available stages (in pipeline order): format, triage, fix, verify_fix
When --stages is given, each issue is advanced repeatedly until it reaches
the last requested stage (or a terminal stage like NEEDS_HUMAN/DONE).
Without --stages, each issue advances exactly one stage (original behaviour).
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
from issue_handler.utils.config import ISSUE_REPO, ISSUE_LABEL, TERMINAL_STAGES, TRACKING_REPO
from issue_handler.utils.body_templates import get_status

# Import e2e report (scripts/ dir, add to path)
_scripts_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from e2e_report import collect_results, build_report, update_tracking_issue

# Map user-facing stage names → the body status marker(s) set after that stage completes
_STAGE_TO_STATUS: dict[str, str | tuple[str, ...]] = {
    "format":     "DISCOVERED",
    "triage":     ("TRIAGED", "NEEDS_HUMAN"),
    "fix":        ("IMPLEMENTING", "IN_REVIEW"),
    "verify_fix": ("IN_REVIEW", "DONE"),
}

# Ordered list of stage names (used for --stages choices)
_STAGE_ORDER = ["format", "triage", "fix", "verify_fix"]


def _get_open_agent_issues() -> list[int]:
    """Fetch all open issues with the ai_generated label."""
    issues = gh.get_issues(ISSUE_REPO, ISSUE_LABEL)
    return [i["number"] for i in issues]


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _target_reached(stage: str | None, target_stages: list[str]) -> bool:
    """Return True if the current status means we've reached the last requested stage."""
    if stage is None:
        return False
    last = target_stages[-1]
    expected = _STAGE_TO_STATUS.get(last)
    if isinstance(expected, tuple):
        return stage in expected
    return stage == expected


def run_cycle(issue_numbers: list[int] | None = None,
              target_stages: list[str] | None = None) -> dict[int, str]:
    """Run one (or multi-stage) advance cycle. Returns {issue_number: result_string}."""
    if issue_numbers:
        issues = issue_numbers
    else:
        issues = _get_open_agent_issues()

    if not issues:
        _log("No issues to process")
        return {}

    _log(f"Processing {len(issues)} issue(s): {issues}")
    if target_stages:
        _log(f"Target stages: {' → '.join(target_stages)}")
    results: dict[int, str] = {}

    for num in issues:
        try:
            detail = gh.get_issue_detail(ISSUE_REPO, num)
            body = detail.get("body", "") or ""
            initial_stage = get_status(body)

            if initial_stage in TERMINAL_STAGES:
                _log(f"  #{num}: at terminal stage {initial_stage}, skipping")
                results[num] = f"skipped:{initial_stage}"
                continue

            _log(f"  #{num}: stage={initial_stage or 'NONE'}, advancing...")
            current_stage = initial_stage

            # Without --stages: advance exactly once (original behaviour)
            max_advances = len(target_stages) if target_stages else 1
            advances_done = 0

            while advances_done < max_advances:
                advance(num)
                advances_done += 1

                # Re-read stage after advance
                detail_after = gh.get_issue_detail(ISSUE_REPO, num)
                current_stage = get_status(detail_after.get("body", "") or "")

                # Stop if terminal regardless of requested stages
                if current_stage in TERMINAL_STAGES:
                    _log(f"  #{num}: reached terminal stage {current_stage}, stopping")
                    break

                # Stop if we've reached the last requested stage
                if target_stages and _target_reached(current_stage, target_stages):
                    break

            _log(f"  #{num}: advanced to {current_stage}")
            results[num] = f"ok:{initial_stage or 'NONE'}->{current_stage}"

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"  #{num}: ERROR — {e}")
            _log(f"  {tb[-500:]}")
            results[num] = f"error:{e}"

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pipeline cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--issues", type=int, nargs="+", help="Specific issue numbers to process")
    parser.add_argument(
        "--stages", nargs="+",
        choices=_STAGE_ORDER,
        metavar="STAGE",
        help=(
            "Stages to run through in order: format triage fix verify_fix. "
            "Each issue advances until it reaches the last stage or a terminal. "
            "Without this flag, each issue advances exactly one stage."
        ),
    )
    parser.add_argument(
        "--batch", type=str, default="Run",
        help="Batch label for the [Run N] section in the dashboard (e.g. 'Batch run — 35 issues')",
    )
    args = parser.parse_args()

    if args.once:
        results = run_cycle(args.issues, target_stages=args.stages)
        _log(f"Cycle complete: {results}")

        # Generate and publish E2E report
        processed_issues = list(results.keys())
        if processed_issues:
            try:
                report_results = collect_results(ISSUE_REPO, processed_issues)
                report = build_report(report_results, repo=ISSUE_REPO)
                tracking_num = update_tracking_issue(TRACKING_REPO, report_results,
                                                       batch_name=args.batch)
                _log(f"E2E report updated: #{tracking_num}")
            except Exception as e:
                _log(f"E2E report failed: {e}")
    else:
        print("Continuous mode not implemented. Use --once with cron.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
