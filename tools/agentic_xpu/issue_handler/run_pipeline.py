#!/usr/bin/env python3
"""Run the issue-handler pipeline: scan issues and advance through stages.

Usage:
  # Run all stages (format → triage → fix → verify_fix) by default
    python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12

  # Run specific stages only
    python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12 --stages format triage
    python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12 --stages fix verify_fix

  # Run all open agent issues through all stages
    python tools/agentic_xpu/issue_handler/run_pipeline.py --once

Available stages (in pipeline order): format, triage, fix, verify_fix
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

SCENARIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_DIR))

from agentic_xpu.issue_handler.orchestrator import advance
from agentic_xpu.issue_handler.utils import git as gh
from agentic_xpu.issue_handler.utils.config import (
    ISSUE_REPO, ISSUE_LABEL, TERMINAL_STAGES, TRACKING_REPO,
)
from agentic_xpu.issue_handler.utils.body_templates import get_status

# Optional e2e report
try:
    _scripts_dir = str(SCENARIO_DIR)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from e2e_report import collect_results, build_report, update_tracking_issue
    _HAS_E2E_REPORT = True
except ImportError:
    _HAS_E2E_REPORT = False

# Pipeline stages in execution order
STAGE_ORDER = ["format", "triage", "fix", "verify_fix"]

# After each stage completes, the issue body status should be one of these
_STAGE_DONE_STATUS: dict[str, tuple[str, ...]] = {
    "format":     ("DISCOVERED",),
    "triage":     ("TRIAGED", "NEEDS_HUMAN"),
    "fix":        ("IMPLEMENTING", "IN_REVIEW"),
    "verify_fix": ("IN_REVIEW", "DONE"),
}

# Stages where no further pipeline work is possible *this cycle*, but the
# issue is NOT terminal (it will move forward later — e.g. WAITING_UPSTREAM
# resumes once the upstream PR merges, BLOCKED clears once a human unblocks).
# Distinct from TERMINAL_STAGES (DONE/SKIPPED/NEEDS_HUMAN) which never
# resume.  Without this set, the per-issue stages loop keeps calling
# ``advance()`` for ``fix`` and ``verify_fix`` even though the dispatcher
# can only short-circuit (e.g. throttle), producing noise in the log and
# wasted ``gh.get_issue_detail`` calls.
_CYCLE_HALT_STAGES: frozenset[str] = frozenset({
    "WAITING_UPSTREAM",
    "BLOCKED",
    "UPSTREAM_VERIFYING",  # mid-verify retry; finishes next cycle
})


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _get_open_agent_issues() -> list[int]:
    """Fetch all open issues with the agent label."""
    issues = gh.get_issues(ISSUE_REPO, ISSUE_LABEL)
    return [i["number"] for i in issues]


def _current_stage_index(status: str | None) -> int:
    """Map an issue body status to the index of the stage that produced it.

    Returns -1 if the issue hasn't been through any stage yet (no status marker).
    """
    if status is None:
        return -1
    for i, stage_name in enumerate(STAGE_ORDER):
        if status in _STAGE_DONE_STATUS[stage_name]:
            return i
    return -1


def run_issue(issue_number: int, stages: list[str]) -> str:
    """Run an issue through the requested stages. Returns a result string."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    status = get_status(body)

    if status in TERMINAL_STAGES:
        _log(f"  #{issue_number}: at terminal stage {status}, skipping")
        return f"skipped:{status}"

    initial_status = status or "NONE"
    _log(f"  #{issue_number}: status={initial_status}, will run stages: {' → '.join(stages)}")

    # If already in a cycle-halt stage at entry (e.g. WAITING_UPSTREAM),
    # don't even dispatch — the dispatcher will only short-circuit and log
    # noise. Resumes naturally next cycle when state changes.
    if status in _CYCLE_HALT_STAGES:
        _log(f"  #{issue_number}: at {status}, no work this cycle (will resume on state change)")
        return f"halt:{status}"

    # Figure out which stages still need running.
    # If the issue is already past a requested stage, skip it.
    current_idx = _current_stage_index(status)

    for stage_name in stages:
        stage_idx = STAGE_ORDER.index(stage_name)

        # Already past this stage
        if stage_idx <= current_idx:
            _log(f"  #{issue_number}: already past '{stage_name}', skipping")
            continue

        _log(f"  #{issue_number}: running '{stage_name}'...")
        advance(issue_number)

        # Re-read status after advance
        detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
        status = get_status(detail.get("body", "") or "")
        current_idx = _current_stage_index(status)

        # Stop if terminal
        if status in TERMINAL_STAGES:
            _log(f"  #{issue_number}: reached terminal stage {status}")
            break

        # Stop if cycle-halt (WAITING_UPSTREAM, BLOCKED, UPSTREAM_VERIFYING).
        # Issue will resume next cycle; no point dispatching the remaining
        # stages now — they'd just short-circuit and pollute the log.
        if status in _CYCLE_HALT_STAGES:
            _log(f"  #{issue_number}: reached {status}, halting cycle")
            break

    _log(f"  #{issue_number}: done — {initial_status} → {status}")
    return f"ok:{initial_status}->{status}"


def run_cycle(issue_numbers: list[int] | None = None,
              stages: list[str] | None = None) -> dict[int, str]:
    """Run a pipeline cycle across issues. Returns {issue_number: result}."""
    issues = issue_numbers or _get_open_agent_issues()
    if not issues:
        _log("No issues to process")
        return {}

    target_stages = stages or STAGE_ORDER
    _log(f"Processing {len(issues)} issue(s): {issues}")
    _log(f"Stages: {' → '.join(target_stages)}")

    results: dict[int, str] = {}
    for num in issues:
        try:
            results[num] = run_issue(num, target_stages)
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"  #{num}: ERROR — {e}")
            _log(f"  {tb[-500:]}")
            results[num] = f"error:{e}"

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the issue-handler pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--issues", type=int, nargs="+",
                        help="Specific issue numbers to process")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER, metavar="STAGE",
        help=(
            "Stages to run (in order): format triage fix verify_fix. "
            "Default: all stages."
        ),
    )
    parser.add_argument(
        "--batch", type=str, default="Run",
        help="Batch label for the dashboard report",
    )
    args = parser.parse_args()

    if args.once:
        results = run_cycle(args.issues, stages=args.stages)
        _log(f"Cycle complete: {results}")

        # E2E report
        processed = list(results.keys())
        if processed and _HAS_E2E_REPORT:
            try:
                report_results = collect_results(ISSUE_REPO, processed)
                build_report(report_results, repo=ISSUE_REPO)
                tracking_num = update_tracking_issue(
                    TRACKING_REPO, report_results, batch_name=args.batch,
                )
                _log(f"E2E report updated: #{tracking_num}")
            except Exception as e:
                _log(f"E2E report failed: {e}")
    else:
        print("Continuous mode not implemented. Use --once with cron.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
