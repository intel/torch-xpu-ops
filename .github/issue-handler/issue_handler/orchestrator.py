"""Issue fixing agent — orchestrate all stages from discovery to close.

Entry point:
  python -m issue_handler.orchestrator --issue 123

Reads stage from issue body (<!-- agent:status:STAGE -->) and dispatches
to the appropriate agent/step.
"""
from __future__ import annotations

import argparse
import traceback

from .utils import git as gh
from .utils.config import ISSUE_REPO
from .utils.body_templates import get_status, sync_labels
from .utils.logger import log


def _run_step(step_name: str, run_fn, issue_number: int) -> None:
    """Run a step with error reporting to the source issue."""
    log("INFO", f"Dispatching {step_name} step", issue=issue_number)
    try:
        run_fn(issue_number)
        log("INFO", f"{step_name} step completed", issue=issue_number)
        # Sync labels after each step (re-read body for new status)
        detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
        new_stage = get_status(detail.get("body", "") or "")
        if new_stage:
            sync_labels(ISSUE_REPO, issue_number, new_stage)
    except Exception as e:
        tb = traceback.format_exc()
        log("ERROR", f"{step_name} step failed: {e}", issue=issue_number, exc=e)
        # Post error to source issue so it's visible remotely
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"❌ **Agent error** in `{step_name}` step:\n\n"
            f"```\n{str(e)[:500]}\n```\n\n"
            f"<details><summary>Full traceback</summary>\n\n"
            f"```\n{tb[-1500:]}\n```\n</details>",
        )
        raise


def advance(issue_number: int) -> None:
    """Advance an issue to the next stage.

    Reads current stage from issue body and dispatches to the right agent.
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    stage = get_status(body)

    if stage is None:
        # No status marker → needs discovery
        stage = "DISCOVERED"

    log("INFO", f"Advancing issue #{issue_number} at stage {stage}",
        issue=issue_number)

    match stage:
        case "DISCOVERED":
            # Issue is formatted, advance to triage
            from .triage_agent import run
            _run_step("triage", run, issue_number)
        case "TRIAGING":
            from .triage_agent import run
            _run_step("triage", run, issue_number)
        case "IMPLEMENTING":
            from .fixing_steps.code_fix import run
            _run_step("code_fix", run, issue_number)
        case "IN_REVIEW" | "PUBLIC_PR" | "CI_WATCH" | "MERGED" | "DONE":
            log("INFO", f"Issue #{issue_number} at stage {stage} — not yet implemented",
                issue=issue_number)
        case "NEEDS_HUMAN":
            log("INFO", f"Issue #{issue_number} needs human intervention, skipping",
                issue=issue_number)
        case _:
            log("WARN", f"Issue #{issue_number} at unexpected stage {stage}",
                issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advance an issue through the pipeline")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    advance(args.issue)


if __name__ == "__main__":
    main()
