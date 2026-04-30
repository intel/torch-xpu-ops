"""Issue fixing agent — orchestrate fixing steps based on current stage.

Entry point:
  python -m pytorch_agent.issue_fixing_agent --issue 123
"""
from __future__ import annotations

import argparse
import traceback

from .utils import github_client as gh
from .utils.config import UPSTREAM_ISSUE_REPO
from .utils.state import load_tracked
from .utils.logger import log


def _run_step(step_name: str, run_fn, tracked, issue_number: int) -> None:
    """Run a fixing step with error reporting to the source issue."""
    log("INFO", f"Dispatching {step_name} step", issue=issue_number)
    try:
        run_fn(tracked)
        log("INFO", f"{step_name} step completed", issue=issue_number)
    except Exception as e:
        tb = traceback.format_exc()
        log("ERROR", f"{step_name} step failed: {e}", issue=issue_number, exc=e)
        # Post error to source issue so it's visible remotely
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, issue_number,
            f"❌ **Agent error** in `{step_name}` step:\n\n"
            f"```\n{str(e)[:500]}\n```\n\n"
            f"<details><summary>Full traceback</summary>\n\n"
            f"```\n{tb[-1500:]}\n```\n</details>",
        )
        raise


def advance(issue_number: int) -> None:
    """Advance a tracked issue to the next stage.

    Each step checks current state and short-circuits if already done.
    """
    tracked = load_tracked(issue_number)
    log("INFO", f"Advancing issue #{issue_number} at stage {tracked.stage}",
        issue=issue_number)

    match tracked.stage:
        case "IMPLEMENTING":
            from .fixing_steps.implement import run
            _run_step("implement", run, tracked, issue_number)
        case "IN_REVIEW":
            from .fixing_steps.private_review import run
            _run_step("private_review", run, tracked, issue_number)
        case "PUBLIC_PR":
            from .fixing_steps.public_submit import run
            _run_step("public_submit", run, tracked, issue_number)
        case "CI_WATCH":
            from .fixing_steps.ci_watch import run
            _run_step("ci_watch", run, tracked, issue_number)
        case "DONE":
            from .fixing_steps.close_issue import run
            _run_step("close_issue", run, tracked, issue_number)
        case "NEEDS_HUMAN":
            log("INFO", f"Issue #{issue_number} needs human intervention, skipping",
                issue=issue_number)
        case _:
            log("WARN", f"Issue #{issue_number} at unexpected stage {tracked.stage}",
                issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advance a tracked issue")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    advance(args.issue)


if __name__ == "__main__":
    main()
