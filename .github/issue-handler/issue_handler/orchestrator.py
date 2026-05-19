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
from .utils.body_templates import get_status, get_metadata, sync_labels
from .utils.logger import log


def _assign_copilot(issue_number: int) -> None:
    """Assign Copilot to a torch-xpu-ops issue and hand off."""
    log("INFO", f"Assigning Copilot to #{issue_number} (torch-xpu-ops fix)",
        issue=issue_number)
    gh.assign_issue(ISSUE_REPO, issue_number, "Copilot")
    gh.add_issue_comment(
        ISSUE_REPO, issue_number,
        "🤖 **Routed to Copilot** — this issue is in torch-xpu-ops scope.\n\n"
        "@Copilot please fix this issue.",
    )


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
        # Sanitize error: strip long command args and local paths
        err_str = str(e)
        # If error contains a full command list, truncate to just the error type
        if "timed out after" in err_str:
            short_err = f"Agent timed out after the configured timeout"
        elif "failed (rc=" in err_str:
            short_err = err_str.split(", log:")[0][:200]
        else:
            # Strip anything that looks like a prompt dump
            short_err = err_str[:200]
        # Strip local paths from traceback
        clean_tb = tb.replace("/home/stonepia/", "~/")
        # Only keep the last few lines of traceback (actual error chain)
        tb_lines = clean_tb.strip().split("\n")
        short_tb = "\n".join(tb_lines[-6:]) if len(tb_lines) > 6 else clean_tb
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"❌ **Agent error** in `{step_name}` step:\n\n"
            f"```\n{short_err}\n```\n\n"
            f"<details><summary>Traceback</summary>\n\n"
            f"```\n{short_tb}\n```\n</details>",
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
        # No status marker → needs formatting first
        from .format_agent import run
        _run_step("format", run, issue_number)
        return

    log("INFO", f"Advancing issue #{issue_number} at stage {stage}",
        issue=issue_number)

    match stage:
        case "DISCOVERED":
            # Verify if issue still reproduces before triaging
            from .verify_existence import run as verify_run
            if verify_run(issue_number):
                # Issue is already fixed — no further work
                return
            from .triage_agent import run
            _run_step("triage", run, issue_number)
        case "TRIAGING":
            from .triage_agent import run
            _run_step("triage", run, issue_number)
        case "IMPLEMENTING":
            # Route based on target_repo: torch-xpu-ops → Copilot, pytorch → code_fix
            target_repo = get_metadata(body, "target_repo") or "pytorch"
            if target_repo == "torch-xpu-ops":
                _assign_copilot(issue_number)
            else:
                from .fixing_steps.code_fix import run
                _run_step("code_fix", run, issue_number)
        case "TRIAGED":
            # Advance status to IMPLEMENTING, then route by target_repo
            detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
            body = detail.get("body", "") or ""
            from .utils.body_templates import set_status as _set_status
            new_body = _set_status(body, "IMPLEMENTING")
            gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
            sync_labels(ISSUE_REPO, issue_number, "IMPLEMENTING")
            target_repo = get_metadata(body, "target_repo") or "pytorch"
            if target_repo == "torch-xpu-ops":
                _assign_copilot(issue_number)
            else:
                from .fixing_steps.code_fix import run
                _run_step("code_fix", run, issue_number)
        case "IN_REVIEW":
            from .verify_fix import run as verify_fix_run
            _run_step("verify_fix", verify_fix_run, issue_number)
        case "PUBLIC_PR" | "CI_WATCH" | "MERGED" | "DONE":
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
