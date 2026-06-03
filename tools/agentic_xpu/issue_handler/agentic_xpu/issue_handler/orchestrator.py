# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Issue fixing agent — orchestrate all stages from discovery to close.

Entry point:
  python -m issue_handler.orchestrator --issue 123

Reads stage from issue body (<!-- agent:status:STAGE -->) and dispatches
to the appropriate agent/step.
"""
from __future__ import annotations

import argparse
import hashlib
import traceback
from pathlib import Path

from .utils import git as gh
from .utils.config import ISSUE_REPO
from .utils.body_templates import get_status, get_metadata
from .utils.git import sync_labels
from .utils.logger import log
from .utils.stages import Stage, TargetRepo


def _error_marker(step_name: str, short_err: str) -> str:
    """Stable HTML-comment marker for a (step, error) pair.

    Used to detect whether we've already posted this exact error on this
    issue.  We hash the sanitised error string so unrelated whitespace /
    traceback line-number drift doesn't defeat the dedup.
    """
    digest = hashlib.sha1(
        f"{step_name}\n{short_err}".encode("utf-8")).hexdigest()[:12]
    return f"<!-- agent:error:{step_name}:{digest} -->"


def _already_reported(issue_detail: dict, marker: str) -> bool:
    """True if any existing comment on the issue contains ``marker``."""
    for c in (issue_detail.get("comments") or []):
        body = c.get("body", "") if isinstance(c, dict) else ""
        if marker in body:
            return True
    return False


def _assign_copilot(issue_number: int) -> None:
    """Assign Copilot to a torch-xpu-ops issue and hand off.

    Idempotent: if Copilot is already on the assignee list we just log and
    move on instead of spamming the issue with another "@Copilot please
    fix this" comment every time the orchestrator polls.
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    assignees = {
        a.get("login", "") if isinstance(a, dict) else a
        for a in (detail.get("assignees") or [])
    }
    if "Copilot" in assignees or "copilot" in {a.lower() for a in assignees}:
        log("INFO", f"#{issue_number} already assigned to Copilot, skipping reassignment",
            issue=issue_number)
        return

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
            short_err = "Agent timed out after the configured timeout"
        elif "failed (rc=" in err_str:
            short_err = err_str.split(", log:")[0][:200]
        else:
            # Strip anything that looks like a prompt dump
            short_err = err_str[:200]
        # Strip local paths from traceback
        clean_tb = tb.replace(str(Path.home()) + "/", "~/")
        # Only keep the last few lines of traceback (actual error chain)
        tb_lines = clean_tb.strip().split("\n")
        short_tb = "\n".join(tb_lines[-6:]) if len(tb_lines) > 6 else clean_tb
        # Dedup: if the same (step, error) was already reported on this
        # issue, skip posting another comment so a permanently-broken issue
        # doesn't accumulate a wall of identical tracebacks on every poll.
        marker = _error_marker(step_name, short_err)
        detail_for_dedup = gh.get_issue_detail(ISSUE_REPO, issue_number)
        if _already_reported(detail_for_dedup, marker):
            log("INFO",
                f"{step_name} error already reported on #{issue_number}, "
                "not posting duplicate comment",
                issue=issue_number)
        else:
            gh.add_issue_comment(
                ISSUE_REPO, issue_number,
                f"{marker}\n"
                f"❌ **Agent error** in `{step_name}` step:\n\n"
                f"```\n{short_err}\n```\n\n"
                f"<details><summary>Traceback</summary>\n\n"
                f"```\n{short_tb}\n```\n</details>",
            )
        raise


def _route_implementation(issue_number: int, body: str) -> None:
    """Dispatch IMPLEMENTING to Copilot (torch-xpu-ops) or code_fix (pytorch)."""
    target = get_metadata(body, "target_repo") or TargetRepo.PYTORCH
    if target == TargetRepo.TORCH_XPU_OPS:
        _assign_copilot(issue_number)
    else:
        from .code_fix import run
        _run_step("code_fix", run, issue_number)


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
        case Stage.DISCOVERED:
            # Verify if issue still reproduces before triaging
            from .verify_existence import run as verify_run
            _run_step("verify_existence", verify_run, issue_number)
            # Re-read body — verify may have set DONE if already fixed
            detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
            new_stage = get_status(detail.get("body", "") or "")
            # verify_existence only ever sets DONE; NEEDS_HUMAN is included
            # defensively in case that contract changes.
            if new_stage in (Stage.DONE, Stage.NEEDS_HUMAN):
                return
            # Try upstream-PR short-circuit before spending triage effort.
            # Called directly (not via _run_step) because we need its
            # return value to decide whether to skip triage this turn.
            from .verify_upstream_pr import run as upstream_run
            try:
                if upstream_run(issue_number):
                    return  # DONE or WAITING_UPSTREAM — skip triage
            except Exception as e:
                # Infra crash (merge-base unrelated histories, fetch failure,
                # cherry-pick conflict, etc.) is NOT the same as "verdict not
                # borne out". Falling through to triage silently hides real
                # bugs in the upstream-PR path and burns a triage budget on
                # an issue that should be looked at by a human. Surface it.
                log("ERROR",
                    f"verify_upstream_pr crashed: {e!r} — marking "
                    f"agent:needs-human and stopping (NOT falling through "
                    f"to triage)",
                    issue=issue_number, exc=e)
                try:
                    from .utils.body_templates import set_status as _set_status
                    fresh = gh.get_issue_detail(ISSUE_REPO, issue_number).get(
                        "body", "") or ""
                    fresh = _set_status(fresh, Stage.NEEDS_HUMAN)
                    gh.update_issue_body(ISSUE_REPO, issue_number, fresh)
                    sync_labels(ISSUE_REPO, issue_number, Stage.NEEDS_HUMAN)
                except Exception as label_exc:
                    log("ERROR",
                        f"Failed to set agent:needs-human after crash: "
                        f"{label_exc!r}",
                        issue=issue_number, exc=label_exc)
                return
            from .triage_agent import run
            _run_step("triage", run, issue_number)
        case Stage.UPSTREAM_VERIFYING:
            # Re-entry after a crash mid-verification — just retry.
            from .verify_upstream_pr import run as upstream_run
            _run_step("verify_upstream_pr", upstream_run, issue_number)
        case Stage.WAITING_UPSTREAM:
            # Re-check periodically; 12h throttle lives inside the module.
            from .verify_upstream_pr import run as upstream_run
            _run_step("verify_upstream_pr", upstream_run, issue_number)
        case Stage.IMPLEMENTING:
            _route_implementation(issue_number, body)
        case Stage.TRIAGED:
            # Advance status to IMPLEMENTING, then route by target_repo
            from .utils.body_templates import set_status as _set_status
            new_body = _set_status(body, Stage.IMPLEMENTING)
            gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
            sync_labels(ISSUE_REPO, issue_number, Stage.IMPLEMENTING)
            _route_implementation(issue_number, new_body)
        case Stage.IN_REVIEW:
            from .verify_fix import run as verify_fix_run
            _run_step("verify_fix", verify_fix_run, issue_number)
        case Stage.PUBLIC_PR | Stage.CI_WATCH | Stage.MERGED | Stage.DONE:
            log("INFO", f"Issue #{issue_number} at stage {stage} — not yet implemented",
                issue=issue_number)
        case Stage.NEEDS_HUMAN:
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
