"""Watch CI on public PR and handle failures.

Entry point:
  python -m pytorch_agent.fixing_steps.ci_watch --issue 123
"""
from __future__ import annotations

import argparse
import subprocess

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PUBLIC_TARGET_REPO, PYTORCH_DIR, REVIEW_REMOTE,
    PRIVATE_REVIEW_REPO, STAGE_TIMEOUTS, MAX_REVIEW_ITERATIONS,
)
from ..utils.state import TrackedIssue, update_stage, save_state, load_tracked
from ..utils.agent_backend import get_backend
from ..utils.logger import log


CI_FIX_PROMPT_TEMPLATE = """CI is failing on the public PR for your PyTorch fix.

## Failing checks:
{failures}

## Instructions
1. Analyze each failure — is it related to your change or pre-existing?
2. If related: fix the issue in code.
3. If unrelated: note it but don't change code for it.
"""


def run(tracked: TrackedIssue) -> None:
    """Check CI status on public PR."""
    if tracked.stage != "CI_WATCH":
        return

    if not tracked.public_pr_number:
        log("WARN", f"No public PR for issue #{tracked.source_number}",
            issue=tracked.source_number)
        return

    # Check if PR is already merged
    pr_status = gh.get_pr_status(PUBLIC_TARGET_REPO, tracked.public_pr_number)
    if pr_status == "merged":
        update_stage(tracked, "DONE",
                     f"Public PR #{tracked.public_pr_number} merged!")
        return

    # Get CI checks
    checks = gh.get_ci_checks(PUBLIC_TARGET_REPO, tracked.public_pr_number)
    if not checks:
        log("INFO", f"No CI checks yet for #{tracked.source_number}",
            issue=tracked.source_number)
        return

    # Categorize
    pending = [c for c in checks if c.get("status") != "completed"]
    failed = [c for c in checks
              if c.get("status") == "completed"
              and c.get("conclusion") not in ("success", "skipped", "neutral")]
    passed = [c for c in checks
              if c.get("status") == "completed"
              and c.get("conclusion") in ("success", "skipped", "neutral")]

    # Post CI status summary to source issue
    ci_summary = (
        f"⏳ {len(pending)} pending" if pending else
        f"✅ {len(passed)} passed, ❌ {len(failed)} failed"
    )
    gh.add_issue_comment(
        UPSTREAM_ISSUE_REPO, tracked.source_number,
        f"🤖 **CI status** for public PR #{tracked.public_pr_number}:\n\n"
        f"{ci_summary} (total: {len(checks)} checks)",
    )

    if pending:
        log("INFO", f"CI still running for #{tracked.source_number}: "
                     f"{len(pending)} pending, {len(passed)} passed, {len(failed)} failed",
            issue=tracked.source_number)
        return

    if not failed:
        # All passed, but PR not yet merged — wait for merge
        log("INFO", f"All CI passed for #{tracked.source_number}, waiting for merge",
            issue=tracked.source_number)
        return

    # Handle failures
    failure_text = "\n".join(
        f"- **{c.get('name', 'unknown')}**: {c.get('conclusion', '?')} — "
        f"{(c.get('output') or {}).get('summary', 'no details')}"
        for c in failed
    )

    # Post failure details to source issue
    gh.add_issue_comment(
        UPSTREAM_ISSUE_REPO, tracked.source_number,
        f"🤖 **CI failures** on public PR #{tracked.public_pr_number}:\n\n"
        f"{failure_text}\n\n"
        f"Agent is analyzing failures...",
    )

    # --- CI iteration limit ---
    ci_iteration = getattr(tracked, "ci_iteration", 0)
    if ci_iteration >= MAX_REVIEW_ITERATIONS:
        update_stage(tracked, "NEEDS_HUMAN",
                     f"CI fix iteration limit ({MAX_REVIEW_ITERATIONS}) reached.")
        return
    tracked.ci_iteration = ci_iteration + 1
    save_state(tracked)

    prompt = CI_FIX_PROMPT_TEMPLATE.format(failures=failure_text)

    from ..utils.notify import post_session_started

    def _post_session_id(sid: str):
        post_session_started(UPSTREAM_ISSUE_REPO, tracked.source_number,
                             "CI fix", sid)

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("CI_WATCH", 600)
    output, log_path, _ = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                    skill="pytorch-ci-triage", timeout=timeout,
                                    issue=tracked.source_number, stage="CI_WATCH",
                                    on_session_start=_post_session_id)
    log("INFO", f"CI fix agent log: {log_path}",
        issue=tracked.source_number)

    # Auto-commit any changes the agent made
    branch = tracked.branch or f"agent/issue-{tracked.source_number}"
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(PYTORCH_DIR), check=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PYTORCH_DIR),
        )
        if result.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m",
                 f"[agent] CI fix iteration {tracked.ci_iteration} "
                 f"for #{tracked.source_number}"],
                cwd=str(PYTORCH_DIR), check=True,
            )
    except subprocess.CalledProcessError:
        pass  # No changes to commit

    # Push fixes (no force push — preserves reviewed commits)
    try:
        subprocess.run(
            ["git", "push", REVIEW_REMOTE, branch],
            cwd=str(PYTORCH_DIR), check=True,
        )
        tracked.last_push_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PYTORCH_DIR),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        save_state(tracked)
        log("INFO", f"Pushed CI fix for #{tracked.source_number}",
            issue=tracked.source_number)
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **CI fix pushed** (iteration {tracked.ci_iteration}) "
            f"for PR #{tracked.public_pr_number}\n\n"
            f"_Agent log: `{log_path.name}`_",
        )
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Failed to push CI fix: {e}", issue=tracked.source_number,
            exc=e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
