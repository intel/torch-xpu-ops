"""Watch CI on public PR and handle failures.

Entry point:
  python -m issue_handler.fixing_steps.ci_watch --issue 123
"""
from __future__ import annotations

import argparse
from subprocess import CalledProcessError

from ..utils import git as gh
from ..utils.git import git, add_and_commit
from ..utils.config import (
    ISSUE_REPO, PUBLIC_TARGET_REPO, PYTORCH_DIR, REVIEW_REMOTE,
    STAGE_TIMEOUTS, MAX_CI_ITERATIONS,
)
from ..utils.body_templates import (
    get_status, set_status, append_log, get_metadata, set_metadata,
)
from ..utils.agent_backend import get_backend
from ..utils.logger import log
from ..utils.notify import post_agent_completed, post_session_started


def run(issue_number: int) -> None:
    """Check CI status on public PR."""
    # Read issue body
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    status = get_status(body)
    if status != "CI_WATCH":
        return

    # Get public PR number from issue body metadata
    public_pr_str = get_metadata(body, "public_pr")
    if not public_pr_str:
        log("WARN", f"No public PR reference in issue #{issue_number} body",
            issue=issue_number)
        return
    public_pr = int(public_pr_str)

    # Check if PR is already merged
    pr_status = gh.get_pr_status(PUBLIC_TARGET_REPO, public_pr)
    if pr_status == "merged":
        new_body = set_status(body, "MERGED")
        new_body = append_log(new_body, "ci-watch", f"Public PR #{public_pr} merged!")
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        return

    # Get CI checks
    checks = gh.get_ci_checks(PUBLIC_TARGET_REPO, public_pr)
    if not checks:
        log("INFO", f"No CI checks yet for #{issue_number}", issue=issue_number)
        return

    # Categorize
    pending = [c for c in checks if c.get("status") != "completed"]
    failed = [c for c in checks
              if c.get("status") == "completed"
              and c.get("conclusion") not in ("success", "skipped", "neutral")]
    passed = [c for c in checks
              if c.get("status") == "completed"
              and c.get("conclusion") in ("success", "skipped", "neutral")]

    if pending:
        log("INFO", f"CI still running for #{issue_number}: "
                     f"{len(pending)} pending, {len(passed)} passed, {len(failed)} failed",
            issue=issue_number)
        return

    if not failed:
        log("INFO", f"All CI passed for #{issue_number}, waiting for merge",
            issue=issue_number)
        return

    # Handle failures — check iteration count
    ci_iter_str = get_metadata(body, "ci_iteration")
    ci_iteration = int(ci_iter_str) if ci_iter_str else 0
    ci_iteration += 1

    if ci_iteration > MAX_CI_ITERATIONS:
        new_body = set_status(body, "NEEDS_HUMAN")
        new_body = append_log(new_body, "ci-watch",
                              f"CI fix iteration limit ({MAX_CI_ITERATIONS}) reached.")
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        return

    # Update iteration counter in body
    new_body = set_metadata(body, "ci_iteration", str(ci_iteration))

    failure_text = "\n".join(
        f"- **{c.get('name', 'unknown')}**: {c.get('conclusion', '?')} — "
        f"{(c.get('output') or {}).get('summary', 'no details')}"
        for c in failed
    )

    # Call agent with skill (no inline prompt)
    prompt = (
        f"Read the pytorch-ci-triage skill and fix CI failures on PR #{public_pr}.\n\n"
        f"## Failing checks:\n{failure_text}"
    )

    def _post_session_id(sid: str):
        post_session_started(ISSUE_REPO, issue_number, "CI fix", sid)

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("CI_WATCH", 600)
    output, log_path, _ = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                    skill="pytorch-ci-triage", timeout=timeout,
                                    issue=issue_number, stage="CI_WATCH",
                                    on_session_start=_post_session_id)

    # Auto-commit and push
    branch = f"agent/issue-{issue_number}"
    committed = add_and_commit(
        f"[agent] CI fix iteration {ci_iteration} for #{issue_number}",
        issue=issue_number,
    )

    if committed:
        try:
            git("push", REVIEW_REMOTE, branch, issue=issue_number)
        except CalledProcessError as e:
            log("ERROR", f"Failed to push CI fix: {e}", issue=issue_number, exc=e)

    # Update issue body with log
    new_body = append_log(new_body, "ci-watch",
                          f"CI fix iteration {ci_iteration}\n"
                          f"Log: `{log_path.name}`")
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"CI fix iteration {ci_iteration} for #{issue_number}",
        issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
