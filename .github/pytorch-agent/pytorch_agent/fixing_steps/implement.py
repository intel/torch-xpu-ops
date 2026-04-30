"""Implement a fix for a triaged issue.

Entry point:
  python -m pytorch_agent.fixing_steps.implement --issue 123
"""
from __future__ import annotations

import argparse
from subprocess import CalledProcessError

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PRIVATE_REVIEW_REPO, PYTORCH_DIR,
    REVIEW_REMOTE, MAX_AGENT_ATTEMPTS, STAGE_TIMEOUTS,
)
from ..utils.state import TrackedIssue, update_stage, save_state, load_tracked
from ..utils.agent_backend import get_backend
from ..utils.git import git, git_out, add_and_commit
from ..utils.logger import log


IMPLEMENT_PROMPT_TEMPLATE = """Fix the following PyTorch CI failure for Intel XPU.

## Issue #{number}: {title}

{body_section}

## Instructions
1. Start by reproducing the failure using the repro commands above (if provided).
2. If a commit scope is given, use `git log --oneline` on that range to identify the likely breaking commit.
3. Read the error log carefully and trace the root cause in the PyTorch codebase.
4. Make the minimal fix. Prefer XPU-specific paths (aten/src/ATen/xpu/, torch/xpu/) when possible.
5. Run the failing test(s) listed above to verify your fix.
6. Ensure no regressions in related tests.

## HARD RULES (violations will be rejected)
- NEVER use @skipIfXpu, @skip, unittest.skip, or any skip decorator. You must FIX the test, not skip it.
- Do NOT commit submodule pointer changes (third_party/*). Use `git add` on specific files only.

Work in the current directory (~/pytorch).
"""


# Issue section parsing — use the shared helper from _issue_format
from ._issue_format import parse_issue_sections as _parse_issue_sections


def _build_body_section(detail: dict) -> str:
    """Build the prompt body from parsed issue sections.

    If the issue follows the CI failure template, produce a structured prompt.
    Otherwise fall back to the raw issue body.
    """
    body = detail.get("body", "")
    sections = _parse_issue_sections(body)

    # If we got structured sections, build a focused prompt
    if sections.get("Failed Tests"):
        parts = []

        if sections.get("Failure Type"):
            parts.append(f"**Failure Type:** {sections['Failure Type']}")

        if sections.get("Commit Scope"):
            parts.append(f"### Commit Scope\n{sections['Commit Scope']}")

        parts.append(f"### Failed Tests\n{sections['Failed Tests']}")

        if sections.get("Error Log"):
            # Truncate very long logs to keep prompt focused
            error_log = sections["Error Log"]
            lines = error_log.split("\n")
            if len(lines) > 80:
                error_log = "\n".join(lines[-80:])
                error_log = f"... (truncated, showing last 80 lines)\n{error_log}"
            parts.append(f"### Error Log\n```\n{error_log}\n```")

        if sections.get("CI Job URL"):
            parts.append(f"**CI Job URL:** {sections['CI Job URL']}")

        if sections.get("Repro Commands"):
            parts.append(f"### Repro Commands\n```bash\n{sections['Repro Commands']}\n```")

        return "\n\n".join(parts)

    # Fallback: raw body
    return body


def run(tracked: TrackedIssue) -> None:
    """Implement a fix: branch, dispatch agent, push."""
    # Idempotency
    if tracked.stage != "IMPLEMENTING":
        log("INFO", f"Issue #{tracked.source_number} not in IMPLEMENTING stage, skipping",
            issue=tracked.source_number)
        return

    # Escalation check
    tracked.attempt_count += 1
    if tracked.attempt_count > MAX_AGENT_ATTEMPTS:
        update_stage(tracked, "NEEDS_HUMAN",
                     f"Exceeded {MAX_AGENT_ATTEMPTS} implementation attempts. Needs human.")
        return
    save_state(tracked)

    branch = f"agent/issue-{tracked.source_number}"
    tracked.branch = branch
    # Sync review/main with the latest upstream main to avoid divergence
    git("fetch", "upstream", issue=tracked.source_number)
    git("fetch", REVIEW_REMOTE, issue=tracked.source_number)
    try:
        # Fast-forward review/main to upstream/main so PRs only show our commits
        git("push", REVIEW_REMOTE, "upstream/main:main", issue=tracked.source_number)
        git("fetch", REVIEW_REMOTE, "main", issue=tracked.source_number)
    except CalledProcessError:
        # review/main might be protected or have diverged; log and continue
        log("WARN", "Could not sync review/main with upstream/main — PR may show extra commits",
            issue=tracked.source_number)
    try:
        git("checkout", "-b", branch, f"{REVIEW_REMOTE}/main", issue=tracked.source_number)
    except CalledProcessError:
        # Branch already exists — just checkout (do NOT reset, preserves prior commits)
        git("checkout", branch, issue=tracked.source_number)

    # Get issue details for prompt
    detail = gh.get_issue_detail(UPSTREAM_ISSUE_REPO, tracked.source_number)

    # Check if a prior run already produced changes (e.g. pipeline crashed after
    # opencode finished but before push/PR).  Skip re-running the agent.
    existing_diff = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                            issue=tracked.source_number).strip()
    if existing_diff:
        log("INFO", f"Branch {branch} already has changes vs {REVIEW_REMOTE}/main, "
                     "skipping agent re-run",
            issue=tracked.source_number)
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **Resuming pipeline** — branch `{branch}` already has changes from a "
            f"prior run, skipping to push + PR creation.",
        )
    else:
        body_section = _build_body_section(detail)
        prompt = IMPLEMENT_PROMPT_TEMPLATE.format(
            number=tracked.source_number,
            title=detail.get("title", ""),
            body_section=body_section,
        )

        # Announce that the agent is starting work
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **Agent starting implementation** (attempt {tracked.attempt_count})\n\n"
            f"Branch: `{branch}`\n"
            f"Working on: `{detail.get('title', 'unknown')}`\n\n"
            f"_Session ID will be posted shortly — you can attach to watch live._",
        )

        # Dispatch agent — post session ID to issue as soon as it's available
        from ..utils.notify import post_session_started

        def _post_session_id(sid: str):
            post_session_started(UPSTREAM_ISSUE_REPO, tracked.source_number,
                                 "Implementation", sid, str(PYTORCH_DIR))

        backend = get_backend()
        timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 1800)
        output, log_path, session_id = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                        skill="xpu-ops-pr-creation", timeout=timeout,
                                        issue=tracked.source_number, stage="IMPLEMENTING",
                                        on_session_start=_post_session_id)
        log("INFO", f"Implementation agent log: {log_path}",
            issue=tracked.source_number)

        # Post log + session info to source issue
        from ..utils.notify import post_agent_completed
        post_agent_completed(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"Attempt {tracked.attempt_count} completed", log_path, output,
        )

    # Verify agent made changes (committed or uncommitted)
    # Auto-commit uncommitted changes (excluding third_party/*)
    add_and_commit(
        f"Fix for intel/torch-xpu-ops#{tracked.source_number}\n\n"
        f"{detail.get('title', 'Agent fix')}",
        issue=tracked.source_number,
    )

    # Now check if we have commits above review/main
    diff = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                   issue=tracked.source_number).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{tracked.source_number}, "
                     f"attempt {tracked.attempt_count}", issue=tracked.source_number)
        update_stage(tracked, "IMPLEMENTING",
                     f"Attempt {tracked.attempt_count}: agent produced no changes, will retry.")
        return

    # Squash all agent commits into one clean commit for a tidy PR
    commit_count = git_out("rev-list", "--count", f"{REVIEW_REMOTE}/main..HEAD",
                           issue=tracked.source_number).strip()
    if int(commit_count) > 1:
        log("INFO", f"Squashing {commit_count} commits into one", issue=tracked.source_number)
        git("reset", "--soft", f"{REVIEW_REMOTE}/main", issue=tracked.source_number)
        git("commit", "-m",
            f"{detail.get('title', f'Fix for issue #{tracked.source_number}')}\n\n"
            f"Fixes intel/torch-xpu-ops#{tracked.source_number}",
            issue=tracked.source_number)

    # Push to review remote
    try:
        git("push", REVIEW_REMOTE, branch, issue=tracked.source_number)
    except CalledProcessError:
        # Branch may not exist yet or diverged from squash — force push is safe
        # since we haven't created the PR yet (or are updating pre-review)
        git("push", "--force-with-lease", "--set-upstream", REVIEW_REMOTE, branch,
            issue=tracked.source_number)

    # Get the push SHA
    sha = git_out("rev-parse", "HEAD", issue=tracked.source_number).strip()
    tracked.last_push_sha = sha

    # Build descriptive PR body from issue details + diff summary
    from ._issue_format import build_pr_body
    diff_stat = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                        issue=tracked.source_number).strip()

    pr_title = detail.get("title", f"Fix for issue #{tracked.source_number}")
    pr_body = build_pr_body(
        upstream_issue_repo=UPSTREAM_ISSUE_REPO,
        source_number=tracked.source_number,
        title=detail.get("title", "N/A"),
        triage_reason=tracked.triage_reason,
        issue_body=detail.get("body", ""),
        include_diff_stat=True,
        diff_stat=diff_stat,
    )
    try:
        pr = gh.create_draft_pr(
            PRIVATE_REVIEW_REPO, title=pr_title, body=pr_body,
            head=branch,
        )
    except CalledProcessError:
        # PR may already exist for this branch — find and reuse it
        existing = gh.list_prs(PRIVATE_REVIEW_REPO, state="open",
                               search=f"head:{branch}")
        if existing:
            pr = existing[0]
            # Update the body to reflect the new push
            gh.update_pr_body(PRIVATE_REVIEW_REPO, pr["number"], pr_body)
        else:
            raise
    tracked.tracking_pr_number = pr.get("number")
    tracked.tracking_pr_url = pr.get("html_url", pr.get("url"))

    # Mark ready for review
    gh.mark_pr_ready(PRIVATE_REVIEW_REPO, tracked.tracking_pr_number)

    update_stage(tracked, "IN_REVIEW",
                 f"Implementation pushed to `{branch}`, PR created: {tracked.tracking_pr_url}")
    log("INFO", f"Implementation complete for #{tracked.source_number}, "
                 f"attempt {tracked.attempt_count}", issue=tracked.source_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
