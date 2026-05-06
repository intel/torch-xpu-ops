"""Implement a fix for a triaged issue.

Entry point:
  python -m pytorch_agent.fixing_steps.implement --issue 123

Slim wrapper: reads structured issue body (which already has root cause,
fix strategy, reproducer), calls LLM with fix skill, then handles
git ops (branch, commit, squash, push, PR creation).
"""
from __future__ import annotations

import argparse
from subprocess import CalledProcessError

from ..utils import github_client as gh
from ..utils.config import (
    ISSUE_REPO, PRIVATE_REVIEW_REPO, PYTORCH_DIR,
    REVIEW_REMOTE, MAX_AGENT_ATTEMPTS, STAGE_TIMEOUTS,
)
from ..utils.issue_body import (
    get_status, set_status, check_action_item, append_log,
    parse_sections,
)
from ..utils.agent_backend import get_backend
from ..utils.git import git, git_out, add_and_commit
from ..utils.logger import log
from ..utils.notify import post_agent_completed, post_session_started
from ._issue_format import build_pr_body


IMPLEMENT_PROMPT_TEMPLATE = """Fix the following PyTorch CI failure for Intel XPU.

## Issue #{number}: {title}

{body}

## Instructions
1. Read the Root Cause Analysis and Proposed Fix Strategy above.
2. Reproduce the failure using the Reproducer commands (if provided).
3. Implement the minimal fix following the proposed strategy.
4. Run the failing test(s) listed above to verify your fix.
5. Ensure no regressions in related tests.

## HARD RULES (violations will be rejected)
- NEVER use @skipIfXpu, @skip, unittest.skip, or any skip decorator. You must FIX the test, not skip it.
- Do NOT commit submodule pointer changes (third_party/*). Use `git add` on specific files only.

Work in the current directory (~/pytorch).
"""


def run(issue_number: int) -> None:
    """Implement a fix: branch, dispatch agent, push, create PR."""
    # Read issue
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    # Idempotency check
    status = get_status(body)
    if status != "IMPLEMENTING":
        log("INFO", f"Issue #{issue_number} not in IMPLEMENTING stage ({status}), skipping",
            issue=issue_number)
        return

    # --- Branch setup ---
    branch = f"agent/issue-{issue_number}"
    git("fetch", "upstream", issue=issue_number)
    git("fetch", REVIEW_REMOTE, issue=issue_number)
    try:
        git("push", REVIEW_REMOTE, "upstream/main:main", issue=issue_number)
        git("fetch", REVIEW_REMOTE, "main", issue=issue_number)
    except CalledProcessError:
        log("WARN", "Could not sync review/main with upstream/main",
            issue=issue_number)
    try:
        git("checkout", "-b", branch, f"{REVIEW_REMOTE}/main", issue=issue_number)
    except CalledProcessError:
        git("checkout", branch, issue=issue_number)

    # --- Check for prior changes ---
    existing_diff = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                            issue=issue_number).strip()
    if existing_diff:
        log("INFO", f"Branch {branch} already has changes, skipping agent re-run",
            issue=issue_number)
    else:
        # --- Call LLM ---
        prompt = IMPLEMENT_PROMPT_TEMPLATE.format(
            number=issue_number,
            title=detail.get("title", ""),
            body=body[:10000],
        )

        def _post_session_id(sid: str):
            post_session_started(ISSUE_REPO, issue_number,
                                 "Implementation", sid, str(PYTORCH_DIR))

        backend = get_backend()
        timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 3600)
        output, log_path, session_id = backend.run(
            prompt, workdir=str(PYTORCH_DIR),
            skill="pytorch-fix", timeout=timeout,
            issue=issue_number, stage="IMPLEMENTING",
            on_session_start=_post_session_id,
        )
        log("INFO", f"Implementation agent log: {log_path}", issue=issue_number)

        post_agent_completed(ISSUE_REPO, issue_number,
                             "Implementation completed", log_path, output)

    # --- Commit ---
    add_and_commit(
        f"Fix for {ISSUE_REPO}#{issue_number}\n\n"
        f"{detail.get('title', 'Agent fix')}",
        issue=issue_number,
    )

    diff = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                   issue=issue_number).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{issue_number}",
            issue=issue_number)
        return

    # --- Squash ---
    commit_count = git_out("rev-list", "--count", f"{REVIEW_REMOTE}/main..HEAD",
                           issue=issue_number).strip()
    if int(commit_count) > 1:
        log("INFO", f"Squashing {commit_count} commits", issue=issue_number)
        git("reset", "--soft", f"{REVIEW_REMOTE}/main", issue=issue_number)
        git("commit", "-m",
            f"{detail.get('title', f'Fix for issue #{issue_number}')}\n\n"
            f"Fixes {ISSUE_REPO}#{issue_number}",
            issue=issue_number)

    # --- Push ---
    try:
        git("push", REVIEW_REMOTE, branch, issue=issue_number)
    except CalledProcessError:
        git("push", "--force-with-lease", "--set-upstream", REVIEW_REMOTE, branch,
            issue=issue_number)

    sha = git_out("rev-parse", "HEAD", issue=issue_number).strip()

    # --- PR creation ---
    diff_stat = git_out("diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD",
                        issue=issue_number).strip()
    sections = parse_sections(body)
    pr_title = detail.get("title", f"Fix for issue #{issue_number}")
    pr_body = build_pr_body(
        upstream_issue_repo=ISSUE_REPO,
        source_number=issue_number,
        title=detail.get("title", "N/A"),
        triage_reason=sections.get("Root Cause Analysis", ""),
        issue_body=body,
        include_diff_stat=True,
        diff_stat=diff_stat,
    )

    try:
        pr = gh.create_draft_pr(PRIVATE_REVIEW_REPO, title=pr_title,
                                body=pr_body, head=branch)
    except CalledProcessError:
        existing = gh.list_prs(PRIVATE_REVIEW_REPO, state="open",
                               search=f"head:{branch}")
        if existing:
            pr = existing[0]
            gh.update_pr_body(PRIVATE_REVIEW_REPO, pr["number"], pr_body)
        else:
            raise

    gh.mark_pr_ready(PRIVATE_REVIEW_REPO, pr.get("number"))

    # --- Update issue body ---
    new_body = body
    new_body = check_action_item(new_body, "Fix implemented")
    new_body = check_action_item(new_body, "Fix verified")
    new_body = set_status(new_body, "IN_REVIEW")
    new_body = append_log(
        new_body, "fix",
        f"Branch: `{branch}`\nSHA: `{sha}`\n"
        f"PR: {pr.get('html_url', pr.get('url', 'N/A'))}",
    )
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"Implementation complete for #{issue_number}",
        issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
