"""Implement a fix for a triaged issue.

Entry point:
  python -m issue_handler.fixing_steps.code_fix --issue 123

Slim wrapper: reads structured issue body (which already has root cause,
fix strategy, reproducer), calls LLM with fix skill, then handles
git ops (branch, commit, squash, push, PR creation).

Supports two target repos:
  - pytorch (default): branch/push to REVIEW_REMOTE (chuanqi129/pytorch)
  - torch-xpu-ops: branch/push to origin (intel/torch-xpu-ops)
"""
from __future__ import annotations

import argparse
from subprocess import CalledProcessError

from ..utils import git as gh
from ..utils.config import (
    ISSUE_REPO, PRIVATE_REVIEW_REPO, PYTORCH_DIR, TORCH_XPU_OPS_DIR,
    REVIEW_REMOTE, MAX_AGENT_ATTEMPTS, STAGE_TIMEOUTS,
)
from ..utils.body_templates import (
    get_status, set_status, check_action_item, append_log, set_metadata,
    get_metadata, parse_sections, render_pr_body,
)
from ..utils.agent_backend import get_backend, TokenUsage
from ..utils.git import git, git_out, add_and_commit
from ..utils.logger import log
from ..utils.notify import post_agent_completed, post_session_started


def _detect_target_repo(body: str) -> str:
    """Determine target repo from triage metadata or fix strategy heuristic."""
    target = get_metadata(body, "target_repo")
    if target and target.strip():
        return target.strip().lower()
    # Fallback: infer from fix strategy section
    sections = parse_sections(body)
    fix_strategy = sections.get("Proposed Fix Strategy", "").lower()
    if "src/aten/native/xpu" in fix_strategy or "torch-xpu-ops" in fix_strategy:
        return "torch-xpu-ops"
    return "pytorch"


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

    # --- Determine target repo ---
    target_repo = _detect_target_repo(body)
    if target_repo == "torch-xpu-ops":
        workdir = TORCH_XPU_OPS_DIR
        remote = "origin"       # intel/torch-xpu-ops (direct push)
        base_ref = "main"
        pr_repo = ISSUE_REPO    # PR goes to intel/torch-xpu-ops
    else:
        workdir = PYTORCH_DIR
        remote = REVIEW_REMOTE  # chuanqi129/pytorch
        base_ref = "main"
        pr_repo = PRIVATE_REVIEW_REPO

    log("INFO", f"Target repo for #{issue_number}: {target_repo} "
        f"(workdir={workdir}, remote={remote})", issue=issue_number)

    # --- Branch setup ---
    branch = f"agent/issue-{issue_number}"

    if target_repo == "torch-xpu-ops":
        # For torch-xpu-ops: fetch origin, branch off origin/main
        git("fetch", "origin", workdir=workdir, issue=issue_number)
        try:
            git("checkout", "-b", branch, "origin/main",
                workdir=workdir, issue=issue_number)
        except CalledProcessError:
            git("checkout", branch, workdir=workdir, issue=issue_number)
    else:
        # For pytorch: existing flow (fetch upstream + review remote)
        git("fetch", "upstream", workdir=workdir, issue=issue_number)
        git("fetch", remote, workdir=workdir, issue=issue_number)
        try:
            git("push", remote, "upstream/main:main",
                workdir=workdir, issue=issue_number)
            git("fetch", remote, "main", workdir=workdir, issue=issue_number)
        except CalledProcessError:
            log("WARN", "Could not sync review/main with upstream/main",
                issue=issue_number)
        try:
            git("checkout", "-b", branch, f"{remote}/main",
                workdir=workdir, issue=issue_number)
        except CalledProcessError:
            git("checkout", branch, workdir=workdir, issue=issue_number)

    # --- Check for prior changes ---
    existing_diff = git_out("diff", "--stat", f"{remote}/{base_ref}..HEAD",
                            workdir=workdir, issue=issue_number).strip()
    if existing_diff:
        log("INFO", f"Branch {branch} already has changes, skipping agent re-run",
            issue=issue_number)
        token_usage = TokenUsage()
    else:
        # --- Call LLM (with retry) ---
        prompt = (
            f"Read the issue-fix skill and fix issue #{issue_number}.\n\n"
            f"## Issue #{issue_number}: {detail.get('title', '')}\n\n"
            f"{body[:10000]}"
        )

        def _post_session_id(sid: str):
            post_session_started(ISSUE_REPO, issue_number,
                                 "Implementation", sid, str(workdir))

        backend = get_backend()
        timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 3600)

        last_error = None
        for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
            try:
                output, log_path, session_id, token_usage = backend.run(
                    prompt, workdir=str(workdir),
                    skill="issue-fix", timeout=timeout,
                    issue=issue_number, stage="IMPLEMENTING",
                    on_session_start=_post_session_id,
                )
                log("INFO", f"Implementation agent log: {log_path} "
                    f"(attempt {attempt}/{MAX_AGENT_ATTEMPTS}) | "
                    f"{token_usage.summary()}",
                    issue=issue_number)
                break
            except Exception as e:
                last_error = e
                log("WARN", f"Agent attempt {attempt}/{MAX_AGENT_ATTEMPTS} "
                    f"failed: {e}", issue=issue_number)
                if attempt == MAX_AGENT_ATTEMPTS:
                    raise
        post_agent_completed(ISSUE_REPO, issue_number,
                             "Implementation completed", log_path, output)

    # --- Commit ---
    add_and_commit(
        f"Fix for {ISSUE_REPO}#{issue_number}\n\n"
        f"{detail.get('title', 'Agent fix')}",
        issue=issue_number,
        workdir=workdir,
    )

    diff = git_out("diff", "--stat", f"{remote}/{base_ref}..HEAD",
                   workdir=workdir, issue=issue_number).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{issue_number}",
            issue=issue_number)
        return

    # --- Squash ---
    commit_count = git_out("rev-list", "--count", f"{remote}/{base_ref}..HEAD",
                           workdir=workdir, issue=issue_number).strip()
    if int(commit_count) > 1:
        log("INFO", f"Squashing {commit_count} commits", issue=issue_number)
        git("reset", "--soft", f"{remote}/{base_ref}",
            workdir=workdir, issue=issue_number)
        git("commit", "-m",
            f"{detail.get('title', f'Fix for issue #{issue_number}')}\n\n"
            f"Fixes {ISSUE_REPO}#{issue_number}",
            workdir=workdir, issue=issue_number)

    # --- Push (never force-push — keeps commit history trackable) ---
    git("push", "--set-upstream", remote, branch,
        workdir=workdir, issue=issue_number)

    sha = git_out("rev-parse", "HEAD", workdir=workdir,
                  issue=issue_number).strip()

    # --- PR creation ---
    diff_stat = git_out("diff", "--stat", f"{remote}/{base_ref}..HEAD",
                        workdir=workdir, issue=issue_number).strip()
    sections = parse_sections(body)
    pr_title = detail.get("title", f"Fix for issue #{issue_number}")
    pr_body = render_pr_body(
        upstream_issue_repo=ISSUE_REPO,
        source_number=issue_number,
        title=detail.get("title", "N/A"),
        triage_reason=sections.get("Root Cause Analysis", ""),
        issue_body=body,
        include_diff_stat=True,
        diff_stat=diff_stat,
    )

    try:
        pr = gh.create_draft_pr(pr_repo, title=pr_title,
                                body=pr_body, head=branch)
    except CalledProcessError:
        existing = gh.list_prs(pr_repo, state="open",
                               search=f"head:{branch}")
        if existing:
            pr = existing[0]
            gh.update_pr_body(pr_repo, pr["number"], pr_body)
        else:
            raise

    # Keep PR as draft until review passes (don't call mark_pr_ready)

    # --- Update issue body ---
    new_body = body
    # Record PR and SHA for downstream stages (private_review, ci_watch)
    tracking_pr_num = pr.get("number")
    new_body = set_metadata(new_body, "tracking_pr", f"#{tracking_pr_num}")
    new_body = set_metadata(new_body, "last_push_sha", sha)
    new_body = check_action_item(new_body, "Fix implemented")
    new_body = check_action_item(new_body, "PR proposed")
    new_body = set_status(new_body, "IN_REVIEW")
    new_body = append_log(
        new_body, "fix",
        f"Target: `{target_repo}`\nBranch: `{branch}`\nSHA: `{sha}`\n"
        f"PR: {pr.get('html_url', pr.get('url', 'N/A'))}\n"
        f"**Tokens:** {token_usage.summary()}",
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
