# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Implement a fix for a triaged issue.

Entry point:
  python -m issue_handler.code_fix --issue 123

Slim wrapper: reads structured issue body (which already has root cause,
fix strategy, reproducer), calls LLM with fix skill, then handles
git ops (branch, commit, squash, push, PR creation).

Supports two target repos:
  - pytorch (default): branch/push to REVIEW_REMOTE (chuanqi129/pytorch)
    - torch-xpu-ops: branch/push to origin (intel/torch-xpu-ops)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from subprocess import CalledProcessError

from .utils import git as gh
from .utils.config import (
    ISSUE_REPO, MAX_AGENT_ATTEMPTS, STAGE_TIMEOUTS, TORCH_XPU_OPS_DIR,
)
from .utils.body_templates import (
    set_status, check_action_item, append_log, set_metadata,
    get_status, get_metadata, parse_sections, render_pr_body,
)
from .utils.agent_backend import get_backend, TokenUsage
from .utils.build import incremental_build
from .utils.git import git, git_out, add_and_commit
from .utils.locks import pytorch_lock
from .utils.logger import log
from .utils.notify import post_agent_completed, post_session_started
from .utils.stages import RepoProfile, Skill, Stage, TargetRepo, repo_profile
from .utils.verification import extract_test_command, run_test


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _run_build(profile: RepoProfile, workdir: Path,
               issue: int) -> tuple[bool, str]:
    """Run incremental build if C++/SYCL files were modified.

    Thin wrapper around ``utils.build.incremental_build`` that keeps the
    torch-xpu-ops early return here (no native binaries on that side of
    the worktree). All real build mechanics — diff vs base_ref,
    incremental→clean fallback, torch-importable guard, build marker
    write — live in the shared helper so ``verify_upstream_pr`` and
    ``verify_existence`` get the same behaviour.
    """
    if profile.target is TargetRepo.TORCH_XPU_OPS:
        return True, ""
    return incremental_build(
        workdir=workdir,
        base_ref=profile.diff_base,
        issue=issue,
    )


# ---------------------------------------------------------------------------
# Repo / branch setup
# ---------------------------------------------------------------------------

def _detect_target_repo(body: str) -> TargetRepo:
    """Determine target repo from triage metadata or fix strategy heuristic."""
    raw = get_metadata(body, "target_repo")
    if raw and raw.strip():
        try:
            return TargetRepo(raw.strip().lower())
        except ValueError:
            pass
    # Fallback: infer from fix strategy section
    sections = parse_sections(body)
    fix_strategy = sections.get("Proposed Fix Strategy", "").lower()
    if "src/aten/native/xpu" in fix_strategy or "torch-xpu-ops" in fix_strategy:
        return TargetRepo.TORCH_XPU_OPS
    return TargetRepo.PYTORCH


def _setup_xpu_ops_worktree(branch: str, issue: int) -> Path:
    """Create a per-issue worktree on the agent branch in torch-xpu-ops."""
    git("fetch", "origin", workdir=TORCH_XPU_OPS_DIR, issue=issue)

    def _mkworktree() -> Path:
        return Path(tempfile.mkdtemp(
            prefix=f"agent-fix-{issue}-",
            dir=os.environ.get("AGENTIC_XPU_TMP_DIR", tempfile.gettempdir()),
        ))

    worktree = _mkworktree()
    try:
        git("worktree", "add", str(worktree), "-b", branch,
            "origin/main", workdir=TORCH_XPU_OPS_DIR, issue=issue)
    except CalledProcessError:
        # Branch exists — attach worktree to it (fresh tempdir because the
        # failed call may have created the directory).
        shutil.rmtree(worktree, ignore_errors=True)
        worktree = _mkworktree()
        git("worktree", "add", str(worktree), branch,
            workdir=TORCH_XPU_OPS_DIR, issue=issue)
    return worktree


def _setup_pytorch_branch(profile: RepoProfile, branch: str, issue: int) -> None:
    """Fetch upstream and check out / create the agent branch off upstream/main.

    The agent fix branch is later pushed to ``review`` (chuanqi129/pytorch)
    for PR; we deliberately do NOT push upstream/main into the review fork
    here. Keeping the fork's main untouched avoids non-fast-forward errors
    when the fork has diverged, and the agent branch built off
    ``upstream/main`` is what reviewers actually care about.

    Caller must already hold the ``pytorch_lock``.
    """
    workdir = profile.workdir

    # Upstream fetch can be very slow for large repos — use timeout
    try:
        subprocess.run(
            ["git", "fetch", "upstream", "--depth=1"],
            cwd=str(workdir), capture_output=True, text=True,
            check=True, timeout=60,
        )
    except (CalledProcessError, subprocess.TimeoutExpired):
        log("WARN", "git fetch upstream timed out or failed, using existing refs",
            issue=issue)

    # Clean working tree before switching branches (prior runs may leave
    # uncommitted changes that block checkout)
    git("checkout", ".", workdir=workdir, issue=issue)
    try:
        git("checkout", "-b", branch, "upstream/main",
            workdir=workdir, issue=issue)
    except CalledProcessError:
        git("checkout", branch, workdir=workdir, issue=issue)


def _cleanup_xpu_ops_worktree(worktree: Path, issue: int) -> None:
    git("worktree", "remove", str(worktree), check=False,
        workdir=TORCH_XPU_OPS_DIR, issue=issue)
    shutil.rmtree(worktree, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(issue_number: int) -> None:
    """Implement a fix: branch, dispatch agent, verify, push, create PR."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    # Idempotency check
    status = get_status(body)
    if status != Stage.IMPLEMENTING:
        log("INFO", f"Issue #{issue_number} not in IMPLEMENTING stage ({status}), skipping",
            issue=issue_number)
        return

    target = _detect_target_repo(body)
    profile = repo_profile(target)

    log("INFO", f"Target repo for #{issue_number}: {target} "
        f"(workdir={profile.workdir}, remote={profile.remote})",
        issue=issue_number)

    if target is TargetRepo.TORCH_XPU_OPS:
        # torch-xpu-ops fixes run in a per-issue worktree, so they don't
        # need the pytorch lock for the agent loop itself.  We'll grab it
        # later only if we need to verify against pytorch.
        _run_xpu_ops_fix(issue_number, body, detail, profile)
    else:
        with pytorch_lock(issue=issue_number):
            _run_pytorch_fix(issue_number, body, detail, profile)


def _run_xpu_ops_fix(issue_number: int, body: str, detail: dict,
                     profile: RepoProfile) -> None:
    branch = f"agent/issue-{issue_number}"
    worktree = _setup_xpu_ops_worktree(branch, issue_number)
    try:
        _execute_fix(issue_number, body, detail, profile, branch,
                     workdir=worktree)
    finally:
        _cleanup_xpu_ops_worktree(worktree, issue_number)


def _run_pytorch_fix(issue_number: int, body: str, detail: dict,
                     profile: RepoProfile) -> None:
    branch = f"agent/issue-{issue_number}"
    _setup_pytorch_branch(profile, branch, issue_number)
    _execute_fix(issue_number, body, detail, profile, branch,
                 workdir=profile.workdir)


def _execute_fix(issue_number: int, body: str, detail: dict,
                 profile: RepoProfile, branch: str, *,
                 workdir: Path) -> None:
    """Agent fix loop + push + PR.  Caller has prepared ``workdir``."""

    remote = profile.remote
    diff_base = profile.diff_base
    pr_repo = profile.pr_repo

    # --- Check for prior changes ---
    verified = False
    verification_error = ""
    token_usage = TokenUsage()
    log_path = None
    output = ""

    existing_diff = git_out(
        "diff", "--stat", f"{diff_base}..HEAD",
        workdir=workdir, issue=issue_number,
    ).strip()

    should_run_agent = not existing_diff

    if existing_diff:
        log("INFO", f"Branch {branch} already has changes, skipping agent re-run",
            issue=issue_number)

        # Still verify the existing fix
        test_cmd = extract_test_command(body, executable=True)
        if test_cmd:
            log("INFO", f"Verifying existing fix: {test_cmd[:200]}",
                issue=issue_number)
            build_ok, build_output = _run_build(
                profile, workdir, issue_number)
            if build_ok:
                verify_ok, verify_output = run_test(
                    workdir, test_cmd, issue=issue_number)
                if verify_ok:
                    log("INFO", "Existing fix verification PASSED",
                        issue=issue_number)
                    verified = True
                else:
                    log("WARN", "Existing fix verification FAILED — resetting "
                        "branch and re-running agent",
                        issue=issue_number)
                    verification_error = verify_output
                    git("reset", "--hard", f"{diff_base}",
                        workdir=workdir, issue=issue_number)
                    should_run_agent = True
            else:
                log("WARN", "Existing fix build FAILED — resetting",
                    issue=issue_number)
                git("reset", "--hard", f"{diff_base}",
                    workdir=workdir, issue=issue_number)
                should_run_agent = True
        else:
            log("INFO", "No verification command, skipping verification",
                issue=issue_number)

    if should_run_agent:
        verified, verification_error, token_usage, log_path, output = (
            _agent_fix_loop(issue_number, body, detail, profile, branch,
                            workdir=workdir)
        )

    # --- Final diff check ---
    diff = git_out("diff", "--stat", f"{diff_base}..HEAD",
                   workdir=workdir, issue=issue_number).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{issue_number}",
            issue=issue_number)
        return

    # --- Squash ---
    commit_count = git_out("rev-list", "--count", f"{diff_base}..HEAD",
                           workdir=workdir, issue=issue_number).strip()
    if int(commit_count) > 1:
        log("INFO", f"Squashing {commit_count} commits", issue=issue_number)
        git("reset", "--soft", f"{diff_base}",
            workdir=workdir, issue=issue_number)
        git("commit", "-m",
            f"{detail.get('title', f'Fix for issue #{issue_number}')}\n\n"
            f"Fixes {ISSUE_REPO}#{issue_number}",
            workdir=workdir, issue=issue_number)

    # --- Push (force-with-lease for agent branches — prior retry attempts
    # may have pushed commits that were then reset, causing divergence) ---
    try:
        git("push", "--set-upstream", remote, branch,
            workdir=workdir, issue=issue_number)
    except CalledProcessError as e:
        log("WARN", f"Normal push failed ({e.stderr.strip() if e.stderr else e}), "
            f"using --force-with-lease for agent branch {branch}",
            issue=issue_number)
        git("push", "--force-with-lease", "--set-upstream", remote, branch,
            workdir=workdir, issue=issue_number)

    sha = git_out("rev-parse", "HEAD", workdir=workdir,
                  issue=issue_number).strip()

    # --- PR creation ---
    diff_stat = git_out("diff", "--stat", f"{diff_base}..HEAD",
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
            # Best-effort body refresh — fine-grained PATs without
            # pull_requests:write on the fork return 403 here. Don't kill
            # the run; the PR already exists and code is pushed.
            try:
                gh.update_pr_body(pr_repo, pr["number"], pr_body)
            except CalledProcessError as e:
                log("WARN",
                    f"Could not refresh PR #{pr['number']} body "
                    f"(continuing — PR exists, branch pushed): "
                    f"{e.stderr.strip() if e.stderr else e}",
                    issue=issue_number)
        else:
            raise

    # Keep PR as draft until review passes (don't call mark_pr_ready)

    # Add disable_all label for torch-xpu-ops PRs (skip CI)
    if profile.target is TargetRepo.TORCH_XPU_OPS:
        gh.add_label(pr_repo, pr["number"], "disable_all")

    # --- Update issue body ---
    new_body = body
    tracking_pr_num = pr.get("number")
    new_body = set_metadata(new_body, "tracking_pr", f"#{tracking_pr_num}")
    new_body = set_metadata(new_body, "last_push_sha", sha)
    new_body = check_action_item(new_body, "Fix implemented")
    if verified:
        new_body = check_action_item(new_body, "Fix verified locally")
    new_body = check_action_item(new_body, "PR proposed")
    if verified:
        new_body = set_status(new_body, Stage.IN_REVIEW)
    else:
        new_body = set_status(new_body, Stage.NEEDS_HUMAN)

    verify_note = "✅ Verified" if verified else "⚠️ Not verified (no test cmd or all attempts failed)"
    verify_detail = ""
    if not verified and verification_error:
        truncated = verification_error[-3000:]
        verify_detail = f"\n<details><summary>Last verification output</summary>\n\n```\n{truncated}\n```\n</details>\n"
    new_body = append_log(
        new_body, "fix",
        f"Target: `{profile.target}`\nBranch: `{branch}`\nSHA: `{sha}`\n"
        f"PR: {pr.get('html_url', pr.get('url', 'N/A'))}\n"
        f"Verification: {verify_note}\n"
        f"{verify_detail}"
        f"**Tokens:** {token_usage.summary()}",
    )
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"Implementation complete for #{issue_number}",
        issue=issue_number)


def _agent_fix_loop(issue_number: int, body: str, detail: dict,
                    profile: RepoProfile, branch: str,
                    *, workdir: Path
                    ) -> tuple[bool, str, TokenUsage, Path | None, str]:
    """Repeatedly call the fix agent + verify until success or MAX attempts.

    Returns (verified, verification_error, token_usage, log_path, last_output).
    """
    remote = profile.remote
    diff_base = profile.diff_base

    test_cmd = extract_test_command(body, executable=True)
    if test_cmd:
        log("INFO", f"Verification command: {test_cmd[:200]}",
            issue=issue_number)
    else:
        log("INFO", "No verification command found, will skip verification",
            issue=issue_number)

    def _post_session_id(sid: str) -> None:
        post_session_started(ISSUE_REPO, issue_number,
                             "Implementation", sid, str(workdir))

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 3600)
    verification_error = ""
    verified = False
    token_usage = TokenUsage()
    log_path: Path | None = None
    output = ""

    for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
        prompt = (
            f"Read the issue-fix skill and fix issue #{issue_number}.\n\n"
            f"## Issue #{issue_number}: {detail.get('title', '')}\n\n"
            f"{body[:10000]}"
        )
        if verification_error:
            prompt += (
                f"\n\n---\n"
                f"## PREVIOUS FIX ATTEMPT FAILED VERIFICATION\n\n"
                f"Your previous fix attempt FAILED verification.\n\n"
                f"Test command: `{test_cmd}`\n\n"
                f"Test output (last 200 lines):\n```\n"
                f"{verification_error[-5000:]}\n```\n\n"
                f"Please analyze the failure and produce a corrected fix. "
                f"Do NOT repeat the same approach.\n"
            )

        try:
            output, log_path, session_id, token_usage = backend.run(
                prompt, workdir=str(workdir),
                skill=Skill.FIX, timeout=timeout,
                issue=issue_number, stage="IMPLEMENTING",
                on_session_start=_post_session_id,
            )
            log("INFO", f"Implementation agent log: {log_path} "
                f"(attempt {attempt}/{MAX_AGENT_ATTEMPTS}) | "
                f"{token_usage.summary()}",
                issue=issue_number)
        except Exception as e:
            log("WARN", f"Agent attempt {attempt}/{MAX_AGENT_ATTEMPTS} "
                f"failed: {e}", issue=issue_number)
            if attempt == MAX_AGENT_ATTEMPTS:
                raise
            continue

        # --- Commit agent changes ---
        add_and_commit(
            f"Fix for {ISSUE_REPO}#{issue_number} (attempt {attempt})\n\n"
            f"{detail.get('title', 'Agent fix')}",
            issue=issue_number,
            workdir=workdir,
        )

        # Check if agent produced changes
        diff = git_out("diff", "--stat", f"{diff_base}..HEAD",
                       workdir=workdir, issue=issue_number).strip()
        if not diff:
            log("WARN", f"Agent produced no changes (attempt {attempt})",
                issue=issue_number)
            if attempt < MAX_AGENT_ATTEMPTS:
                verification_error = "Agent produced no code changes."
                continue
            break

        # --- Build (pytorch only) ---
        build_ok, build_output = _run_build(
            profile, workdir, issue_number)
        if not build_ok:
            log("WARN", f"Build failed (attempt {attempt})",
                issue=issue_number)
            if attempt < MAX_AGENT_ATTEMPTS:
                # Reset to base for retry — applies to both target repos so a
                # bad torch-xpu-ops attempt doesn't pollute the next diff.
                git("reset", "--hard", f"{diff_base}",
                    workdir=workdir, issue=issue_number)
                verification_error = f"Build failed:\n{build_output}"
                continue
            break

        # --- Verify ---
        if not test_cmd:
            # No test command — check for agent-created repro
            repro_files = list(
                Path(workdir).glob("test/repro/test_*.py"))
            if repro_files:
                test_cmd = f"pytest -xvs {repro_files[0]}"
                log("INFO", f"Found agent repro: {test_cmd}",
                    issue=issue_number)

        if test_cmd:
            verify_ok, verify_output = run_test(
                workdir, test_cmd, issue=issue_number)
            if verify_ok:
                log("INFO", f"Verification PASSED (attempt {attempt})",
                    issue=issue_number)
                verified = True
                break
            else:
                # EXTRACTION_BUG: malformed reproducer — never a fix failure.
                # Bail immediately, do NOT reset+rebuild (would loop forever).
                if verify_output.startswith("EXTRACTION_BUG:"):
                    log("ERROR",
                        f"Verification ABORTED — extraction bug detected; "
                        f"will not rebuild. Details: {verify_output[:300]}",
                        issue=issue_number)
                    verification_error = verify_output
                    break
                log("WARN", f"Verification FAILED (attempt {attempt})",
                    issue=issue_number)
                if attempt < MAX_AGENT_ATTEMPTS:
                    git("reset", "--hard", f"{diff_base}",
                        workdir=workdir, issue=issue_number)
                    verification_error = verify_output
                    continue
                break
        else:
            log("INFO", "No verification command, skipping verification",
                issue=issue_number)
            break

    if log_path is not None:
        post_agent_completed(ISSUE_REPO, issue_number,
                             "Implementation completed", log_path, output)

    return verified, verification_error, token_usage, log_path, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
