"""Implement a fix for a triaged issue.

Entry point:
  python -m issue_handler.fixing_steps.code_fix --issue 123

Slim wrapper: reads structured issue body (which already has root cause,
fix strategy, reproducer), calls LLM with fix skill, then handles
git ops (branch, commit, squash, push, PR creation).

Supports two target repos:
  - pytorch (default): branch/push to REVIEW_REMOTE (chuanqi129/pytorch)
  - torch-xpu-ops: branch/push to origin (intel-sandbox/torch-xpu-ops-exp)
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
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
from ..utils.xpu_env import ENV_SETUP, XPU_BUILD_FLAGS
from ..utils.logger import log
from ..utils.notify import post_agent_completed, post_session_started


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _get_test_command(body: str) -> str | None:
    """Extract test command from issue body.

    Priority:
      1. "Reproducer" section — use verbatim bash block
      2. "Failed Tests" section — construct pytest command from test names
      3. None (skip verification)
    """
    sections = parse_sections(body)

    # 1. Reproducer section
    reproducer = sections.get("Reproducer", "").strip()
    if reproducer:
        # Extract bash code block if present
        m = re.search(r"```(?:bash|sh)?\s*\n(.+?)```", reproducer, re.DOTALL)
        if m:
            cmd = m.group(1).strip()
            # Strip nested fence markers (LLM sometimes double-wraps)
            cmd = re.sub(r'^```(?:bash|sh)?\s*\n', '', cmd)
            cmd = re.sub(r'\n```\s*$', '', cmd)
            cmd = cmd.strip()
            if cmd:
                # Strip 'cd <pytorch> &&' or 'cd /path/to/pytorch &&' prefix
                # from ALL lines (multi-line reproducers have it on each line)
                lines = cmd.splitlines()
                cleaned = []
                for line in lines:
                    line = re.sub(
                        r'^cd\s+(?:<[^>]+>|/\S+)\s*&&\s*', '', line).strip()
                    if line:
                        cleaned.append(line)
                cmd = "\n".join(cleaned)
                # Path resolution is now handled by the verification agent.
                # The Reproducer section will contain the refined command.
                return cmd
        # If no code block but non-empty text that looks like a command
        if reproducer and not reproducer.startswith("_Pending"):
            lines = [l.strip() for l in reproducer.splitlines()
                     if l.strip() and not l.strip().startswith("#")
                     and not l.strip().startswith("```")]
            if lines:
                return "\n".join(lines)

    # 2. Failed Tests section
    failed_tests = sections.get("Failed Tests", "").strip()
    if failed_tests:
        # Extract test paths like `test_ops.py::TestClass::test_method`
        tests = re.findall(
            r"`([^`]+(?:::|-k\s+)[^`]*)`", failed_tests)
        if not tests:
            # Try lines starting with - that look like test paths
            tests = re.findall(
                r"[-*]\s+`?(\S+::\S+)`?", failed_tests)
        if tests:
            test_args = " ".join(f'"{t}"' for t in tests)
            return f"pytest -v {test_args}"

    return None


def _run_build(workdir: Path, target_repo: str, remote: str,
               base_ref: str, issue: int) -> tuple[bool, str]:
    """Run incremental build if C++/SYCL files were modified.

    Returns (success, output). For torch-xpu-ops, no build needed.
    """
    if target_repo == "torch-xpu-ops":
        return True, ""

    # Check if any C++/SYCL files were modified
    diff_files = git_out("diff", "--name-only", f"{remote}/{base_ref}..HEAD",
                         workdir=workdir, issue=issue).strip()
    if not diff_files:
        return True, ""

    cpp_extensions = {'.cpp', '.h', '.cu', '.cuh', '.hpp', '.sycl'}
    needs_build = any(
        Path(f).suffix in cpp_extensions
        for f in diff_files.splitlines()
    )

    if not needs_build:
        return True, ""  # Python-only change, no build needed

    log("INFO", "C++/SYCL files modified, running incremental build",
        issue=issue)

    from ..utils.xpu_env import ENV_SETUP, XPU_BUILD_FLAGS
    env_setup = ENV_SETUP
    build_cmd = f"cd {workdir} && {XPU_BUILD_FLAGS} python setup.py develop"

    # Incremental build
    try:
        result = subprocess.run(
            env_setup + build_cmd,
            cwd=str(workdir), capture_output=True, text=True,
            timeout=1800, shell=True, executable="/bin/bash",
        )
        if result.returncode == 0:
            return True, result.stdout[-2000:]
        output = (result.stdout + result.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        output = "Build timed out (30min)"
        log("WARN", "Incremental build timed out", issue=issue)
        # Fall through to clean build

    # Clean build fallback
    log("INFO", "Incremental build failed, trying clean build", issue=issue)
    try:
        clean_cmd = f"cd {workdir} && {XPU_BUILD_FLAGS} python setup.py clean && {XPU_BUILD_FLAGS} python setup.py develop"
        result = subprocess.run(
            env_setup + clean_cmd,
            cwd=str(workdir), capture_output=True, text=True,
            timeout=2400, shell=True, executable="/bin/bash",
        )
        if result.returncode == 0:
            return True, result.stdout[-2000:]
        return False, (result.stdout + result.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        return False, "Clean build timed out (40min)"


def _run_verification(workdir: Path, test_cmd: str,
                      issue: int) -> tuple[bool, str]:
    """Run verification test command. Returns (success, output)."""
    log("INFO", f"Running verification: {test_cmd[:200]}", issue=issue)
    # For multi-line reproducers, join with && so each line runs in sequence
    if "\n" in test_cmd:
        cmd = " && ".join(line.strip() for line in test_cmd.splitlines() if line.strip())
    else:
        cmd = test_cmd
    # Prepend XPU environment setup
    full_cmd = ENV_SETUP + cmd
    try:
        result = subprocess.run(
            full_cmd, cwd=str(workdir),
            capture_output=True, text=True,
            timeout=600, shell=True, executable="/bin/bash",
        )
        output = (result.stdout + result.stderr)[-5000:]
        if result.returncode == 0:
            return True, output
        return False, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out (10min)"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

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
    """Implement a fix: branch, dispatch agent, verify, push, create PR."""
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
        remote = "origin"       # intel-sandbox/torch-xpu-ops-exp (direct push)
        base_ref = "main"
        pr_repo = ISSUE_REPO    # PR goes to intel-sandbox/torch-xpu-ops-exp
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
        # For torch-xpu-ops: use a worktree so we don't disturb the pipeline's
        # checked-out branch in the main repo.
        import tempfile
        worktree_dir = Path(tempfile.mkdtemp(
            prefix=f"agent-fix-{issue_number}-", dir="/tmp"))
        git("fetch", "origin", workdir=TORCH_XPU_OPS_DIR, issue=issue_number)
        try:
            git("worktree", "add", str(worktree_dir), "-b", branch,
                "origin/main", workdir=TORCH_XPU_OPS_DIR, issue=issue_number)
        except CalledProcessError:
            # Branch exists — attach worktree to existing branch
            import shutil
            shutil.rmtree(worktree_dir, ignore_errors=True)
            worktree_dir = Path(tempfile.mkdtemp(
                prefix=f"agent-fix-{issue_number}-", dir="/tmp"))
            git("worktree", "add", str(worktree_dir), branch,
                workdir=TORCH_XPU_OPS_DIR, issue=issue_number)
        workdir = worktree_dir
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
        # Clean working tree before switching branches (prior runs may leave
        # uncommitted changes that block checkout)
        git("checkout", ".", workdir=workdir, issue=issue_number)
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

        # Still verify the existing fix
        test_cmd = _get_test_command(body)
        verified = False
        if test_cmd:
            log("INFO", f"Verifying existing fix: {test_cmd[:200]}",
                issue=issue_number)
            build_ok, build_output = _run_build(
                workdir, target_repo, remote, base_ref, issue_number)
            if build_ok:
                verify_ok, verify_output = _run_verification(
                    workdir, test_cmd, issue_number)
                if verify_ok:
                    log("INFO", "Existing fix verification PASSED",
                        issue=issue_number)
                    verified = True
                else:
                    log("WARN", "Existing fix verification FAILED — resetting "
                        "branch and re-running agent",
                        issue=issue_number)
                    git("reset", "--hard", f"{remote}/{base_ref}",
                        workdir=workdir, issue=issue_number)
                    # Fall through to the fix loop below
                    existing_diff = None
            else:
                log("WARN", "Existing fix build FAILED — resetting",
                    issue=issue_number)
                git("reset", "--hard", f"{remote}/{base_ref}",
                    workdir=workdir, issue=issue_number)
                existing_diff = None
        else:
            log("INFO", "No verification command, skipping verification",
                issue=issue_number)

    if not existing_diff:
        # --- Extract test command for verification ---
        test_cmd = _get_test_command(body)
        if test_cmd:
            log("INFO", f"Verification command: {test_cmd[:200]}",
                issue=issue_number)
        else:
            log("INFO", "No verification command found, will skip verification",
                issue=issue_number)

        # --- Fix + Verify loop ---
        def _post_session_id(sid: str):
            post_session_started(ISSUE_REPO, issue_number,
                                 "Implementation", sid, str(workdir))

        backend = get_backend()
        timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 3600)
        verification_error = ""
        verified = False

        for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
            # --- Call agent ---
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
                    skill="issue-fix", timeout=timeout,
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
            diff = git_out("diff", "--stat", f"{remote}/{base_ref}..HEAD",
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
                workdir, target_repo, remote, base_ref, issue_number)
            if not build_ok:
                log("WARN", f"Build failed (attempt {attempt})",
                    issue=issue_number)
                if attempt < MAX_AGENT_ATTEMPTS:
                    # Reset to base for retry
                    git("reset", "--hard", f"{remote}/{base_ref}",
                        workdir=workdir, issue=issue_number)
                    verification_error = (
                        f"Build failed:\n{build_output}")
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
                verify_ok, verify_output = _run_verification(
                    workdir, test_cmd, issue_number)
                if verify_ok:
                    log("INFO", f"Verification PASSED (attempt {attempt})",
                        issue=issue_number)
                    verified = True
                    break
                else:
                    log("WARN", f"Verification FAILED (attempt {attempt})",
                        issue=issue_number)
                    if attempt < MAX_AGENT_ATTEMPTS:
                        # Reset to base for retry
                        git("reset", "--hard", f"{remote}/{base_ref}",
                            workdir=workdir, issue=issue_number)
                        verification_error = verify_output
                        continue
                    # Last attempt — keep whatever we have
                    break
            else:
                log("INFO", "No verification command, skipping verification",
                    issue=issue_number)
                break

        post_agent_completed(ISSUE_REPO, issue_number,
                             "Implementation completed", log_path, output)

    # --- Final diff check ---
    diff = git_out("diff", "--stat", f"{remote}/{base_ref}..HEAD",
                   workdir=workdir, issue=issue_number).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{issue_number}",
            issue=issue_number)
        # Cleanup worktree if used
        if target_repo == "torch-xpu-ops":
            import shutil
            git("worktree", "remove", str(workdir), check=False,
                workdir=TORCH_XPU_OPS_DIR, issue=issue_number)
            shutil.rmtree(workdir, ignore_errors=True)
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

    # Add disable_all label for torch-xpu-ops PRs (skip CI)
    if target_repo == "torch-xpu-ops":
        gh.add_label(pr_repo, pr["number"], "disable_all")

    # --- Update issue body ---
    new_body = body
    # Record PR and SHA for downstream stages (private_review, ci_watch)
    tracking_pr_num = pr.get("number")
    new_body = set_metadata(new_body, "tracking_pr", f"#{tracking_pr_num}")
    new_body = set_metadata(new_body, "last_push_sha", sha)
    new_body = check_action_item(new_body, "Fix implemented")
    if verified:
        new_body = check_action_item(new_body, "Fix verified locally")
    new_body = check_action_item(new_body, "PR proposed")
    new_body = set_status(new_body, "IN_REVIEW")

    verify_note = "✅ Verified" if verified else "⚠️ Not verified (no test cmd or all attempts failed)"
    new_body = append_log(
        new_body, "fix",
        f"Target: `{target_repo}`\nBranch: `{branch}`\nSHA: `{sha}`\n"
        f"PR: {pr.get('html_url', pr.get('url', 'N/A'))}\n"
        f"Verification: {verify_note}\n"
        f"**Tokens:** {token_usage.summary()}",
    )
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"Implementation complete for #{issue_number}",
        issue=issue_number)

    # --- Cleanup worktree if used ---
    if target_repo == "torch-xpu-ops":
        import shutil
        git("worktree", "remove", str(workdir), check=False,
            workdir=TORCH_XPU_OPS_DIR, issue=issue_number)
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
