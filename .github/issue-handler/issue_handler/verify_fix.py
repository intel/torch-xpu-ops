"""Post-PR verification — check out the PR branch and run tests locally.

Entry point:
  python -m issue_handler.verify_fix --issue 123

Runs when an issue is at IN_REVIEW stage. Checks out the PR branch,
builds if needed (C++ changes), and runs the reproducer tests.

If tests pass  → checks "Fix verified locally" action item.
If tests fail  → comments with failure details, moves to NEEDS_HUMAN.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .utils import git as gh
from .utils.config import (
    ISSUE_REPO, PYTORCH_DIR, TORCH_XPU_OPS_DIR,
    PRIVATE_REVIEW_REPO, REVIEW_REMOTE,
)
from .utils.body_templates import (
    get_status, set_status, get_metadata, parse_sections,
    check_action_item, append_log,
)
from .utils.git import git, git_out
from .utils.logger import log
from .verify_existence import _get_test_command, _run_test


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _needs_rebuild(diff_files: str) -> bool:
    """Check if any C++/SYCL files were modified that require a rebuild."""
    cpp_extensions = {'.cpp', '.h', '.cu', '.cuh', '.hpp', '.sycl'}
    return any(
        Path(f).suffix in cpp_extensions
        for f in diff_files.splitlines()
        if f.strip()
    )


def _rebuild_pytorch(workdir: Path, issue: int) -> tuple[bool, str]:
    """Run incremental rebuild via python setup.py develop.

    This handles CMake reconfiguration and incremental builds correctly.
    """
    log("INFO", "Running incremental pytorch rebuild", issue=issue)
    env_setup = (
        "source ~/intel/oneapi/setvars.sh --force 2>/dev/null; "
        "source ~/pytorch/.venv/bin/activate; "
    )
    cmd = env_setup + f"cd {workdir} && TORCH_XPU_ARCH_LIST=pvc USE_XPU=1 python setup.py develop"
    try:
        result = subprocess.run(
            cmd,
            cwd=str(workdir), capture_output=True, text=True,
            timeout=1800, shell=True, executable="/bin/bash",
        )
        output = (result.stdout + result.stderr)[-5000:]
        if result.returncode == 0:
            return True, output
        return False, output
    except subprocess.TimeoutExpired:
        return False, "Build timed out (30min)"


# ---------------------------------------------------------------------------
# Branch checkout helpers
# ---------------------------------------------------------------------------

def _checkout_pytorch_pr(branch: str, issue: int) -> tuple[Path, str]:
    """Fetch and check out a pytorch PR branch. Returns (workdir, base_ref).

    Uses a detached checkout to avoid disturbing the main working tree state.
    """
    workdir = PYTORCH_DIR
    remote = REVIEW_REMOTE
    base_ref = "main"

    git("fetch", remote, workdir=workdir, issue=issue)
    # Checkout the PR branch
    git("checkout", f"{remote}/{branch}", workdir=workdir, issue=issue)

    return workdir, base_ref


def _checkout_xpu_ops_pr(branch: str, issue: int) -> tuple[Path, str]:
    """Fetch and check out a torch-xpu-ops PR branch via worktree.

    Creates a temporary worktree for verification so we don't disturb
    the main repo checkout.

    For torch-xpu-ops, tests run from ~/pytorch, so we also need to
    sync the fix into third_party/torch-xpu-ops/.
    """
    base_ref = "main"

    # Fetch the branch
    git("fetch", "origin", branch, workdir=TORCH_XPU_OPS_DIR, issue=issue)

    # Create a temp worktree
    worktree_dir = Path(tempfile.mkdtemp(
        prefix=f"verify-{issue}-", dir="/tmp"))
    try:
        git("worktree", "add", str(worktree_dir), f"origin/{branch}",
            "--detach", workdir=TORCH_XPU_OPS_DIR, issue=issue)
    except subprocess.CalledProcessError:
        shutil.rmtree(worktree_dir, ignore_errors=True)
        raise

    return worktree_dir, base_ref


def _sync_to_pytorch(worktree_dir: Path, branch: str, issue: int) -> None:
    """Copy only the changed files from the PR branch into pytorch's submodule.

    Instead of doing a full checkout (which changes mtimes on all files and
    triggers a massive ninja rebuild), we copy only the files that differ
    between main and the PR branch. This keeps ninja's incremental build
    fast — only the actually-changed translation units get recompiled.
    """
    submodule = PYTORCH_DIR / "third_party" / "torch-xpu-ops"
    if not submodule.exists():
        log("WARN", f"third_party/torch-xpu-ops not found at {submodule}",
            issue=issue)
        return

    # Get list of changed files relative to origin/main
    diff_files = git_out(
        "diff", "--name-only", "origin/main..HEAD",
        workdir=worktree_dir, issue=issue,
    ).strip()

    if not diff_files:
        log("INFO", "No changed files to sync", issue=issue)
        return

    copied = 0
    for fname in diff_files.splitlines():
        fname = fname.strip()
        if not fname:
            continue
        src = worktree_dir / fname
        dst = submodule / fname
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            copied += 1
            log("INFO", f"Synced {fname} → third_party/torch-xpu-ops/",
                issue=issue)
    log("INFO", f"Synced {copied} file(s) from PR branch", issue=issue)


def _restore_submodule(issue: int) -> None:
    """Restore third_party/torch-xpu-ops to its clean state."""
    submodule = PYTORCH_DIR / "third_party" / "torch-xpu-ops"
    if submodule.exists():
        git("checkout", ".", workdir=submodule, issue=issue)
        log("INFO", "Restored third_party/torch-xpu-ops to clean state",
            issue=issue)


def _cleanup_xpu_ops_worktree(worktree_dir: Path, issue: int) -> None:
    """Remove the temporary worktree."""
    git("worktree", "remove", str(worktree_dir), check=False,
        workdir=TORCH_XPU_OPS_DIR, issue=issue)
    shutil.rmtree(worktree_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main verification logic
# ---------------------------------------------------------------------------

def run(issue_number: int) -> bool:
    """Verify a PR fix locally.

    Returns True if verification passed, False otherwise.
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    # --- Stage check ---
    status = get_status(body)
    if status != "IN_REVIEW":
        log("INFO", f"Issue #{issue_number} not in IN_REVIEW stage ({status}), "
            "skipping verification", issue=issue_number)
        return False

    # Already verified?
    sections = parse_sections(body)
    action_items = sections.get("Action Items", "")
    if re.search(r"\[x\].*Fix verified locally", action_items):
        log("INFO", f"Issue #{issue_number} already verified locally, skipping",
            issue=issue_number)
        return True

    # --- Get test command ---
    test_cmd = _get_test_command(body)
    if not test_cmd:
        log("WARN", f"No test command found for #{issue_number}",
            issue=issue_number)
        new_body = append_log(
            body, "verification",
            "⚠️ **Verification skipped** — no test command found in issue body.\n"
            "Manual verification required.",
        )
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        return False

    # --- Determine target and branch ---
    target_repo = get_metadata(body, "target_repo") or "pytorch"
    branch = f"agent/issue-{issue_number}"

    # Check if a tracking PR exists
    tracking_pr = get_metadata(body, "tracking_pr")
    if not tracking_pr:
        log("WARN", f"No tracking PR for #{issue_number}", issue=issue_number)
        return False

    log("INFO", f"Verifying #{issue_number}: target={target_repo}, "
        f"branch={branch}, test_cmd={test_cmd[:200]}",
        issue=issue_number)

    # --- Checkout PR branch ---
    worktree_dir = None
    _did_rebuild = False  # Track if we rebuilt so finally can restore build state
    try:
        if target_repo == "torch-xpu-ops":
            worktree_dir, base_ref = _checkout_xpu_ops_pr(
                branch, issue_number)
            # Sync changes into pytorch's third_party
            _sync_to_pytorch(worktree_dir, branch, issue_number)
            # Tests always run from pytorch dir
            test_workdir = PYTORCH_DIR
            # Check if we need a rebuild
            diff_files = git_out(
                "diff", "--name-only", f"origin/main..HEAD",
                workdir=worktree_dir, issue=issue_number,
            ).strip()
        else:
            test_workdir, base_ref = _checkout_pytorch_pr(
                branch, issue_number)
            remote = REVIEW_REMOTE
            diff_files = git_out(
                "diff", "--name-only",
                f"{remote}/{base_ref}..{remote}/{branch}",
                workdir=test_workdir, issue=issue_number,
            ).strip()

        # --- Build if needed ---
        if diff_files and _needs_rebuild(diff_files):
            do_build = True
        else:
            do_build = False

        if do_build:
            log("INFO", "C++ files changed, rebuilding", issue=issue_number)
            build_ok, build_output = _rebuild_pytorch(
                PYTORCH_DIR, issue_number)
            _did_rebuild = True
            if not build_ok:
                log("ERROR", f"Rebuild failed for #{issue_number}",
                    issue=issue_number)
                new_body = set_status(body, "NEEDS_HUMAN")
                new_body = append_log(
                    new_body, "verification",
                    f"❌ **Verification failed** — rebuild error.\n\n"
                    f"```\n{build_output[-2000:]}\n```",
                )
                gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
                gh.add_issue_comment(
                    ISSUE_REPO, issue_number,
                    f"❌ **Local verification failed** — rebuild error.\n\n"
                    f"The PR branch `{branch}` modifies C++ files but the "
                    f"rebuild failed. Manual investigation needed.",
                )
                return False

        # --- Run tests ---
        passed, output = _run_test(test_workdir, test_cmd, issue_number)

        if passed:
            log("INFO", f"Verification PASSED for #{issue_number}",
                issue=issue_number)
            new_body = check_action_item(body, "Fix verified locally")
            new_body = append_log(
                new_body, "verification",
                f"✅ **Fix verified locally**\n"
                f"Test command: `{test_cmd[:200]}`\n"
                f"Result: PASSED",
            )
            gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
            gh.add_issue_comment(
                ISSUE_REPO, issue_number,
                f"✅ **Local verification passed**\n\n"
                f"```\n{test_cmd}\n```\n\n"
                f"The fix has been verified locally. Ready for human review.\n\n"
                f"<details><summary>Test output (last 2000 chars)</summary>\n\n"
                f"```\n{output[-2000:]}\n```\n</details>",
            )
            return True
        else:
            log("WARN", f"Verification FAILED for #{issue_number}",
                issue=issue_number)
            new_body = set_status(body, "NEEDS_HUMAN")
            # Mark action item as checked with failure indicator
            new_body = new_body.replace(
                "- [ ] ✅ Fix verified locally",
                "- [x] ❌ Fix verified locally — FAILED",
            )
            new_body = append_log(
                new_body, "verification",
                f"❌ **Verification failed**\n"
                f"Test command: `{test_cmd[:200]}`\n"
                f"Result: FAILED\n\n"
                f"```\n{output[-2000:]}\n```",
            )
            gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
            gh.add_issue_comment(
                ISSUE_REPO, issue_number,
                f"❌ **Local verification failed**\n\n"
                f"The fix in PR branch `{branch}` does not pass the "
                f"reproducer test:\n"
                f"```\n{test_cmd}\n```\n\n"
                f"Moving to NEEDS_HUMAN for investigation.\n\n"
                f"<details><summary>Test output (last 2000 chars)</summary>\n\n"
                f"```\n{output[-2000:]}\n```\n</details>",
            )
            return False

    finally:
        # Cleanup
        if worktree_dir and worktree_dir.exists():
            _cleanup_xpu_ops_worktree(worktree_dir, issue_number)
            _restore_submodule(issue_number)
            # If we rebuilt with the PR's C++ changes, restore the build to
            # main state so the next issue starts from a clean binary.
            if _did_rebuild:
                log("INFO",
                    "Rebuild happened — restoring build to main state",
                    issue=issue_number)
                _rebuild_pytorch(PYTORCH_DIR, issue_number)
        elif target_repo == "pytorch":
            # Restore pytorch to its previous state
            try:
                git("checkout", "-", workdir=PYTORCH_DIR, issue=issue_number)
            except subprocess.CalledProcessError:
                pass  # Best effort
            # If we rebuilt with the PR's C++ changes, rebuild from restored
            # main source so the next issue starts from a clean binary.
            if _did_rebuild:
                log("INFO",
                    "Rebuild happened — restoring build to main state",
                    issue=issue_number)
                _rebuild_pytorch(PYTORCH_DIR, issue_number)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a PR fix locally")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    passed = run(args.issue)
    print(f"Issue #{args.issue}: {'VERIFIED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
