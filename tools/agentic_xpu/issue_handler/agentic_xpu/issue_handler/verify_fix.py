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
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .utils import git as gh
from .utils.config import (
    ISSUE_REPO, PYTORCH_DIR, TORCH_XPU_OPS_DIR,
    REVIEW_REMOTE,
)
from .utils.body_templates import (
    get_status, set_status, get_metadata, parse_sections,
    check_action_item, append_log,
)
from .utils.git import git, git_out
from .utils.locks import pytorch_lock
from .utils.logger import log
from .utils.stages import Stage, TargetRepo
from .utils.verification import extract_test_command, run_test
from .utils.xpu_env import ENV_SETUP, XPU_BUILD_FLAGS


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

CPP_EXTENSIONS = {'.cpp', '.h', '.cu', '.cuh', '.hpp', '.sycl'}


def _needs_rebuild(diff_files: str) -> bool:
    """Check if any C++/SYCL files were modified that require a rebuild."""
    return any(
        Path(f).suffix in CPP_EXTENSIONS
        for f in diff_files.splitlines()
        if f.strip()
    )


def _rebuild_pytorch(workdir: Path, issue: int) -> tuple[bool, str]:
    """Run incremental rebuild via python setup.py develop.

    This handles CMake reconfiguration and incremental builds correctly.
    """
    log("INFO", "Running incremental pytorch rebuild", issue=issue)
    cmd = ENV_SETUP + f"cd {workdir} && {XPU_BUILD_FLAGS} python setup.py develop"
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
    except subprocess.TimeoutExpired as e:
        # Surface partial output so reviewers can see what compiled.
        partial = ""
        for stream in (e.stdout, e.stderr):
            if not stream:
                continue
            partial += stream.decode("utf-8", errors="replace") if isinstance(
                stream, (bytes, bytearray)) else stream
        return False, f"Build timed out (30min)\n{partial[-4000:]}"


# ---------------------------------------------------------------------------
# Branch checkout helpers
# ---------------------------------------------------------------------------

def _checkout_pytorch_pr(branch: str, issue: int) -> tuple[Path, str]:
    """Fetch and check out a pytorch PR branch. Returns (workdir, base_ref).

    Uses a detached checkout to avoid disturbing the main working tree state.
    The previous HEAD is *not* captured here — callers that need to restore
    it should use :func:`_capture_head_ref` before invoking this helper, and
    :func:`_restore_head_ref` to roll back.
    """
    workdir = PYTORCH_DIR
    remote = REVIEW_REMOTE
    base_ref = "main"

    git("fetch", remote, workdir=workdir, issue=issue)
    # Checkout the PR branch
    git("checkout", f"{remote}/{branch}", workdir=workdir, issue=issue)

    return workdir, base_ref


def _capture_head_ref(workdir: Path, issue: int) -> str:
    """Capture the current HEAD as something we can return to later.

    Prefers a symbolic branch name (``main``, ``rfc/foo``, …) so subsequent
    ``git checkout`` re-attaches HEAD.  Falls back to the raw commit SHA
    when HEAD is already detached — ``git checkout -`` cannot reliably
    handle that case, which is the bug this captures around.
    """
    branch = git_out(
        "symbolic-ref", "--short", "-q", "HEAD",
        workdir=workdir, issue=issue, check=False,
    ).strip()
    if branch:
        return branch
    return git_out(
        "rev-parse", "HEAD", workdir=workdir, issue=issue,
    ).strip()


def _restore_head_ref(workdir: Path, ref: str, issue: int) -> None:
    """Best-effort restore of HEAD to a ref captured by :func:`_capture_head_ref`."""
    if not ref:
        return
    try:
        git("checkout", ref, workdir=workdir, issue=issue)
    except subprocess.CalledProcessError:
        log("WARN", f"Failed to restore HEAD to {ref!r} in {workdir}",
            issue=issue)


def _checkout_xpu_ops_pr(branch: str, issue: int) -> tuple[Path, str]:
    """Fetch and check out a torch-xpu-ops PR branch via worktree.

    Creates a temporary worktree for verification so we don't disturb
    the main repo checkout.

    For torch-xpu-ops, tests run from PYTORCH_DIR, so we also need to
    sync the fix into third_party/torch-xpu-ops/.
    """
    base_ref = "main"

    # Fetch the branch
    git("fetch", "origin", branch, workdir=TORCH_XPU_OPS_DIR, issue=issue)

    # Create a temp worktree
    worktree_dir = Path(tempfile.mkdtemp(
        prefix=f"verify-{issue}-",
        dir=os.environ.get("AGENTIC_XPU_TMP_DIR", tempfile.gettempdir()),
    ))
    try:
        git("worktree", "add", str(worktree_dir), f"origin/{branch}",
            "--detach", workdir=TORCH_XPU_OPS_DIR, issue=issue)
    except subprocess.CalledProcessError:
        shutil.rmtree(worktree_dir, ignore_errors=True)
        raise

    return worktree_dir, base_ref


def _sync_to_pytorch(worktree_dir: Path, branch: str, issue: int) -> None:
    """Mirror the PR's diff onto pytorch's torch-xpu-ops submodule.

    A full checkout would update every file's mtime and trigger a huge
    ninja rebuild, so we instead apply the file deltas: copy added /
    modified files and unlink deleted ones.  Without the delete handling
    (which the previous version skipped) a removed source file would
    silently remain and get re-compiled, polluting the verification.
    """
    submodule = PYTORCH_DIR / "third_party" / "torch-xpu-ops"
    if not submodule.exists():
        log("WARN", f"third_party/torch-xpu-ops not found at {submodule}",
            issue=issue)
        return

    # Use --name-status so we know which files were Added/Modified vs Deleted.
    diff_status = git_out(
        "diff", "--name-status", "origin/main..HEAD",
        workdir=worktree_dir, issue=issue,
    ).strip()

    if not diff_status:
        log("INFO", "No changed files to sync", issue=issue)
        return

    copied = 0
    deleted = 0
    for line in diff_status.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status_code = parts[0]
        # Renames produce "R<score>\told\tnew": the old path is deleted, the
        # new path is created.
        if status_code.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
            old_dst = submodule / old_path
            if old_dst.exists():
                old_dst.unlink()
                deleted += 1
            src = worktree_dir / new_path
            if src.is_file():
                dst = submodule / new_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                copied += 1
            continue

        fname = parts[1].strip()
        if not fname:
            continue
        dst = submodule / fname
        if status_code == "D":
            if dst.exists():
                dst.unlink()
                deleted += 1
                log("INFO", f"Removed {fname} from third_party/torch-xpu-ops/",
                    issue=issue)
            continue
        src = worktree_dir / fname
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            copied += 1
            log("INFO", f"Synced {fname} → third_party/torch-xpu-ops/",
                issue=issue)
    log("INFO", f"Synced {copied} file(s), removed {deleted} file(s) from PR branch",
        issue=issue)


def _restore_submodule(issue: int) -> None:
    """Restore third_party/torch-xpu-ops to its clean state."""
    submodule = PYTORCH_DIR / "third_party" / "torch-xpu-ops"
    if submodule.exists():
        # `git checkout .` only restores tracked-and-modified files; any
        # files our sync deleted but git already tracks need an explicit
        # restore. `git restore --source=HEAD --staged --worktree .` would
        # be ideal but for compatibility we do a two-step.
        git("checkout", "HEAD", "--", ".", workdir=submodule, issue=issue,
            check=False)
        # And remove any new files we copied that aren't tracked by HEAD.
        git("clean", "-fd", workdir=submodule, issue=issue, check=False)
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
    if status != Stage.IN_REVIEW:
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
    # We want a runnable command (cd-stripped + pytest-synthesised) — by the
    # time we hit IN_REVIEW the Reproducer was usually already refined by
    # verify_existence, but executable=True makes the call robust either way.
    test_cmd = extract_test_command(body, executable=True)
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
    target_repo = get_metadata(body, "target_repo") or TargetRepo.PYTORCH
    is_xpu_ops = target_repo == TargetRepo.TORCH_XPU_OPS
    branch = f"agent/issue-{issue_number}"

    # Check if a tracking PR exists
    tracking_pr = get_metadata(body, "tracking_pr")
    if not tracking_pr:
        log("WARN", f"No tracking PR for #{issue_number}", issue=issue_number)
        return False

    log("INFO", f"Verifying #{issue_number}: target={target_repo}, "
        f"branch={branch}, test_cmd={test_cmd[:200]}",
        issue=issue_number)

    # All paths below mutate either the pytorch checkout or its build
    # artefacts, so serialise concurrent issues against each other.
    with pytorch_lock(issue=issue_number):
        return _run_locked(issue_number, body, test_cmd, branch, is_xpu_ops, detail)


def _run_locked(issue_number: int, body: str, test_cmd: str, branch: str,
                is_xpu_ops: bool, detail: dict) -> bool:
    """Verification body — runs under the pytorch lock."""

    # --- Checkout PR branch ---
    worktree_dir: Path | None = None
    did_rebuild = False  # restore build state in finally if we rebuilt
    # For pytorch issues we mutate the main checkout's HEAD.  Capture
    # whatever HEAD pointed at *before* we touched it so the finally block
    # can put it back regardless of whether it was attached or detached.
    saved_head: str | None = None
    if not is_xpu_ops:
        saved_head = _capture_head_ref(PYTORCH_DIR, issue_number)
    try:
        if is_xpu_ops:
            worktree_dir, base_ref = _checkout_xpu_ops_pr(
                branch, issue_number)
            # Sync changes into pytorch's third_party
            _sync_to_pytorch(worktree_dir, branch, issue_number)
            # Tests always run from pytorch dir
            test_workdir = PYTORCH_DIR
            # Check if we need a rebuild
            diff_files = git_out(
                "diff", "--name-only", "origin/main..HEAD",
                workdir=worktree_dir, issue=issue_number,
            ).strip()
        else:
            test_workdir, base_ref = _checkout_pytorch_pr(
                branch, issue_number)
            remote = REVIEW_REMOTE
            # Use merge-base to get only the agent's actual changes,
            # not upstream drift between remote/main and the branch.
            merge_base = git_out(
                "merge-base", f"{remote}/{base_ref}", f"{remote}/{branch}",
                workdir=test_workdir, issue=issue_number,
            ).strip()
            diff_files = git_out(
                "diff", "--name-only",
                f"{merge_base}..{remote}/{branch}",
                workdir=test_workdir, issue=issue_number,
            ).strip()

        # --- Build if needed ---
        do_build = bool(diff_files) and _needs_rebuild(diff_files)

        if do_build:
            log("INFO", "C++ files changed, rebuilding", issue=issue_number)
            build_ok, build_output = _rebuild_pytorch(
                PYTORCH_DIR, issue_number)
            did_rebuild = True
            if not build_ok:
                log("ERROR", f"Rebuild failed for #{issue_number}",
                    issue=issue_number)
                new_body = set_status(body, Stage.NEEDS_HUMAN)
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
        passed, output = run_test(test_workdir, test_cmd, issue=issue_number)

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
            new_body = set_status(body, Stage.NEEDS_HUMAN)
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
        # Branch-specific cleanup. We dispatch on the same condition we used
        # to enter — relying on ``worktree_dir is not None`` would silently
        # skip cleanup if _checkout_xpu_ops_pr raised before assignment, and
        # would run the wrong pytorch-restore branch for an xpu-ops issue.
        if is_xpu_ops:
            if worktree_dir is not None and worktree_dir.exists():
                _cleanup_xpu_ops_worktree(worktree_dir, issue_number)
            _restore_submodule(issue_number)
        else:
            # ``git checkout -`` works only when the prior HEAD was a branch
            # — it raises if HEAD was already detached.  Use the explicit
            # ref captured above instead.
            if saved_head:
                _restore_head_ref(PYTORCH_DIR, saved_head, issue_number)
        # If we rebuilt with the PR's C++ changes, restore the build to
        # main state so the next issue starts from a clean binary.
        if did_rebuild:
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
