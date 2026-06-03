"""XPU environment helpers.

Ensures oneAPI is sourced and PyTorch has XPU support before running tests.
Used by verify_existence, verify_fix, and code_fix.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
from pathlib import Path

from .config import PYTORCH_DIR
from .logger import log

# Shell preamble that sources oneAPI + activates the pytorch venv.
# Use this as prefix for any subprocess that needs XPU support.
ONEAPI_SET_VARS = os.environ.get("ONEAPI_SET_VARS", "~/intel/oneapi/setvars.sh")
ENV_SETUP = (
    f"source {ONEAPI_SET_VARS} --force 2>/dev/null; "
    f"source {PYTORCH_DIR}/.venv/bin/activate; "
)

# Build flags for XPU-enabled PyTorch builds.
XPU_BUILD_FLAGS = "TORCH_XPU_ARCH_LIST=pvc USE_XPU=1"

# Auto-stash retention policy. Stashes created by _ensure_clean_worktree
# use the AUTOSTASH_PREFIX so _prune_old_stashes can find them without
# touching user/manual stashes.
AUTOSTASH_PREFIX = "agent-autoclean-"
AGENT_BRANCH_PREFIX = "agent/"   # see code_fix.py:279,290
MAX_STASH_AGE_DAYS = 7
MAX_STASH_COUNT = 10


def _ensure_clean_worktree(pytorch: str, *, issue: int | None = None
                            ) -> tuple[bool, str | None, str | None]:
    """Make ~/pytorch ready for ``git pull``.

    If the working tree is dirty, ``git stash push -u`` is run unconditionally
    with an ``AUTOSTASH_PREFIX`` message so the pruner can find it and the
    stash is recoverable for ``MAX_STASH_AGE_DAYS``.

    After stashing, if HEAD is on an ``agent/*`` branch (leftover from a
    previous fix-agent run), switch to ``main`` so the subsequent
    ``git pull`` operates on a real tracked branch.

    Submodule modifications are intentionally ignored
    (``--ignore-submodules=all``) — they're already handled by
    ``git submodule update`` + the explicit ``git checkout .`` block
    in sync_pytorch.

    Returns:
        (ok, stash_ref, switched_from).
        ``stash_ref`` is ``stash@{0}`` when a stash was created, else None.
        ``switched_from`` is the previous branch name when we switched
        off an agent/* branch, else None.

    Raises:
        RuntimeError: git status / stash / checkout failed.
    """
    # 1. Fast path: is anything dirty (ignoring submodules)?
    r = subprocess.run(
        f"cd {pytorch} && git status --porcelain --ignore-submodules=all",
        shell=True, executable="/bin/bash",
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git status failed: {r.stderr.strip()}")
    dirty = r.stdout.strip()

    # Current branch — needed for both stash log and post-stash switch.
    br = subprocess.run(
        f"cd {pytorch} && git rev-parse --abbrev-ref HEAD",
        shell=True, executable="/bin/bash",
        capture_output=True, text=True, timeout=10,
    )
    branch = br.stdout.strip() if br.returncode == 0 else "<unknown>"

    stash_ref: str | None = None
    if dirty:
        # 2. Auto-stash (unconditional — recoverable via git stash apply).
        file_lines = dirty.splitlines()
        sample = ", ".join(line[3:] for line in file_lines[:5])
        if len(file_lines) > 5:
            sample += f", ... (+{len(file_lines) - 5} more)"
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        msg = (f"{AUTOSTASH_PREFIX}{ts}-issue-"
               f"{issue if issue is not None else 'none'}")
        log("INFO",
            f"Worktree dirty on '{branch}' ({len(file_lines)} file(s)): "
            f"{sample} — auto-stashing as '{msg}'",
            issue=issue)
        r = subprocess.run(
            f"cd {pytorch} && git stash push -u -m {msg!r}",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"git stash push failed (rc={r.returncode}): "
                f"{(r.stdout + r.stderr).strip()[-500:]}"
            )
        stash_ref = "stash@{0}"
        log("INFO",
            f"Stashed. Recover with: cd {pytorch} && "
            f"git stash apply '{stash_ref}'",
            issue=issue)

    # 3. If HEAD is on an agent/* branch (leftover from a prior fix-agent
    #    run), switch to main so the next git pull works.
    switched_from: str | None = None
    if branch.startswith(AGENT_BRANCH_PREFIX):
        log("INFO",
            f"On agent branch '{branch}' — switching to main before pull",
            issue=issue)
        sw = subprocess.run(
            f"cd {pytorch} && git checkout main",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30,
        )
        if sw.returncode != 0:
            raise RuntimeError(
                f"git checkout main failed (rc={sw.returncode}): "
                f"{(sw.stdout + sw.stderr).strip()[-500:]}"
            )
        switched_from = branch

    return True, stash_ref, switched_from


def _prune_old_stashes(pytorch: str, *, issue: int | None = None) -> int:
    """Drop autoclean stashes older than MAX_STASH_AGE_DAYS or beyond
    MAX_STASH_COUNT (whichever cuts).  Best-effort; never raises.

    Only stashes whose message starts with AUTOSTASH_PREFIX are
    considered — user/manual stashes are left alone.

    Returns count pruned (0 on any failure).
    """
    try:
        r = subprocess.run(
            f"cd {pytorch} && git stash list --format='%gd%x09%ct%x09%s'",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log("WARN", f"git stash list failed: {r.stderr.strip()}",
                issue=issue)
            return 0
        now = _dt.datetime.now(_dt.timezone.utc).timestamp()
        cutoff = now - MAX_STASH_AGE_DAYS * 86400
        # Parse: (gd_ref, timestamp, msg) for autoclean stashes only.
        autos: list[tuple[str, int, str]] = []
        for line in r.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            gd, ts_s, msg = parts
            if AUTOSTASH_PREFIX not in msg:
                continue
            try:
                ts = int(ts_s)
            except ValueError:
                continue
            autos.append((gd, ts, msg))
        # Stash list is already newest-first.  Mark for drop:
        # any beyond MAX_STASH_COUNT, OR older than cutoff.
        to_drop: list[str] = []
        for idx, (gd, ts, msg) in enumerate(autos):
            if idx >= MAX_STASH_COUNT or ts < cutoff:
                to_drop.append(gd)
        if not to_drop:
            return 0
        # IMPORTANT: drop in reverse order — dropping a low-index stash
        # shifts all higher refs down by one.  Sort by numeric stash index
        # descending.
        def _idx(gd: str) -> int:
            try:
                return int(gd.split("{")[1].rstrip("}"))
            except (IndexError, ValueError):
                return -1
        to_drop.sort(key=_idx, reverse=True)
        pruned = 0
        for gd in to_drop:
            dr = subprocess.run(
                f"cd {pytorch} && git stash drop {gd!r}",
                shell=True, executable="/bin/bash",
                capture_output=True, text=True, timeout=15,
            )
            if dr.returncode == 0:
                pruned += 1
            else:
                log("WARN", f"git stash drop {gd} failed: {dr.stderr.strip()}",
                    issue=issue)
        log("INFO",
            f"Pruned {pruned} old autoclean stash(es) "
            f"(>{MAX_STASH_AGE_DAYS}d or beyond {MAX_STASH_COUNT} kept)",
            issue=issue)
        return pruned
    except Exception as e:
        log("WARN", f"_prune_old_stashes failed: {e}", issue=issue)
        return 0


def _record_stash_in_body(issue: int, stash_ref: str,
                          *, switched_from: str | None = None) -> None:
    """Best-effort append of stash recovery hint into the issue body's
    <!-- agent:env-log --> section.  Never raises.
    """
    if issue is None:
        return
    try:
        # Local imports to avoid a top-level cycle (xpu_env <- git <- ...).
        from . import git as _git
        from .body_templates import append_log
        from .config import ISSUE_REPO
        repo = str(ISSUE_REPO)
        body = _git._gh(["issue", "view", str(issue), "--repo", repo,
                         "--json", "body", "-q", ".body"]) or ""
        if not body:
            return
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        switch_note = (f" Also switched HEAD off `{switched_from}` → `main` "
                       f"so pull can proceed.") if switched_from else ""
        note = (
            f"[{ts}] Auto-stashed dirty pytorch worktree before sync."
            f"{switch_note} "
            f"Recover with: `cd ~/pytorch && git stash apply '{stash_ref}'` "
            f"(stash list grep `{AUTOSTASH_PREFIX}`)"
        )
        new_body = append_log(body, "env", note)
        if new_body == body:
            return
        _git._gh(["issue", "edit", str(issue), "--repo", repo,
                  "--body", new_body])
    except Exception as e:
        log("WARN", f"_record_stash_in_body(issue={issue}) failed: {e}",
            issue=issue)


def sync_pytorch(ref: str = "main", *, pull: bool = True,
                 issue: int | None = None) -> bool:
    """Sync pytorch and rebuild if installed binary is stale.

    Args:
        ref: git ref to compare ``.so`` mtime against (default: ``main``).
            When ``pull=False`` we assume the caller has already checked out
            this ref (e.g. an upstream PR checkout); we just verify staleness
            against its commit timestamp.
        pull: When True (default), runs ``git pull`` on the current branch
            and re-syncs submodules. Set False when the working tree is
            already on a specific ref the caller does not want disturbed.

    Returns True if pytorch is up-to-date (or successfully rebuilt).
    """
    pytorch = str(PYTORCH_DIR)

    if pull:
        # 0. Make worktree clean before pulling — auto-stash any dirty
        #    files and switch off agent/* branches so pull can proceed.
        try:
            _ok, stash_ref, switched_from = _ensure_clean_worktree(
                pytorch, issue=issue)
        except RuntimeError as e:
            log("ERROR", f"Cannot auto-clean worktree: {e}", issue=issue)
            return False
        if stash_ref:
            _prune_old_stashes(pytorch, issue=issue)
            if issue is not None:
                _record_stash_in_body(issue, stash_ref,
                                       switched_from=switched_from)

        # 1. git pull + submodule sync/update
        log("INFO", "Syncing pytorch repo", issue=issue)
        sync_cmd = (
            f"cd {pytorch} && git pull && git submodule sync && "
            "git submodule update --init --recursive"
        )
        try:
            r = subprocess.run(
                sync_cmd, shell=True, executable="/bin/bash",
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                log("ERROR", f"git sync failed:\n{(r.stdout + r.stderr)[-1000:]}",
                    issue=issue)
                return False
        except subprocess.TimeoutExpired:
            log("ERROR", "git sync timed out (10min)", issue=issue)
            return False
    else:
        log("INFO", f"sync_pytorch(pull=False, ref={ref}) — skipping git pull",
            issue=issue)

    # 2. Clean any dirty files in torch-xpu-ops submodule (leftover from
    #    previous agent runs that removed skip decorators etc.)
    xpu_ops = PYTORCH_DIR / "third_party" / "torch-xpu-ops"
    if xpu_ops.exists():
        subprocess.run(
            f"cd {xpu_ops} && git checkout .", shell=True,
            executable="/bin/bash", capture_output=True, timeout=30,
        )

    # 3. Staleness check: compare .so mtime against the requested ref's
    #    commit timestamp.
    #    torch.version.git_version is unreliable — it gets set during cmake
    #    configure even if the actual compilation (ninja) fails, so a matching
    #    git_version does NOT mean the binary is up-to-date.
    so_glob = list(PYTORCH_DIR.glob("torch/_C.cpython-*.so"))
    if so_glob:
        so_mtime = so_glob[0].stat().st_mtime
    else:
        so_mtime = 0  # no .so at all → definitely stale

    try:
        head_ts = int(subprocess.run(
            f"cd {pytorch} && git log -1 --format=%ct {ref}",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
    except Exception:
        head_ts = int(1e18)  # force rebuild on error

    log("INFO",
        f"Binary .so mtime: {int(so_mtime)}, {ref} commit time: {head_ts} "
        f"({'stale' if so_mtime < head_ts else 'up-to-date'})",
        issue=issue)

    if so_mtime >= head_ts:
        log("INFO", f"PyTorch binary is newer than {ref} — no rebuild needed",
            issue=issue)
        return True

    # 4. Rebuild
    log("WARN", "PyTorch binary is stale — rebuilding", issue=issue)
    ok, _ = _do_rebuild(issue=issue)
    return ok


def _do_rebuild(issue: int | None = None, clean_cmake: bool = False) -> tuple[bool, str]:
    """Shared rebuild logic for sync_pytorch and rebuild_pytorch.

    Args:
        clean_cmake: If True, remove CMakeCache.txt to force cmake reconfigure.
    Returns (success, output_tail).
    """
    if clean_cmake:
        cmake_cache = PYTORCH_DIR / "build" / "CMakeCache.txt"
        if cmake_cache.exists():
            log("INFO", "Removing CMakeCache.txt to force reconfigure", issue=issue)
            cmake_cache.unlink()
    cmd = (
        ENV_SETUP +
        f"cd {PYTORCH_DIR} && {XPU_BUILD_FLAGS} pip install -e . -v --no-build-isolation"
    )
    try:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=3600,
        )
        output = (result.stdout + result.stderr)[-5000:]
        if result.returncode == 0:
            log("INFO", "PyTorch rebuild succeeded", issue=issue)
            return True, output
        log("ERROR", f"PyTorch rebuild failed:\n{output[-1000:]}", issue=issue)
        return False, output
    except subprocess.TimeoutExpired:
        return False, "PyTorch rebuild timed out (60min)"


def check_xpu_available() -> bool:
    """Check if PyTorch has XPU support and a device is visible.

    Sources oneAPI first, then imports torch and checks xpu.is_available().
    Returns True if XPU device is available.
    """
    cmd = (
        ENV_SETUP +
        "python -c 'import torch; print(torch.xpu.is_available())'"
    )
    try:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=60,
        )
        return "True" in result.stdout
    except Exception:
        return False


def rebuild_pytorch(issue: int | None = None) -> tuple[bool, str]:
    """Rebuild PyTorch with USE_XPU=1.

    Removes CMakeCache.txt first to force cmake reconfigure — otherwise
    cmake reuses the cached USE_XPU=OFF and the build is a no-op.

    Returns (success, output).
    """
    log("WARN", "XPU not available — rebuilding PyTorch with USE_XPU=1",
        issue=issue)
    return _do_rebuild(issue=issue, clean_cmake=True)


def ensure_xpu_ready(issue: int | None = None) -> bool:
    """Ensure oneAPI is sourced and PyTorch has XPU support.

    If XPU is not available, triggers a rebuild with USE_XPU=1.
    Returns True if XPU is ready after all attempts, False if unrecoverable.
    """
    if check_xpu_available():
        return True

    # Auto-rebuild
    ok, output = rebuild_pytorch(issue=issue)
    if not ok:
        log("ERROR", f"Cannot get XPU ready: rebuild failed\n{output[-500:]}",
            issue=issue)
        return False

    # Re-check after rebuild
    if check_xpu_available():
        return True

    log("ERROR", "XPU still not available after rebuild", issue=issue)
    return False
