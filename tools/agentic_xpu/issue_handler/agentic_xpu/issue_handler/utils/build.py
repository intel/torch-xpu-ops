"""Shared incremental-build helper for PyTorch XPU.

Extracted from ``code_fix._run_build`` so ``verify_upstream_pr`` and
``verify_existence`` can reuse the same incremental → clean-fallback
build path without invoking the heavy full-rebuild dance that
``utils.xpu_env.sync_pytorch`` triggers on every staleness check.

Marker file
-----------
On successful build, the helper writes the just-built commit SHA to::

    PYTORCH_DIR / "build" / "agentic_last_build_ref"

``build/`` is gitignored by pytorch and wiped by ``setup.py clean``,
so the marker auto-invalidates with the ``.so`` it describes. No
``.hermes/`` directory is introduced anywhere — the pipeline owns
its own files.

Callers that don't have a natural ``base_ref`` (e.g. ``verify_existence``
checking whether trunk reproduces a bug) can pass ``base_ref=None`` and
the helper will read the marker; if missing it falls back to a clean
build (the correct one-time cost).
"""
from __future__ import annotations

import hashlib
import os
import select
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import PYTORCH_DIR
from .git import git_out
from .logger import log
from .xpu_env import ENV_SETUP, XPU_BUILD_FLAGS


CPP_EXTENSIONS = {".cpp", ".h", ".cu", ".cuh", ".hpp", ".sycl"}

# Marker file co-located with the build artifacts it describes.
# build/ is gitignored by pytorch and wiped on `setup.py clean`.
BUILD_MARKER_REL = Path("build") / "agentic_last_build_ref"

INCREMENTAL_BUILD_TIMEOUT = 1800  # 30 min — wall-clock cap
CLEAN_BUILD_TIMEOUT = 2400        # 40 min — wall-clock cap
TORCH_IMPORT_TIMEOUT = 60
BUILD_STALL_TIMEOUT = 600         # 10 min of no stdout → SIGTERM
BUILD_LOG_DIR = Path(os.environ.get(
    "AGENTIC_XPU_BUILD_LOG_DIR",
    str(Path(tempfile.gettempdir()) / "agentic_xpu_runs" / "build_logs"),
))
BUILD_PROGRESS_INTERVAL = 60      # seconds between "still building" pings to main log


def _build_log_path(label: str, issue: int | None) -> Path:
    """Pick a unique side-log path for one build invocation.

    Format: ``/tmp/agentic_xpu_runs/build_logs/issue<N>-<label>-<ts>-<hash>.log``
    Hash is a short digest of (issue, label, monotonic ns) so concurrent or
    rapid sequential builds don't collide.
    """
    BUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed = f"{issue}-{label}-{time.monotonic_ns()}"
    digest = hashlib.sha1(seed.encode()).hexdigest()[:8]
    safe_label = label.replace(":", "-").replace("/", "-")
    issue_part = f"issue{issue}" if issue is not None else "noissue"
    return BUILD_LOG_DIR / f"{issue_part}-{safe_label}-{ts}-{digest}.log"


def _run_build_streaming(
    cmd: str,
    *,
    cwd: Path,
    wall_timeout: int,
    stall_timeout: int = BUILD_STALL_TIMEOUT,
    issue: int | None = None,
    label: str = "build",
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` with side-log streaming + stall-based kill.

    Replaces ``subprocess.run(capture_output=True)`` for long builds where
    blind capture hides progress (no monitoring), defeats stall detection
    (a hung ninja is indistinguishable from a slow link), and surfaces
    nothing in the run log until exit.

    Output policy — the main pipeline log must stay readable:
      - Every line of build stdout goes to a per-invocation side-log under
        ``/tmp/agentic_xpu_runs/build_logs/``. The path is logged once at
        start so a human or agent can ``tail -f`` it for live monitoring.
      - Only milestones (start, periodic heartbeat, kill, exit) go to the
        main log via ``log(...)``. The full build chatter never pollutes it.

    Behavior:
      - merges stderr into stdout, line-buffered
      - if ``stall_timeout`` seconds pass with no output → SIGTERM, then
        SIGKILL after 10 s grace
      - if ``wall_timeout`` is exceeded → same SIGTERM/SIGKILL escalation
      - returns a ``CompletedProcess`` with the FULL captured stdout in
        ``.stdout`` (callers truncate as they see fit); ``.stderr`` empty.
    """
    side_log = _build_log_path(label, issue)
    log("INFO",
        f"[{label}] starting (wall={wall_timeout}s stall={stall_timeout}s) — "
        f"build log: {side_log}",
        issue=issue)
    side_fh = side_log.open("w", buffering=1)  # line-buffered
    side_fh.write(
        f"# build label={label} issue={issue} cwd={cwd}\n"
        f"# wall_timeout={wall_timeout}s stall_timeout={stall_timeout}s\n"
        f"# started={datetime.now().isoformat(timespec='seconds')}\n"
        f"# cmd={cmd}\n"
        f"# ---\n"
    )
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        preexec_fn=os.setsid,  # own process group so kill takes children too
    )
    assert proc.stdout is not None  # stdout=PIPE guarantees this
    captured: list[str] = []
    start = time.monotonic()
    last_output = start
    last_progress_ping = start
    killed_reason: str | None = None

    try:
        while True:
            rlist, _, _ = select.select([proc.stdout], [], [], 5.0)
            now = time.monotonic()
            if rlist:
                line = proc.stdout.readline()
                if line == "":
                    break  # EOF
                captured.append(line)
                side_fh.write(line)
                last_output = now
            else:
                if now - last_output > stall_timeout:
                    killed_reason = (
                        f"stalled — no stdout for {int(now - last_output)}s "
                        f"(threshold {stall_timeout}s)"
                    )
                    break
                if now - start > wall_timeout:
                    killed_reason = (
                        f"wall-clock timeout after {int(now - start)}s "
                        f"(limit {wall_timeout}s)"
                    )
                    break
                if proc.poll() is not None:
                    break

            # Periodic heartbeat to main log so humans know we're alive.
            if now - last_progress_ping > BUILD_PROGRESS_INTERVAL:
                elapsed = int(now - start)
                idle = int(now - last_output)
                log("INFO",
                    f"[{label}] still running — elapsed={elapsed}s "
                    f"idle={idle}s, see {side_log}",
                    issue=issue)
                last_progress_ping = now

        if killed_reason:
            log("ERROR",
                f"[{label}] killing build: {killed_reason} (see {side_log})",
                issue=issue)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
            try:
                tail = proc.stdout.read()
                if tail:
                    captured.append(tail)
                    side_fh.write(tail)
            except Exception:
                pass
            kill_note = f"\n[{label} KILLED: {killed_reason}]\n"
            captured.append(kill_note)
            side_fh.write(kill_note)
        else:
            proc.wait()
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        try:
            side_fh.write(
                f"# ---\n"
                f"# finished={datetime.now().isoformat(timespec='seconds')} "
                f"rc={proc.returncode}\n"
            )
            side_fh.close()
        except Exception:
            pass

    rc = proc.returncode if proc.returncode is not None else -1
    elapsed = int(time.monotonic() - start)
    log("INFO",
        f"[{label}] finished rc={rc} elapsed={elapsed}s — build log: {side_log}",
        issue=issue)
    output = "".join(captured)
    return subprocess.CompletedProcess(
        args=cmd, returncode=rc, stdout=output, stderr="",
    )


def _marker_path(workdir: Path) -> Path:
    return Path(workdir) / BUILD_MARKER_REL


def read_build_marker(workdir: Path = PYTORCH_DIR) -> str | None:
    """Return the SHA the current ``_C.so`` was built against, or None.

    Validated only as "file exists and is non-empty". Caller is
    responsible for checking the SHA still resolves in the repo
    (a force-pushed/garbage-collected SHA → fall back to clean build).
    """
    marker = _marker_path(workdir)
    try:
        sha = marker.read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    return sha or None


def write_build_marker(sha: str, workdir: Path = PYTORCH_DIR) -> None:
    """Persist the just-built SHA next to the build artifacts."""
    marker = _marker_path(workdir)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(sha.strip() + "\n")
    except OSError as e:
        # Don't fail the build over a marker write — just log.
        log("WARN", f"Failed to write build marker {marker}: {e}")


def _torch_importable(workdir: Path, issue: int | None = None) -> bool:
    """Check if torch is importable in ``workdir``.

    A failed clean build can destroy ``.so`` artifacts, leaving torch
    un-importable even after git reset. This guard catches that.
    """
    cmd = ENV_SETUP + "python -c 'import torch; print(torch.__version__)'"
    try:
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True,
            timeout=TORCH_IMPORT_TIMEOUT, shell=True, executable="/bin/bash",
        )
        if result.returncode == 0:
            return True
        log("WARN", "torch is not importable — build artifacts may be missing",
            issue=issue)
        return False
    except subprocess.TimeoutExpired:
        log("WARN", "torch import check timed out", issue=issue)
        return False


def _format_timeout(exc: subprocess.TimeoutExpired, header: str) -> str:
    """Concatenate any partial output captured by ``TimeoutExpired``."""
    partial = ""
    for stream in (exc.stdout, exc.stderr):
        if not stream:
            continue
        partial += stream.decode("utf-8", errors="replace") if isinstance(
            stream, (bytes, bytearray)) else stream
    if not partial:
        return header
    return f"{header}\n--- partial output ---\n{partial[-4000:]}"


def _resolve_base_ref(base_ref: str | None, workdir: Path,
                      issue: int | None) -> str | None:
    """Return a base ref usable for ``git diff base_ref..HEAD``.

    If ``base_ref`` is provided, use it. Otherwise read the build marker
    and validate the SHA still exists in the repo.
    """
    if base_ref:
        return base_ref
    marker_sha = read_build_marker(workdir)
    if not marker_sha:
        return None
    try:
        git_out("cat-file", "-e", f"{marker_sha}^{{commit}}",
                workdir=workdir, issue=issue)
        return marker_sha
    except subprocess.CalledProcessError:
        log("WARN", f"Build marker SHA {marker_sha[:12]} no longer resolves "
            "in repo — treating as fresh build", issue=issue)
        return None


def _head_sha(workdir: Path, issue: int | None) -> str:
    return git_out("rev-parse", "HEAD", workdir=workdir, issue=issue).strip()


def incremental_build(
    workdir: Path,
    base_ref: str | None,
    issue: int,
    *,
    force_rebuild: bool = False,
    cpp_extensions: frozenset[str] = frozenset(CPP_EXTENSIONS),
) -> tuple[bool, str]:
    """Run an incremental PyTorch build if any C++/SYCL files changed.

    Parameters
    ----------
    workdir : Path
        PyTorch worktree to build in. Usually ``PYTORCH_DIR``.
    base_ref : str or None
        Ref to diff against (e.g. ``"upstream/main"`` or a SHA). If
        ``None``, the helper reads the build marker; if that's also
        missing, it does a clean build.
    issue : int
        Issue number for log correlation.
    force_rebuild : bool
        Skip the diff/extension check and always build. Used by callers
        that know a rebuild is needed regardless of diff (e.g. PR
        cherry-picked but the diff happens to be Python-only yet the
        installed .so is stale vs current main).
    cpp_extensions : set[str]
        File suffixes that trigger a rebuild.

    Returns
    -------
    (success, output_tail) where ``output_tail`` is the last ~2-3 KB of
    combined stdout/stderr for inclusion in agent logs / issue comments.

    On success, writes ``PYTORCH_DIR/build/agentic_last_build_ref``
    with the just-built HEAD SHA.
    """
    workdir = Path(workdir)

    # Guard: if torch isn't importable, force a rebuild regardless of diff.
    if not force_rebuild:
        force_rebuild = not _torch_importable(workdir, issue)

    # Resolve the diff base. None → clean build (no marker, no caller hint).
    effective_base = _resolve_base_ref(base_ref, workdir, issue)

    if not force_rebuild and effective_base:
        try:
            diff_files = git_out(
                "diff", "--name-only", f"{effective_base}..HEAD",
                workdir=workdir, issue=issue,
            ).strip()
        except subprocess.CalledProcessError as e:
            log("WARN", f"git diff {effective_base}..HEAD failed ({e}); "
                "forcing rebuild", issue=issue)
            diff_files = ""
            force_rebuild = True

        if not force_rebuild:
            if not diff_files:
                return True, ""  # No changes since last build.
            needs_build = any(
                Path(f).suffix in cpp_extensions
                for f in diff_files.splitlines()
            )
            if not needs_build:
                return True, ""  # Python-only change, no native rebuild.
            log("INFO", "C++/SYCL files modified, running incremental build",
                issue=issue)
    elif force_rebuild:
        log("WARN", "Forcing rebuild (torch un-importable or caller requested)",
            issue=issue)
    else:
        log("INFO", "No base ref available, running clean build", issue=issue)

    build_cmd = (
        f"cd {workdir} && {XPU_BUILD_FLAGS} pip install -e . -v --no-build-isolation"
    )

    # Try incremental first (unless we already know base_ref is missing,
    # in which case skip straight to clean).
    if effective_base and not force_rebuild:
        result = _run_build_streaming(
            ENV_SETUP + build_cmd, cwd=workdir,
            wall_timeout=INCREMENTAL_BUILD_TIMEOUT,
            issue=issue, label="build:incremental",
        )
        if result.returncode == 0:
            write_build_marker(_head_sha(workdir, issue), workdir)
            return True, result.stdout[-2000:]
        output = result.stdout[-3000:]
        log("WARN", "Incremental build failed, falling back to clean build",
            issue=issue)

    # Clean build (either fallback path or force_rebuild requested it).
    log("INFO", "Running clean build (clean + pip install)", issue=issue)
    clean_cmd = (
        f"cd {workdir} && {XPU_BUILD_FLAGS} python setup.py clean && "
        f"{XPU_BUILD_FLAGS} pip install -e . -v --no-build-isolation"
    )
    result = _run_build_streaming(
        ENV_SETUP + clean_cmd, cwd=workdir,
        wall_timeout=CLEAN_BUILD_TIMEOUT,
        issue=issue, label="build:clean",
    )
    if result.returncode == 0:
        write_build_marker(_head_sha(workdir, issue), workdir)
        return True, result.stdout[-2000:]
    return False, result.stdout[-3000:]
