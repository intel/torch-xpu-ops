"""Inter-process locks for shared working directories.

``code_fix`` and ``verify_fix`` both mutate ``PYTORCH_DIR`` directly
(checkout, reset --hard, rebuild).  If the orchestrator is ever invoked
for two issues concurrently (cron, parallel workers) those operations
will trample each other and trigger thrash rebuilds.

We use a coarse advisory file lock (``fcntl.flock``) on a sentinel file
under ``/tmp`` — coarse-grained but sufficient: the slowest single
operation (a clean rebuild) is ~40 minutes, and a fair queue is better
than corruption.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .logger import log


_DEFAULT_LOCK_DIR = Path(os.environ.get("AGENTIC_XPU_LOCK_DIR", "/tmp"))
_PYTORCH_LOCK_PATH = _DEFAULT_LOCK_DIR / "agentic-xpu-pytorch.lock"


@contextmanager
def file_lock(path: Path, *, issue: int | None = None,
              description: str = "file"):
    """Acquire an exclusive ``flock`` on ``path``, blocking until available.

    Logs at WARN level if we wait more than 5s so long queues are visible.
    The lock file is created if missing; it's never removed because that
    would race with another process about to lock it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    start = time.monotonic()
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("WARN",
                f"Waiting on {description} lock at {path} (held by another worker)",
                issue=issue)
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        waited = time.monotonic() - start
        if waited > 5:
            log("INFO", f"Acquired {description} lock after {waited:.1f}s",
                issue=issue)
        # Best-effort identifier so a stale lock can be inspected.
        try:
            fh.write(f"pid={os.getpid()} issue={issue}\n")
            fh.flush()
        except OSError:
            pass
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def pytorch_lock(issue: int | None = None) -> contextlib.AbstractContextManager:
    """Exclusive lock for any operation that mutates ``PYTORCH_DIR``."""
    return file_lock(_PYTORCH_LOCK_PATH, issue=issue, description="pytorch")
