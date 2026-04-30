"""Shared git helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PYTORCH_DIR
from .logger import log


def git(*args: str, workdir: Path | None = None, check: bool = True,
        issue: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command with logging."""
    cwd = str(workdir or PYTORCH_DIR)
    cmd_str = "git " + " ".join(args)
    log("INFO", f"$ {cmd_str}", issue=issue)
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check,
        capture_output=True, text=True,
    )


def add_and_commit(message: str, *, issue: int | None = None,
                   workdir: Path | None = None) -> bool:
    """Stage tracked files (excluding third_party/*) and commit if dirty.

    Returns True if a commit was made, False if tree was clean.
    """
    cwd = workdir or PYTORCH_DIR
    status = git("status", "--porcelain", workdir=cwd, issue=issue).stdout
    if not status.strip():
        return False

    # Filter out submodule pointer changes (third_party/*)
    files = []
    for line in status.splitlines():
        # porcelain format: XY filename  or  XY old -> new
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        fname = parts[1].strip()
        if fname.startswith("third_party/"):
            log("INFO", f"Skipping submodule change: {fname}", issue=issue)
            continue
        files.append(fname)

    if not files:
        return False

    git("add", "--", *files, workdir=cwd, issue=issue)
    git("commit", "-m", message, workdir=cwd, issue=issue)
    return True
