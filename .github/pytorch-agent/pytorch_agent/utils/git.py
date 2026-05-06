"""Shared git helpers for the pytorch-agent pipeline."""
from __future__ import annotations

import subprocess
from pathlib import Path


def git(*args: str, workdir: Path | None = None, check: bool = True,
        issue: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command inside PYTORCH_DIR (or *workdir*) with logging."""
    raise NotImplementedError("git.py: stub — implementation added in PR 2")


def git_out(*args: str, **kwargs) -> str:
    """Run git and return stdout string."""
    raise NotImplementedError("git.py: stub — implementation added in PR 2")


def add_and_commit(message: str, *,
                   issue: int | None = None,
                   workdir: Path | None = None) -> bool:
    """Stage all tracked files (excluding third_party/*) and commit if dirty.

    Returns True if a commit was made, False if the tree was already clean.
    Handles renamed files (porcelain R old -> new) correctly.
    """
    raise NotImplementedError("git.py: stub — implementation added in PR 2")
