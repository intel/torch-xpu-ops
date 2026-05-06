"""Helpers for posting agent status comments to GitHub issues."""
from __future__ import annotations

from pathlib import Path


def post_session_started(repo: str, issue: int, stage: str, session_id: str) -> None:
    """Comment on *issue* that an agent session has started.

    Args:
        repo:       Source issue repo (e.g. "intel/torch-xpu-ops").
        issue:      Issue number.
        stage:      Human-readable stage name (e.g. "Implement").
        session_id: opencode session ID for cross-referencing logs.
    """
    raise NotImplementedError


def post_agent_completed(
    repo: str, issue: int, header: str, log_path: Path,
    output: str, *, tail: int = 50,
) -> None:
    """Comment on *issue* with a summary of agent output + last *tail* lines of log.

    Args:
        repo:      Source issue repo.
        issue:     Issue number.
        header:    One-line summary shown at the top of the comment.
        log_path:  Path to the agent run log file.
        output:    Full agent output text (trimmed to *tail* lines in comment).
        tail:      Number of log lines to include in the comment.
    """
    raise NotImplementedError
