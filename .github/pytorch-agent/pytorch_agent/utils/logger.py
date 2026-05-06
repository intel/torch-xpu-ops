"""Structured logger: writes to a daily log file and stderr."""
from __future__ import annotations


def log(level: str, message: str, *,
        issue: int | None = None, exc: Exception | None = None) -> None:
    """Append a timestamped log line to the daily log file and print to stderr.

    Args:
        level:   Log level string (INFO / WARN / ERROR).
        message: Human-readable message.
        issue:   Source issue number for correlation.
        exc:     Optional exception — full traceback is appended when set.
    """
    raise NotImplementedError("logger.py: stub — implementation added in PR 2")
