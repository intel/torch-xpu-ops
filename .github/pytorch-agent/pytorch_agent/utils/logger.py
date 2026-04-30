"""Simple structured file + console logger for pytorch-agent."""
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .config import LOG_DIR


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(level: str, message: str, *, issue: int | None = None,
        exc: Exception | None = None) -> None:
    """Append a log line to daily log file AND print to stderr.

    Format: YYYY-MM-DD HH:MM:SS [LEVEL] (issue #N) message
    If exc is provided, appends the full traceback.
    """
    _ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"(issue #{issue}) " if issue is not None else ""
    line = f"{ts} [{level.upper()}] {prefix}{message}"

    # Append traceback if exception provided
    tb = ""
    if exc is not None:
        tb = "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    full_line = line + tb + "\n"

    # Write to file
    log_file = LOG_DIR / f"pipeline-{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a") as f:
        f.write(full_line)

    # Also print to stderr for live monitoring
    print(line, file=sys.stderr, flush=True)
    if tb:
        print(tb, file=sys.stderr, flush=True)
