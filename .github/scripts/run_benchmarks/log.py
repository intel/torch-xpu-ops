"""Logging helpers — consistent, readable output."""

import time


def log(msg: str, *, level: str = "INFO", worker: int | None = None) -> None:
    """Print a timestamped, consistently-formatted log line."""
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}] [{level}]"
    if worker is not None:
        prefix += f" [Worker {worker}]"
    print(f"{prefix} {msg}", flush=True)


def banner(title: str) -> None:
    """Print a visible section separator."""
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{line}", flush=True)


def fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
