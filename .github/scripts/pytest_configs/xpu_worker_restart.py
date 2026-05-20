"""Pytest plugin: restart xdist workers on XPU OOM.

Loaded via ``-p xpu_worker_restart`` (``PYTHONPATH = .../pytest_configs/``).

Roles
-----
* **Controller** (or non-xdist run): starts :mod:`xpu_mem_monitor`'s writer
  thread and publishes the snapshot path / owner pid via env.
* **Worker**: only attaches to that snapshot file. On test teardown, if any
  visible card crosses ``_GPU_MEM_THRESHOLD``, or a known-fatal pattern
  shows up in failure output, the worker calls torch.xpu cleanup, prints a
  ``!RESTART`` line, and exits with ``_WORKER_RESTART_CODE`` so
  pytest-xdist respawns it.

Hooks never raise: any unexpected exception is logged and swallowed so the
watchdog can't break the test session.
"""
from __future__ import annotations

import atexit
import os
import re
import sys
from contextlib import suppress

import pytest

import xpu_mem_monitor

# --- tunables -----------------------------------------------------------------

_WORKER_RESTART_CODE = 101
_GPU_MEM_THRESHOLD = 80.0   # percent
_XPU_SMI_INTERVAL = 3       # seconds between snapshots
_XPU_SMI_TIMEOUT = 60       # per `xpu-smi dump -n 1` invocation
_LOG_PATH_ENV = "XPU_SMI_LOG_PATH"
_LOG_OWNER_ENV = "XPU_SMI_LOG_OWNER_PID"

# Single combined regex for fatal failure markers in test output.
_FATAL_RE = re.compile(
    r"ur_result_error"
    r"|segmentation\s+fault"
    r"|bus\s+error"
    r"|kernel\s+died"
    r"|illegal\s+memory"
    r"|failed\s+on\s+setup\s+with.*crashed\s+while\s+running"
    r"|out.*of.*memory",
    re.IGNORECASE | re.DOTALL,
)

# --- module state -------------------------------------------------------------

_worker_id: str | None = None
# Resolved target cards for the memory watchdog:
#   None       -> ZE_AFFINITY_MASK unset; monitor every visible card.
#   list[int]  -> the (non-empty) cards parsed from the env var.
#   []         -> env var was set but parsed to nothing; mem-based
#                 restart disabled (we won't second-guess the user's
#                 mask). Fatal-pattern restart still works.
_target_cards: list[int] | None = None
_restart_armed: bool = True


def _log(msg: str) -> None:
    print(f"[xpu-watchdog] {msg}", file=sys.stderr, flush=True)


def _parse_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def _parse_affinity_part(part: str) -> list[int]:
    """Parse a single ZE_AFFINITY_MASK token (``'3'`` or ``'2-5'``)."""
    part = part.strip()
    if not part:
        return []
    if "-" in part:
        lo_s, hi_s = part.split("-", 1)
        lo, hi = _parse_int(lo_s), _parse_int(hi_s)
        if lo is None or hi is None:
            return []
        return list(range(lo, hi + 1))
    val = _parse_int(part)
    return [val] if val is not None else []


def _parse_affinity_mask() -> list[int] | None:
    """Parse ``ZE_AFFINITY_MASK``.

    Returns:
        ``None``   -- env var unset/empty; monitor every visible card.
        ``list``   -- parsed card ids (non-empty).
        ``[]``     -- env var was set but no token parsed; the caller
                      should disable memory-based restart (we don't
                      know which cards this worker actually uses).
    """
    raw = os.environ.get("ZE_AFFINITY_MASK")
    if raw is None:
        return None
    mask = raw.strip()
    if not mask:
        return None
    cards = [c for part in mask.split(",") for c in _parse_affinity_part(part)]
    if not cards:
        _log(f"ZE_AFFINITY_MASK={raw!r} parsed to no valid card ids; "
             "disabling memory-based restart for this worker")
    return cards


def _cleanup_torch_xpu() -> None:
    """Best-effort GPU memory cleanup before bailing out."""
    try:
        import gc
        import torch
    except ImportError:
        return
    with suppress(Exception):
        gc.collect()
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return
    for fn_name in ("synchronize", "empty_cache"):
        if (fn := getattr(xpu, fn_name, None)) is not None:
            with suppress(Exception):
                fn()


def _restart_worker(reason: str) -> None:
    global _restart_armed
    if not _restart_armed:
        return
    _restart_armed = False
    _cleanup_torch_xpu()
    _log(f"!RESTART {_worker_id} ({reason})")
    os._exit(_WORKER_RESTART_CODE)


# --- pytest hooks -------------------------------------------------------------

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    global _worker_id, _target_cards
    try:
        _worker_id = getattr(config, "workerinput", {}).get("workerid")
        _target_cards = _parse_affinity_mask()

        if _worker_id is None:
            # Controller (or non-xdist): own the writer.
            path = xpu_mem_monitor.start(
                interval=_XPU_SMI_INTERVAL, timeout=_XPU_SMI_TIMEOUT,
            )
            if path:
                os.environ[_LOG_PATH_ENV] = path
                os.environ[_LOG_OWNER_ENV] = str(os.getpid())
                atexit.register(xpu_mem_monitor.stop)
        else:
            # Worker: read-only attach.
            xpu_mem_monitor.attach(os.environ.get(_LOG_PATH_ENV) or None)
    except Exception as e:  # noqa: BLE001 - watchdog must not break pytest
        _log(f"pytest_configure failed: {e!r}")


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    try:
        if os.environ.get(_LOG_OWNER_ENV) == str(os.getpid()):
            xpu_mem_monitor.stop()
    except Exception as e:  # noqa: BLE001
        _log(f"pytest_unconfigure failed: {e!r}")


@pytest.hookimpl
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _worker_id is None or not _restart_armed:
        return  # only restart workers, and only once
    try:
        if report.failed and report.longrepr is not None:
            if _FATAL_RE.search(str(report.longrepr)):
                _restart_worker("fatal pattern matched")
                return  # _restart_worker returns only when disarmed

        if report.when == "teardown" and _target_cards != []:
            hot = xpu_mem_monitor.get_max_mem_util(_target_cards)
            if hot is not None and hot[1] >= _GPU_MEM_THRESHOLD:
                card, mem = hot
                _restart_worker(
                    f"card {card} mem {mem:.2f}% >= {_GPU_MEM_THRESHOLD}%"
                )
    except Exception as e:  # noqa: BLE001
        _log(f"logreport handler failed: {e!r}")
