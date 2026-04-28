# xpu_worker_restart.py - Pytest plugin: restart xdist workers on XPU OOM.
#
# Loaded via `-p xpu_worker_restart`. The xdist controller starts a
# background xpu-smi poller (xpu_mem_monitor.start) and publishes the log
# path via env. Workers attach to that log and exit with code 101 whenever
# their target device crosses _GPU_MEM_THRESHOLD or a test fails with a
# known fatal pattern. pytest-xdist respawns workers under
# --max-worker-restart.

import atexit
import os
import re
import sys

import pytest

import xpu_mem_monitor

# --- tunables -----------------------------------------------------------------

_WORKER_RESTART_CODE = 101
_GPU_MEM_THRESHOLD = 80.0   # percent
_XPU_SMI_INTERVAL = 3       # seconds between snapshots
_XPU_SMI_TIMEOUT = 60       # per `xpu-smi dump -n 1` invocation
_LOG_PATH_ENV = 'XPU_SMI_LOG_PATH'
_LOG_OWNER_ENV = 'XPU_SMI_LOG_OWNER_PID'

_FATAL_PATTERNS = [
    'ur_result_error',
    'segmentation fault',
    'bus error',
    'kernel died',
    'illegal memory',
    re.compile(r'failed on setup with.*crashed while running'),
    re.compile(r'out.*of.*memory'),
]

# --- module state -------------------------------------------------------------

_worker_id = None
_target_device = '0'


# --- helpers ------------------------------------------------------------------

def _log(msg):
    sys.stderr.write(f"[xpu-watchdog] {msg}\n")
    sys.stderr.flush()


def _get_target_device():
    """Pick device id from ZE_AFFINITY_MASK (first entry); default to '0'."""
    mask = os.environ.get('ZE_AFFINITY_MASK', '').strip()
    if mask:
        first = mask.split(',', 1)[0].strip()
        if first.isdigit():
            return first
    return '0'


def _cleanup_torch_xpu():
    try:
        import gc
        import torch
        gc.collect()
        if hasattr(torch, 'xpu'):
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
    except ImportError:
        pass


def _restart_worker(reason):
    _cleanup_torch_xpu()
    _log(f"!RESTART {_worker_id} ({reason})")
    os._exit(_WORKER_RESTART_CODE)


# --- pytest hooks -------------------------------------------------------------

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id, _target_device

    try:
        _worker_id = config.workerinput.get('workerid')
    except AttributeError:
        _worker_id = None

    _target_device = _get_target_device()

    if _worker_id is None:
        # xdist controller (or non-xdist run): own the poller.
        path = xpu_mem_monitor.start(
            interval=_XPU_SMI_INTERVAL, timeout=_XPU_SMI_TIMEOUT,
        )
        if path:
            os.environ[_LOG_PATH_ENV] = path
            os.environ[_LOG_OWNER_ENV] = str(os.getpid())
        atexit.register(xpu_mem_monitor.stop)
    else:
        # xdist worker: attach to the controller's snapshot file.
        xpu_mem_monitor.attach(os.environ.get(_LOG_PATH_ENV) or None)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    # Only the owner cleans up the shared log/process.
    owner_pid = os.environ.get(_LOG_OWNER_ENV)
    if owner_pid and owner_pid == str(os.getpid()):
        xpu_mem_monitor.stop()


@pytest.hookimpl
def pytest_runtest_logreport(report):
    if _worker_id is None:
        return  # only restart workers, never the controller

    if report.failed and report.longrepr is not None:
        err = str(report.longrepr).lower()
        for p in _FATAL_PATTERNS:
            if (p.search(err) if hasattr(p, 'search') else p in err):
                _restart_worker("error pattern matched")

    if report.when == 'teardown':
        mem = xpu_mem_monitor.get_device_mem_util(_target_device)
        if mem is not None and mem >= _GPU_MEM_THRESHOLD:
            _restart_worker(
                f"device {_target_device} mem {mem:.2f}% >= {_GPU_MEM_THRESHOLD}%"
            )
