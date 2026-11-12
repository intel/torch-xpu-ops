"""Ultra-minimal xdist worker restart on fatal XPU failures.

When running under pytest-xdist, an unrecoverable failure (device crash, OOM,
segfault, etc.) can leave a worker in a bad state. This plugin detects such
failures from the test report and forcibly exits the worker process so xdist
respawns a fresh one, while still forwarding the real failure to the controller.
"""

import os
import re
import sys

import pytest

_WORKER_RESTART_CODE = 101

# Populated in pytest_configure when running inside an xdist worker.
_worker_id = None

patterns = [
    "ur_result_error",
    "segmentation fault",
    "bus error",
    "kernel died",
    "illegal memory",
    re.compile(r"failed on setup with.*crashed while running"),
    re.compile(r"out.*of.*memory"),
]


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id
    try:
        _worker_id = config.workerinput.get("workerid")
    except Exception:
        pass


# trylast: let xdist send this failing report to the controller BEFORE we
# os._exit, so results show the real failure reason instead of a worker crash.
@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if not _worker_id or not report.failed:
        return

    err_msg = str(report.longrepr).lower() if report.longrepr else ""

    # Direct inline pattern checks (fastest)
    if any(p in err_msg if isinstance(p, str) else p.search(err_msg) for p in patterns):
        try:
            import gc

            import torch

            # Only force GC when XPU is actually used
            gc.collect()

            # Direct XPU operations without try-catch if possible
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        except Exception:
            # Silent fail - XPU ops might fail in some states
            pass

        sys.stderr.write(f"\n!RESTART {_worker_id} {report.nodeid}\n")
        sys.stderr.flush()
        os._exit(_WORKER_RESTART_CODE)
