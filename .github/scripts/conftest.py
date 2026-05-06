# conftest.py - Ultra-minimal worker restart

import os
import sys
import re
import pytest

_WORKER_RESTART_CODE = 101

# Single global variable
_worker_id = None

patterns = [
    'ur_result_error',
    'segmentation fault',
    'bus error',
    'kernel died',
    'illegal memory',
    re.compile(r'failed on setup with.*crashed while running'),
    re.compile(r'out.*of.*memory'),
]

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id
    try:
        _worker_id = config.workerinput.get('workerid')
    except Exception:
        pass

@pytest.hookimpl
def pytest_runtest_logreport(report):
    if not _worker_id or not report.failed:
        return

    err_msg = str(report.longrepr).lower() if report.longrepr else ''

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

        sys.stderr.write(f"\n!RESTART {_worker_id}\n")
        os._exit(_WORKER_RESTART_CODE)
