# conftest.py - worker restart + crash-reason capture

import os
import sys
import re
import faulthandler
import pytest

_WORKER_RESTART_CODE = 101

# Shared directory where workers record why a test took the worker down, so the
# log parser can replace xdist's generic "crashed while running" with the real
# reason. check-ut.py runs in the same job/container, so a plain path suffices.
_CRASH_DIR = (
    os.environ.get("UT_CRASH_DIR")
    or (
        os.path.join(os.environ["GITHUB_WORKSPACE"], "ut_log", "crash_reasons")
        if os.environ.get("GITHUB_WORKSPACE")
        else "/tmp/ut_crash"
    )
)

# (regex, reason) ordered most-specific first; matched against the failure text.
_REASON_PATTERNS = [
    (re.compile(r'out.*of.*memory'), 'oom'),
    (re.compile(r'ur_result_error'), 'ur_error'),
    (re.compile(r'illegal memory'), 'illegal_memory'),
    (re.compile(r'segmentation fault'), 'segfault'),
    (re.compile(r'bus error'), 'bus_error'),
    (re.compile(r'kernel died'), 'kernel_died'),
    (re.compile(r'failed on setup with.*crashed while running'), 'setup_crash'),
]

# Single global variable
_worker_id = None


def _breadcrumb_path():
    return os.path.join(_CRASH_DIR, f"current_{_worker_id}.txt")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id
    # Labels fatal signals (SIGSEGV/SIGABRT) in the worker error log, so a raw
    # crash is distinguishable from a timeout during reason resolution.
    faulthandler.enable()
    try:
        _worker_id = config.workerinput.get('workerid')
    except Exception:
        pass
    if _worker_id:
        try:
            os.makedirs(_CRASH_DIR, exist_ok=True)
        except OSError:
            pass


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logstart(nodeid, location):
    # Record the in-flight test so a thread-mode timeout, which os._exit()s
    # without producing a report, can still be attributed to the right case.
    if not _worker_id:
        return
    try:
        with open(_breadcrumb_path(), 'w', encoding='utf-8') as f:
            f.write(nodeid)
    except OSError:
        pass


def _record_reason(nodeid, reason):
    # A single short append is atomic across workers on POSIX (O_APPEND).
    try:
        with open(os.path.join(_CRASH_DIR, "reasons.tsv"), 'a', encoding='utf-8') as f:
            f.write(f"{nodeid}\t{reason}\n")
    except OSError:
        pass


# trylast: let xdist send this failing report to the controller BEFORE we
# os._exit, so results show the real failure reason instead of a worker crash.
@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if not _worker_id or not report.failed:
        return

    err_msg = str(report.longrepr).lower() if report.longrepr else ''

    reason = next((name for rx, name in _REASON_PATTERNS if rx.search(err_msg)), None)
    if reason is None:
        return

    _record_reason(report.nodeid, reason)

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
