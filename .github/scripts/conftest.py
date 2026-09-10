# conftest.py - Ultra-minimal worker restart

import os
import sys
import re
import time
import shutil
import hashlib
import tempfile
import threading
import traceback
import pytest

_WORKER_RESTART_CODE = 101

# Single global variable
_worker_id = None

# Directory holding one reason file per case that hit the pytest timeout.
# Chosen on the controller and forwarded to every xdist worker.
_sentinel_dir = None
# nodeid -> threading.Timer that records a pending timeout for that case.
_sentinel_timers = {}
_sentinel_lock = threading.Lock()

patterns = [
    'ur_result_error',
    'segmentation fault',
    'bus error',
    'kernel died',
    'illegal memory',
    re.compile(r'failed on setup with.*crashed while running'),
    re.compile(r'out.*of.*memory'),
]


def _default_sentinel_dir():
    return os.path.join(
        tempfile.gettempdir(),
        f"pytest_timeout_sentinel_{os.getpid()}_{int(time.time())}",
    )


def _reason_file(nodeid):
    digest = hashlib.sha1(nodeid.encode('utf-8', 'replace')).hexdigest()
    return os.path.join(_sentinel_dir, digest)


def _classify_reason(nodeid, timeout, budget):
    # Sample the test thread's Python stack across a short window (kept within the
    # lead time before pytest-timeout aborts the worker) to tell the two cases
    # apart: a thread that is dead or never advances is a Hang; a thread that is
    # still moving but ran past the limit is a (slow) Timeout.
    main_ident = threading.main_thread().ident
    window = min(3.0, max(budget - 1.0, 0.2))
    samples = 4
    interval = window / (samples - 1)
    previous = None
    last_stack = ''
    progressed = False
    dead = False
    for index in range(samples):
        if not threading.main_thread().is_alive():
            dead = True
        frame = sys._current_frames().get(main_ident)
        if frame is None:
            dead = True
            current = None
        else:
            current = ''.join(traceback.format_stack(frame)[-40:])
            last_stack = current
        if previous is not None and current is not None and current != previous:
            progressed = True
            break
        previous = current
        if index < samples - 1:
            time.sleep(interval)

    if dead:
        label, detail = 'Hang', 'the test thread is gone (dead)'
    elif progressed:
        label, detail = 'Timeout', 'the test thread was still progressing but ran past the limit'
    else:
        label, detail = 'Hang', 'the test thread made no progress across sampling (stuck)'
    return (
        f"{label}: {nodeid} exceeded the configured pytest timeout of {timeout:g}s; "
        f"{detail}. The worker was aborted and the case was not rerun.\n"
        "Hung stack (captured shortly before the timeout):\n"
        f"{last_stack}"
    )


def _record_timeout(nodeid, reason):
    if not _sentinel_dir:
        return
    try:
        path = _reason_file(nodeid)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(reason)
        os.replace(tmp, path)
    except Exception:
        pass


def _clear_timeout(nodeid):
    if not _sentinel_dir:
        return
    try:
        os.remove(_reason_file(nodeid))
    except OSError:
        pass


def _timeout_reason(crashitem):
    if not _sentinel_dir:
        return None
    try:
        with open(_reason_file(crashitem), 'r', encoding='utf-8') as handle:
            return handle.read()
    except OSError:
        return None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id, _sentinel_dir
    if hasattr(config, "workerinput"):
        # xdist worker: reuse the sentinel directory handed down by the controller.
        _worker_id = config.workerinput.get('workerid')
        _sentinel_dir = config.workerinput.get('timeout_sentinel')
    else:
        # Controller (or non-xdist main process): pick/clean the sentinel dir.
        _sentinel_dir = os.environ.get('PYTEST_TIMEOUT_SENTINEL') or _default_sentinel_dir()
        os.environ['PYTEST_TIMEOUT_SENTINEL'] = _sentinel_dir
        shutil.rmtree(_sentinel_dir, ignore_errors=True)
        try:
            os.makedirs(_sentinel_dir, exist_ok=True)
        except OSError:
            pass


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    # Controller side: share the sentinel directory with each xdist worker.
    if _sentinel_dir:
        node.workerinput['timeout_sentinel'] = _sentinel_dir


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    # Fire slightly before pytest-timeout kills the worker so the real reason
    # (hang vs timeout, duration and stack) is captured before the process dies.
    timeout = getattr(settings, 'timeout', None)
    if _sentinel_dir and timeout and timeout > 0:
        nodeid = item.nodeid
        lead = 5 if timeout > 10 else timeout * 0.1
        timer = threading.Timer(
            max(timeout - lead, 0.0),
            lambda: _record_timeout(nodeid, _classify_reason(nodeid, timeout, lead)),
        )
        timer.daemon = True
        with _sentinel_lock:
            previous = _sentinel_timers.pop(nodeid, None)
            if previous is not None:
                previous.cancel()
            _sentinel_timers[nodeid] = timer
        timer.start()
    # Return None so pytest-timeout still installs its real (killing) timer.
    return None


@pytest.hookimpl(trylast=True, optionalhook=True)
def pytest_timeout_cancel_timer(item):
    # The case returned control, so it did not time out: cancel our timer and
    # drop any reason file the timer may have already written. Only genuinely
    # aborted (timed-out) cases keep a reason file for the controller to read.
    with _sentinel_lock:
        timer = _sentinel_timers.pop(item.nodeid, None)
    if timer is not None:
        timer.cancel()
    _clear_timeout(item.nodeid)
    return None


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_handlecrashitem(crashitem, report, sched):
    # A thread-method timeout hard-kills the xdist worker, so xdist reports it as
    # "crashed while running" - identical to a real crash. Replace that generic
    # message with the captured timeout reason: this drops the phrase the
    # "--only-rerun 'crashed while running'" filter matches (so timeouts are not
    # rerun) and surfaces the real reason in the pytest XML report.
    reason = _timeout_reason(crashitem)
    if reason:
        report.longrepr = reason
    return None

# trylast: let xdist send this failing report to the controller BEFORE we
# os._exit, so results show the real failure reason instead of a worker crash.
@pytest.hookimpl(trylast=True)
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

        sys.stderr.write(f"\n!RESTART {_worker_id} {report.nodeid}\n")
        sys.stderr.flush()
        os._exit(_WORKER_RESTART_CODE)
