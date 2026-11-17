"""Ultra-minimal xdist worker restart on fatal XPU failures.

When running under pytest-xdist, an unrecoverable failure (device crash, OOM,
segfault, etc.) can leave a worker in a bad state. This plugin detects such
failures from the test report and forcibly exits the worker process so xdist
respawns a fresh one, while still forwarding the real failure to the controller.

On the controller, controller.XpuController health-checks the worker's GPU
before respawning it and recovers broken cards (see README).
"""

import os
import re
import sys
import threading
import traceback

import pytest

_WORKER_RESTART_CODE = 101
# Verdicts a worker puts on a pytest-timeout failure (lowercase: matched as such).
GPU_HANG_TAG = "[xpu-timeout] gpu hang"
TEST_TIMEOUT_TAG = "[xpu-timeout] test timeout"

# Populated in pytest_configure when running inside an xdist worker.
_worker_id = None
# Set once a timed-out worker is reporting and exiting on its own.
_exiting = False

patterns = [
    "ur_result_error",
    "segmentation fault",
    "bus error",
    "kernel died",
    "illegal memory",
    "device lost",
    "device_lost",
    GPU_HANG_TAG,
    re.compile(r"failed on setup with.*crashed while running"),
    re.compile(r"out.*of.*memory"),
]


def is_fatal(report):
    """A failure after which the worker restarts itself."""
    err_msg = str(report.longrepr).lower() if report.failed and report.longrepr else ""
    return any(
        p in err_msg if isinstance(p, str) else p.search(err_msg) for p in patterns
    )


def pytest_addoption(parser):
    group = parser.getgroup("xpu-recovery", "XPU worker GPU recovery (xdist only)")
    group.addoption(
        "--xpu-crash-reruns",
        type=int,
        default=1,
        help="rerun a test whose worker crashed or hung this many times on another worker "
        "(default 1, 0 disables)",
    )
    group.addoption(
        "--xpu-hang-timeout",
        type=int,
        default=None,
        help="kill a worker stuck on one test this long, seconds "
        "(default: pytest-timeout + max(300, pytest-timeout); 0 disables)",
    )
    group.addoption(
        "--xpu-offline-timeout",
        type=int,
        default=600,
        help="how long a worker with a broken GPU waits for recovery before the card is "
        "excluded, seconds (default 600)",
    )
    group.addoption(
        "--xpu-recover-cmd",
        default=None,
        help="command that recovers a card given its PCI BDF "
        "(default: $XPU_RECOVER_CMD, set by detect-devices.sh)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id
    try:
        _worker_id = config.workerinput.get("workerid")
    except Exception:
        pass


# tryfirst: patch the xdist session before its own sessionstart spawns workers.
@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    pm = session.config.pluginmanager
    dsession = pm.getplugin("dsession")
    if _worker_id or dsession is None or pm.has_plugin("xpu_controller"):
        return
    from .controller import XpuController

    controller = XpuController(session.config, dsession)
    pm.register(controller, "xpu_controller")
    controller.start()


# Workers take over pytest-timeout's thread timer to tell a GPU hang from a slow test.
@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    if not _worker_id or settings.method != "thread":
        return None
    timer = threading.Timer(settings.timeout, _on_timeout, (item, settings))
    timer.daemon = True
    item._xpu_timer = timer
    timer.start()
    return True


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_cancel_timer(item):
    timer = getattr(item, "_xpu_timer", None)
    if timer is None:
        return None
    item._xpu_cancelled = True
    timer.cancel()
    return True


def _on_timeout(item, settings):
    """Report the timed-out test with a GPU verdict and stacks, then exit the worker."""
    global _exiting
    from . import gpu

    stacks = _thread_stacks()
    ok, detail, _ = gpu.probe(os.environ.get("ZE_AFFINITY_MASK"))
    if getattr(item, "_xpu_cancelled", False):
        return
    _exiting = True
    if ok:
        verdict = f"{TEST_TIMEOUT_TAG}: the test is slow or stuck, the GPU still runs kernels ({detail})"
    else:
        verdict = f"{GPU_HANG_TAG}: the GPU does not run kernels any more ({detail})"
    sections = []
    capman = item.config.pluginmanager.getplugin("capturemanager")
    try:
        capman.suspend_global_capture(item)
        out, err = capman.read_global_capture()
        sections = [("Captured stdout call", out), ("Captured stderr call", err)]
    except Exception:
        pass
    report = pytest.TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords=dict.fromkeys(item.keywords, 1),
        outcome="failed",
        longrepr=f"Failed: Timeout >{settings.timeout}s\n{verdict}\n\n{stacks}",
        when="call",
        sections=[s for s in sections if s[1]],
        duration=float(settings.timeout),
        user_properties=list(item.user_properties),
    )
    try:
        item.ihook.pytest_runtest_logreport(report=report)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _thread_stacks():
    names = {t.ident: t.name for t in threading.enumerate()}
    current = threading.get_ident()
    return "\n".join(
        f"Stack of {names.get(ident, '<unknown>')} ({ident}):\n"
        + "".join(traceback.format_stack(frame))
        for ident, frame in sys._current_frames().items()
        if ident != current
    )


# trylast: let xdist send this failing report to the controller BEFORE we
# os._exit, so results show the real failure reason instead of a worker crash.
@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if not _worker_id or _exiting or not is_fatal(report):
        return

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
