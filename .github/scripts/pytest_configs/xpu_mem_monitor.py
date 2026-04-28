# xpu_mem_monitor.py - Background poller for `xpu-smi` GPU memory utilization.
#
# Public API:
#   start(interval=3, timeout=60) -> Optional[str]
#       Spawn a daemon thread that runs `xpu-smi dump -m 5 -n 1` every
#       `interval` seconds and atomically rewrites a small snapshot file
#       containing only the latest sample. Returns the log path, or None
#       if `xpu-smi` is unavailable. Idempotent.
#   stop() -> None
#       Stop the poller and remove the snapshot file.
#   attach(path) -> None
#       Tell this process to read from an existing snapshot file (e.g. a
#       worker inheriting the controller's log path via env).
#   get_device_mem_util(device_id) -> Optional[float]
#       Return the latest GPU memory utilization (%) for `device_id`
#       (string), or None if the snapshot is unavailable / row missing.
#   log_path() -> Optional[str]
#       Current snapshot file path, or None.

import os
import re
import subprocess
import sys
import tempfile
import threading

from xpu_smi_install import ensure_xpu_smi

# Matches an xpu-smi CSV row like:  "23:59:21.296,    0, 0.85"
_ROW_RE = re.compile(
    r'^\s*\d{1,2}:\d{2}:\d{2}\.\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*$'
)

# Module state -----------------------------------------------------------------

_log_path = None         # snapshot file (writer or reader, both processes)
_poller_thread = None    # only set in the process that owns the poller
_poller_stop = None      # threading.Event to signal the poller to exit
_interval = 3
_timeout = 60


def _log(msg):
    sys.stderr.write(f"[xpu-mem] {msg}\n")


# Snapshot writer (controller side) --------------------------------------------

def _take_snapshot():
    """Run `xpu-smi dump -m 5 -n 1` once and return its stdout, or None."""
    try:
        out = subprocess.check_output(
            ['xpu-smi', 'dump', '-m', '5', '-n', '1'],
            stderr=subprocess.DEVNULL,
            timeout=_timeout,
        )
    except (subprocess.SubprocessError, OSError) as e:
        _log(f"snapshot failed: {e}")
        return None
    return out.decode('utf-8', errors='ignore')


def _atomic_write(path, text):
    """Write `text` to `path` atomically (rename within the same dir)."""
    dirname = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.xpu_smi_', suffix='.tmp', dir=dirname)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError as e:
        _log(f"snapshot write failed: {e}")
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _poller_loop(stop_event, path):
    """Take a snapshot every `_interval` seconds and overwrite `path`."""
    while not stop_event.is_set():
        snap = _take_snapshot()
        if snap:
            _atomic_write(path, snap)
        if stop_event.wait(_interval):
            return


# Public API -------------------------------------------------------------------

def start(interval=3, timeout=60):
    """Start the snapshot poller. Returns the log path, or None on failure."""
    global _poller_thread, _poller_stop, _log_path, _interval, _timeout

    if _poller_thread is not None:
        return _log_path

    if not ensure_xpu_smi():
        _log("xpu-smi unavailable; memory monitoring disabled")
        return None

    _interval = interval
    _timeout = timeout

    path = None
    try:
        fd, path = tempfile.mkstemp(prefix='xpu_smi_', suffix='.log')
        os.close(fd)
    except OSError as e:
        _log(f"failed to create snapshot file: {e}")
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        return None

    _poller_stop = threading.Event()
    _poller_thread = threading.Thread(
        target=_poller_loop, args=(_poller_stop, path),
        name='xpu-smi-poller', daemon=True,
    )
    _poller_thread.start()
    _log_path = path
    _log(f"started poller every {_interval}s, log={path}")
    return path


def stop():
    """Stop the poller (if owned by this process) and remove the log."""
    global _poller_thread, _poller_stop, _log_path

    stop_event, _poller_stop = _poller_stop, None
    if stop_event is not None:
        stop_event.set()

    thread, _poller_thread = _poller_thread, None
    if thread is not None:
        thread.join(timeout=_timeout + 1)

    path, _log_path = _log_path, None
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def attach(path):
    """Reader-side: bind this process to an already-running poller's log."""
    global _log_path
    _log_path = path or None


def log_path():
    return _log_path


# Snapshot reader --------------------------------------------------------------

def _read_snapshot(path):
    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except OSError:
        return ''


def get_device_mem_util(device_id):
    """Return latest GPU memory utilization (%) for `device_id`, or None."""
    if not _log_path or not os.path.exists(_log_path):
        return None
    text = _read_snapshot(_log_path)
    if not text:
        return None
    for line in reversed(text.splitlines()):
        m = _ROW_RE.match(line)
        if m and m.group(1) == device_id:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    return None
