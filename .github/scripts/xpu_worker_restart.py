# xpu_worker_restart.py - Pytest plugin: restart xdist workers on XPU OOM
#
# Loaded via `-p scripts.xpu_worker_restart`. On the xdist controller a
# background thread polls `xpu-smi dump -m 5 -n 1` every _XPU_SMI_INTERVAL
# seconds and atomically overwrites a small snapshot file with the latest
# sample. Workers read that file and restart themselves (os._exit(101))
# whenever their target device crosses _GPU_MEM_THRESHOLD or a test fails
# with a known fatal pattern. pytest-xdist respawns workers under
# --max-worker-restart.

import atexit
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile

import pytest

# --- tunables -----------------------------------------------------------------

_WORKER_RESTART_CODE = 101
_GPU_MEM_THRESHOLD = 85.0   # percent
_XPU_SMI_INTERVAL = 3       # seconds between snapshots
_XPU_SMI_TIMEOUT = 10       # seconds, per `xpu-smi dump -n 1` invocation
_LOG_PATH_ENV = 'XPU_SMI_LOG_PATH'
_LOG_OWNER_ENV = 'XPU_SMI_LOG_OWNER_PID'

# Fallback download (Windows only). On Linux we use the system package manager.
# The SHA-256 is pinned so a tampered or substituted asset is rejected before
# extraction, and ZipFile members are validated to mitigate "zip slip".
_XPU_SMI_WIN_URL = (
    'https://github.com/intel/xpumanager/releases/download/v1.3.6/'
    'xpu-smi-1.3.6-20260206.143316.1004f6cb_win.zip'
)
_XPU_SMI_WIN_SHA256 = (
    'd1b44b5820d65317f453db070779332228372f8ebb3aae8c8f2e7937ba21f9de'
)
_XPU_SMI_WIN_MAX_BYTES = 50 * 1024 * 1024  # hard cap on download size
_XPU_SMI_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'xpu_smi_cache')

# Matches a CSV row like:  "23:59:21.296,    0, 0.85"
_ROW_RE = re.compile(
    r'^\s*\d{1,2}:\d{2}:\d{2}\.\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*$'
)

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

_worker_id = None          # xdist worker id, or None on controller
_target_device = '0'       # this worker's GPU index (string)
_log_path = None           # shared xpu-smi snapshot file
_poller_thread = None      # only set in the process that owns the poller
_poller_stop = None        # threading.Event used to ask the poller to exit


# --- helpers ------------------------------------------------------------------

def _log(msg):
    sys.stderr.write(f"[xpu-watchdog] {msg}\n")


def _is_windows():
    return platform.system() == 'Windows'


def _which_xpu_smi():
    return shutil.which('xpu-smi') or shutil.which('xpu-smi.exe')


def _get_target_device():
    """Pick device id from ZE_AFFINITY_MASK (first entry); default to '0'."""
    mask = os.environ.get('ZE_AFFINITY_MASK', '').strip()
    if mask:
        first = mask.split(',', 1)[0].strip()
        if first.isdigit():
            return first
    return '0'


# --- xpu-smi installation -----------------------------------------------------

def _download(url, dest, expected_sha256=None, max_bytes=None):
    """Download `url` to `dest`, enforcing https, size cap, and sha256."""
    if not url.lower().startswith('https://'):
        raise ValueError(f"refusing non-https download URL: {url!r}")
    req = urllib.request.Request(url, headers={'User-Agent': 'xpu-watchdog'})
    h = hashlib.sha256()
    written = 0
    # nosec B310 - URL is hard-coded to a pinned GitHub release asset above
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as f:
        while True:
            chunk = r.read(64 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise OSError(
                    f"download exceeded {max_bytes} bytes (got >= {written})"
                )
            h.update(chunk)
            f.write(chunk)
    if expected_sha256:
        digest = h.hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise OSError(
                f"sha256 mismatch for {url}: got {digest}, "
                f"expected {expected_sha256}"
            )


def _safe_extract_zip(zf, dest_dir):
    """Extract `zf` into `dest_dir`, rejecting absolute paths, '..', and
    symlinks (zip slip mitigation)."""
    dest_real = os.path.realpath(dest_dir)
    for info in zf.infolist():
        name = info.filename
        # Normalize separators and forbid drive letters / absolute paths.
        if not name or name.startswith(('/', '\\')) or (len(name) > 1 and name[1] == ':'):
            raise OSError(f"refusing absolute path in archive: {name!r}")
        # External attribute high bits indicate a unix symlink (mode 0xA000).
        mode = (info.external_attr >> 16) & 0xFFFF
        if (mode & 0xF000) == 0xA000:
            raise OSError(f"refusing symlink in archive: {name!r}")
        target = os.path.realpath(os.path.join(dest_real, name))
        if os.path.commonpath([dest_real, target]) != dest_real:
            raise OSError(f"refusing path traversal in archive: {name!r}")
    zf.extractall(dest_dir)  # noqa: S202 - members validated above


def _find_in_dir(root, target):
    target_l = target.lower()
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower() == target_l:
                return os.path.join(dirpath, name)
    return None


def _install_linux():
    """Install xpu-smi via apt-get or dnf, with sudo when not root."""
    is_root = (hasattr(os, 'geteuid') and os.geteuid() == 0)
    sudo = [] if is_root else (['sudo', '-n'] if shutil.which('sudo') else [])
    plans = []
    if shutil.which('apt-get'):
        plans.append([
            sudo + ['apt-get', 'update'],
            sudo + ['apt-get', 'install', '-y', 'xpu-smi'],
        ])
    if shutil.which('dnf'):
        plans.append([sudo + ['dnf', 'install', '-y', 'xpu-smi']])
    for plan in plans:
        try:
            for cmd in plan:
                subprocess.check_call(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            if _which_xpu_smi():
                return True
        except (subprocess.CalledProcessError, OSError) as e:
            _log(f"package install failed ({plan[-1][-1]}): {e}")
    return False


def _install_windows():
    """Download + extract xpu-smi zip, prepend its dir to PATH."""
    exe = 'xpu-smi.exe'
    try:
        os.makedirs(_XPU_SMI_CACHE_DIR, exist_ok=True)
    except OSError as e:
        _log(f"cache dir creation failed: {e}")
        return False

    cached = _find_in_dir(_XPU_SMI_CACHE_DIR, exe)
    if not cached:
        archive = os.path.join(
            _XPU_SMI_CACHE_DIR, os.path.basename(_XPU_SMI_WIN_URL)
        )
        try:
            _download(
                _XPU_SMI_WIN_URL,
                archive,
                expected_sha256=_XPU_SMI_WIN_SHA256,
                max_bytes=_XPU_SMI_WIN_MAX_BYTES,
            )
            with zipfile.ZipFile(archive) as zf:
                _safe_extract_zip(zf, _XPU_SMI_CACHE_DIR)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError, ValueError) as e:
            _log(f"xpu-smi download/extract failed: {e}")
            # Best-effort: drop a partial/untrusted archive.
            if os.path.exists(archive):
                try:
                    os.unlink(archive)
                except OSError:
                    pass
            return False
        cached = _find_in_dir(_XPU_SMI_CACHE_DIR, exe)
    if not cached:
        _log("xpu-smi.exe not found after extraction")
        return False
    os.environ['PATH'] = os.path.dirname(cached) + os.pathsep + os.environ.get('PATH', '')
    return True


def _ensure_xpu_smi():
    if _which_xpu_smi():
        return True
    return _install_windows() if _is_windows() else _install_linux()


# --- xpu-smi snapshot poller (controller only) --------------------------------

def _take_snapshot():
    """Run `xpu-smi dump -m 5 -n 1` once and return its stdout, or None."""
    try:
        out = subprocess.check_output(
            ['xpu-smi', 'dump', '-m', '5', '-n', '1'],
            stderr=subprocess.DEVNULL,
            timeout=_XPU_SMI_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError) as e:
        _log(f"xpu-smi snapshot failed: {e}")
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
    """Take an xpu-smi snapshot every _XPU_SMI_INTERVAL seconds and
    overwrite `path` with just the latest snapshot (so the file stays small)."""
    while not stop_event.is_set():
        snap = _take_snapshot()
        if snap:
            _atomic_write(path, snap)
        # stop_event.wait returns True if set during the wait -> exit promptly.
        if stop_event.wait(_XPU_SMI_INTERVAL):
            return


def _start_xpu_smi():
    """Start the snapshot poller and publish its log path via env."""
    global _poller_thread, _poller_stop, _log_path

    if _poller_thread is not None:
        return  # already running in this process

    if not _ensure_xpu_smi():
        _log("xpu-smi unavailable; memory monitoring disabled")
        return

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
        return

    _poller_stop = threading.Event()
    _poller_thread = threading.Thread(
        target=_poller_loop, args=(_poller_stop, path),
        name='xpu-smi-poller', daemon=True,
    )
    _poller_thread.start()
    _log_path = path
    os.environ[_LOG_PATH_ENV] = path
    os.environ[_LOG_OWNER_ENV] = str(os.getpid())
    _log(f"started xpu-smi poller every {_XPU_SMI_INTERVAL}s, log={path}")


def _stop_xpu_smi():
    """Owner-only: stop the poller and remove the log."""
    global _poller_thread, _poller_stop, _log_path

    stop_event, _poller_stop = _poller_stop, None
    if stop_event is not None:
        stop_event.set()

    thread, _poller_thread = _poller_thread, None
    if thread is not None:
        thread.join(timeout=_XPU_SMI_TIMEOUT + 1)

    path, _log_path = _log_path, None
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# --- log reader (all processes) -----------------------------------------------

def _read_snapshot(path):
    """Read the full snapshot file (one xpu-smi sample, ~few hundred bytes)."""
    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except OSError:
        return ''


def _get_device_mem_util(device_id):
    """Return latest GPU memory utilization (%) for `device_id`, or None."""
    if not _log_path or not os.path.exists(_log_path):
        return None
    tail = _read_snapshot(_log_path)
    if not tail:
        return None
    for line in reversed(tail.splitlines()):
        m = _ROW_RE.match(line)
        if m and m.group(1) == device_id:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    return None


# --- restart ------------------------------------------------------------------

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
    sys.stderr.write(f"\n!RESTART {_worker_id} ({reason})\n")
    sys.stderr.flush()
    os._exit(_WORKER_RESTART_CODE)


# --- pytest hooks -------------------------------------------------------------

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id, _target_device, _log_path

    try:
        _worker_id = config.workerinput.get('workerid')
    except AttributeError:
        _worker_id = None

    _target_device = _get_target_device()

    if _worker_id is None:
        # xdist controller (or non-xdist run): own the xpu-smi process.
        _start_xpu_smi()
        atexit.register(_stop_xpu_smi)
    else:
        # xdist worker: inherit log path from controller via env.
        _log_path = os.environ.get(_LOG_PATH_ENV) or None


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    # Only the owner cleans up the shared log/process.
    owner_pid = os.environ.get(_LOG_OWNER_ENV)
    if owner_pid and owner_pid == str(os.getpid()):
        _stop_xpu_smi()


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
        mem = _get_device_mem_util(_target_device)
        if mem is not None and mem >= _GPU_MEM_THRESHOLD:
            _restart_worker(
                f"device {_target_device} mem {mem:.2f}% >= {_GPU_MEM_THRESHOLD}%"
            )
