# xpu_worker_restart.py - Pytest plugin: restart xdist workers on XPU OOM
#
# Loaded via `-p scripts.xpu_worker_restart`. Spawns one `xpu-smi dump`
# process per host (the xdist controller), tails its CSV log, and restarts
# any worker whose target device crosses _GPU_MEM_THRESHOLD or whose test
# fails with a known fatal pattern. Workers are restarted via os._exit(101);
# pytest-xdist respawns them when --max-worker-restart allows.

import atexit
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import pytest

# --- tunables -----------------------------------------------------------------

_WORKER_RESTART_CODE = 101
_GPU_MEM_THRESHOLD = 80.0   # percent
_XPU_SMI_INTERVAL = 3       # seconds (xpu-smi -i)
_LOG_TAIL_BYTES = 8192      # bytes read from end of log per check
_LOG_PATH_ENV = 'XPU_SMI_LOG_PATH'
_LOG_OWNER_ENV = 'XPU_SMI_LOG_OWNER_PID'

# Fallback download URL (Windows only). On Linux we use the system package manager.
_XPU_SMI_WIN_URL = (
    'https://github.com/intel/xpumanager/releases/download/v1.3.6/'
    'xpu-smi-1.3.6-20260206.143316.1004f6cb_win.zip'
)
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
_log_path = None           # shared xpu-smi CSV log
_xpu_smi_proc = None       # only set in the process that owns xpu-smi


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

def _download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'xpu-watchdog'})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)


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
            _download(_XPU_SMI_WIN_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(_XPU_SMI_CACHE_DIR)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError) as e:
            _log(f"xpu-smi download/extract failed: {e}")
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


# --- xpu-smi process lifecycle (controller only) ------------------------------

def _start_xpu_smi():
    """Spawn xpu-smi in the background and publish its log path via env."""
    global _xpu_smi_proc, _log_path

    if _xpu_smi_proc is not None:
        return  # already running in this process

    if not _ensure_xpu_smi():
        _log("xpu-smi unavailable; memory monitoring disabled")
        return

    fd = None
    path = None
    try:
        fd, path = tempfile.mkstemp(prefix='xpu_smi_', suffix='.log')
        proc = subprocess.Popen(
            ['ZE_AFFINITY_MASK=', 'xpu-smi', 'dump', '-m', '5', '-i', str(_XPU_SMI_INTERVAL)],
            stdout=fd,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError) as e:
        _log(f"failed to spawn xpu-smi: {e}")
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        return
    finally:
        # Popen dup'd the fd; we can close our copy now.
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    _xpu_smi_proc = proc
    _log_path = path
    os.environ[_LOG_PATH_ENV] = path
    os.environ[_LOG_OWNER_ENV] = str(os.getpid())
    _log(f"started xpu-smi pid={proc.pid} log={path}")


def _stop_xpu_smi():
    """Owner-only: terminate xpu-smi and remove the log."""
    global _xpu_smi_proc, _log_path

    proc, _xpu_smi_proc = _xpu_smi_proc, None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        except OSError as e:
            _log(f"failed to stop xpu-smi: {e}")

    path, _log_path = _log_path, None
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# --- log reader (all processes) -----------------------------------------------

def _read_log_tail(path):
    try:
        with open(path, 'rb') as f:
            try:
                f.seek(-_LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode('utf-8', errors='ignore')
    except OSError:
        return ''


def _get_device_mem_util(device_id):
    """Return latest GPU memory utilization (%) for `device_id`, or None."""
    if not _log_path or not os.path.exists(_log_path):
        return None
    tail = _read_log_tail(_log_path)
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
