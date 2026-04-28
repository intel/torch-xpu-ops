# conftest.py - Ultra-minimal worker restart

import os
import sys
import re
import shutil
import subprocess
import tempfile
import atexit
import platform
import urllib.request
import zipfile
import pytest

_WORKER_RESTART_CODE = 101
_GPU_MEM_THRESHOLD = 80.0  # percent
_XPU_SMI_INTERVAL = 3      # seconds, passed to xpu-smi -i

# Fallback download URL (Windows only). On Linux we use the system package manager.
_XPU_SMI_WIN_URL = (
    'https://github.com/intel/xpumanager/releases/download/v1.3.6/'
    'xpu-smi-1.3.6-20260206.143316.1004f6cb_win.zip'
)
_XPU_SMI_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'xpu_smi_cache')

# Single global variables
_worker_id = None
_xpu_smi_proc = None
_xpu_smi_log_path = None

patterns = [
    'ur_result_error',
    'segmentation fault',
    'bus error',
    'kernel died',
    'illegal memory',
    re.compile(r'failed on setup with.*crashed while running'),
    re.compile(r'out.*of.*memory'),
]


def _get_device_id():
    """Pick device id from ZE_AFFINITY_MASK (first entry); default to 0."""
    mask = os.environ.get('ZE_AFFINITY_MASK', '').strip()
    if mask:
        first = mask.split(',', 1)[0].strip()
        if first.isdigit():
            return first
    return '0'


def _download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'conftest-xpu-smi'})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)


def _find_in_dir(root, target):
    """Return absolute path to a file named `target` (case-insensitive) under root, or None."""
    target_l = target.lower()
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower() == target_l:
                return os.path.join(dirpath, name)
    return None


def _install_xpu_smi_linux():
    """Install xpu-smi via system package manager (apt-get or dnf, with sudo if needed)."""
    sudo = [] if os.geteuid() == 0 else (['sudo', '-n'] if shutil.which('sudo') else [])
    candidates = []
    if shutil.which('apt-get'):
        candidates.append([
            sudo + ['apt-get', 'update'],
            sudo + ['apt-get', 'install', '-y', 'xpu-smi'],
        ])
    if shutil.which('dnf'):
        candidates.append([
            sudo + ['dnf', 'install', '-y', 'xpu-smi'],
        ])
    for cmds in candidates:
        try:
            for cmd in cmds:
                subprocess.check_call(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            if shutil.which('xpu-smi'):
                return True
        except Exception:
            continue
    return False


def _install_xpu_smi_windows():
    """Download + extract xpu-smi zip into a cache dir and prepend its dir to PATH."""
    exe_name = 'xpu-smi.exe'
    os.makedirs(_XPU_SMI_CACHE_DIR, exist_ok=True)

    cached = _find_in_dir(_XPU_SMI_CACHE_DIR, exe_name)
    if not cached:
        try:
            archive = os.path.join(_XPU_SMI_CACHE_DIR, os.path.basename(_XPU_SMI_WIN_URL))
            _download(_XPU_SMI_WIN_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(_XPU_SMI_CACHE_DIR)
        except Exception as e:
            sys.stderr.write(f"[conftest] xpu-smi download failed: {e}\n")
            return False
        cached = _find_in_dir(_XPU_SMI_CACHE_DIR, exe_name)
    if not cached:
        return False
    bin_dir = os.path.dirname(cached)
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    return True


def _ensure_xpu_smi():
    """Make sure `xpu-smi` is callable; install/download if missing."""
    if shutil.which('xpu-smi') or shutil.which('xpu-smi.exe'):
        return True
    if platform.system() == 'Windows':
        return _install_xpu_smi_windows()
    return _install_xpu_smi_linux()


def _start_xpu_smi():
    """Spawn `xpu-smi dump` in the background, streaming to a temp log file."""
    global _xpu_smi_proc, _xpu_smi_log_path
    if _xpu_smi_proc is not None:
        return
    if not _ensure_xpu_smi():
        sys.stderr.write("[conftest] xpu-smi not available; memory monitoring disabled\n")
        return
    try:
        fd, _xpu_smi_log_path = tempfile.mkstemp(prefix='xpu_smi_', suffix='.log')
        _xpu_smi_proc = subprocess.Popen(
            ['xpu-smi', 'dump', '-d', _get_device_id(), '-m', '5', '-i', str(_XPU_SMI_INTERVAL)],
            stdout=fd,
            stderr=subprocess.DEVNULL,
        )
        os.close(fd)
    except Exception:
        _xpu_smi_proc = None
        if _xpu_smi_log_path:
            try:
                os.unlink(_xpu_smi_log_path)
            except Exception:
                pass
            _xpu_smi_log_path = None


def _stop_xpu_smi():
    """Terminate xpu-smi and clean up the log file."""
    global _xpu_smi_proc, _xpu_smi_log_path
    if _xpu_smi_proc is not None:
        try:
            _xpu_smi_proc.terminate()
            try:
                _xpu_smi_proc.wait(timeout=2)
            except Exception:
                _xpu_smi_proc.kill()
        except Exception:
            pass
        _xpu_smi_proc = None
    if _xpu_smi_log_path:
        try:
            os.unlink(_xpu_smi_log_path)
        except Exception:
            pass
        _xpu_smi_log_path = None


def _get_xpu_mem_util():
    """Read the latest GPU memory utilization (%) from the xpu-smi log, or None."""
    if not _xpu_smi_log_path:
        return None
    try:
        with open(_xpu_smi_log_path, 'rb') as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode('utf-8', errors='ignore')
    except Exception:
        return None
    for line in reversed(tail.strip().splitlines()):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None


def _restart_worker(reason):
    try:
        import gc
        import torch
        gc.collect()
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    except Exception:
        pass
    sys.stderr.write(f"\n!RESTART {_worker_id} ({reason})\n")
    _stop_xpu_smi()
    os._exit(_WORKER_RESTART_CODE)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _worker_id
    try:
        _worker_id = config.workerinput.get('workerid')
    except Exception:
        pass
    # Start xpu-smi once per pytest process (master + each xdist worker)
    _start_xpu_smi()
    atexit.register(_stop_xpu_smi)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    _stop_xpu_smi()


@pytest.hookimpl
def pytest_runtest_logreport(report):
    if not _worker_id:
        return

    if report.failed:
        err_msg = str(report.longrepr).lower() if report.longrepr else ''
        # Direct inline pattern checks (fastest)
        if any(p in err_msg if isinstance(p, str) else p.search(err_msg) for p in patterns):
            _restart_worker("error pattern matched")

    # Check GPU memory utilization at the end of each test phase
    if report.when == 'teardown':
        mem = _get_xpu_mem_util()
        if mem is not None and mem >= _GPU_MEM_THRESHOLD:
            _restart_worker(f"GPU mem {mem:.2f}% >= {_GPU_MEM_THRESHOLD}%")
