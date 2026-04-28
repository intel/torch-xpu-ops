# xpu_smi_install.py - Locate / install the `xpu-smi` binary.
#
# Public API:
#   ensure_xpu_smi() -> bool
#       Returns True if `xpu-smi` is available on PATH after the call,
#       False otherwise. On Linux uses apt-get/dnf (with sudo when not
#       root). On Windows downloads a pinned release zip from the
#       Intel xpumanager GitHub release with sha256 verification, size
#       cap, and zip-slip member validation, then prepends the extract
#       directory to PATH.

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

# --- pinned Windows release ---------------------------------------------------

_XPU_SMI_WIN_URL = (
    'https://github.com/intel/xpumanager/releases/download/v1.3.6/'
    'xpu-smi-1.3.6-20260206.143316.1004f6cb_win.zip'
)
_XPU_SMI_WIN_SHA256 = (
    'd1b44b5820d65317f453db070779332228372f8ebb3aae8c8f2e7937ba21f9de'
)
_XPU_SMI_WIN_MAX_BYTES = 50 * 1024 * 1024  # hard cap on download size
_XPU_SMI_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'xpu_smi_cache')


# --- helpers ------------------------------------------------------------------

def _log(msg):
    sys.stderr.write(f"[xpu-smi-install] {msg}\n")


def _is_windows():
    return platform.system() == 'Windows'


def which_xpu_smi():
    """Return the resolved path to xpu-smi[.exe] if it's on PATH, else None."""
    return shutil.which('xpu-smi') or shutil.which('xpu-smi.exe')


# --- secure download + extract ------------------------------------------------

def _download(url, dest, expected_sha256=None, max_bytes=None):
    """Download `url` to `dest`, enforcing https, size cap, and sha256."""
    if not url.lower().startswith('https://'):
        raise ValueError(f"refusing non-https download URL: {url!r}")
    req = urllib.request.Request(url, headers={'User-Agent': 'xpu-smi-install'})
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
        if not name or name.startswith(('/', '\\')) or (len(name) > 1 and name[1] == ':'):
            raise OSError(f"refusing absolute path in archive: {name!r}")
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


# --- platform installers ------------------------------------------------------

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
            if which_xpu_smi():
                return True
        except (subprocess.CalledProcessError, OSError) as e:
            _log(f"package install failed ({plan[-1][-1]}): {e}")
    return False


def _install_windows():
    """Download + extract the pinned xpu-smi zip and prepend its dir to PATH."""
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


# --- public entry point -------------------------------------------------------

def ensure_xpu_smi():
    """Make sure `xpu-smi` is on PATH; install if missing. Returns bool."""
    if which_xpu_smi():
        return True
    return _install_windows() if _is_windows() else _install_linux()


if __name__ == '__main__':
    ok = ensure_xpu_smi()
    print(which_xpu_smi() or '(not found)')
    sys.exit(0 if ok else 1)
