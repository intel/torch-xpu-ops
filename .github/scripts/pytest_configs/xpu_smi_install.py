"""Locate or install the ``xpu-smi`` binary.

Public API
----------
which_xpu_smi() -> str | None
    Resolved path to ``xpu-smi``/``xpu-smi.exe`` on ``PATH``, else ``None``.
ensure_xpu_smi() -> bool
    ``True`` if ``xpu-smi`` is on ``PATH`` after the call. Installs from the
    distro package manager (Linux) or a pinned, sha256-verified GitHub
    release zip (Windows). Never raises.
"""
from __future__ import annotations

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
from contextlib import suppress
from pathlib import Path

# --- pinned Windows release ---------------------------------------------------

_WIN_URL = (
    "https://github.com/intel/xpumanager/releases/download/v1.3.6/"
    "xpu-smi-1.3.6-20260206.143316.1004f6cb_win.zip"
)
_WIN_SHA256 = "d1b44b5820d65317f453db070779332228372f8ebb3aae8c8f2e7937ba21f9de"
_WIN_MAX_BYTES = 50 * 1024 * 1024
_CACHE_DIR = Path(tempfile.gettempdir()) / "xpu_smi_cache"


def _log(msg: str) -> None:
    print(f"[xpu-smi-install] {msg}", file=sys.stderr)


def which_xpu_smi() -> str | None:
    return shutil.which("xpu-smi") or shutil.which("xpu-smi.exe")


# --- secure download + extract ------------------------------------------------

def _download(url: str, dest: Path, *, sha256: str | None = None,
              max_bytes: int | None = None) -> None:
    """Stream ``url`` to ``dest``: https-only, size-capped, sha256-pinned.

    Removes ``dest`` on any failure.
    """
    if not url.lower().startswith("https://"):
        raise ValueError(f"refusing non-https download URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "xpu-smi-install"})
    digest = hashlib.sha256()
    written = 0
    try:
        # nosec B310 - URL is hard-coded to a pinned GitHub release asset above
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fp:
            while chunk := resp.read(64 * 1024):
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise OSError(
                        f"download exceeded {max_bytes} bytes "
                        f"(got >= {written})"
                    )
                digest.update(chunk)
                fp.write(chunk)
        if sha256 and digest.hexdigest().lower() != sha256.lower():
            raise OSError(
                f"sha256 mismatch for {url}: got {digest.hexdigest()}, "
                f"expected {sha256}"
            )
    except BaseException:
        with suppress(OSError):
            dest.unlink(missing_ok=True)
        raise


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Validate every member, then extract (zip-slip / symlink mitigation)."""
    dest_real = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if not name or name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
            raise OSError(f"refusing absolute path in archive: {name!r}")
        if (info.external_attr >> 16) & 0xF000 == 0xA000:
            raise OSError(f"refusing symlink in archive: {name!r}")
        target = (dest_real / name).resolve()
        if dest_real != target and dest_real not in target.parents:
            raise OSError(f"refusing path traversal in archive: {name!r}")
    zf.extractall(dest)  # noqa: S202 - members validated above


def _find_in_dir(root: Path, target: str) -> Path | None:
    target_l = target.lower()
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == target_l:
            return path
    return None


# --- platform installers ------------------------------------------------------

def _sudo_prefix() -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    return ["sudo", "-n"] if shutil.which("sudo") else []


def _install_linux() -> bool:
    sudo = _sudo_prefix()
    plans: list[list[list[str]]] = []
    if shutil.which("apt-get"):
        plans.append([
            sudo + ["apt-get", "update"],
            sudo + ["apt-get", "install", "-y", "xpu-smi"],
        ])
    if shutil.which("dnf"):
        plans.append([sudo + ["dnf", "install", "-y", "xpu-smi"]])

    for plan in plans:
        try:
            for cmd in plan:
                subprocess.run(
                    cmd, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except (subprocess.CalledProcessError, OSError) as e:
            _log(f"package install failed ({plan[-1][-1]}): {e}")
            continue
        if which_xpu_smi():
            return True
    return False


def _install_windows() -> bool:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log(f"cache dir creation failed: {e}")
        return False

    if (cached := _find_in_dir(_CACHE_DIR, "xpu-smi.exe")) is None:
        archive = _CACHE_DIR / Path(_WIN_URL).name
        try:
            _download(_WIN_URL, archive, sha256=_WIN_SHA256, max_bytes=_WIN_MAX_BYTES)
            with zipfile.ZipFile(archive) as zf:
                _safe_extract_zip(zf, _CACHE_DIR)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError, ValueError) as e:
            _log(f"xpu-smi download/extract failed: {e}")
            return False
        cached = _find_in_dir(_CACHE_DIR, "xpu-smi.exe")

    if cached is None:
        _log("xpu-smi.exe not found after extraction")
        return False
    os.environ["PATH"] = os.pathsep.join(
        [str(cached.parent), os.environ.get("PATH", "")]
    )
    return True


# --- public entry point -------------------------------------------------------

def ensure_xpu_smi() -> bool:
    """Make sure ``xpu-smi`` is on ``PATH``; install if missing. Never raises."""
    if which_xpu_smi():
        return True
    try:
        if platform.system() == "Windows":
            return _install_windows()
        return _install_linux()
    except Exception as e:  # noqa: BLE001 - last-resort guard for CI
        _log(f"unexpected install failure: {e!r}")
        return False


if __name__ == "__main__":
    ok = ensure_xpu_smi()
    print(which_xpu_smi() or "(not found)")
    sys.exit(0 if ok else 1)
