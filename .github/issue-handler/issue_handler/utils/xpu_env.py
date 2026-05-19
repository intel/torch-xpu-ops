"""XPU environment helpers.

Ensures oneAPI is sourced and PyTorch has XPU support before running tests.
Used by verify_existence, verify_fix, and code_fix.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PYTORCH_DIR
from .logger import log

# Shell preamble that sources oneAPI + activates the pytorch venv.
# Use this as prefix for any subprocess that needs XPU support.
ENV_SETUP = (
    "source ~/intel/oneapi/setvars.sh --force 2>/dev/null; "
    "source ~/pytorch/.venv/bin/activate; "
)

# Build flags for XPU-enabled PyTorch builds.
XPU_BUILD_FLAGS = "TORCH_XPU_ARCH_LIST=pvc USE_XPU=1"


def check_xpu_available() -> bool:
    """Check if PyTorch has XPU support and a device is visible.

    Sources oneAPI first, then imports torch and checks xpu.is_available().
    Returns True if XPU device is available.
    """
    cmd = (
        ENV_SETUP +
        "python -c 'import torch; print(torch.xpu.is_available())'"
    )
    try:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=60,
        )
        return "True" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def rebuild_pytorch(issue: int | None = None) -> tuple[bool, str]:
    """Rebuild PyTorch with USE_XPU=1.

    Removes CMakeCache.txt first to force cmake reconfigure — otherwise
    cmake reuses the cached USE_XPU=OFF and the build is a no-op.

    Returns (success, output).
    """
    log("WARN", "XPU not available — rebuilding PyTorch with USE_XPU=1",
        issue=issue)
    cmake_cache = PYTORCH_DIR / "build" / "CMakeCache.txt"
    if cmake_cache.exists():
        log("INFO", "Removing CMakeCache.txt to force reconfigure", issue=issue)
        cmake_cache.unlink()
    cmd = (
        ENV_SETUP +
        f"cd {PYTORCH_DIR} && {XPU_BUILD_FLAGS} python setup.py develop"
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=3600, shell=True, executable="/bin/bash",
        )
        output = (result.stdout + result.stderr)[-5000:]
        if result.returncode == 0:
            log("INFO", "PyTorch rebuilt with XPU support", issue=issue)
            return True, output
        log("ERROR", "PyTorch XPU rebuild failed", issue=issue)
        return False, output
    except subprocess.TimeoutExpired:
        return False, "XPU rebuild timed out (60min)"


def ensure_xpu_ready(issue: int | None = None) -> bool:
    """Ensure oneAPI is sourced and PyTorch has XPU support.

    If XPU is not available, triggers a rebuild with USE_XPU=1.
    Returns True if XPU is ready after all attempts, False if unrecoverable.
    """
    if check_xpu_available():
        return True

    # Auto-rebuild
    ok, output = rebuild_pytorch(issue=issue)
    if not ok:
        log("ERROR", f"Cannot get XPU ready: rebuild failed\n{output[-500:]}",
            issue=issue)
        return False

    # Re-check after rebuild
    if check_xpu_available():
        return True

    log("ERROR", "XPU still not available after rebuild", issue=issue)
    return False
