"""GPU and host memory monitoring via xpu-smi and /proc/meminfo."""

import re
import shutil
import subprocess
import threading

from .config import TestTask, gpu_memory_monitor_enabled, error_patterns
from .log import log

# Cache for xpu-smi availability
_xpu_smi_available: bool | None = None
_xpu_smi_lock = threading.Lock()

# Cache for tiles-per-device (probed once from device 0)
_tiles_per_device: int | None = None
_tiles_per_device_lock = threading.Lock()


def parse_memory_threshold() -> float | None:
    """Extract memory threshold from error_patterns (e.g. 'Memory>0.9' → 0.9)."""
    for pattern in error_patterns:
        m = re.match(r'Memory>(\d+\.?\d*)', pattern)
        if m:
            return float(m.group(1))
    return None


def _is_xpu_smi_available() -> bool:
    global _xpu_smi_available
    with _xpu_smi_lock:
        if _xpu_smi_available is None:
            _xpu_smi_available = shutil.which("xpu-smi") is not None
        return _xpu_smi_available


def _get_tiles_per_device() -> int:
    """Return tiles-per-device, probed once from device 0 via xpu-smi stats."""
    global _tiles_per_device
    with _tiles_per_device_lock:
        if _tiles_per_device is None:
            try:
                result = subprocess.run(
                    ["xpu-smi", "stats", "-d", "0"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    tile_ids = {int(m.group(1)) for m in re.finditer(r'Tile\s+(\d+)', result.stdout)}
                    _tiles_per_device = max(len(tile_ids), 1)
                else:
                    _tiles_per_device = 1
            except Exception:
                _tiles_per_device = 1
            log(f"Detected {_tiles_per_device} tile(s) per device")
        return _tiles_per_device


def get_gpu_memory_utilization(
    card: int,
    memory_threshold: float | None = None,
    task: TestTask | None = None,
) -> float | None:
    """Return GPU memory utilisation for *card* as a 0.0-1.0 fraction, or *None*.

    Multi-tile layout (N tiles/device): card C → device C//N, tile C%N.
    Single-tile layout: card C → device C.
    Also logs GPU Utilization, Frequency, Core Temperature, and Memory Bandwidth.
    """
    if not gpu_memory_monitor_enabled:
        return None
    if not _is_xpu_smi_available():
        return None

    try:
        tiles = _get_tiles_per_device()
        device_id, tile_id = (card // tiles, card % tiles) if tiles > 1 else (card, None)

        cmd = ["xpu-smi", "dump", "-d", str(device_id)]
        if tile_id is not None:
            cmd += ["-t", str(tile_id)]
        # -m 0: GPU Utilization, 2: GPU Frequency, 3: GPU Core Temperature,
        # 5: GPU Memory Utilization, 17: GPU Memory Bandwidth Utilization
        cmd += ["-m", "0,2,3,5,17", "-n", "1"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None

        # Parse output, ignoring error/warning lines (e.g. "lspci: ...")
        # Header starts with "Timestamp", data lines start with a timestamp pattern
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        header_line = None
        data_line = None
        for l in lines:
            if l.startswith("Timestamp"):
                header_line = l
            elif header_line is not None and re.match(r'\d{2}:\d{2}:\d{2}', l):
                data_line = l
        if header_line is None or data_line is None:
            return None

        headers = [h.strip() for h in header_line.split(',')]
        values = [v.strip() for v in data_line.split(',')]

        def _get_col(keyword: str) -> str | None:
            try:
                idx = next(i for i, h in enumerate(headers) if keyword in h)
                if idx < len(values) and values[idx] not in ("N/A", ""):
                    return values[idx]
            except StopIteration:
                pass
            return None

        mem_util_str = _get_col("GPU Memory Utilization")
        if mem_util_str is None:
            return None

        val = float(mem_util_str)
        if val > 0.0:  # normalise 0-100 → 0-1
            val /= 100.0

        # Gather additional metrics for logging
        gpu_util = _get_col("GPU Utilization")
        gpu_freq = _get_col("GPU Frequency")
        gpu_temp = _get_col("GPU Core Temperature")
        mem_bw = _get_col("Memory Bandwidth Utilization")

        loc = f"GPU {card} (Device {device_id} Tile {tile_id})" if tiles > 1 else f"GPU {card}"
        thr = f" ({memory_threshold:.0%})" if memory_threshold is not None else ""
        model_name = task.model if task is not None else "N/A"
        extras = []
        if gpu_util is not None:
            extras.append(f"GPU Util={gpu_util}%")
        if gpu_freq is not None:
            extras.append(f"GPU Freq={gpu_freq}MHz")
        if gpu_temp is not None:
            extras.append(f"GPU Temp={gpu_temp}℃")
        if mem_bw is not None:
            extras.append(f"GPU Mem BW Util={mem_bw}%")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        log(f"{loc} GPU Memory Util: {val:.2%}{thr}{extra_str} | {model_name}")
        return val
    except Exception:
        return None


def get_host_memory_utilization(
    memory_threshold: float | None = None,
    task: TestTask | None = None,
) -> float | None:
    """Return host RAM utilisation as a 0.0-1.0 fraction, or *None*.

    Reads /proc/meminfo on Linux; falls back to os-level heuristics on Windows.
    """
    from .config import IS_WINDOWS

    try:
        if IS_WINDOWS:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = stat.ullTotalPhys
            avail = stat.ullAvailPhys
        else:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024  # kB → bytes
            total = meminfo.get("MemTotal", 0)
            avail = meminfo.get("MemAvailable", 0)

        if total <= 0:
            return None
        val = 1.0 - avail / total

        thr = f" (threshold {memory_threshold:.0%})" if memory_threshold is not None else ""
        model_name = task.model if task is not None else "N/A"
        log(f"Host memory: {val:.2%}{thr} | {model_name}")
        return val
    except Exception:
        return None
