#!/usr/bin/env python3
import argparse
import csv
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

from packaging import version

# Constants & globals
IS_WINDOWS = platform.system() == "Windows"


@dataclass
class TestTask:
    suite: str
    dt: str
    mode: str
    scenario: str
    model: str


# Patterns that trigger process kill when detected in output or GPU metrics.
# Text patterns are matched case-insensitively; "Memory>N" triggers GPU memory
# utilisation polling with threshold N.
gpu_memory_threshold = 0.8 if IS_WINDOWS else 0.9
error_patterns: list[str] = [
    "out of memory",
    "OutOfMemory",
    "UR_RESULT_ERROR",
    f"Memory>{gpu_memory_threshold}",
]

_csv_lock = threading.Lock()


# Logging helpers — consistent, readable output

def _log(msg: str, *, level: str = "INFO", worker: int | None = None) -> None:
    """Print a timestamped, consistently-formatted log line."""
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}] [{level}]"
    if worker is not None:
        prefix += f" [Worker {worker}]"
    print(f"{prefix} {msg}", flush=True)


def _banner(title: str) -> None:
    """Print a visible section separator."""
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{line}", flush=True)


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# Parse memory utilization threshold from error_patterns (e.g. "Memory>0.95" → 0.95)
def _parse_memory_threshold() -> float | None:
    for pattern in error_patterns:
        m = re.match(r'Memory>(\d+\.?\d*)', pattern)
        if m:
            return float(m.group(1))
    return None


# Cache for xpu-smi availability (None = not checked yet)
_xpu_smi_available: bool | None = None
_xpu_smi_lock = threading.Lock()

# Cache for tiles-per-device (probed once from device 0)
_tiles_per_device: int | None = None
_tiles_per_device_lock = threading.Lock()


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
            _log(f"Detected {_tiles_per_device} tile(s) per device")
        return _tiles_per_device


def _get_gpu_memory_utilization(
    card: int,
    memory_threshold: float | None = None,
    task: TestTask | None = None,
) -> float | None:
    """Return GPU memory utilisation for *card* as a 0.0-1.0 fraction, or *None*.

    Multi-tile layout (N tiles/device): card C → device C//N, tile C%N.
    Single-tile layout: card C → device C.
    """
    if not _is_xpu_smi_available():
        return None

    try:
        tiles = _get_tiles_per_device()
        device_id, tile_id = (card // tiles, card % tiles) if tiles > 1 else (card, None)

        cmd = ["xpu-smi", "dump", "-d", str(device_id)]
        if tile_id is not None:
            cmd += ["-t", str(tile_id)]
        cmd += ["-m", "5", "-n", "1"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None

        headers = [h.strip() for h in lines[0].split(',')]
        try:
            col_idx = next(i for i, h in enumerate(headers) if "GPU Memory Utilization" in h)
        except StopIteration:
            return None

        values = [v.strip() for v in lines[-1].split(',')]
        if col_idx >= len(values) or values[col_idx] in ("N/A", ""):
            return None

        val = float(values[col_idx])
        if val > 1.0:  # normalise 0-100 → 0-1
            val /= 100.0

        loc = f"GPU {card} (Device {device_id} Tile {tile_id})" if tiles > 1 else f"GPU {card}"
        thr = f" (threshold {memory_threshold:.0%})" if memory_threshold is not None else ""
        model_name = task.model if task is not None else "N/A"
        _log(f"{loc} memory: {val:.2%}{thr} | {model_name}")
        return val
    except Exception:
        return None


def _get_host_memory_utilization(
    memory_threshold: float | None = None,
    task: TestTask | None = None,
) -> float | None:
    """Return host RAM utilisation as a 0.0-1.0 fraction, or *None*.

    Reads /proc/meminfo on Linux; falls back to os-level heuristics on Windows.
    """
    try:
        if IS_WINDOWS:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
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
        _log(f"Host memory: {val:.2%}{thr} | {model_name}")
        return val
    except Exception:
        return None


# Helper to parse string to list
def parse_string_list(s: str) -> list[str]:
    """Split a string, return non-empty trimmed items."""
    if not s:
        return []
    parts = re.split(r'\s*-k\s*|,|\s+', s)
    return [token.strip() for token in parts if token.strip()]


# Model list extraction
def get_model_list(suite: str, mode: str, model_only: str | None) -> list[str]:
    """Retrieve a list of model names from either a CSV file, a string list, or a default text file."""
    if model_only is not None:
        if os.path.isfile(model_only):
            import pandas as pd
            df = pd.read_csv(model_only)
            col_header = suite if suite != "torchbench" else f"{suite} {mode}"
            if col_header not in df.columns:
                available = ", ".join(df.columns)
                raise ValueError(f"Column '{col_header}' not found in {model_only}. Available: {available}")
            return df[col_header].dropna().astype(str).str.strip().tolist()

        return parse_string_list(model_only)

    base = suite.replace('_models', '')
    list_file = Path(f"benchmarks/dynamo/{base}_models_list.txt")
    if not list_file.exists():
        raise FileNotFoundError(f"Model list file not found: {list_file}")

    with list_file.open() as f:
        models = [
            parts[0].strip()
            for raw in f
            if (line := raw.split('#', 1)[0].strip())
            and (parts := parse_string_list(line))
            and parts[0].strip()
        ]
    return models


@lru_cache(maxsize=1)
def _get_torch_version() -> str:
    pip_cmd = "pip.exe" if IS_WINDOWS else "pip"
    try:
        result = subprocess.run([pip_cmd, "list", "--format=freeze"], capture_output=True, text=True, check=True)
        torch_line = next((l for l in result.stdout.splitlines() if l.startswith("torch==")), None)
        return torch_line.split("==")[1].split("+")[0] if torch_line else "0.0.0"
    except Exception:
        return "0.0.0"


# Benchmark execution (merged from inductor_xpu_test.sh)
def _build_benchmark_cmd(task: TestTask, device: str, shape: str, output_csv: str) -> str:
    """Build the benchmark command string for a single model run."""
    torch_ver = _get_torch_version()

    # Determine mode flag
    if task.mode == "training":
        mode_flag = "--training"
    elif version.parse(torch_ver) >= version.parse("2.0.2"):
        mode_flag = "--inference"
    else:
        mode_flag = ""

    # Resolve dtype: amp_bf16/amp_fp16 → --amp --amp-dtype ...
    dt_map = {"amp_bf16": ("amp", "--amp-dtype bfloat16"),
              "amp_fp16": ("amp", "--amp-dtype float16")}
    real_dt, dt_extra = dt_map.get(task.dt, (task.dt, ""))

    shape_flags = "--dynamic-shapes --dynamic-batch-only" if shape == "dynamic" else ""

    parts = [
        f"python benchmarks/dynamo/{task.suite}.py",
        f"-d {device}",
        f"--{task.scenario}",
        f"--{real_dt}",
        dt_extra,
        mode_flag,
        shape_flags,
        f"--only {task.model}" if task.model else "",
        "--backend=inductor",
        "--cold-start-latency",
        "-n10",
        "--timeout=10800",
        "--disable-cudagraphs",
        f"--output={output_csv}",
    ]
    return " ".join(p for p in parts if p)


def _check_success_from_log(log_path: Path) -> tuple[str, bool]:
    """Check the last line of the log to determine pass/fail.

    Returns (test_result, True) if last token is 'pass', 'pass_due_to_skip',
    or a speedup like '1.23x'.
    """
    if not log_path.exists():
        return "None", False
    last_non_empty = ""
    with open(log_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_non_empty = stripped
    if not last_non_empty:
        return "None", False
    test_result = last_non_empty.split()[-1]
    success = test_result in ('pass', 'pass_due_to_skip') or bool(re.fullmatch(r'[0-9.]+x', test_result))
    return test_result, success


def _build_cmd_list(cmd: str, cmd_prefix: str) -> tuple[list[str], str]:
    """Split command (with optional prefix) into an argv list and display string."""
    _posix = not IS_WINDOWS
    if cmd_prefix:
        cmd_list = shlex.split(cmd_prefix, posix=_posix) + shlex.split(cmd, posix=_posix)
        cmd_str = f"{cmd_prefix} {cmd}"
    else:
        cmd_list = shlex.split(cmd, posix=_posix)
        cmd_str = cmd
    return cmd_list, cmd_str


def run_benchmark_with_prefix(
    task: TestTask,
    card: int,
    cmd_prefix: str,
    env_vars: dict,
    log_dir: Path,
    worker_id: int,
    device: str,
    shape: str,
) -> tuple[int, bool, str, str | None]:
    """Run a single benchmark and return `(exit_code, success, test_result, kill_reason)`."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inductor-logs-{task.model.replace('/', '_')}-worker{worker_id}-card{card}.log"
    log_csv = log_dir / f"inductor-results-{task.suite}-{task.dt}-{task.mode}-{device}-{task.scenario}.csv"
    tmp_fd, tmp_log_csv = tempfile.mkstemp(prefix='tmp_', suffix='.csv')
    os.close(tmp_fd)

    cmd = _build_benchmark_cmd(task, device, shape, tmp_log_csv)
    full_cmd_list, full_cmd_str = _build_cmd_list(cmd, cmd_prefix)
    _log(f"Running: {full_cmd_str[:200]}...", worker=worker_id)

    env = {**os.environ, **env_vars}
    if device != "cpu":
        env["ZE_AFFINITY_MASK"] = str(card)

    # Shared state for threads
    log_buffer: list[str] = []
    log_buffer_lock = threading.Lock()
    kill_reason: list[str | None] = [None]
    kill_reason_lock = threading.Lock()

    popen_kwargs = {
        "shell": False, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
        "text": True, "env": env, "bufsize": 1,
    }
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    log_f = None
    try:
        log_f = open(log_file, "w")
        proc = subprocess.Popen(full_cmd_list, **popen_kwargs)
    except Exception as e:
        if log_f is not None and not log_f.closed:
            log_f.close()
        raise RuntimeError(f"Failed to start benchmark for {task.model}: {e}") from e

    # --- Output reader thread ---
    def output_reader():
        if proc.stdout is None:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                formatted = f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} {line.rstrip()}"
                print(formatted, flush=True)
                with log_buffer_lock:
                    log_buffer.append(formatted)
                try:
                    log_f.write(formatted + "\n")
                    log_f.flush()
                except (ValueError, OSError):
                    pass
        except (ValueError, OSError):
            pass

    # --- Error monitor thread ---
    def error_monitor():
        scan_pos = 0
        memory_threshold = _parse_memory_threshold()
        deadline = time.time() + 10800

        while proc.poll() is None and time.time() < deadline:
            # Check log buffer for text-based error patterns
            with log_buffer_lock:
                new_lines = log_buffer[scan_pos:]
                scan_pos = len(log_buffer)
            if new_lines:
                content = "\n".join(new_lines).lower()
                for pattern in error_patterns:
                    if pattern.startswith("Memory>"):
                        continue
                    if pattern.lower() in content:
                        _log(f"Detected '{pattern}' — killing process", level="WARN", worker=worker_id)
                        with kill_reason_lock:
                            kill_reason[0] = pattern
                        proc.kill()
                        return

            # Poll memory utilization (GPU or host depending on device)
            if memory_threshold is not None:
                try:
                    if device == "cpu":
                        mem_util = _get_host_memory_utilization(memory_threshold, task)
                    else:
                        mem_util = _get_gpu_memory_utilization(card, memory_threshold, task)
                except Exception as e:
                    _log(f"Memory poll error: {e}", level="WARN", worker=worker_id)
                    mem_util = None
                if mem_util is not None and mem_util >= memory_threshold:
                    mem_kind = "Host" if device == "cpu" else "GPU"
                    _log(
                        f"{mem_kind} memory {mem_util:.2%} >= threshold {memory_threshold:.0%} — killing process",
                        level="WARN", worker=worker_id,
                    )
                    with kill_reason_lock:
                        kill_reason[0] = f"Memory>{memory_threshold}"
                    proc.kill()
                    return

            time.sleep(3)

        # Deadline reached — kill the process if still running
        if proc.poll() is None:
            _log("Monitor deadline reached — killing process", level="WARN", worker=worker_id)
            with kill_reason_lock:
                kill_reason[0] = "timeout"
            proc.kill()

    # Launch threads
    reader_thread = threading.Thread(target=output_reader, daemon=True)
    monitor_thread = threading.Thread(target=error_monitor, daemon=True)
    reader_thread.start()
    monitor_thread.start()

    # Wait for process and threads
    try:
        exit_code = proc.wait()
    except Exception:
        exit_code = -1
    reader_thread.join(timeout=10)
    if log_f is not None and not log_f.closed:
        log_f.close()
    monitor_thread.join(timeout=1)

    # Collect CSV results and clean up temp file
    with kill_reason_lock:
        matched_pattern = kill_reason[0]
    try:
        collect_csv_results(log_csv, tmp_log_csv, device, task, kill_reason=matched_pattern)
    except Exception as e:
        _log(f"CSV collection error for {task.model}: {e}", level="ERROR", worker=worker_id)
    finally:
        with suppress(OSError):
            os.unlink(tmp_log_csv)

    test_result, success = _check_success_from_log(log_file)
    return exit_code, success, test_result, matched_pattern


def _read_last_matching_row(tmp_log_csv: str, device: str) -> tuple[list[str], list[str]]:
    """Read header and last device-matching row from a temp CSV. Returns (header, row)."""
    if not os.path.isfile(tmp_log_csv):
        return [], []
    try:
        with open(tmp_log_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
        if not rows:
            return [], []
        header = rows[0]
        matching = [r for r in rows[1:] if r and r[0] == device]
        return header, matching[-1] if matching else []
    except Exception as e:
        _log(f"Error reading temp CSV {tmp_log_csv}: {e}", level="WARN")
        return [], []


def _build_fallback_row(
    device: str, task: TestTask, kill_reason: str | None,
) -> tuple[list[str], list[str]]:
    """Build a fallback (header, row) when the benchmark produced no CSV output."""
    fail_status = kill_reason or "core_dump"
    fallbacks = {
        "accuracy": (
            ["dev", "name", "batch_size", "accuracy"],
            [device, task.model, "0", fail_status],
        ),
        "performance": (
            ["dev", "name", "batch_size", "speedup", "abs_latency"],
            [device, task.model, "0", "0", "0"],
        ),
    }
    if task.scenario not in fallbacks:
        raise ValueError(f"Unknown task.scenario: {task.scenario}")
    return fallbacks[task.scenario]


def collect_csv_results(
    log_csv: Path,
    tmp_log_csv: str,
    device: str,
    task: TestTask,
    kill_reason: str | None = None,
) -> None:
    """Collect benchmark results from *tmp_log_csv* and append to *log_csv*."""
    header, row = _read_last_matching_row(tmp_log_csv, device)
    if not row:
        header, row = _build_fallback_row(device, task, kill_reason)

    prefix = [task.scenario, task.suite, task.dt, task.mode]
    final_header = ["scenario", "suite", "dtype", "mode"] + header
    final_row = prefix + row

    with _csv_lock:
        write_header = not log_csv.exists()
        with open(log_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(final_header)
            writer.writerow(final_row)


# CPU topology and command prefix generation
@lru_cache(maxsize=1)
def get_cpu_topology() -> int:
    """Return number of physical cores."""
    if IS_WINDOWS:
        return _get_cpu_topology_windows()
    return _get_cpu_topology_linux()


def _get_cpu_topology_windows() -> int:
    """Return number of physical cores on Windows via wmic or os.cpu_count fallback."""
    try:
        output = subprocess.check_output(
            ["wmic", "cpu", "get", "NumberOfCores"], text=True
        )
        for line in output.splitlines():
            line = line.strip()
            if line.isdigit():
                physical_cores = int(line)
                logical = os.cpu_count() or physical_cores
                _log(f"CPU topology (Windows): {logical} logical, {physical_cores} physical cores")
                return physical_cores
    except Exception:
        pass
    # Fallback: assume half of logical CPUs are physical cores
    logical = os.cpu_count() or 1
    physical_cores = max(logical // 2, 1)
    _log(f"CPU topology (Windows fallback): {logical} logical, ~{physical_cores} physical cores")
    return physical_cores


def _parse_cpu_range(range_str: str) -> int:
    """Count CPUs from a range string like '0-7,12-19'."""
    total = 0
    for seg in range_str.split(','):
        seg = seg.strip()
        if '-' in seg:
            lo, hi = map(int, seg.split('-'))
            total += hi - lo + 1
        else:
            total += 1
    return total


def _get_cpu_topology_linux() -> int:
    """Return number of physical cores on Linux via lscpu."""
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except Exception as e:
        sys.exit(f"ERROR: Could not run lscpu: {e}")

    fields: dict[str, str] = {}
    for line in output.splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            fields[key.strip()] = val.strip()

    try:
        online_cpus = _parse_cpu_range(fields["On-line CPU(s) list"])
        threads_per_core = int(fields["Thread(s) per core"])
    except (KeyError, ValueError) as e:
        sys.exit(f"ERROR: Could not parse CPU topology: {e}")

    physical_cores = online_cpus // threads_per_core
    _log(f"CPU topology: {online_cpus} logical, {threads_per_core} threads/core → {physical_cores} physical cores")
    return physical_cores


def _parse_ze_affinity_mask(num_gpus: int) -> list[int]:
    """Parse ZE_AFFINITY_MASK into a list of GPU indices, defaulting to 0..num_gpus-1."""
    ze_mask = os.environ.get("ZE_AFFINITY_MASK", "")
    if not ze_mask:
        _log(f"ZE_AFFINITY_MASK not set — using all {num_gpus} GPUs")
        return list(range(num_gpus))

    gpu_list: list[int] = []
    try:
        for part in ze_mask.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                lo, hi = part.split('-', maxsplit=1)
                gpu_list.extend(range(int(lo), int(hi) + 1))
            else:
                gpu_list.append(int(part))
    except (ValueError, IndexError) as e:
        sys.exit(f"ERROR: Invalid ZE_AFFINITY_MASK format '{ze_mask}': {e}")

    if not gpu_list:
        sys.exit("ERROR: ZE_AFFINITY_MASK produced no GPUs")
    _log(f"ZE_AFFINITY_MASK → GPUs {gpu_list}")
    return gpu_list


def generate_gpu_workers(num_gpus: int) -> list[tuple[int, str, dict]]:
    """Return (card, cmd_prefix, env_vars) per GPU with numactl CPU binding.

    GPU list comes from ZE_AFFINITY_MASK (if set) or 0..num_gpus-1.
    CPU cores are always distributed evenly by num_gpus.
    """
    gpu_list = _parse_ze_affinity_mask(num_gpus)
    physical_cores = get_cpu_topology()
    cores_per_gpu = max(physical_cores // num_gpus, 1)

    workers: list[tuple[int, str, dict]] = []
    for gpu in gpu_list:
        start = gpu * cores_per_gpu
        end = min(start + cores_per_gpu - 1, physical_cores - 1)
        if IS_WINDOWS:
            _log(f"  GPU {gpu}: OMP_NUM_THREADS={cores_per_gpu} (numactl N/A on Windows)")
            workers.append((gpu, "", {"OMP_NUM_THREADS": str(cores_per_gpu)}))
        else:
            core_range = f"{start}-{end}"
            _log(f"  GPU {gpu}: cores {core_range}, OMP_NUM_THREADS={cores_per_gpu}")
            workers.append((gpu, f"numactl -l -C {core_range}", {"OMP_NUM_THREADS": str(cores_per_gpu)}))
    return workers


def _get_numa_nodes() -> list[list[int]]:
    """Return a list of NUMA nodes, each containing its physical core IDs.

    Falls back to a single node with all cores on Windows or parse failure.
    """
    if IS_WINDOWS:
        n = get_cpu_topology()
        return [list(range(n))]

    try:
        output = subprocess.check_output(["lscpu", "--parse=CPU,NODE,CORE"], text=True)
    except Exception:
        n = get_cpu_topology()
        return [list(range(n))]

    # lscpu --parse outputs "# ..." comment lines, then "cpu,node,core" data lines
    # We want one CPU ID per physical core (deduplicate by node+core to skip hyperthreads)
    seen_cores: set[tuple[int, int]] = set()
    node_cpus: dict[int, list[int]] = {}
    for line in output.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue
        try:
            cpu_id, node_id, core_id = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        key = (node_id, core_id)
        if key not in seen_cores:
            seen_cores.add(key)
            node_cpus.setdefault(node_id, []).append(cpu_id)

    if not node_cpus:
        n = get_cpu_topology()
        return [list(range(n))]

    result = [sorted(cpus) for _, cpus in sorted(node_cpus.items())]
    for i, cores in enumerate(result):
        _log(f"  NUMA node {i}: {len(cores)} physical cores")
    return result


def generate_cpu_workers(cores_per_instance: int | None = None) -> list[tuple[int, str, dict]]:
    """Return (worker_id, cmd_prefix, env_vars) for CPU-only benchmarking.

    Workers are created per NUMA node. If *cores_per_instance* is set,
    total physical cores are split into chunks of that size instead.
    """
    if IS_WINDOWS:
        physical_cores = get_cpu_topology()
        cpi = cores_per_instance or physical_cores
        num_workers = max(physical_cores // cpi, 1)
        workers: list[tuple[int, str, dict]] = []
        for i in range(num_workers):
            _log(f"  CPU worker {i}: OMP_NUM_THREADS={cpi} (numactl N/A on Windows)")
            workers.append((i, "", {"OMP_NUM_THREADS": str(cpi)}))
        return workers

    numa_nodes = _get_numa_nodes()

    if cores_per_instance is not None:
        # Flatten all physical cores and chunk by cores_per_instance
        all_cores: list[int] = []
        for cores in numa_nodes:
            all_cores.extend(cores)
        all_cores.sort()
        num_workers = max(len(all_cores) // cores_per_instance, 1)
        workers = []
        for i in range(num_workers):
            start = i * cores_per_instance
            end = min(start + cores_per_instance, len(all_cores))
            chunk = all_cores[start:end]
            if not chunk:
                break
            core_range = _cores_to_range_str(chunk)
            _log(f"  CPU worker {i}: cores {core_range}, OMP_NUM_THREADS={len(chunk)}")
            workers.append((
                i,
                f"numactl -l -C {core_range}",
                {"OMP_NUM_THREADS": str(len(chunk))},
            ))
        return workers

    # Default: one worker per NUMA node
    workers = []
    for node_idx, cores in enumerate(numa_nodes):
        core_range = _cores_to_range_str(cores)
        _log(f"  CPU worker {node_idx} (NUMA {node_idx}): cores {core_range}, OMP_NUM_THREADS={len(cores)}")
        workers.append((
            node_idx,
            f"numactl -m {node_idx} -C {core_range}",
            {"OMP_NUM_THREADS": str(len(cores))},
        ))
    return workers


def _cores_to_range_str(cores: list[int]) -> str:
    """Convert a sorted list of core IDs to a compact range string like '0-7,12-19'."""
    if not cores:
        return ""
    ranges: list[str] = []
    start = prev = cores[0]
    for c in cores[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


# Argument validation helpers

VALID_SUITES: set[str] = {"huggingface", "timm_models", "torchbench"}
VALID_DT: set[str] = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
VALID_MODES: set[str] = {"inference", "training"}
VALID_SCENARIOS: set[str] = {"accuracy", "performance"}


def _filter_valid(items: list[str], valid_set: set[str], name: str) -> list[str]:
    """Keep only items in *valid_set*, warning about invalid ones."""
    if invalid := sorted(set(items) - valid_set):
        _log(f"Skipping invalid {name}: {', '.join(invalid)}", level="WARN")
    return [i for i in items if i in valid_set]


def _is_numeric(s: str) -> bool:
    """Return True if *s* looks like a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _load_tasks_from_file(path: str) -> list[TestTask]:
    """Load tasks from a delimited file (comma, tab, semicolon, or whitespace).

    Expected columns: suite, dtype, mode, model, result
    The *result* column determines scenario: numeric → performance, else → accuracy.
    A header row is auto-detected and skipped.
    """
    file_path = Path(path)
    if not file_path.is_file():
        sys.exit(f"ERROR: Task file not found: {path}")

    tasks: list[TestTask] = []
    with open(file_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Split by comma, semicolon, tab, or whitespace
            fields = re.split(r'[,;\t]\s*|\s+', line)
            fields = [f.strip() for f in fields if f.strip()]
            if len(fields) < 5:
                _log(f"Skipping line {lineno}: expected 5+ fields, got {len(fields)}: {line!r}", level="WARN")
                continue
            suite, dt, mode, model, result = fields[0], fields[1], fields[2], fields[3], fields[4]
            # Skip header row
            if suite.lower() == "suite":
                continue
            scenario = "performance" if _is_numeric(result) else "accuracy"
            tasks.append(TestTask(suite, dt, mode, scenario, model))

    # Deduplicate while preserving order
    seen: set[tuple] = set()
    unique: list[TestTask] = []
    for t in tasks:
        key = (t.suite, t.dt, t.mode, t.scenario, t.model)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _get_num_gpus(required: bool = True) -> int | None:
    """Read and validate the NUM_GPUS environment variable.

    If *required* is False, returns None when the variable is unset.
    """
    raw = os.getenv('NUM_GPUS')
    if raw is None:
        if required:
            sys.exit("ERROR: Environment variable NUM_GPUS is not set.")
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"ERROR: NUM_GPUS must be an integer, got '{raw}'.")
    if n <= 0:
        sys.exit(f"ERROR: NUM_GPUS must be positive, got {n}.")
    return n


def _parse_numactl_args(numactl_args: str, num_gpus: int) -> list[tuple[int, str, dict]]:
    """Parse user-provided --numactl-args into worker tuples."""
    prefix_strings = [p.strip().rstrip(';') for p in numactl_args.split(';') if p.strip()]
    if not prefix_strings:
        sys.exit("ERROR: --numactl-args produced no valid prefixes.")
    if len(prefix_strings) < num_gpus:
        prefix_strings += [prefix_strings[-1]] * (num_gpus - len(prefix_strings))
    workers: list[tuple[int, str, dict]] = []
    for i, ps in enumerate(prefix_strings):
        m = re.search(r'ZE_AFFINITY_MASK=(\d+)', ps)
        workers.append((int(m.group(1)) if m else i, ps, {}))
    _log(f"Using user-provided NUMACTL_ARGS ({len(workers)} workers)")
    return workers


# Main orchestrator

def main():
    parser = argparse.ArgumentParser(
        description="Run E2E tests with per-GPU job queue and hang detection",
    )
    parser.add_argument("--suite", default="",
                        help="Comma- or space-separated suites (huggingface, timm_models, torchbench). Empty = all.")
    parser.add_argument("--dt", "--dtype", default="",
                        help="Comma- or space-separated dtypes (float32, bfloat16, float16, amp_bf16, amp_fp16). Empty = all.")
    parser.add_argument("--mode", default="",
                        help="Comma- or space-separated modes (inference, training). Empty = all.")
    parser.add_argument("--scenario", default="",
                        help="Comma- or space-separated scenarios (accuracy, performance). Empty = all.")
    parser.add_argument("--model-only", default=None,
                        help="Run only specific model(s) (overrides list file).")
    parser.add_argument("--device", default="xpu", help="Device type (xpu, cuda, or cpu)")
    parser.add_argument("--shape", default="static", help="Shape mode (static or dynamic)")
    parser.add_argument("--task-file", default=None,
                        help="Load tasks from a delimited file (columns: suite dtype mode model result). "
                             "Overrides --suite/--dt/--mode/--scenario/--model-only.")
    parser.add_argument("--cores-per-instance", type=int, default=None,
                        help="CPU-only: cores per worker instance. Default = all cores per NUMA node.")
    parser.add_argument("--numactl-args", default="",
                        help="Override NUMACTL_ARGS (semicolon-separated per GPU)")
    args = parser.parse_args()

    is_cpu = args.device == "cpu"
    num_gpus = _get_num_gpus(required=not is_cpu)

    if not Path("benchmarks/dynamo").is_dir():
        sys.exit("ERROR: benchmarks/dynamo directory not found. Are you in the pytorch directory?")

    if args.task_file:
        # Load tasks from file — overrides suite/dt/mode/scenario/model-only
        tasks = _load_tasks_from_file(args.task_file)
        if not tasks:
            sys.exit(f"No valid tasks found in {args.task_file}.")
        suites = sorted({t.suite for t in tasks})
        dts = sorted({t.dt for t in tasks})
        modes = sorted({t.mode for t in tasks})
        scenarios = sorted({t.scenario for t in tasks})
    else:
        # Parse & validate parameter lists
        suites = _filter_valid(parse_string_list(args.suite) or list(VALID_SUITES), VALID_SUITES, "suite")
        dts = _filter_valid(parse_string_list(args.dt) or list(VALID_DT), VALID_DT, "data type")
        modes = _filter_valid(parse_string_list(args.mode) or list(VALID_MODES), VALID_MODES, "mode")
        scenarios = _filter_valid(parse_string_list(args.scenario) or list(VALID_SCENARIOS), VALID_SCENARIOS, "scenario")

        if not all([suites, dts, modes, scenarios]):
            sys.exit("ERROR: No valid combinations left after filtering.")

        # Build task list
        tasks = [
            TestTask(suite, dt, mode, scenario, model)
            for suite, mode, dt, scenario in product(suites, modes, dts, scenarios)
            for model in sorted(set(get_model_list(suite, mode, args.model_only)))
        ]
        if not tasks:
            sys.exit("No valid tasks generated.")

    _banner("Configuration")
    if num_gpus is not None:
        _log(f"NUM_GPUS:   {num_gpus}")
    _log(f"Suites:     {', '.join(suites)}")
    _log(f"Dtypes:     {', '.join(dts)}")
    _log(f"Modes:      {', '.join(modes)}")
    _log(f"Scenarios:  {', '.join(scenarios)}")
    _log(f"Device:     {args.device}")
    _log(f"Shape:      {args.shape}")
    if args.cores_per_instance:
        _log(f"Cores/inst: {args.cores_per_instance}")
    _log(f"Tasks:      {len(tasks)}")

    # Determine workers and CPU binding
    _banner("Worker Setup")
    if args.numactl_args and args.numactl_args.strip():
        if num_gpus is None:
            sys.exit("ERROR: --numactl-args requires NUM_GPUS to be set.")
        workers = _parse_numactl_args(args.numactl_args, num_gpus)
    elif is_cpu:
        workers = generate_cpu_workers(args.cores_per_instance)
        _log(f"Auto-generated {len(workers)} CPU worker(s)")
    else:
        if num_gpus is None:
            sys.exit("ERROR: NUM_GPUS must be set for GPU devices.")
        workers = generate_gpu_workers(num_gpus)
        _log(f"Auto-generated {len(workers)} GPU worker(s)")

    _banner("Running Benchmarks")
    _run_workers(tasks, workers, args.device, args.shape)


def _run_workers(
    tasks: list[TestTask],
    workers: list[tuple[int, str, dict]],
    device: str,
    shape: str,
) -> None:
    total = len(tasks)
    task_queue: queue.Queue[TestTask] = queue.Queue()
    for t in tasks:
        task_queue.put(t)

    results: queue.Queue[tuple[TestTask, bool, str, str | None]] = queue.Queue()
    completed = 0
    completed_lock = threading.Lock()
    base_log_dir = Path.cwd().resolve() / "inductor_log"
    wall_start = time.monotonic()

    def _worker(worker_id: int, card: int, cmd_prefix: str, env_vars: dict) -> None:
        nonlocal completed
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return

            task_start = time.monotonic()
            log_dir = base_log_dir / task.suite / task.dt / task.mode / task.scenario
            kill_reason: str | None = None
            try:
                exit_code, success, test_result, kill_reason = run_benchmark_with_prefix(
                    task=task, card=card, cmd_prefix=cmd_prefix, env_vars=env_vars,
                    log_dir=log_dir, worker_id=worker_id, device=device, shape=shape,
                )
            except Exception as e:
                _log(f"Exception running {task.model}: {e}", level="ERROR", worker=worker_id)
                exit_code, success, test_result = -1, False, "exception"

            elapsed = _fmt_duration(time.monotonic() - task_start)
            with completed_lock:
                completed += 1
                n = completed

            pct = n * 100 // total
            progress = f"[{n}/{total} {pct}%]"
            model_info = f"{task.suite}/{task.model} ({task.dt}, {task.mode}, {task.scenario})"
            if success:
                _log(f"{progress} PASS  {model_info} ({elapsed}) → {test_result}", worker=worker_id)
            elif kill_reason:
                _log(f"{progress} KILL  {model_info} ({elapsed}) → {kill_reason}", level="WARN", worker=worker_id)
            elif exit_code == 0:
                _log(f"{progress} FAIL  {model_info} ({elapsed}) → {test_result}", level="WARN", worker=worker_id)
            else:
                _log(f"{progress} FAIL  {model_info} ({elapsed}) → exit code {exit_code}", level="ERROR", worker=worker_id)

            results.put((task, success, test_result, kill_reason))

    threads = [
        threading.Thread(target=_worker, args=(idx, card, pfx, ev), daemon=True)
        for idx, (card, pfx, ev) in enumerate(workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Summary
    wall_time = _fmt_duration(time.monotonic() - wall_start)
    all_results = [results.get() for _ in range(results.qsize())]
    failed = [(task, tr, kr) for task, ok, tr, kr in all_results if not ok]
    passed = total - len(failed)

    _banner("Summary")
    _log(f"Total:     {total}")
    _log(f"Passed:    {passed} ({passed * 100 // total}%)")
    _log(f"Failed:    {len(failed)} ({len(failed) * 100 // total}%)")
    _log(f"Wall time: {wall_time}")
    if failed:
        print(flush=True)
        _log("Failed tasks:")
        for task, test_result, kill_reason in failed:
            reason = kill_reason or test_result
            _log(f"  ✗ {task.suite}/{task.model} ({task.dt}, {task.mode}, {task.scenario}) → {reason}")
    else:
        print(flush=True)
        _log("All tasks completed successfully.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
