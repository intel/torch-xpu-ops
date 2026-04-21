#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import re
import platform
from pathlib import Path
from dataclasses import dataclass
import argparse
import tempfile
import queue
import shlex
import shutil
from packaging import version

IS_WINDOWS = platform.system() == "Windows"
import pandas as pd


# Data structures
@dataclass
class TestTask:
    suite: str
    dt: str
    mode: str
    scenario: str
    model: str

# Monitor for OutOfMemoryError or UR_RESULT_ERROR
error_patterns = ["out of memory", "OutOfMemory", "UR_RESULT_ERROR", "Memory>0.95"]

# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


# Parse memory utilization threshold from error_patterns (e.g. "Memory>0.95" → 0.95)
def _parse_memory_threshold() -> float | None:
    for pattern in error_patterns:
        m = re.match(r'Memory>(\d+\.?\d*)', pattern)
        if m:
            return float(m.group(1))
    return None


# Cache for tile count per device (device_id → number of tiles)
_tile_count_cache: dict[int, int] = {}
_tile_count_cache_lock = threading.Lock()

# Cache for xpu-smi availability (None = not checked yet)
_xpu_smi_available: bool | None = None
_xpu_smi_lock = threading.Lock()


def _is_xpu_smi_available() -> bool:
    global _xpu_smi_available
    with _xpu_smi_lock:
        if _xpu_smi_available is None:
            _xpu_smi_available = shutil.which("xpu-smi") is not None
        return _xpu_smi_available


def _detect_tile_count(device_id: int) -> int:
    """Detect the number of tiles for a device using xpu-smi stats."""
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 1
        # Count distinct "Tile N" references in the output
        tile_ids = {int(m.group(1)) for m in re.finditer(r'Tile\s+(\d+)', result.stdout)}
        return max(len(tile_ids), 1)
    except Exception:
        return 1


def _get_tile_count(device_id: int) -> int:
    """Cached tile count for a device."""
    with _tile_count_cache_lock:
        if device_id not in _tile_count_cache:
            _tile_count_cache[device_id] = _detect_tile_count(device_id)
        return _tile_count_cache[device_id]


def _get_gpu_memory_utilization(card: int, memory_threshold: float | None = None) -> float | None:
    """
    Get GPU memory utilization for the given card using ``xpu-smi dump``.
    Returns utilization as a fraction (0.0 – 1.0), or *None* on failure.

    Multi-tile layout (N tiles/device): card C maps to device C//N, tile C%N.
    Single-tile layout: card C maps to device C directly.
    """
    if not _is_xpu_smi_available():
        return None

    try:
        # Probe device 0 to learn tiles-per-device, then compute mapping
        tiles_per_device = _get_tile_count(card)
        if tiles_per_device <= 0:
            tiles_per_device = 1
        if tiles_per_device > 1:
            device_id = card // tiles_per_device
            tile_id = card % tiles_per_device
            cmd = ["xpu-smi", "dump", "-d", str(device_id),
                   "-t", str(tile_id), "-m", "5", "-n", "1"]
        else:
            device_id = card
            tile_id = None
            cmd = ["xpu-smi", "dump", "-d", str(device_id),
                   "-m", "5", "-n", "1"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None

        # Find the column index for GPU Memory Utilization
        headers = [h.strip() for h in lines[0].split(',')]
        mem_util_idx = None
        for i, h in enumerate(headers):
            if "GPU Memory Utilization" in h:
                mem_util_idx = i
                break
        if mem_util_idx is None:
            return None

        # Parse the last data line
        values = [v.strip() for v in lines[-1].split(',')]
        if mem_util_idx >= len(values):
            return None

        val_str = values[mem_util_idx]
        if val_str in ("N/A", ""):
            return None

        val = float(val_str)
        # Normalise: values > 1 are in 0-100 percentage scale
        if val > 1.0:
            val = val / 100.0
        if tiles_per_device > 1:
            loc = f"GPU {card} (Device {device_id} Tile {tile_id})"
        else:
            loc = f"GPU {card}"
        threshold_info = f"; threshold: {memory_threshold:.2%}" if memory_threshold is not None else ""
        print(f"{loc} memory utilization: {val:.2%}{threshold_info}")
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
def get_model_list(suite: str, mode: str, model_only: str) -> list[str]:
    """Retrieve a list of model names from either a CSV file, a string list, or a default text file."""
    if model_only is not None:
        if os.path.isfile(model_only):
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

    models = []
    with list_file.open() as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = parse_string_list(line)
            if parts:
                model_name = parts[0].strip()
                if model_name:
                    models.append(model_name)
    return models


# Cached torch version (computed once)
_torch_version_cache = None

def _get_torch_version() -> str:
    global _torch_version_cache
    if _torch_version_cache is not None:
        return _torch_version_cache
    pip_cmd = "pip.exe" if IS_WINDOWS else "pip"
    try:
        result = subprocess.run([pip_cmd, "list", "--format=freeze"], capture_output=True, text=True, check=True)
        torch_line = [line for line in result.stdout.splitlines() if line.startswith("torch==")]
        _torch_version_cache = torch_line[0].split("==")[1].split("+")[0] if torch_line else "0.0.0"
    except Exception:
        _torch_version_cache = "0.0.0"
    return _torch_version_cache


# Benchmark execution (merged from inductor_xpu_test.sh)
def run_benchmark_with_prefix(
    task: TestTask,
    card: int,
    cmd_prefix: str,
    env_vars: dict,
    log_dir: Path,
    worker_id: int,
    device: str,
    shape: str,
) -> tuple[int, bool, Path]:
    """Run a single benchmark and return `(exit_code, success, test_result)`."""
    model_only = task.model
    suite = task.suite
    dt = task.dt
    mode = task.mode
    scenario = task.scenario

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inductor-logs-{task.model.replace('/', '_')}-worker{worker_id}-card{card}.log"
    log_csv = log_dir / f"inductor-results-{suite}-{dt}-{mode}-{device}-{scenario}.csv"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, prefix='tmp_', suffix='.csv')
    tmp_log_csv = tmp_file.name
    tmp_file.close()

    # Model only extra
    model_only_extra = ""
    if model_only:
        model_only_extra = f"--only {model_only}"

    # Version check
    torch_ver = _get_torch_version()

    mode_extra = ""
    if version.parse(torch_ver) >= version.parse("2.0.2"):
        mode_extra = "--inference "
    if mode == "training":
        mode_extra = "--training "

    real_dt = dt
    dt_extra = ""
    if dt == "amp_bf16":
        real_dt = "amp"
        dt_extra = "--amp-dtype bfloat16 "
    elif dt == "amp_fp16":
        real_dt = "amp"
        dt_extra = "--amp-dtype float16 "

    shape_extra = ""
    if shape == "dynamic":
        shape_extra = "--dynamic-shapes --dynamic-batch-only "

    # Partition flags: we are not using sharding per model, so set to empty
    partition_flags = ""

    cmd = (
        f"python benchmarks/dynamo/{suite}.py  -d {device} --{scenario} --{real_dt} "
        f"{dt_extra}{mode_extra}{shape_extra}{partition_flags}{model_only_extra} "
        f"--backend=inductor --cold-start-latency "
        f"-n10 --timeout=10800 --disable-cudagraphs --output={tmp_log_csv} "
    )
    cmd = re.sub(r'\s+', ' ', cmd).strip()

    # Build command as list for safe execution (no shell=True)
    _posix = not IS_WINDOWS
    if cmd_prefix:
        full_cmd_list = shlex.split(cmd_prefix, posix=_posix) + shlex.split(cmd, posix=_posix)
        full_cmd_str = f"{cmd_prefix} {cmd}"
    else:
        full_cmd_list = shlex.split(cmd, posix=_posix)
        full_cmd_str = cmd

    print(f" [Worker {worker_id}] Running: {full_cmd_str[:200]}...")

    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = str(card)
    env.update(env_vars)

    # Thread-safe buffer for error monitoring
    log_buffer = []
    log_buffer_lock = threading.Lock()

    popen_kwargs = dict(
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
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

    # Output reader with timestamps
    def output_reader():
        if proc.stdout is None:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
                formatted = f"{timestamp} {line.rstrip()}"
                print(formatted)

                with log_buffer_lock:
                    log_buffer.append(formatted)

                try:
                    log_f.write(formatted + "\n")
                    log_f.flush()
                except (ValueError, OSError):
                    pass  # file already closed or I/O error
        except (ValueError, OSError):
            pass  # stdout closed unexpectedly

    reader_thread = threading.Thread(target=output_reader, daemon=True)
    reader_thread.start()

    def error_monitor():
        timeout = 10800  # seconds
        start = time.time()
        scan_pos = 0
        memory_threshold = _parse_memory_threshold()
        while proc.poll() is None and (time.time() - start) < timeout:
            # --- Check log buffer for text-based error patterns ---
            with log_buffer_lock:
                new_lines = log_buffer[scan_pos:]
                scan_pos = len(log_buffer)
            if new_lines:
                content = "\n".join(new_lines).lower()
                for pattern in error_patterns:
                    # Skip Memory>N patterns; handled via GPU polling below
                    if pattern.startswith("Memory>"):
                        continue
                    if pattern.lower() in content:
                        print(f"  [Worker {worker_id}] Detected '{pattern}', killing process")
                        proc.kill()
                        return

            # --- Poll GPU memory utilization via xpu-smi ---
            if memory_threshold is not None:
                try:
                    mem_util = _get_gpu_memory_utilization(card, memory_threshold)
                except Exception as e:
                    print(f"  [Worker {worker_id}] GPU memory poll error: {e}")
                    mem_util = None
                if mem_util is not None and mem_util >= memory_threshold:
                    print(
                        f"  [Worker {worker_id}] GPU memory utilization "
                        f"{mem_util:.2%} >= threshold {memory_threshold:.0%}, "
                        f"killing process"
                    )
                    proc.kill()
                    return

            time.sleep(3)

    monitor_thread = threading.Thread(target=error_monitor, daemon=True)
    monitor_thread.start()

    try:
        exit_code = proc.wait()
    except Exception:
        exit_code = -1
    reader_thread.join(timeout=10)
    if log_f is not None and not log_f.closed:
        log_f.close()
    monitor_thread.join(timeout=1)

    try:
        collect_csv_results(log_csv, tmp_log_csv, device, task)
    except Exception as e:
        print(f"  [Worker {worker_id}] CSV collection error for {task.model}: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_log_csv)
        except OSError:
            pass

    # Determine success based on last non-empty line of log file
    def check_success_from_log(log_path: Path) -> tuple[str, bool]:
        """
        Return (test_result, True) if last line ends with 'pass' or 'pass_due_to_skip'.
        test_result is the last whitespace-separated part of the last non-empty line.
        """
        if not log_path.exists():
            return "None", False
        last_non_empty = ""
        with open(log_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_non_empty = stripped
        if last_non_empty:
            # Split by whitespace and take the last part
            parts = last_non_empty.split()
            test_result = parts[-1] if parts else ""
            if test_result in ('pass', 'pass_due_to_skip') or re.fullmatch(r'[0-9.]+x', test_result):
                return test_result, True
            return test_result, False
        return "None", False

    test_result, success = check_success_from_log(log_file)

    return exit_code, success, test_result


def collect_csv_results(
    log_csv: Path,
    tmp_log_csv: str,
    device: str,
    task: TestTask,
) -> None:
    """Append a crash row to the CSV file."""
    import csv

    header = []
    last_match = []

    def condition(row: list[str]) -> bool:
        return row and row[0] == device

    if os.path.isfile(tmp_log_csv):
        try:
            with open(tmp_log_csv, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Find first non-empty row as header
                for row in reader:
                    if row:  # skip empty rows
                        header = row
                        break
                # Continue scanning for matching row
                for row in reader:
                    if row and condition(row):
                        last_match = row
        except Exception as e:
            print(f"Warning: Error reading temp CSV {tmp_log_csv}: {e}")

    if not last_match:
        if task.scenario == "accuracy":
            header = ["dev","name","batch_size","accuracy"]
            last_match = [device, task.model, 0, "crashed"]
        elif task.scenario == "performance":
            header = ["dev","name","batch_size","speedup","abs_latency"]
            last_match = [device, task.model, 0, 0, 0]
        else:
            raise ValueError(f"Unknown task.scenario: {task.scenario}")

    # Build the final row
    final_header = ["scenario", "suite", "dtype", "mode"] + header
    final_row = [task.scenario, task.suite, task.dt, task.mode] + last_match

    with _csv_lock:
        try:
            exists = os.path.isfile(log_csv)
            mode = 'a' if exists else 'w'
            with open(log_csv, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(final_header)
                writer.writerow(final_row)
        except Exception as e:
            raise RuntimeError(f"Error writing to destination file: {e}") from e


# CPU topology and command prefix generation
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
                print(f"Detected (Windows): {logical} logical CPUs, {physical_cores} physical cores")
                return physical_cores
    except Exception:
        pass
    # Fallback: assume half of logical CPUs are physical cores
    logical = os.cpu_count() or 1
    physical_cores = max(logical // 2, 1)
    print(f"Detected (Windows fallback): {logical} logical CPUs, estimated {physical_cores} physical cores")
    return physical_cores


def _get_cpu_topology_linux() -> int:
    """Return number of physical cores on Linux via lscpu."""
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except Exception as e:
        print(f"ERROR: Could not run lscpu: {e}")
        sys.exit(1)

    online_cpus = None
    threads_per_core = None

    for line in output.splitlines():
        if "On-line CPU(s) list:" in line:
            parts = line.split(':')
            if len(parts) >= 2:
                cpu_range = parts[1].strip()
                count = 0
                for segment in cpu_range.split(','):
                    segment = segment.strip()
                    if '-' in segment:
                        lo, hi = map(int, segment.split('-'))
                        count += hi - lo + 1
                    else:
                        count += 1
                online_cpus = count
        elif "Thread(s) per core:" in line:
            parts = line.split(':')
            if len(parts) >= 2:
                threads_per_core = int(parts[1].strip())

    if online_cpus is None or threads_per_core is None:
        print("ERROR: Could not parse CPU topology")
        sys.exit(1)

    physical_cores = online_cpus // threads_per_core
    print(f"Detected: {online_cpus} logical CPUs, {threads_per_core} threads/core → {physical_cores} physical cores")
    return physical_cores


def generate_command_prefixes(num_gpus: int) -> list[tuple[int, str, dict]]:
    """
    Read ZE_AFFINITY_MASK and CPU topology.
    Returns list of (card_number, command_prefix, env_vars) for each GPU to use.
    command_prefix includes numactl CPU binding; env_vars includes OMP_NUM_THREADS.
    """
    ze_mask = os.environ.get("ZE_AFFINITY_MASK", "")
    if not ze_mask:
        gpu_list = list(range(num_gpus))
        print(f"ZE_AFFINITY_MASK not set, using all {num_gpus} GPUs")
    else:
        gpu_list = []
        try:
            for part in ze_mask.split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    parts = part.split('-')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid range format: '{part}'")
                    start, end = int(parts[0]), int(parts[1])
                    gpu_list.extend(range(start, end+1))
                else:
                    gpu_list.append(int(part))
        except (ValueError, IndexError) as e:
            print(f"ERROR: Invalid ZE_AFFINITY_MASK format '{ze_mask}': {e}")
            sys.exit(1)
        if not gpu_list:
            print("ERROR: ZE_AFFINITY_MASK produced no GPUs")
            sys.exit(1)
        print(f"ZE_AFFINITY_MASK: using GPUs {gpu_list}")

    physical_cores = get_cpu_topology()
    cores_per_gpu = physical_cores // num_gpus
    if cores_per_gpu < 1:
        cores_per_gpu = 1

    prefixes = []
    for gpu in gpu_list:
        start_core = gpu * cores_per_gpu
        end_core = min(start_core + cores_per_gpu - 1, physical_cores - 1)
        core_range = f"{start_core}-{end_core}"
        env_vars = {}
        if IS_WINDOWS:
            prefix = ""
            print(f"GPU {gpu} → OMP_NUM_THREADS={cores_per_gpu} (numactl not available on Windows)")
        else:
            env_vars = {"OMP_NUM_THREADS": str(cores_per_gpu)}
            prefix = f"numactl -l -C {core_range}"
            print(f"GPU {gpu} → cores {core_range} (OMP_NUM_THREADS={cores_per_gpu})")
        prefixes.append((gpu, prefix, env_vars))
    return prefixes


# Main orchestrator
def main():
    parser = argparse.ArgumentParser(description="Run E2E tests with per‑GPU job queue and hang detection")
    parser.add_argument("--suite", default="",
                        help="Comma- or space-separated list of suites. Valid: huggingface, timm_models, torchbench. "
                             "Empty means all.")
    parser.add_argument("--dt", default="",
                        help="Comma- or space-separated list of data types. Valid: float32, bfloat16, float16, "
                             "amp_bf16, amp_fp16. Empty means all.")
    parser.add_argument("--mode", default="",
                        help="Comma- or space-separated list of modes. Valid: inference, training. Empty means all.")
    parser.add_argument("--scenario", default="",
                        help="Comma- or space-separated list of scenarios. Valid: accuracy, performance. Empty means all.")
    parser.add_argument("--model-only", default=None,
                        help="Run only a single model (overrides list file).")
    parser.add_argument("--device", default="xpu", help="Device type (xpu or cuda)")
    parser.add_argument("--shape", default="static", help="Shape mode (static or dynamic)")
    parser.add_argument("--numactl-args", default="",
                        help="Override NUMACTL_ARGS (semicolon-separated per GPU)")
    args = parser.parse_args()

    # Check required environment variables
    num_gpus_str = os.getenv('NUM_GPUS')
    if num_gpus_str is None:
        sys.exit("ERROR: Environment variable NUM_GPUS is not set.")
    try:
        num_gpus = int(num_gpus_str)
    except ValueError:
        sys.exit(f"ERROR: NUM_GPUS must be an integer, got '{num_gpus_str}'.")
    if num_gpus <= 0:
        sys.exit(f"ERROR: NUM_GPUS must be positive, got {num_gpus}.")

    print(f"NUM_GPUS = {num_gpus}")

    # Change to pytorch directory (workflow already cd's there)
    if not Path("benchmarks/dynamo").is_dir():
        print("ERROR: benchmarks/dynamo directory not found. Are you in the pytorch directory?")
        sys.exit(1)

    VALID_SUITES = {"huggingface", "timm_models", "torchbench"}
    VALID_DT = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
    VALID_MODES = {"inference", "training"}
    VALID_SCENARIOS = {"accuracy", "performance"}

    # Parse arguments with support for both comma and space separation
    suites = parse_string_list(args.suite) if args.suite else list(VALID_SUITES)
    dts = parse_string_list(args.dt) if args.dt else list(VALID_DT)
    modes = parse_string_list(args.mode) if args.mode else list(VALID_MODES)
    scenarios = parse_string_list(args.scenario) if args.scenario else list(VALID_SCENARIOS)

    # Validate and filter
    def filter_valid(items, valid_set, name):
        valid_items = [i for i in items if i in valid_set]
        invalid = [i for i in items if i not in valid_set]
        if invalid:
            print(f"Warning: Skipping invalid {name}: {', '.join(invalid)}")
        return valid_items

    suites = filter_valid(suites, VALID_SUITES, "suite")
    dts = filter_valid(dts, VALID_DT, "data type")
    modes = filter_valid(modes, VALID_MODES, "mode")
    scenarios = filter_valid(scenarios, VALID_SCENARIOS, "scenario")

    if not suites or not dts or not modes or not scenarios:
        print("ERROR: No valid combinations left after filtering.")
        sys.exit(1)

    # Build task list
    tasks: list[TestTask] = []
    for suite in suites:
        for mode in modes:
            for dt in dts:
                for scenario in scenarios:
                    models = get_model_list(suite, mode, args.model_only)
                    for model in sorted(set(models)):
                        tasks.append(TestTask(suite, dt, mode, scenario, model))

    total_tasks = len(tasks)

    if total_tasks == 0:
        print("No valid tasks generated.")
        sys.exit(1)

    print(f"Total tasks to run: {total_tasks}")

    # Determine GPU workers and CPU binding
    if args.numactl_args and args.numactl_args.strip():
        prefix_strings = [p.strip().rstrip(';') for p in args.numactl_args.split(';') if p.strip()]
        if len(prefix_strings) < num_gpus:
            prefix_strings += [prefix_strings[-1]] * (num_gpus - len(prefix_strings))
        # Extract card number from ZE_AFFINITY_MASK=N in each prefix, fallback to index
        workers = []
        for i, ps in enumerate(prefix_strings):
            m = re.search(r'ZE_AFFINITY_MASK=(\d+)', ps)
            card = int(m.group(1)) if m else i
            workers.append((card, ps, {}))
        print(f"Using user-provided NUMACTL_ARGS prefixes ({len(workers)} workers)")
    else:
        workers = generate_command_prefixes(num_gpus)
        print(f"Auto-generated {len(workers)} workers from ZE_AFFINITY_MASK")

    _run_workers(tasks, workers, args)


def _run_workers(tasks, workers, args):
    total_tasks = len(tasks)

    # Job queue
    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results_queue = queue.Queue()  # (task, success)
    completed_counter = [0]
    counter_lock = threading.Lock()

    def worker(worker_id: int, card: int, cmd_prefix: str, env_vars: dict,
            total_tasks: int, completed_counter: list, counter_lock: threading.Lock,
            device: str, shape: str):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break

            log_dir = Path.cwd().resolve() / "inductor_log" / task.suite / task.dt / task.mode / task.scenario
            try:
                exit_code, success, test_result = run_benchmark_with_prefix(
                    task=task,
                    card=card,
                    cmd_prefix=cmd_prefix,
                    env_vars=env_vars,
                    log_dir=log_dir,
                    worker_id=worker_id,
                    device=device,
                    shape=shape,
                )
            except Exception as e:
                print(f"  [Worker {worker_id}] Exception running {task.model}: {e}")
                exit_code, success, test_result = -1, False, "exception"

            # Increment completed counter
            with counter_lock:
                completed_counter[0] += 1
                new_completed = completed_counter[0]
            prefix = f"[Tasks {new_completed}/{total_tasks}]"

            if success:
                print(f"  {prefix} [Worker {worker_id}] {task} successfully {test_result}.")
            elif exit_code == 0:
                print(f"  {prefix} [Worker {worker_id}] {task} failed {test_result}.")
            elif exit_code == -9:
                print(f"  {prefix} [Worker {worker_id}] {task} killed due to hang issue.")
            else:
                print(f"  {prefix} [Worker {worker_id}] {task} failed with exit code {exit_code}.")

            results_queue.put((task, success))

    threads = []
    for idx, (card, prefix, env_vars) in enumerate(workers):
        t = threading.Thread(
            target=worker,
            args=(idx, card, prefix, env_vars,
                  total_tasks, completed_counter, counter_lock,
                  args.device, args.shape),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Collect results
    failed_tasks = []
    while not results_queue.empty():
        task, success = results_queue.get()
        if not success:
            failed_tasks.append(task)

    print("\n" + "=" * 40)
    print("Test execution finished")
    print(f"Total tasks: {total_tasks}")
    print(f"Failed tasks: {len(failed_tasks)}")
    if failed_tasks:
        print("Failed tasks:")
        for task in failed_tasks:
            print(f"  - {task.suite} {task.dt} {task.mode} {task.scenario} {task.model}")
    else:
        print(f"[Tasks {total_tasks}/{total_tasks}] All tasks completed successfully.")


if __name__ == "__main__":
    main()
