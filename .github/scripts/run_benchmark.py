#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import re
from pathlib import Path
from dataclasses import dataclass
import argparse
import tempfile
import queue
import shlex
import shutil
from packaging import version
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
error_patterns = ["out of memory", "OutOfMemory", "UR_RESULT_ERROR"]

# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


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
) -> tuple[bool, Path]:
    """Run a single benchmark with optional numactl prefix and extra environment variables."""
    model_only = task.model
    suite = task.suite
    dt = task.dt
    mode = task.mode
    scenario = task.scenario

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inductor-logs-{task.model.replace('/', '_')}-worker{worker_id}-card{card}.log"
    log_csv = log_dir / f"inductor-results-{suite}-{dt}-{mode}-{device}-{scenario}.csv"
    tmp_file = tempfile.NamedTemporaryFile(delete=True, prefix='tmp_', suffix='.csv')
    tmp_log_csv = tmp_file.name
    tmp_file.close()

    # Model only extra
    model_only_extra = ""
    if model_only:
        model_only_extra = f"--only {model_only}"

    # Version check
    try:
        result = subprocess.run(["pip", "list", "--format=freeze"], capture_output=True, text=True, check=True)
        torch_line = [line for line in result.stdout.splitlines() if line.startswith("torch==")]
        torch_ver = torch_line[0].split("==")[1].split("+")[0] if torch_line else "0.0.0"
    except Exception:
        torch_ver = "0.0.0"

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
    if cmd_prefix:
        full_cmd_list = shlex.split(cmd_prefix) + shlex.split(cmd)
        full_cmd_str = f"{cmd_prefix} {cmd}"
    else:
        full_cmd_list = shlex.split(cmd)
        full_cmd_str = cmd

    print(f" [Worker {worker_id}] Running: {full_cmd_str[:200]}...")

    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = str(card)
    env.update(env_vars)

    # Thread-safe buffer for error monitoring
    log_buffer = []
    log_buffer_lock = threading.Lock()

    try:
        log_f = open(log_file, "w")
        proc = subprocess.Popen(
            full_cmd_list,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
    except Exception as e:
        if 'log_f' in locals() and not log_f.closed:
            log_f.close()
        raise RuntimeError(f"Failed to start benchmark for {task.model}: {e}") from e

    # Output reader with timestamps
    def output_reader():
        for line in iter(proc.stdout.readline, ""):
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
            formatted = f"{timestamp} {line.rstrip()}"
            print(formatted)

            with log_buffer_lock:
                log_buffer.append(formatted)

            if not log_f.closed:
                log_f.write(formatted + "\n")
                log_f.flush()

    reader_thread = threading.Thread(target=output_reader, daemon=True)
    reader_thread.start()

    def error_monitor():
        timeout = 10800  # seconds
        start = time.time()
        while proc.poll() is None and (time.time() - start) < timeout:
            with log_buffer_lock:
                content = "\n".join(log_buffer)
            for pattern in error_patterns:
                if pattern.lower() in content.lower():
                    print(f"  [Worker {worker_id}] Detected '{pattern}', killing process")
                    proc.kill()
                    return
            time.sleep(30)

    monitor_thread = threading.Thread(target=error_monitor, daemon=True)
    monitor_thread.start()

    exit_code = proc.wait()
    reader_thread.join(timeout=1)
    monitor_thread.join(timeout=1)
    log_f.close()
    collect_csv_results(log_csv, tmp_log_csv, device, task)

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
            raise RuntimeError(f"Error reading source file: {e}") from e

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
                if '-' in cpu_range:
                    start, end = map(int, cpu_range.split('-'))
                    online_cpus = end - start + 1
                else:
                    online_cpus = len(cpu_range.split(','))
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
        prefix = f"numactl -C {core_range}"
        env_vars = {"OMP_NUM_THREADS": str(cores_per_gpu)}
        prefixes.append((gpu, prefix, env_vars))
        print(f"GPU {gpu} → cores {core_range} (OMP_NUM_THREADS={cores_per_gpu})")
    return prefixes


# Earlyoom handling
def start_earlyoom() -> subprocess.Popen | None:
    """Start earlyoom process if available. Return Popen object or None."""
    earlyoom_path = shutil.which("earlyoom")
    if not earlyoom_path:
        print("INFO: earlyoom not found in PATH. Memory pressure monitoring disabled.")
        return None

    # Arguments as requested: -m 3 -s 100 -r 3600 --prefer '^(python|pytest)'
    cmd = [
        earlyoom_path,
        "-m", "3",
        "-s", "100",
        "-r", "600",
        "--prefer", "^(python|pytest)"
    ]
    try:
        # Start earlyoom; redirect stdout/stderr to /dev/null to avoid clutter,
        # but we could also let it print to console for visibility. We'll keep it quiet.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # so it can be killed cleanly later
        )
        print(f"Started earlyoom (PID {proc.pid}) to monitor memory pressure.")
        return proc
    except Exception as e:
        print(f"WARNING: Failed to start earlyoom: {e}. Continuing without it.")
        return None


def stop_earlyoom(proc: subprocess.Popen | None) -> None:
    """Terminate earlyoom process if it is running."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
        print("earlyoom process terminated.")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print("earlyoom process killed (timeout).")
    except Exception as e:
        print(f"Warning while stopping earlyoom: {e}")


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
    num_gpus = os.getenv('NUM_GPUS')
    if num_gpus is None:
        sys.exit("ERROR: Environment variable NUM_GPUS is not set.")
    num_gpus = int(num_gpus)

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
        sys.exit(0)

    print(f"Total tasks to run: {total_tasks}")

    # Determine GPU workers and CPU binding
    if args.numactl_args and args.numactl_args.strip():
        prefix_strings = [p.strip().rstrip(';') for p in args.numactl_args.split(';') if p.strip()]
        if len(prefix_strings) < num_gpus:
            prefix_strings += [prefix_strings[-1]] * (num_gpus - len(prefix_strings))
        # Assign sequential card numbers 0..len(prefix_strings)-1
        workers = [(i, prefix_strings[i], {}) for i in range(len(prefix_strings))]
        print(f"Using user-provided NUMACTL_ARGS prefixes ({len(workers)} workers)")
    else:
        workers = generate_command_prefixes(num_gpus)
        print(f"Auto-generated {len(workers)} workers from ZE_AFFINITY_MASK")

    # Start earlyoom if available
    earlyoom_proc = start_earlyoom()

    try:
        _run_workers(tasks, workers, args, earlyoom_proc)
    finally:
        stop_earlyoom(earlyoom_proc)


def _run_workers(tasks, workers, args, earlyoom_proc):
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
        sys.exit(1)
    else:
        print(f"[Tasks {total_tasks}/{total_tasks}] All tasks completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
