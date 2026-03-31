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
import queue
from packaging import version


# Data structures
@dataclass
class TestTask:
    suite: str
    dt: str
    mode: str
    scenario: str
    model: str


# Model list extraction (parse model names from file)
_SPLIT_RE = re.compile(r'[,\s]+')  # Regex to handle comma or space separation
_K_RE = re.compile(r'-k\s+(\S+)')  # Regex to handle -k flags
def get_model_list(suite: str, model_only: str | None = None) -> list[str]:
    """Get list of model names.
    Priority:
      1. If model_only is provided, parse it
      2. Otherwise read from benchmarks/dynamo/<base>_models_list.txt
    model_only supports:
      - "model1,model2,model3" -> ["model1", "model2", "model3"]
      - "-k model1 -k model2" -> ["model1", "model2"]
      - "model1 model2 model3" -> ["model1", "model2", "model3"]
    File format per line:
      model_name[,<number>] or model_name <number>
    """

    models: list[str] = []

    # --- Case 1: explicit override ---
    if model_only:
        model_only = model_only.strip()

        # Handle -k flags
        matches = _K_RE.findall(model_only)
        if matches:
            # If -k flags are found, split by commas within each -k value
            return [m for match in matches for m in match.split(',') if m]

        # Comma-separated (handles "model1,model2,model3")
        if ',' in model_only:
            return [m.strip() for m in model_only.split(',') if m.strip()]

        # Space-separated (handles "model1 model2 model3")
        return [m.strip() for m in model_only.split() if m.strip()]

    # --- Case 2: fallback to file ---
    base = suite.replace('_models', '')
    list_file = Path(f"benchmarks/dynamo/{base}_models_list.txt")

    if not list_file.exists():
        raise FileNotFoundError(f"Model list file not found: {list_file}")

    with list_file.open() as f:
        for line in f:
            line = line.split('#', 1)[0].strip()  # Remove comments
            if not line:
                continue

            parts = _SPLIT_RE.split(line)  # Split by comma or whitespace
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
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"model_{task.model.replace('/', '_')}_worker{worker_id}.log"

    model_only = task.model
    suite = task.suite
    dt = task.dt
    mode = task.mode
    scenario = task.scenario

    # Model only extra
    model_only_extra = ""
    if model_only:
        if " -k " in model_only:
            model_only_extra = f" {model_only} "
        else:
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
        f"python benchmarks/dynamo/{suite}.py --{scenario} --{real_dt} "
        f"-d {device} -n10 {dt_extra}{mode_extra}{shape_extra}{partition_flags}{model_only_extra} "
        f"--backend=inductor --cold-start-latency --timeout=10800 --disable-cudagraphs "
        f"--output={log_dir}/inductor_{suite}_{dt}_{mode}_{device}_{scenario}.csv"
    )
    cmd = re.sub(r'\s+', ' ', cmd).strip()

    # Prepend numactl prefix if provided
    if cmd_prefix:
        full_cmd = f"{cmd_prefix} {cmd}"
    else:
        full_cmd = cmd

    print(f"  [Worker {worker_id}] Running on card {card}: {full_cmd[:200]}...")

    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = str(card)
    env.update(env_vars)

    log_f = open(log_file, "w")
    proc = subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    # Output reader with timestamps
    def output_reader():
        for line in iter(proc.stdout.readline, ""):
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
            formatted = f"{timestamp} {line.rstrip()}"
            print(formatted)

            # Check if the log file is still open before writing
            if not log_f.closed:
                log_f.write(formatted + "\n")
                log_f.flush()

    reader_thread = threading.Thread(target=output_reader, daemon=True)
    reader_thread.start()

    # Monitor for OutOfMemoryError or UR_RESULT_ERROR
    error_patterns = ["OutOfMemoryError", "UR_RESULT_ERROR"]

    def error_monitor():
        timeout = 300  # seconds
        start = time.time()
        while proc.poll() is None and (time.time() - start) < timeout:
            try:
                with open(log_file) as f:
                    content = f.read()
                    for pattern in error_patterns:
                        if pattern in content:
                            print(f"  [Worker {worker_id}] Detected '{pattern}', killing process")
                            proc.kill()
                            return
            except Exception:
                pass
            time.sleep(5)

    monitor_thread = threading.Thread(target=error_monitor, daemon=True)
    monitor_thread.start()

    exit_code = proc.wait()
    reader_thread.join(timeout=1)
    monitor_thread.join(timeout=1)
    log_f.close()

    # Determine success based on last non-empty line of log file
    def check_success_from_log(log_path: Path) -> tuple[str, bool]:
        """
        Return (last_token, True) if last line ends with 'pass' or 'pass_due_to_skip'.
        last_token is the last whitespace-separated part of the last non-empty line.
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
            last_token = parts[-1] if parts else ""
            if last_token in ('pass', 'pass_due_to_skip') or re.fullmatch(r'[0-9.]+x', last_token):
                return last_token, True
            return last_token, False
        return "None", False

    last_token, success = check_success_from_log(log_file)

    if success:
        print(f"  [Worker {worker_id}] successfully and Model {model_only} successfully {last_token}.")
        return True, log_file

    if exit_code == 0:
        print(f"  [Worker {worker_id}] successfully but Model {model_only} failed {last_token}.")
        return False, log_file
    else:
        try:
            with open(log_file) as f:
                content = f.read()
                for pattern in error_patterns:
                    if pattern in content:
                        print(f"  [Worker {worker_id}] failed and Model {model_only} killed due to {pattern}")
                        return False, log_file
                print(f"  [Worker {worker_id}] failed and Model {model_only} failed with exit code {exit_code}. Check {log_file}")
        except Exception:
            pass
        return False, log_file

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
        for part in ze_mask.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                gpu_list.extend(range(start, end+1))
            else:
                gpu_list.append(int(part))
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


# Main orchestrator

def main():
    parser = argparse.ArgumentParser(description="Run E2E tests with per‑GPU job queue and hang detection")
    parser.add_argument("--suite", default="", help="Comma-separated list of suites (empty = all)")
    parser.add_argument("--dt", default="", help="Comma-separated list of data types (empty = all)")
    parser.add_argument("--mode", default="", help="Comma-separated list of modes (empty = all)")
    parser.add_argument("--scenario", default="", help="Comma-separated list of scenarios (empty = all)")
    parser.add_argument("--model-only", default=None, help="Run only a single model (overrides list file)")
    parser.add_argument("--device", default="xpu", help="Device type (xpu or cuda)")
    parser.add_argument("--shape", default="static", help="Shape mode (static or dynamic)")
    parser.add_argument("--numactl-args", default="", help="Override NUMACTL_ARGS (semicolon-separated per GPU)")
    args = parser.parse_args()

    # Read NUM_GPUS from environment, default 8
    num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    print(f"NUM_GPUS = {num_gpus}")

    # Change to pytorch directory (workflow already cd's there)
    if not Path("benchmarks/dynamo").is_dir():
        print("ERROR: benchmarks/dynamo directory not found. Are you in the pytorch directory?")
        sys.exit(1)

    VALID_SUITES = {"huggingface", "timm_models", "torchbench"}
    VALID_DT = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
    VALID_MODES = {"inference", "training"}
    VALID_SCENARIOS = {"accuracy", "performance"}

    suites = [s.strip() for s in args.suite.split(',') if s.strip()] if args.suite else list(VALID_SUITES)
    dts = [d.strip() for d in args.dt.split(',') if d.strip()] if args.dt else list(VALID_DT)
    modes = [m.strip() for m in args.mode.split(',') if m.strip()] if args.mode else list(VALID_MODES)
    scenarios = [sc.strip() for sc in args.scenario.split(',') if sc.strip()] if args.scenario else list(VALID_SCENARIOS)

    # Build task list
    tasks: list[TestTask] = []
    for suite in suites:
        if suite not in VALID_SUITES:
            print(f"Skipping invalid suite: {suite}")
            continue
        models = get_model_list(suite, args.model_only)
        for dt in dts:
            if dt not in VALID_DT:
                print(f"Skipping invalid dt: {dt}")
                continue
            for mode in modes:
                if mode not in VALID_MODES:
                    print(f"Skipping invalid mode: {mode}")
                    continue
                for scenario in scenarios:
                    if scenario not in VALID_SCENARIOS:
                        print(f"Skipping invalid scenario: {scenario}")
                        continue
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

    # Job queue
    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results_queue = queue.Queue()  # (task, success)

    def worker(worker_id: int, card: int, cmd_prefix: str, env_vars: dict):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break
            log_dir = Path.cwd().resolve() / "inductor_log" / task.suite / task.dt / task.mode / task.scenario
            success, _ = run_benchmark_with_prefix(
                task=task,
                card=card,
                cmd_prefix=cmd_prefix,
                env_vars=env_vars,
                log_dir=log_dir,
                worker_id=worker_id,
                device=args.device,
                shape=args.shape,
            )
            results_queue.put((task, success))

    threads = []
    for idx, (card, prefix, env_vars) in enumerate(workers):
        t = threading.Thread(target=worker, args=(idx, card, prefix, env_vars), daemon=True)
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
        print("All tasks completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
