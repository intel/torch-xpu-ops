"""Entry point for `python -m benchmarks`."""

import argparse
import sys
from pathlib import Path

from . import config
from .log import banner, log
from .runner import run_all
from .tasks import (
    filter_valid,
    generate_tasks,
    load_tasks_from_file,
    load_tasks_from_scope,
    parse_string_list,
)
from .workers import (
    generate_cpu_workers,
    generate_gpu_workers,
    get_num_gpus,
    parse_numactl_args,
)



def _explicit(value: str) -> set[str] | None:
    """Return the CLI filter values, or None when unset (all)."""
    return set(parse_string_list(value)) or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E2E benchmarks with per-GPU job queue and hang detection",
    )
    parser.add_argument(
        "--suite",
        default="",
        help=(
            "Comma- or space-separated suites "
            "(huggingface, timm_models, torchbench, pt2e). Empty = all."
        ),
    )
    parser.add_argument(
        "--dt",
        "--dtype",
        default="",
        help=(
            "Comma- or space-separated dtypes "
            "(float32, bfloat16, float16, amp_bf16, amp_fp16, int8). Empty = all."
        ),
    )
    parser.add_argument("--mode", default="",
                        help="Comma- or space-separated modes (inference, training). Empty = all.")
    parser.add_argument("--scenario", default="",
                        help="Comma- or space-separated scenarios (accuracy, performance). Empty = all.")
    parser.add_argument("--model-only", default=None,
                        help="Run only specific model(s). Filters --task-file tasks when given.")
    parser.add_argument("--device", default="xpu", help="Device type (xpu, cuda, or cpu)")
    parser.add_argument("--backend", default="default", choices=sorted(config.VALID_BACKENDS),
                        help="Inductor backend variant (default, dynamic, cpp_wrapper, freezing, "
                             "aot_inductor, inductor_deterministic_perf). Applies to inductor suites only.")
    parser.add_argument("--task-file", default=None,
                        help="Comma-separated task sources, run one by one. A tier config .yaml uses its own "
                             "backends/precisions/scenarios/models (CLI --suite/--dt/--mode/--scenario filter it); "
                             ".csv/.txt holds task rows and uses --backend. --model-only filters the result.")
    parser.add_argument("--cores-per-instance", type=int, default=None,
                        help="CPU-only: cores per worker instance. Default = all cores per NUMA node.")
    parser.add_argument("--numactl-args", default="",
                        help="Override NUMACTL_ARGS (semicolon-separated per GPU)")
    parser.add_argument("--dataset-dir", default="",
                        help="Path to ImageNet dataset directory (required for pt2e accuracy)")
    parser.add_argument("--gpu-memory-threshold", type=float, default=None,
                        help="GPU memory utilisation threshold (0.0-1.0) to kill a process. "
                             "Default: 1.0.")
    parser.add_argument("--no-gpu-memory-monitor", action="store_true", default=False,
                        help="Disable GPU memory monitoring (enabled by default).")
    parser.add_argument("--task-timeout", type=int, default=None,
                        help="Max wall-clock seconds a single task may run before being killed "
                             f"(default: {config.task_timeout_seconds}).")
    args = parser.parse_args()

    # Apply GPU memory monitoring settings
    if args.no_gpu_memory_monitor:
        config.gpu_memory_monitor_enabled = False
    if args.gpu_memory_threshold is not None:
        config.gpu_memory_threshold = args.gpu_memory_threshold
        config.error_patterns = [
            p for p in config.error_patterns if not p.startswith("Memory>")
        ] + [f"Memory>{config.gpu_memory_threshold}"]
    if args.task_timeout is not None:
        config.task_timeout_seconds = args.task_timeout

    is_cpu = args.device == "cpu"
    num_gpus = get_num_gpus(required=not is_cpu)

    if not Path("benchmarks/dynamo").is_dir() and not args.task_file:
        requested = parse_string_list(args.suite)
        if not requested or set(requested) != {"pt2e"}:
            sys.exit("ERROR: benchmarks/dynamo directory not found. Are you in the pytorch directory?")

    task_files = [p.strip() for p in args.task_file.split(",") if p.strip()] if args.task_file else []

    if task_files:
        runs = []
        for path in task_files:
            if path.endswith((".yaml", ".yml")):
                tasks = load_tasks_from_scope(
                    path, _explicit(args.suite), _explicit(args.dt), _explicit(args.mode), _explicit(args.scenario))
            else:
                tasks = load_tasks_from_file(path)
                for t in tasks:
                    t.backend = args.backend
            if args.model_only:
                names = set(parse_string_list(args.model_only))
                tasks = [t for t in tasks if t.model in names]
            if not tasks:
                sys.exit(f"No valid tasks found in {path}.")
            label = None
            if len(task_files) > 1:
                label = Path(path).parent.name if Path(path).stem == "config" else Path(path).stem
            runs.append((label, tasks))
        all_tasks = [t for _, ts in runs for t in ts]
        suites = sorted({t.suite for t in all_tasks})
        dts = sorted({t.dt for t in all_tasks})
        modes = sorted({t.mode for t in all_tasks})
        scenarios = sorted({t.scenario for t in all_tasks})
        backends = sorted({t.backend for t in all_tasks})
    else:
        suites = filter_valid(
            parse_string_list(args.suite) or list(config.VALID_SUITES), config.VALID_SUITES, "suite")
        dts = filter_valid(
            parse_string_list(args.dt) or list(config.VALID_DT), config.VALID_DT, "data type")
        modes = filter_valid(
            parse_string_list(args.mode) or list(config.VALID_MODES), config.VALID_MODES, "mode")
        scenarios = filter_valid(
            parse_string_list(args.scenario) or list(config.VALID_SCENARIOS), config.VALID_SCENARIOS, "scenario")

        if not all([suites, dts, modes, scenarios]):
            sys.exit("ERROR: No valid combinations left after filtering.")

        tasks = generate_tasks(suites, dts, modes, scenarios, args.model_only)
        if not tasks:
            sys.exit("No valid tasks generated.")
        for t in tasks:
            t.backend = args.backend
        runs = [(None, tasks)]
        backends = [args.backend]

    banner("Configuration")
    if num_gpus is not None:
        log(f"NUM_GPUS:   {num_gpus}")
    log(f"Suites:     {', '.join(suites)}")
    log(f"Dtypes:     {', '.join(dts)}")
    log(f"Modes:      {', '.join(modes)}")
    log(f"Scenarios:  {', '.join(scenarios)}")
    log(f"Device:     {args.device}")
    log(f"Backends:   {', '.join(backends)}")
    if args.cores_per_instance:
        log(f"Cores/inst: {args.cores_per_instance}")
    log(f"Tasks:      {sum(len(t) for _, t in runs)}")

    # Determine workers
    banner("Worker Setup")
    if args.numactl_args and args.numactl_args.strip():
        if num_gpus is None:
            sys.exit("ERROR: --numactl-args requires NUM_GPUS to be set.")
        workers = parse_numactl_args(args.numactl_args, num_gpus)
    elif is_cpu:
        workers = generate_cpu_workers(args.cores_per_instance)
        log(f"Auto-generated {len(workers)} CPU worker(s)")
    else:
        if num_gpus is None:
            sys.exit("ERROR: NUM_GPUS must be set for GPU devices.")
        workers = generate_gpu_workers(num_gpus)
        log(f"Auto-generated {len(workers)} GPU worker(s)")

    banner("Running Benchmarks")
    total_failed = 0
    for label, tasks in runs:
        if label:
            banner(f"Task file {label}")
        log_root = Path.cwd().resolve() / "inductor_log" / (label or "")
        total_failed += run_all(tasks, workers, args.device, args.dataset_dir, base_log_dir=log_root)

    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
