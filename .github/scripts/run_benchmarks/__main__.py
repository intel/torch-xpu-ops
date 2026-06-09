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
    parse_string_list,
)
from .workers import (
    generate_cpu_workers,
    generate_gpu_workers,
    get_num_gpus,
    parse_numactl_args,
)


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
                        help="Run only specific model(s) (overrides list file).")
    parser.add_argument("--device", default="xpu", help="Device type (xpu, cuda, or cpu)")
    parser.add_argument("--shape", default="static", help="Shape mode (static or dynamic)")
    parser.add_argument("--task-file", default=None,
                        help="Load tasks from a delimited file. Overrides --suite/--dt/--mode/--scenario/--model-only.")
    parser.add_argument("--cores-per-instance", type=int, default=None,
                        help="CPU-only: cores per worker instance. Default = all cores per NUMA node.")
    parser.add_argument("--numactl-args", default="",
                        help="Override NUMACTL_ARGS (semicolon-separated per GPU)")
    parser.add_argument("--dataset-dir", default="",
                        help="Path to ImageNet dataset directory (required for pt2e accuracy)")
    parser.add_argument("--gpu-memory-threshold", type=float, default=None,
                        help="GPU memory utilisation threshold (0.0-1.0) to kill a process. "
                             "Default: 0.8 on Windows, 0.9 on Linux.")
    parser.add_argument("--no-gpu-memory-monitor", action="store_true", default=False,
                        help="Disable GPU memory monitoring (enabled by default).")
    args = parser.parse_args()

    # Apply GPU memory monitoring settings
    if args.no_gpu_memory_monitor:
        config.gpu_memory_monitor_enabled = False
    if args.gpu_memory_threshold is not None:
        config.gpu_memory_threshold = args.gpu_memory_threshold
        config.error_patterns = [
            p for p in config.error_patterns if not p.startswith("Memory>")
        ] + [f"Memory>{config.gpu_memory_threshold}"]

    is_cpu = args.device == "cpu"
    num_gpus = get_num_gpus(required=not is_cpu)

    if not Path("benchmarks/dynamo").is_dir() and not args.task_file:
        requested = parse_string_list(args.suite)
        if not requested or set(requested) != {"pt2e"}:
            sys.exit("ERROR: benchmarks/dynamo directory not found. Are you in the pytorch directory?")

    if args.task_file:
        tasks = load_tasks_from_file(args.task_file)
        if not tasks:
            sys.exit(f"No valid tasks found in {args.task_file}.")
        suites = sorted({t.suite for t in tasks})
        dts = sorted({t.dt for t in tasks})
        modes = sorted({t.mode for t in tasks})
        scenarios = sorted({t.scenario for t in tasks})
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

    banner("Configuration")
    if num_gpus is not None:
        log(f"NUM_GPUS:   {num_gpus}")
    log(f"Suites:     {', '.join(suites)}")
    log(f"Dtypes:     {', '.join(dts)}")
    log(f"Modes:      {', '.join(modes)}")
    log(f"Scenarios:  {', '.join(scenarios)}")
    log(f"Device:     {args.device}")
    log(f"Shape:      {args.shape}")
    if args.cores_per_instance:
        log(f"Cores/inst: {args.cores_per_instance}")
    log(f"Tasks:      {len(tasks)}")

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
    run_all(tasks, workers, args.device, args.shape, args.dataset_dir)


if __name__ == "__main__":
    main()
