"""Inductor benchmark suite: command building, result parsing, and CSV collection."""

import csv
import os
import re
from pathlib import Path

from packaging import version

from .config import IS_WINDOWS, TestTask, csv_lock
from .log import log
from .tasks import get_torch_version


def build_command(task: TestTask, device: str, shape: str, output_csv: str) -> str:
    """Build the shell command for an inductor model run."""
    torch_ver = get_torch_version()

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


def check_success(log_path: Path) -> tuple[str, bool]:
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


def _read_last_matching_row(tmp_log_csv: str, device: str) -> tuple[list[str], list[str]]:
    """Read header and last device-matching row from a temp CSV."""
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
        log(f"Error reading temp CSV {tmp_log_csv}: {e}", level="WARN")
        return [], []


def _build_fallback_row(
    device: str, task: TestTask, kill_reason: str | None,
) -> tuple[list[str], list[str]]:
    """Build a fallback row when the benchmark produced no CSV output."""
    fail_status = kill_reason or "core_dump"
    prefix_header = ["dev", "suite", "name", "dtype", "mode", "scenario", "batch_size"]
    prefix_row = [device, task.suite, task.model, task.dt, task.mode, task.scenario, "0"]
    fallbacks = {
        "accuracy": (
            prefix_header + ["accuracy"],
            prefix_row + [fail_status],
        ),
        "performance": (
            prefix_header + ["speedup", "inductor_latency", "eager_latency"],
            prefix_row + ["0", "0", "0"],
        ),
    }
    if task.scenario not in fallbacks:
        raise ValueError(f"Unknown task.scenario: {task.scenario}")
    return fallbacks[task.scenario]


def collect_results(
    log_csv: Path,
    tmp_log_csv: str,
    device: str,
    task: TestTask,
    kill_reason: str | None = None,
) -> None:
    """Collect benchmark results from *tmp_log_csv* and append to *log_csv*."""
    header, row = _read_last_matching_row(tmp_log_csv, device)
    if not row:
        final_header, final_row = _build_fallback_row(device, task, kill_reason)
    else:
        # Benchmark output: dev, name, batch_size, <result_cols...>
        # Reorder to: dev, suite, name, dtype, mode, scenario, batch_size, <result_cols...>
        result_header = header[3:]  # columns after batch_size
        result_row = row[3:]
        final_header = ["dev", "suite", "name", "dtype", "mode", "scenario", "batch_size"] + result_header
        final_row = [device, task.suite, task.model, task.dt, task.mode, task.scenario, row[2]] + result_row

    with csv_lock:
        write_header = not log_csv.exists()
        with open(log_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(final_header)
            writer.writerow(final_row)
