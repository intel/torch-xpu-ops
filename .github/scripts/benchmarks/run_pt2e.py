"""PT2E benchmark suite: command building, result parsing, and CSV collection."""

import csv
import os
import re
import shutil
from pathlib import Path

from .config import TestTask, csv_lock
from .log import log


def build_command(task: TestTask, device: str, dataset_dir: str) -> str:
    """Build the shell command for a pt2e model run."""
    if task.scenario == "accuracy":
        parts = [
            "python pt2e-accuracy/scripts/modelbench/quant/inductor_quant_acc.py",
            f"--device {device}",
            f"--dataset_dir {dataset_dir}",
            f"--model_list {task.model}",
        ]
        if task.dt == "float32":
            parts.append("--is_fp32")
    else:  # performance
        parts = [
            "python pt2e-performance/run_benchmark.py",
            device,
            "--test eval",
            "--channels-last",
            "--metrics throughputs",
            "--torchdynamo inductor",
            f"-m {task.model}",
        ]
        if task.quant:
            parts.append("--quantization pt2e")
    return " ".join(parts)


def prepare_run(task: TestTask, device: str, dataset_dir: str) -> str:
    """Clean previous results and return the command string."""
    if task.scenario == "performance":
        shutil.rmtree("pt2e-performance/.userbenchmark", ignore_errors=True)
    return build_command(task, device, dataset_dir)


def _parse_accuracy_log(log_file: Path) -> tuple[str, str]:
    """Parse Acc@1 and Acc@5 from a pt2e accuracy log."""
    acc1, acc5 = "failed", "failed"
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if re.search(r'Acc.1.*Acc.5', line, re.IGNORECASE):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        acc1 = parts[-3]
                        acc5 = parts[-1]
    return acc1, acc5


def _parse_performance_throughput() -> str:
    """Extract eval_throughput from .userbenchmark output files."""
    userbenchmark_dir = Path("pt2e-performance/.userbenchmark")
    try:
        for root, _dirs, files in os.walk(userbenchmark_dir):
            for fname in files:
                filepath = Path(root) / fname
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                for content_line in content.splitlines():
                    if "eval_throughput" in content_line:
                        return content_line.strip().split()[-1]
    except Exception:
        pass
    return "failed"


def collect_results(
    log_csv: Path,
    log_file: Path,
    device: str,
    task: TestTask,
    kill_reason: str | None = None,
) -> None:
    """Collect pt2e benchmark results and append to log_csv."""
    if task.scenario == "accuracy":
        if kill_reason:
            acc1, acc5 = "failed", "failed"
        else:
            acc1, acc5 = _parse_accuracy_log(log_file)
        header = ["dev", "suite", "name", "dtype", "mode", "scenario", "batch_size", "top1", "top5"]
        row = [device, task.suite, task.model, task.dt, task.mode, task.scenario, "0", acc1, acc5]
    else:  # performance
        if kill_reason:
            throughput = "failed"
        else:
            throughput = _parse_performance_throughput()
        header = ["dev", "suite", "name", "dtype", "mode", "scenario", "batch_size", "throughput", "quantization"]
        row = [device, task.suite, task.model, task.dt, task.mode, task.scenario, "0", throughput, task.quant]

    with csv_lock:
        write_header = not log_csv.exists()
        with open(log_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


def check_success(log_file: Path, task: TestTask) -> tuple[str, bool]:
    """Determine pass/fail from parsed pt2e results."""
    if task.scenario == "accuracy":
        acc1, acc5 = _parse_accuracy_log(log_file)
        if acc1 != "failed":
            return f"acc1={acc1}", True
        return "failed", False
    else:
        throughput = _parse_performance_throughput()
        if throughput != "failed":
            return f"{throughput}", True
        return "failed", False
