"""Benchmark suites: command building, result parsing, and CSV collection.

Every suite — the TorchDynamo inductor suites (huggingface, timm_models,
torchbench) and the post-training-quantization ``pt2e`` suite — implements the
common :class:`BenchmarkSuite` interface, so the runner can drive them
uniformly without branching on the suite name.
"""

import csv
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from packaging import version

from .config import TestTask, csv_lock
from .log import log
from .tasks import get_torch_version

# Common leading columns shared by every result CSV.
CSV_PREFIX = ["dev", "elapsed", "suite", "dtype", "mode", "name", "scenario", "batch_size"]


def _prefix_row(device: str, task: TestTask, batch_size: str = "0", elapsed: float = 0.0) -> list[str]:
    """Build the leading CSV cells common to every suite."""
    return [device, f"{elapsed:.3f}", task.suite, task.dt, task.mode, task.model, task.scenario, batch_size]


def _append_row(log_csv: Path, header: list[str], row: list[str]) -> None:
    """Append *row* to *log_csv*, writing *header* first when the file is new."""
    with csv_lock:
        write_header = not log_csv.exists()
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


class BenchmarkSuite(ABC):
    """Common interface implemented by each benchmark suite."""

    #: When True, the runner allocates a temp CSV and passes it as *output_csv*.
    uses_temp_csv: bool = False

    def prepare(self, task: TestTask) -> None:
        """Optional hook to clean stale state before a run."""

    def env_overrides(self, task: TestTask) -> dict[str, str]:
        """Optional per-task environment variables for the subprocess."""
        return {}

    @abstractmethod
    def build_command(
        self, task: TestTask, device: str, shape: str,
        dataset_dir: str, output_csv: str | None,
    ) -> str:
        """Return the shell command that runs *task*."""

    @abstractmethod
    def collect_results(
        self, log_csv: Path, log_file: Path, tmp_csv: str | None,
        device: str, task: TestTask, kill_reason: str | None, elapsed: float,
    ) -> None:
        """Parse the run output and append one row to *log_csv*."""

    @abstractmethod
    def check_success(self, log_file: Path, task: TestTask) -> tuple[str, bool]:
        """Return ``(test_result, success)`` for a finished run."""


class InductorSuite(BenchmarkSuite):
    """TorchDynamo inductor suites: huggingface, timm_models, torchbench."""

    uses_temp_csv = True

    def build_command(self, task, device, shape, dataset_dir, output_csv):
        torch_ver = get_torch_version()
        if task.mode == "training":
            mode_flag = "--training"
        elif version.parse(torch_ver) >= version.parse("2.0.2"):
            mode_flag = "--inference"
        else:
            mode_flag = ""

        # amp_bf16/amp_fp16 → --amp --amp-dtype ...
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

    def check_success(self, log_file, task):
        """Pass when the last log token is 'pass', 'pass_due_to_skip', or a speedup."""
        if not log_file.exists():
            return "None", False
        last = ""
        with open(log_file) as f:
            for line in f:
                if stripped := line.strip():
                    last = stripped
        if not last:
            return "None", False
        test_result = last.split()[-1]
        success = test_result in ("pass", "pass_due_to_skip") or bool(re.fullmatch(r"[0-9.]+x", test_result))
        return test_result, success

    def collect_results(self, log_csv, log_file, tmp_csv, device, task, kill_reason, elapsed):
        header, row = self._read_last_matching_row(tmp_csv, device)
        if not row:
            final_header, final_row = self._fallback_row(device, task, kill_reason, elapsed)
        else:
            # Benchmark CSV row: dev, name, batch_size, <result_cols...>
            # Reorder to the common prefix plus the result columns.
            final_header = CSV_PREFIX + header[3:]
            final_row = _prefix_row(device, task, row[2], elapsed) + row[3:]
        _append_row(log_csv, final_header, final_row)

    @staticmethod
    def _read_last_matching_row(tmp_csv, device):
        """Read header and last device-matching row from a temp CSV."""
        if not tmp_csv or not os.path.isfile(tmp_csv):
            return [], []
        try:
            with open(tmp_csv, newline="", encoding="utf-8") as f:
                rows = [row for row in csv.reader(f) if row]
            if not rows:
                return [], []
            matching = [r for r in rows[1:] if r and r[0] == device]
            return rows[0], (matching[-1] if matching else [])
        except Exception as e:
            log(f"Error reading temp CSV {tmp_csv}: {e}", level="WARN")
            return [], []

    @staticmethod
    def _fallback_row(device, task, kill_reason, elapsed=0.0):
        """Build a placeholder row when the benchmark produced no CSV output."""
        fail_status = kill_reason or "core_dump"
        fallbacks = {
            "accuracy": (
                CSV_PREFIX + ["accuracy"],
                _prefix_row(device, task, elapsed=elapsed) + [fail_status],
            ),
            "performance": (
                CSV_PREFIX + ["speedup", "inductor_latency", "eager_latency"],
                _prefix_row(device, task, elapsed=elapsed) + ["0", "0", "0"],
            ),
        }
        if task.scenario not in fallbacks:
            raise ValueError(f"Unknown task.scenario: {task.scenario}")
        return fallbacks[task.scenario]


class PT2ESuite(BenchmarkSuite):
    """Post-training-quantization (pt2e) accuracy and performance benchmarks."""

    #: Sentinel stored on a task to remember its private performance output dir.
    _OUTPUT_ATTR = "_pt2e_output_dir"

    def env_overrides(self, task):
        return {"XPU_QUANT_CONFIG": task.quant.upper()} if task.quant else {}

    def build_command(self, task, device, shape, dataset_dir, output_csv):
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
            # Each run writes metrics to its own directory so parallel workers
            # never clobber one another's results.
            output_dir = tempfile.mkdtemp(prefix="pt2e-perf-")
            setattr(task, self._OUTPUT_ATTR, output_dir)
            parts = [
                "python pt2e-performance/run_benchmark.py",
                device,
                "--test eval",
                "--channels-last",
                "--metrics throughputs",
                "--torchdynamo inductor",
                f"--output {output_dir}",
                f"-m {task.model}",
            ]
            if task.quant:
                parts.append("--quantization pt2e")
        return " ".join(parts)

    def collect_results(self, log_csv, log_file, tmp_csv, device, task, kill_reason, elapsed):
        if task.scenario == "accuracy":
            acc1, acc5 = ("failed", "failed") if kill_reason else self._parse_accuracy(log_file)
            header = CSV_PREFIX + ["top1", "top5"]
            row = _prefix_row(device, task, elapsed=elapsed) + [acc1, acc5]
        else:  # performance
            output_dir = getattr(task, self._OUTPUT_ATTR, None)
            try:
                throughput = "failed" if kill_reason else self._parse_throughput(output_dir)
            finally:
                if output_dir:
                    shutil.rmtree(output_dir, ignore_errors=True)
            header = CSV_PREFIX + ["throughput", "quantization"]
            row = _prefix_row(device, task, elapsed=elapsed) + [throughput, task.quant]
        _append_row(log_csv, header, row)

    def check_success(self, log_file, task):
        if task.scenario == "accuracy":
            acc1, _ = self._parse_accuracy(log_file)
            return (f"acc1={acc1}", True) if acc1 != "failed" else ("failed", False)
        throughput = self._parse_throughput(getattr(task, self._OUTPUT_ATTR, None))
        return (throughput, True) if throughput != "failed" else ("failed", False)

    @staticmethod
    def _parse_accuracy(log_file: Path) -> tuple[str, str]:
        """Parse Acc@1 and Acc@5 from a pt2e accuracy log."""
        acc1, acc5 = "failed", "failed"
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    if re.search(r"Acc.1.*Acc.5", line, re.IGNORECASE):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            acc1, acc5 = parts[-3], parts[-1]
        return acc1, acc5

    @staticmethod
    def _parse_throughput(output_dir: str | None) -> str:
        """Extract eval_throughput from the run's ``--output`` directory."""
        if not output_dir:
            return "failed"
        try:
            for root, _dirs, files in os.walk(output_dir):
                for fname in files:
                    content = (Path(root) / fname).read_text(encoding="utf-8", errors="ignore")
                    for line in content.splitlines():
                        if "eval_throughput" in line:
                            return line.strip().split()[-1]
        except Exception:
            pass
        return "failed"


# Registry: pt2e is special-cased; every other suite is inductor-based.
_INDUCTOR_SUITE = InductorSuite()
_SUITES: dict[str, BenchmarkSuite] = {"pt2e": PT2ESuite()}


def get_suite(task: TestTask) -> BenchmarkSuite:
    """Return the :class:`BenchmarkSuite` handler for *task*."""
    return _SUITES.get(task.suite, _INDUCTOR_SUITE)
