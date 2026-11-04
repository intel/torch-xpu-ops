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

from .config import BACKEND_FLAGS, TestTask, csv_lock
from .log import log
from .tasks import get_torch_version

# Common leading columns shared by every result CSV; "status" is "pass" or the failure reason.
CSV_PREFIX = ["dev", "backend", "suite", "dtype", "mode", "name", "scenario", "batch_size", "elapsed", "status"]

# Fixed result columns for suites whose output shape is known up front.
INDUCTOR_PERFORMANCE_COLS = ["speedup", "abs_latency", "eager_latency"]
PT2E_ACCURACY_COLS = ["top1", "top5"]
PT2E_PERFORMANCE_COLS = ["throughput", "quantization"]


def _prefix_row(device: str, task: TestTask, status: str, batch_size: str = "0", elapsed: float = 0.0) -> list[str]:
    """Build the leading CSV cells common to every suite."""
    return [device, task.backend, task.suite, task.dt, task.mode, task.model, task.scenario,
            batch_size, f"{elapsed:.3f}", status]


# Columns identifying a task; a re-run replaces the existing row with the same key.
_ROW_KEY = ["dev", "backend", "suite", "dtype", "mode", "name", "scenario", "quantization"]


def _append_row(log_csv: Path, header: list[str], row: list[str]) -> None:
    """Add *row* to *log_csv* so the header and every row share the same columns.

    The file header is the union of all headers seen (i.e. the full success
    shape); when a new column appears the file is rewritten and earlier rows
    (e.g. a failed first row) are padded with ``0``. A row for a task already
    in the file replaces the old one in place.
    """
    def key(rec: dict[str, str]) -> tuple:
        return tuple(rec.get(c) or "0" for c in _ROW_KEY)

    with csv_lock:
        records: list[dict[str, str]] = []
        merged: list[str] = []
        if log_csv.exists():
            with open(log_csv, newline="", encoding="utf-8") as f:
                lines = [r for r in csv.reader(f) if r]
            if lines:
                merged = list(lines[0])
                records = [dict(zip(merged, r)) for r in lines[1:]]
        merged += [c for c in header if c not in merged]
        new = dict(zip(header, row))
        idx = next((i for i, rec in enumerate(records) if key(rec) == key(new)), None)
        if idx is None:
            records.append(new)
        else:
            records[idx] = new
        # Write to a sibling temp file and swap in so a crash never truncates results.
        tmp_path = log_csv.with_name(log_csv.name + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(merged)
            for rec in records:
                writer.writerow([rec.get(c) or "0" for c in merged])
        os.replace(tmp_path, log_csv)


class BenchmarkSuite(ABC):
    """Common interface implemented by each benchmark suite."""

    #: When True, the runner allocates a temp CSV and passes it as *output_csv*.
    uses_temp_csv: bool = False

    def prepare(self, task: TestTask) -> None:
        """Optional hook to clean stale state before a run."""
        return None

    def env_overrides(self, task: TestTask) -> dict[str, str]:
        """Optional per-task environment variables for the subprocess."""
        return {}

    @abstractmethod
    def build_command(
        self, task: TestTask, device: str, backend: str,
        dataset_dir: str, output_csv: str | None,
    ) -> str:
        """Return the shell command that runs *task*."""

    @abstractmethod
    def collect_results(
        self, log_csv: Path, log_file: Path, tmp_csv: str | None,
        device: str, task: TestTask, kill_reason: str | None, elapsed: float,
    ) -> tuple[str, bool]:
        """Parse the run output, append one row to *log_csv*, and return
        ``(test_result, success)``."""


class InductorSuite(BenchmarkSuite):
    """TorchDynamo inductor suites: huggingface, timm_models, torchbench."""

    uses_temp_csv = True

    def build_command(self, task, device, backend, dataset_dir, output_csv):
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

        # Dynamic shapes are selected via the "dynamic" backend, not a separate flag.
        backend_flags = BACKEND_FLAGS.get(backend, ["--backend=inductor"])

        parts = [
            f"python benchmarks/dynamo/{task.suite}.py",
            f"-d {device}",
            f"--{task.scenario}",
            f"--{real_dt}",
            dt_extra,
            mode_flag,
            f"--only {task.model}" if task.model else "",
            " ".join(backend_flags),
            "--cold-start-latency",
            "-n10",
            "--timeout=10800",
            "--disable-cudagraphs",
            f"--output={output_csv}",
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _last_log_token(log_file) -> str:
        """Return the last whitespace-separated token of the run log, or ''."""
        if not log_file.exists():
            return ""
        last = ""
        with open(log_file, errors="ignore") as f:
            for line in f:
                if stripped := line.strip():
                    last = stripped
        return last.split()[-1] if last else ""

    def collect_results(self, log_csv, log_file, tmp_csv, device, task, kill_reason, elapsed):
        header, row = self._read_last_matching_row(tmp_csv, device)

        def _col(name):
            try:
                return row[header.index(name)] or "0"
            except (ValueError, IndexError):
                return "0"

        batch_size = _col("batch_size")
        log_result = self._last_log_token(log_file)
        # A failure keyword printed last by the benchmark (e.g. fail_to_run) beats a generic reason.
        log_reason = log_result if re.fullmatch(r"\w*(fail|error)\w*|timeout|oom", log_result, re.I) else ""
        failure = kill_reason or log_reason or ("fail_to_run" if row else "core_dump")
        # The CSV is the single source of truth: a measured result wins over a late kill.
        if task.scenario == "performance":
            speedup, abs_latency = _col("speedup"), _col("abs_latency")
            try:
                ok = float(speedup) > 0 and float(abs_latency) > 0
            except ValueError:
                ok = False
            if ok:
                eager = f"{float(abs_latency) * float(speedup):.6f}"
                values, status, result = [speedup, abs_latency, eager], "pass", f"{speedup}x"
            else:
                values, status, result = ["0", "0", "0"], failure, failure
            final_header = CSV_PREFIX + INDUCTOR_PERFORMANCE_COLS
        else:
            # Benchmark CSV row: dev, name, batch_size, <result_cols...>
            if row:
                result_cols = header[3:]
                values = [_col(c) for c in result_cols]
                status = _col("accuracy")
            else:
                result_cols = ["accuracy"]
                status = failure
                values = [status]
            final_header = CSV_PREFIX + result_cols
            ok = status in ("pass", "pass_due_to_skip")
            result = status
        _append_row(log_csv, final_header, _prefix_row(device, task, status, batch_size, elapsed) + values)
        return result, ok

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


class PT2ESuite(BenchmarkSuite):
    """Post-training-quantization (pt2e) accuracy and performance benchmarks."""

    #: Sentinel stored on a task to remember its private performance output dir.
    _OUTPUT_ATTR = "_pt2e_output_dir"

    def env_overrides(self, task):
        return {"XPU_QUANT_CONFIG": task.quant.upper()} if task.quant else {}

    def build_command(self, task, device, backend, dataset_dir, output_csv):
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
            # Each run gets its own scratch dir so parallel workers never clobber
            # one another. run_benchmark.py writes the aggregated metrics JSON to
            # the *parent* of --output, so point --output at a nested subdir to
            # keep that file inside our scratch dir (where _parse_throughput looks).
            base_dir = tempfile.mkdtemp(prefix="pt2e-perf-")
            setattr(task, self._OUTPUT_ATTR, base_dir)
            output_dir = os.path.join(base_dir, "run")
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
            ok = acc1 != "failed"
            status = "pass" if ok else kill_reason or "failed"
            header = CSV_PREFIX + PT2E_ACCURACY_COLS
            row = _prefix_row(device, task, status, elapsed=elapsed) + ([acc1, acc5] if ok else ["0", "0"])
            result = (f"acc1={acc1}", True) if ok else (status, False)
        else:  # performance
            output_dir = getattr(task, self._OUTPUT_ATTR, None)
            try:
                throughput = "failed" if kill_reason else self._parse_throughput(output_dir)
            finally:
                if output_dir:
                    shutil.rmtree(
                        output_dir,
                        onerror=lambda func, path, exc: log(
                            f"Failed to remove {path}: {exc[1]}", level="WARN"),
                    )
            ok = throughput != "failed"
            status = "pass" if ok else kill_reason or "failed"
            header = CSV_PREFIX + PT2E_PERFORMANCE_COLS
            row = _prefix_row(device, task, status, elapsed=elapsed) + [
                throughput if ok else "0", task.quant or "0"]
            result = (throughput, True) if ok else (status, False)
        _append_row(log_csv, header, row)
        return result

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
        """Extract the ``*_throughput`` metric from the run's scratch dir.

        ``run_benchmark.py`` aggregates results into a ``metrics-*.json`` file
        whose keys look like ``<model>-eval_throughput``; the value is the last
        whitespace-separated token on that line.
        """
        if not output_dir:
            return "failed"
        try:
            for root, _dirs, files in os.walk(output_dir):
                for fname in files:
                    content = (Path(root) / fname).read_text(encoding="utf-8", errors="ignore")
                    for line in content.splitlines():
                        if "eval_throughput" in line:
                            return line.strip().split()[-1].rstrip(",")
        except Exception:
            pass
        return "failed"


# Registry: pt2e is special-cased; every other suite is inductor-based.
_INDUCTOR_SUITE = InductorSuite()
_SUITES: dict[str, BenchmarkSuite] = {"pt2e": PT2ESuite()}


def get_suite(task: TestTask) -> BenchmarkSuite:
    """Return the :class:`BenchmarkSuite` handler for *task*."""
    return _SUITES.get(task.suite, _INDUCTOR_SUITE)
