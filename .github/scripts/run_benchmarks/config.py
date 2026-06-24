"""Shared configuration: types, constants, and mutable global settings."""

import platform
import threading
from dataclasses import dataclass

IS_WINDOWS = platform.system() == "Windows"


@dataclass
class TestTask:
    """A single benchmark task to execute."""

    suite: str
    dt: str
    mode: str
    scenario: str
    model: str
    quant: str = ""


# Valid parameter sets
VALID_SUITES: set[str] = {"huggingface", "timm_models", "torchbench", "pt2e"}
VALID_DT: set[str] = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16", "int8"}
VALID_MODES: set[str] = {"inference", "training"}
VALID_SCENARIOS: set[str] = {"accuracy", "performance"}

# Per-suite dtype support: pt2e only runs float32/int8; int8 is pt2e-only.
PT2E_DT: set[str] = {"float32", "int8"}
INDUCTOR_DT: set[str] = VALID_DT - {"int8"}

# GPU memory monitoring settings (mutable at runtime via CLI args)
gpu_memory_threshold: float = 0.90 if IS_WINDOWS else 0.95
gpu_memory_monitor_enabled: bool = True

# Maximum wall-clock time (seconds) a single task's process may run before the
# monitor kills it. Mutable at runtime via the --task-timeout CLI arg.
task_timeout_seconds: int = 10800

# Patterns that trigger process kill when detected in output or GPU metrics.
# Text patterns are matched case-insensitively; "Memory>N" triggers GPU memory
# utilisation polling with threshold N.
error_patterns: list[str] = [
    "out of memory",
    "OutOfMemory",
    "UR_RESULT_ERROR",
    f"Memory>{gpu_memory_threshold}",
]

# Thread-safe lock for CSV file writes
csv_lock = threading.Lock()
