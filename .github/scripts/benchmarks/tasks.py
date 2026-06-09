"""Model list loading, task file parsing, and task generation."""

import os
import re
import subprocess
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

from .config import (
    IS_WINDOWS,
    INDUCTOR_DT,
    PT2E_DT,
    TestTask,
    VALID_DT,
    VALID_MODES,
    VALID_SCENARIOS,
    VALID_SUITES,
)
from .log import log


def parse_string_list(s: str) -> list[str]:
    """Split a comma/space/'-k'-separated string into non-empty trimmed items."""
    if not s:
        return []
    parts = re.split(r'\s*-k\s*|,|\s+', s)
    return [token.strip() for token in parts if token.strip()]


def filter_valid(items: list[str], valid_set: set[str], name: str) -> list[str]:
    """Keep only items in *valid_set*, warning about invalid ones."""
    if invalid := sorted(set(items) - valid_set):
        log(f"Skipping invalid {name}: {', '.join(invalid)}", level="WARN")
    return [i for i in items if i in valid_set]


@lru_cache(maxsize=1)
def get_torch_version() -> str:
    """Return the installed PyTorch version string."""
    pip_cmd = "pip.exe" if IS_WINDOWS else "pip"
    try:
        result = subprocess.run([pip_cmd, "list", "--format=freeze"], capture_output=True, text=True, check=True)
        torch_line = next((l for l in result.stdout.splitlines() if l.startswith("torch==")), None)
        return torch_line.split("==")[1].split("+")[0] if torch_line else "0.0.0"
    except Exception:
        return "0.0.0"


def get_model_list(suite: str, mode: str, model_only: str | None, scenario: str | None = None) -> list[str]:
    """Retrieve model names from a CSV file, string list, or default text file."""
    if model_only is not None:
        if os.path.isfile(model_only):
            import pandas as pd
            df = pd.read_csv(model_only)
            if suite == "pt2e":
                col_header = f"pt2e {scenario}" if scenario else "pt2e accuracy"
            elif suite == "torchbench":
                col_header = f"{suite} {mode}"
            else:
                col_header = suite
            if col_header not in df.columns:
                available = ", ".join(df.columns)
                raise ValueError(f"Column '{col_header}' not found in {model_only}. Available: {available}")
            return df[col_header].dropna().astype(str).str.strip().tolist()

        return parse_string_list(model_only)

    base = suite.replace('_models', '')
    list_file = Path(f"benchmarks/dynamo/{base}_models_list.txt")
    if not list_file.exists():
        raise FileNotFoundError(f"Model list file not found: {list_file}")

    with list_file.open() as f:
        models = [
            parts[0].strip()
            for raw in f
            if (line := raw.split('#', 1)[0].strip())
            and (parts := parse_string_list(line))
            and parts[0].strip()
        ]
    return models


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_tasks_from_file(path: str) -> list[TestTask]:
    """Load tasks from a delimited file (comma, tab, semicolon, or whitespace).

    Supports two formats:
    1. Legacy: suite, dtype, mode, model, result (5 cols)
    2. Output CSV: dev, elapsed, suite, dtype, mode, name, scenario, batch_size, result[, extra] (8+ cols)

    Format is auto-detected from the header row or column count.
    """
    file_path = Path(path)
    if not file_path.is_file():
        sys.exit(f"ERROR: Task file not found: {path}")

    tasks: list[TestTask] = []
    output_format: bool | None = None
    with open(file_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = re.split(r'[,;\t]\s*|\s+', line)
            fields = [f.strip() for f in fields if f.strip()]

            # Auto-detect format from header row
            if output_format is None and fields[0].lower() in ("dev", "suite"):
                output_format = fields[0].lower() == "dev"
                continue  # skip header

            if output_format is None:
                output_format = len(fields) >= 8

            if output_format:
                # Output CSV: dev, elapsed, suite, dtype, mode, name, scenario, batch_size, ...
                if len(fields) < 8:
                    log(f"Skipping line {lineno}: expected 8+ fields, got {len(fields)}: {line!r}", level="WARN")
                    continue
                suite, dt, mode, model, scenario = fields[2], fields[3], fields[4], fields[5], fields[6]
                quant = fields[9] if len(fields) >= 10 else ""
                tasks.append(TestTask(suite, dt, mode, scenario, model, quant))
            else:
                # Legacy: suite, dtype, mode, model, result
                if len(fields) < 5:
                    log(f"Skipping line {lineno}: expected 5+ fields, got {len(fields)}: {line!r}", level="WARN")
                    continue
                suite, dt, mode, model, result = fields[0], fields[1], fields[2], fields[3], fields[4]
                scenario = "performance" if _is_numeric(result) else "accuracy"
                tasks.append(TestTask(suite, dt, mode, scenario, model))

    # Deduplicate while preserving order
    seen: set[tuple] = set()
    unique: list[TestTask] = []
    for t in tasks:
        key = (t.suite, t.dt, t.mode, t.scenario, t.model, t.quant)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def generate_tasks(
    suites: list[str],
    dts: list[str],
    modes: list[str],
    scenarios: list[str],
    model_only: str | None,
) -> list[TestTask]:
    """Generate the full task list from parameter combinations."""
    non_pt2e_suites = [s for s in suites if s != "pt2e"]
    inductor_dts = [d for d in dts if d in INDUCTOR_DT]
    tasks = [
        TestTask(suite, dt, mode, scenario, model)
        for suite, mode, dt, scenario in product(non_pt2e_suites, modes, inductor_dts, scenarios)
        for model in sorted(set(get_model_list(suite, mode, model_only)))
    ]

    # PT2E tasks (inference only, dtypes: float32/int8)
    if "pt2e" in suites:
        pt2e_dts = [d for d in dts if d in PT2E_DT]
        for scenario in scenarios:
            for dt in pt2e_dts:
                models = sorted(set(get_model_list("pt2e", "inference", model_only, scenario)))
                for model in models:
                    if dt == "int8" and scenario == "performance":
                        tasks.append(TestTask("pt2e", dt, "inference", scenario, model, "symm"))
                        tasks.append(TestTask("pt2e", dt, "inference", scenario, model, "asymm"))
                    else:
                        tasks.append(TestTask("pt2e", dt, "inference", scenario, model))
    return tasks
