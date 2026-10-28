"""Model list loading, task file parsing, and task generation."""

import os
import re
import sys
from functools import lru_cache, cache
from itertools import product
from pathlib import Path

from .config import (
    INDUCTOR_DT,
    PT2E_DT,
    TestTask,
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
    """Return the PyTorch version visible to the running interpreter (local suffix stripped)."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("torch").split("+")[0]
    except PackageNotFoundError:
        return "0.0.0"


@cache
def get_skipped_models(suite: str) -> frozenset[str]:
    """Return the models that ``benchmarks/dynamo/<suite>.yaml`` skips on XPU."""
    yaml_file = Path(f"benchmarks/dynamo/{suite}.yaml")
    if not yaml_file.exists():
        return frozenset()

    import yaml

    with yaml_file.open(encoding="utf-8") as f:
        skip = (yaml.safe_load(f) or {}).get("skip") or {}

    def flatten(items):
        # Anchors such as *DETECTRON2_MODELS expand to a nested list, "~" to None.
        for item in items or []:
            if isinstance(item, list):
                yield from flatten(item)
            elif item is not None:
                yield str(item)

    return frozenset(flatten(skip.get("all"))) | frozenset(flatten((skip.get("device") or {}).get("xpu")))


@cache
def get_model_list(suite: str, mode: str, model_only: str | None, scenario: str | None = None) -> list[str]:
    """Retrieve model names from a tier config.yaml, CSV file, string list, or default text file.

    Cached: callers must treat the returned list as read-only.
    """
    if model_only is not None:
        yaml_files = [p.strip() for p in model_only.split(",")]
        if all(p.endswith((".yaml", ".yml")) and os.path.isfile(p) for p in yaml_files):
            import yaml

            models = []
            for path in yaml_files:
                with open(path, encoding="utf-8") as f:
                    models_by_suite = (yaml.safe_load(f) or {}).get("models") or {}
                if mode not in models_by_suite.get(suite, {}):
                    raise ValueError(f"No '{suite} {mode}' models in {path}")
                models.extend(str(m).strip() for m in models_by_suite[suite][mode])
            models = list(dict.fromkeys(models))
        elif os.path.isfile(model_only):
            import pandas as pd
            df = pd.read_csv(model_only)
            if suite == "pt2e":
                col_header = f"pt2e {scenario}" if scenario else "pt2e accuracy"
            else:
                col_header = f"{suite} {mode}"
            if col_header not in df.columns:
                available = ", ".join(df.columns)
                raise ValueError(f"Column '{col_header}' not found in {model_only}. Available: {available}")
            models = df[col_header].dropna().astype(str).str.strip().tolist()
        else:
            models = parse_string_list(model_only)
    else:
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

    skipped = get_skipped_models(suite)
    if dropped := sorted(set(models) & skipped):
        log(f"Skipping {suite} models disabled in {suite}.yaml: {', '.join(dropped)}", level="WARN")
    return [model for model in models if model not in skipped]


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
    2. Output CSV: dev, backend, suite, dtype, mode, name, scenario, batch_size, elapsed, result[, extra] (9+ cols)

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
                # Output CSV: dev, backend, suite, dtype, mode, name, scenario, batch_size, elapsed, ...
                if len(fields) < 8:
                    log(f"Skipping line {lineno}: expected 8+ fields, got {len(fields)}: {line!r}", level="WARN")
                    continue
                suite, dt, mode, model, scenario = fields[2], fields[3], fields[4], fields[5], fields[6]
                backend = fields[1] if len(fields) >= 2 else "default"
                quant = fields[10] if len(fields) >= 11 else ""
                tasks.append(TestTask(suite, dt, mode, scenario, model, quant, backend))
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
        key = (t.suite, t.dt, t.mode, t.scenario, t.model, t.quant, t.backend)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _pt2e_tasks(dt: str, scenario: str, model: str) -> list[TestTask]:
    """Build pt2e inference tasks; int8 performance runs both symmetric and asymmetric quantization."""
    if dt == "int8" and scenario == "performance":
        return [TestTask("pt2e", dt, "inference", scenario, model, quant) for quant in ("symm", "asymm")]
    return [TestTask("pt2e", dt, "inference", scenario, model)]


def load_tasks_from_scope(
    path: str,
    suites: set[str] | None = None,
    dts: set[str] | None = None,
    modes: set[str] | None = None,
    scenarios: set[str] | None = None,
) -> list[TestTask]:
    """Build tasks from a tier config.yaml using its own backends, precisions, scenarios and models."""
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    backends = cfg.get("backends") or ["default"]
    precisions = cfg.get("precision") or []
    scenario_list = [s for s in cfg.get("scenario") or [] if not scenarios or s in scenarios]

    suite_list = cfg.get("suite") or []

    tasks: list[TestTask] = []
    for suite, by_mode in (cfg.get("models") or {}).items():
        if suite_list and suite not in suite_list:
            continue
        if suites and suite not in suites:
            continue
        for mode in (by_mode or {}):
            if modes and mode not in modes:
                continue
            models = dict.fromkeys(get_model_list(suite, mode, path))
            precision_list = [
                p["dtype"] for p in precisions
                if p.get("mode", mode) == mode and (not dts or p["dtype"] in dts)
            ]
            for scenario, dt, model in product(scenario_list, precision_list, models):
                if suite == "pt2e":
                    # pt2e has no backend axis; generating it per backend would duplicate runs.
                    tasks.extend(_pt2e_tasks(dt, scenario, model))
                else:
                    tasks.extend(TestTask(suite, dt, mode, scenario, model, backend=backend) for backend in backends)
    return tasks


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

    # PT2E tasks are inference-only (dtypes: float32/int8), so skip them for other modes.
    if "pt2e" in suites and "inference" in modes:
        pt2e_dts = [d for d in dts if d in PT2E_DT]
        for scenario in scenarios:
            for dt in pt2e_dts:
                models = sorted(set(get_model_list("pt2e", "inference", model_only, scenario)))
                for model in models:
                    tasks.extend(_pt2e_tasks(dt, scenario, model))
    return tasks
