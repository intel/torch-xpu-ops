"""Model list loading, task file parsing, and task generation."""

import csv
import math
import os
import re
import sys
from functools import lru_cache, cache
from itertools import product
from pathlib import Path

from .config import (
    INDUCTOR_DT,
    PT2E_DT,
    VALID_DT,
    VALID_MODES,
    VALID_SCENARIOS,
    VALID_SUITES,
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
                models.extend(str(m).strip() for m in models_by_suite[suite][mode] or [] if m is not None)
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

    return _drop_skipped(suite, models)


def _drop_skipped(suite: str, models: list[str]) -> list[str]:
    return list(_drop_skipped_cached(suite, tuple(models)))


@cache
def _drop_skipped_cached(suite: str, models: tuple[str, ...]) -> tuple[str, ...]:
    skipped = get_skipped_models(suite)
    if dropped := sorted(set(models) & skipped):
        log(f"Skipping {suite} models disabled in {suite}.yaml: {', '.join(dropped)}", level="WARN")
    return tuple(model for model in models if model not in skipped)


def _is_numeric(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


_TXT_COLUMNS = ["suite", "dtype", "mode", "model", "result"]
_COLUMN_ALIASES = {"dt": "dtype", "name": "model", "quantization": "quant"}


def _read_task_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV (header-named columns) or TXT (comma/space separated, header optional) rows."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix.lower() == ".csv":
            raw_rows = list(csv.DictReader(f))
        else:
            raw_rows, header = [], None
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = [x for x in re.split(r'\s*,\s*|\s+', line) if x]
                if header is None:
                    if "suite" in (x.lower() for x in fields):
                        header = [x.lower() for x in fields]
                        continue
                    header = _TXT_COLUMNS
                raw_rows.append(dict(zip(header, fields)))

    rows = []
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            if key:
                name = key.strip().lower()
                row[_COLUMN_ALIASES.get(name, name)] = (value or "").strip()
        rows.append(row)
    return rows


def _task_from_row(row: dict[str, str], default_backend: str) -> TestTask | None:
    """Build a task from one normalized row; returns None (with a warning) if it is invalid."""
    suite, dt, mode, model = row.get("suite"), row.get("dtype"), row.get("mode"), row.get("model")
    if not all([suite, dt, mode, model]):
        log(f"Skipping row missing suite/dtype/mode/model: {row}", level="WARN")
        return None
    scenario = row.get("scenario") or ("performance" if _is_numeric(row.get("result", "")) else "accuracy")
    if suite not in VALID_SUITES or dt not in VALID_DT or mode not in VALID_MODES or scenario not in VALID_SCENARIOS:
        log(f"Skipping row with invalid suite/dtype/mode/scenario: {row}", level="WARN")
        return None
    # Quantization only applies to pt2e performance; "0" means none.
    quant = row.get("quant", "") if suite == "pt2e" and scenario == "performance" else ""
    return TestTask(suite, dt, mode, scenario, model, "" if quant == "0" else quant,
                    row.get("backend") or default_backend)


def load_tasks_from_file(path: str, default_backend: str = "default") -> list[TestTask]:
    """Load tasks from a CSV or TXT file.

    CSV: header-named columns (suite, dtype, mode, model, scenario, backend, quant...), so both
    standalone CSVs and output summary CSVs work. Missing scenario is derived from result.
    TXT: suite, dtype, mode, model, result, separated by commas or whitespace; the header row is
    optional. A numeric result means performance, otherwise accuracy.
    The backend comes from the file when present, else from *default_backend*.
    """
    file_path = Path(path)
    if not file_path.is_file():
        sys.exit(f"ERROR: Task file not found: {path}")

    tasks = [t for row in _read_task_rows(file_path) if (t := _task_from_row(row, default_backend))]

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


def _as_list(value, default: list) -> list:
    if value is None:
        return list(default)
    return [value] if isinstance(value, str) else list(value)


def _resolve_group(group: dict | None, cfg: dict) -> dict:
    """Resolve one ``group`` entry; omitted keys inherit the top level, and omitted mode means both modes."""
    group = group or {}
    return {
        "suite": _as_list(group.get("suite"), _as_list(cfg.get("suite"), [])),
        "mode": _as_list(group.get("mode"), _as_list(cfg.get("mode"), ["training", "inference"])),
        "dtype": _as_list(group.get("dtype"), _as_list(cfg.get("dtype"), [])),
        "scenario": _as_list(group.get("scenario"), _as_list(cfg.get("scenario"), [])),
        "backends": _as_list(group.get("backends"), _as_list(cfg.get("backends"), ["default"])),
        "models": group.get("models"),  # None = use the top-level models
    }


def _combos(group: dict, cfg: dict, path: str, dts, modes):
    """Yield (suite, mode, dt, models) for every combination a group selects."""
    top_models = cfg.get("models") or {}
    for suite in group["suite"] or list(top_models if group["models"] is None else group["models"]):
        models_by_mode = (top_models if group["models"] is None else group["models"]).get(suite) or {}
        for mode in group["mode"]:
            if (modes and mode not in modes) or mode not in models_by_mode:
                continue
            if group["models"] is None:
                models = get_model_list(suite, mode, path)
            else:
                models = _drop_skipped(suite, [str(m).strip() for m in models_by_mode[mode] or [] if m is not None])
            for dt in group["dtype"]:
                if not dts or dt in dts:
                    yield suite, mode, dt, models


def load_tasks_from_scope(
    path: str,
    suites: set[str] | None = None,
    dts: set[str] | None = None,
    modes: set[str] | None = None,
    scenarios: set[str] | None = None,
) -> list[TestTask]:
    """Build tasks from a tier config.yaml; combinations not claimed by any group come from the top level."""
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    planned = [
        (group, list(_combos(group, cfg, path, dts, modes)))
        for group in (_resolve_group(g, cfg) for g in (cfg.get("group") or {}).values())
    ]
    covered = {(suite, mode, dt) for _, combos in planned for suite, mode, dt, _ in combos}
    baseline = _resolve_group(None, cfg)
    planned.append((baseline, [c for c in _combos(baseline, cfg, path, dts, modes) if c[:3] not in covered]))

    tasks: list[TestTask] = []
    for group, combos in planned:
        for suite, mode, dt, models in combos:
            if suites and suite not in suites:
                continue
            scenario_list = [s for s in group["scenario"] if not scenarios or s in scenarios]
            for scenario, model in product(scenario_list, dict.fromkeys(models)):
                if suite == "pt2e":
                    # pt2e has no backend axis; generating it per backend would duplicate runs.
                    tasks.extend(_pt2e_tasks(dt, scenario, model))
                else:
                    tasks.extend(
                        TestTask(suite, dt, mode, scenario, model, backend=backend)
                        for backend in group["backends"]
                    )

    seen: set[tuple] = set()
    unique: list[TestTask] = []
    for t in tasks:
        key = (t.suite, t.dt, t.mode, t.scenario, t.model, t.quant, t.backend)
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

    # PT2E tasks are inference-only (dtypes: float32/int8), so skip them for other modes.
    if "pt2e" in suites and "inference" in modes:
        pt2e_dts = [d for d in dts if d in PT2E_DT]
        for scenario in scenarios:
            for dt in pt2e_dts:
                models = sorted(set(get_model_list("pt2e", "inference", model_only, scenario)))
                for model in models:
                    tasks.extend(_pt2e_tasks(dt, scenario, model))
    return tasks
