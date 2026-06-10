"""File discovery, filename parsing, and CSV loading for benchmark comparison."""

import logging
import os
from glob import glob

import numpy as np
import pandas as pd

from .constants import (
    ACC_CSV_COLS,
    KNOWN_DATA_TYPES,
    KNOWN_MODES,
    KNOWN_SUITES,
    PERF_CSV_COLS,
)

log = logging.getLogger(__name__)


# ── File Discovery ────────────────────────────────────────────────────
def find_result_files(root_dir: str) -> list[str]:
    """Recursively find *performance.csv and *accuracy.csv files."""
    files = []
    for suffix in ("*performance.csv", "*accuracy.csv"):
        files.extend(glob(os.path.join(root_dir, "**", suffix), recursive=True))
    return files


# ── Filename Parsing ──────────────────────────────────────────────────
def parse_filename(filename: str) -> tuple[str, str, str, str]:
    """
    Extract (suite, data_type, mode, result_type) from a CSV filename.

    Supported patterns:
        inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
        inductor-<suite>-<data_type>-<mode>-<device>-<result_type>.csv
        inductor-results-<suite>-<data_type>-<mode>-<device>-<result_type>.csv

    Raises ValueError on unrecognised filenames.
    """
    if not filename.endswith(".csv"):
        raise ValueError(f"Not a CSV file: {filename}")

    # ── Pattern 1: dash-separated ──
    # Parse from the end so both the runner output
    # (inductor-<suite>-<dt>-<mode>-<device>-<result>) and the legacy
    # (inductor-results-<suite>-<dt>-<mode>-<device>-<result>) layouts work.
    if "inductor-" in filename:
        parts = filename[:-4].split("-")
        if len(parts) < 6:
            raise ValueError(f"Too few dash-separated parts in: {filename}")
        result_type = parts[-1]
        if result_type not in ("accuracy", "performance"):
            raise ValueError(f"Unknown result type '{result_type}' in {filename}")
        device = parts[-2]
        if device not in ("xpu", "cuda", "cpu"):
            raise ValueError(f"Unknown device '{device}' in {filename}")
        mode = parts[-3]
        data_type = parts[-4]
        suite = parts[-5]
        return suite, data_type, mode, result_type

    # ── Pattern 2: underscore-separated ──
    base = filename[:-4]
    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor': {filename}")
    rest = base[len("inductor_"):]

    def _match_prefix(rest: str, candidates: set[str]) -> tuple[str | None, str]:
        """Match longest known prefix followed by '_', return (match, remainder)."""
        for c in sorted(candidates, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c, rest[len(c) + 1:]
        return None, rest

    suite, rest = _match_prefix(rest, KNOWN_SUITES)
    if suite is None:
        raise ValueError(f"Unknown suite in {filename}")

    data_type, rest = _match_prefix(rest, KNOWN_DATA_TYPES)
    data_type = data_type or ""

    mode, rest = _match_prefix(rest, KNOWN_MODES)
    if mode is None:
        raise ValueError(f"Could not find mode (inference/training) in {filename}")

    # rest should be xpu_[optional_extra_]<result_type>
    parts = rest.split("_")
    if len(parts) < 2 or parts[0] != "xpu":
        raise ValueError(f"Expected '_xpu_<result_type>' after mode in {filename}")
    result_type = parts[-1]
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Unknown result type '{result_type}' in {filename}")

    return suite, data_type, mode, result_type


# ── Record Deduplication ──────────────────────────────────────────────
def _best_accuracy_record(records: list[dict]) -> dict:
    """Pick best accuracy record: prefer 'pass' over 'fail' over others."""
    for predicate in (
        lambda r: "pass" in str(r.get("accuracy", "")),
        lambda r: "fail" in str(r.get("accuracy", "")),
    ):
        matches = [r for r in records if predicate(r)]
        if matches:
            return matches[0]
    return records[0]


def _best_performance_record(records: list[dict]) -> dict:
    """
    Pick best performance record.
    Priority: both positive > one positive > zero > NaN/negative.
    Among equally-ranked records, prefer smaller inductor latency.
    """
    def _sort_key(r):
        ind = r.get("inductor")
        eag = r.get("eager")
        ind_ok = pd.notna(ind) and ind > 0
        eag_ok = pd.notna(eag) and eag > 0
        if ind_ok and eag_ok:
            return (0, ind, eag)
        if ind_ok or eag_ok:
            return (1, ind if ind_ok else float("inf"),
                    eag if eag_ok else float("inf"))
        if (pd.notna(ind) and ind == 0) or (pd.notna(eag) and eag == 0):
            return (2, 0, 0)
        return (3, float("inf"), float("inf"))

    return min(records, key=_sort_key)


# ── CSV Loading ───────────────────────────────────────────────────────
def _read_csv_safe(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV with graceful fallback for older pandas versions."""
    try:
        return pd.read_csv(
            path, usecols=usecols, on_bad_lines="skip",
            engine="c", encoding="utf-8",
        )
    except TypeError:
        # Older pandas without on_bad_lines
        return pd.read_csv(
            path, usecols=usecols, error_bad_lines=False,
            warn_bad_lines=True, engine="python", encoding="utf-8",
        )


def load_results(file_list: list[str], result_type: str) -> list[dict]:
    """
    Load CSV files of *result_type* ('accuracy' or 'performance'),
    deduplicate by (suite, data_type, mode, model), and return a list
    of record dicts.
    """
    usecols = ACC_CSV_COLS if result_type == "accuracy" else PERF_CSV_COLS
    raw: dict[tuple, list[dict]] = {}

    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = parse_filename(os.path.basename(fpath))
        except ValueError as exc:
            log.debug("Skipping %s: %s", fpath, exc)
            continue
        if res_type != result_type:
            continue

        is_pt2e = suite == "pt2e"
        try:
            if is_pt2e:
                df = _read_csv_safe(fpath)
            else:
                df = _read_csv_safe(fpath, usecols)
        except Exception as exc:
            log.warning("Failed to read %s: %s", fpath, exc)
            continue

        for _, row in df.iterrows():
            dev = row.get("dev")
            if pd.isna(dev) or str(dev).strip() not in ("cpu", "xpu", "cuda"):
                continue

            # For PT2E int8 performance, append quantization type to data_type
            rec_data_type = data_type
            if is_pt2e and data_type == "int8" and result_type == "performance":
                quant = str(row.get("quantization", "")).strip()
                if quant:
                    rec_data_type = f"int8_{quant}"

            key = (suite, rec_data_type, mode, row["name"])
            rec = {
                "suite": suite, "data_type": rec_data_type,
                "mode": mode, "model": row["name"],
                "batch_size": row["batch_size"],
            }

            if result_type == "accuracy":
                if is_pt2e:
                    # PT2E accuracy: preserve top1 as raw numeric value
                    top1 = pd.to_numeric(row.get("top1"), errors="coerce")
                    rec["top1"] = top1 if pd.notna(top1) else np.nan
                    rec["accuracy"] = f"pass_top1={top1:.3f}" if pd.notna(top1) else "fail_to_run"
                else:
                    rec["accuracy"] = row["accuracy"]
            else:
                if is_pt2e:
                    # PT2E performance: preserve throughput and quantization
                    throughput = pd.to_numeric(row.get("throughput"), errors="coerce")
                    quant = str(row.get("quantization", "")).strip()
                    rec["throughput"] = throughput if pd.notna(throughput) and throughput > 0 else np.nan
                    rec["quantization"] = quant
                    if pd.isna(throughput) or throughput <= 0:
                        rec.update(eager=np.nan, inductor=np.nan, speedup=np.nan)
                    else:
                        rec.update(eager=np.nan, inductor=throughput, speedup=np.nan)
                else:
                    speedup = pd.to_numeric(row.get("speedup"), errors="coerce")
                    abs_lat = pd.to_numeric(row.get("abs_latency"), errors="coerce")
                    if pd.isna(speedup) or pd.isna(abs_lat):
                        log.debug(
                            "Missing speedup/abs_latency for %s in %s",
                            row.get("name"), fpath,
                        )
                        rec.update(eager=np.nan, inductor=np.nan, speedup=np.nan)
                    else:
                        rec.update(
                            eager=speedup * abs_lat,
                            inductor=abs_lat,
                            speedup=speedup,
                        )

            raw.setdefault(key, []).append(rec)

    # Deduplicate
    picker = _best_accuracy_record if result_type == "accuracy" else _best_performance_record
    return [picker(recs) for recs in raw.values()]
