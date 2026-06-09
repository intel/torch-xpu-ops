#!/usr/bin/env python3
"""
generate_fleet_summary.py

Generate fleet-level summary report across all models.

Sections:
  1. Overall — geomean R per platform, model count, T2 ratios
  2. Per-Suite Geomean — torchbench / timm / huggingface breakdown
  3. Model Scorecard — one row per model, sorted by XPU R ascending
  4. Worst 10 Models — with top gap op and diagnosis
  5. Op Priority Ranking — which op optimization improves fleet geomean R the most
  6. Projection Accuracy — overcounting, undercounting, uncovered ops per platform
  7. Graph Consistency — fleet-wide calcflops CUDA vs XPU comparison

Usage:
    python generate_fleet_summary.py \
        --b580-dir /path/to/b580/results \
        --4080s-dir /path/to/4080s/results \
        --b70-dir /path/to/b70/results \
        [--config /path/to/hardware_specs.yaml] \
        [--suite-dir /path/to/oob_suite_yamls] \
        -o reports/oob300/summary_eager_inference.md
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from compare_projection_vs_actual import load_platform_specs
from compare_graphs import compare_model as compare_graphs_model
from generate_report import collect_platform_data
from generate_all_reports import discover_models as discover_models_eager


def _extract_t2_compile(t2_path):
    """Extract T2 from compile t2 file."""
    with open(t2_path) as f:
        for line in f:
            if "GPU Time per batch:" in line:
                m = re.search(r"GPU Time per batch:\s*([\d.]+)", line)
                if m:
                    return float(m.group(1))
    return None


def discover_models_compile(result_dir):
    """Discover models from compile result files."""
    models = {}
    if not result_dir or not os.path.isdir(result_dir):
        return models

    pattern = re.compile(r"^(.+)_bs(\d+)_compile_result\.txt$")
    for fname in os.listdir(result_dir):
        m = pattern.match(fname)
        if not m:
            continue
        model_name = m.group(1)
        bs = m.group(2)
        key = f"{model_name}_bs{bs}"

        result_path = os.path.join(result_dir, fname)
        # Verify it has valid data (Machine-Readable section)
        has_r = False
        with open(result_path) as f:
            for line in f:
                if "R_COMPILE=" in line:
                    has_r = True
                    break
        if not has_r:
            continue

        t2_path = os.path.join(result_dir, f"{key}_t2_compile.txt")
        calcflops_path = os.path.join(result_dir, f"{key}_calcflops.txt")
        trace_path = os.path.join(result_dir, f"{key}_trace_compile.json")
        unitrace_path = os.path.join(result_dir, f"{key}_unitrace_compile.json")

        t2 = _extract_t2_compile(t2_path) if os.path.exists(t2_path) else None
        if t2 is None:
            continue

        info = {
            "model": model_name,
            "bs": bs,
            "t2": t2,
            "calcflops": calcflops_path if os.path.exists(calcflops_path) else None,
            "trace": trace_path if os.path.exists(trace_path) else None,
            "unitrace": unitrace_path if os.path.exists(unitrace_path) else None,
        }
        models[key] = info

    return models


def discover_models(result_dir, mode="eager"):
    """Dispatch to eager or compile discovery."""
    if mode == "compile":
        return discover_models_compile(result_dir)
    return discover_models_eager(result_dir)


# ---------------------------------------------------------------------------
# Suite classification
# ---------------------------------------------------------------------------

def load_suite_map(suite_dir, test_mode="inference"):
    """Load model → suite mapping from YAML files in suite_dir.

    Args:
        suite_dir: directory containing suite YAML files
        test_mode: "inference" or "training" — selects *_inference.yaml or *_training.yaml

    Returns dict: model_name → suite_name (e.g. "resnet18" → "torchbench").
    """
    suffix = f"_{test_mode}.yaml"
    suite_map = {}
    if not suite_dir or not os.path.isdir(suite_dir):
        return suite_map

    for fname in os.listdir(suite_dir):
        if not fname.endswith(suffix):
            continue
        path = os.path.join(suite_dir, fname)
        if not HAS_YAML:
            suite_name = fname.replace(suffix, "")
            suite_map.setdefault("__default__", suite_name)
            continue

        with open(path) as f:
            data = yaml.safe_load(f)
        suite_name = data.get("suite", fname.replace(suffix, ""))
        for entry in data.get("models", []):
            name = entry["name"]
            suite_map[name] = suite_name

    return suite_map


def classify_model(model_name, suite_map):
    """Return suite name for a model, or 'unknown'."""
    if model_name in suite_map:
        return suite_map[model_name]
    # Try prefix match (some model names have extra suffixes in data)
    for registered_name, suite in suite_map.items():
        if model_name.startswith(registered_name + "_") or \
           model_name == registered_name:
            return suite
    return "unknown"


# ---------------------------------------------------------------------------
# Geometric mean
# ---------------------------------------------------------------------------

def geomean(values):
    """Compute geometric mean of positive values."""
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Collect all model data
# ---------------------------------------------------------------------------

def collect_all_models(platform_dirs, config_path, suite_map, mode="eager"):
    """Collect per-model data across all platforms.

    Returns list of dicts with keys:
        key, model, bs, suite,
        {platform}_r, {platform}_t2, {platform}_t1,
        plat_data (full plat_data_list for op-level analysis)
    """
    # Discover models per platform
    all_models = {}
    for plat_id, result_dir in platform_dirs:
        models = discover_models(result_dir, mode=mode)
        all_models[plat_id] = models
        print(f"[{plat_id}] Found {len(models)} models in {result_dir}")

    # Find models on ALL platforms
    all_keys = None
    for plat_id, models in all_models.items():
        keys = set(models.keys())
        if all_keys is None:
            all_keys = keys
        else:
            all_keys &= keys

    if not all_keys:
        all_keys = set()
        for models in all_models.values():
            all_keys |= set(models.keys())

    all_keys = sorted(all_keys)
    print(f"\nProcessing {len(all_keys)} models...\n")

    results = []
    for i, key in enumerate(all_keys, 1):
        model_name = None
        bs = None
        plat_data_list = []

        for plat_id, models in all_models.items():
            if key not in models:
                continue
            info = models[key]
            if model_name is None:
                model_name = info["model"]
                bs = info["bs"]

            try:
                spec = load_platform_specs(plat_id, config_path=config_path)
                data = collect_platform_data(
                    plat_id, spec,
                    info["calcflops"], info["trace"],
                    unitrace_path=info.get("unitrace"),
                    t2_ms=info["t2"],
                )
                plat_data_list.append(data)
            except Exception as e:
                print(f"  [{i}/{len(all_keys)}] {key} {plat_id}: ERROR {e}")
                continue

        if not plat_data_list:
            continue

        suite = classify_model(model_name, suite_map)

        row = {
            "key": key,
            "model": model_name,
            "bs": bs,
            "suite": suite,
            "plat_data": plat_data_list,
        }
        for d in plat_data_list:
            p = d["platform"]
            row[f"{p}_r"] = d["r"]
            row[f"{p}_t2"] = d["t2"]
            row[f"{p}_t1"] = d["t1"]

        # Find top gap op (largest |actual - proj| on first XPU platform)
        xpu_d = next((d for d in plat_data_list
                      if d["actual_source"] == "unitrace"), None)
        if xpu_d:
            top_gap_op = max(xpu_d["agg"].items(),
                             key=lambda x: abs(x[1]["diff_ms"]),
                             default=(None, None))
            if top_gap_op[0]:
                row["top_gap_op"] = top_gap_op[0]
                row["top_gap_ms"] = top_gap_op[1]["diff_ms"]
            else:
                row["top_gap_op"] = "—"
                row["top_gap_ms"] = 0
        else:
            row["top_gap_op"] = "—"
            row["top_gap_ms"] = 0

        results.append(row)
        r_strs = [f"{d['platform']}:R={d['r']:.3f}" for d in plat_data_list]
        print(f"  [{i}/{len(all_keys)}] {key}: {', '.join(r_strs)}")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_fleet_report(results, platform_ids, precision="fp16",
                          test_mode="eval"):
    """Generate fleet-level markdown report.

    platform_ids: ordered list like ["B580", "4080", "G31"]
    """
    lines = []
    mode_label = f"{test_mode.capitalize()} ({precision})"
    lines.append(f"# Fleet Summary: OOB Models — {mode_label}\n")

    # Determine which platform is CUDA vs XPU
    cuda_plat = "4080"
    xpu_plats = [p for p in platform_ids if p != cuda_plat]
    primary_xpu = xpu_plats[0] if xpu_plats else platform_ids[0]

    # -----------------------------------------------------------------------
    # Section 1. Overall
    # -----------------------------------------------------------------------
    lines.append("## 1. Overall\n")

    n_models = len(results)
    r_per_plat = {}
    t2_per_plat = {}
    for p in platform_ids:
        rs = [r[f"{p}_r"] for r in results if r.get(f"{p}_r")]
        t2s = [r[f"{p}_t2"] for r in results if r.get(f"{p}_t2")]
        r_per_plat[p] = rs
        t2_per_plat[p] = t2s

    header = "| Metric |"
    sep = "|--------|"
    for p in platform_ids:
        header += f" {p} |"
        sep += "------:|"
    lines.append(header)
    lines.append(sep)

    lines.append("| Models profiled | " +
                 " | ".join(str(len(r_per_plat[p])) for p in platform_ids) + " |")

    gm_row = "| Geomean R |"
    for p in platform_ids:
        gm = geomean(r_per_plat[p])
        gm_row += f" **{gm:.3f}** |" if gm else " — |"
    lines.append(gm_row)

    # Geomean T2 ratio (XPU / CUDA)
    if cuda_plat in platform_ids:
        for xp in xpu_plats:
            ratios = []
            for r in results:
                xpu_t2 = r.get(f"{xp}_t2")
                cuda_t2 = r.get(f"{cuda_plat}_t2")
                if xpu_t2 and cuda_t2 and cuda_t2 > 0:
                    ratios.append(xpu_t2 / cuda_t2)
            gm_ratio = geomean(ratios) if ratios else None
            if gm_ratio:
                lines.append(f"| Geomean T2 ratio ({xp}/4080S) | "
                             f"**{gm_ratio:.2f}x** | | |")

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 2. Per-Suite Geomean
    # -----------------------------------------------------------------------
    lines.append("## 2. Per-Suite Geomean\n")

    suites = sorted({r["suite"] for r in results})
    header = "| Suite | # Models |"
    sep = "|-------|--------:|"
    for p in platform_ids:
        header += f" {p} Geomean R |"
        sep += "--------:|"
    if cuda_plat in platform_ids and xpu_plats:
        for xp in xpu_plats:
            header += f" {xp}/4080S T2 |"
            sep += "--------:|"
    lines.append(header)
    lines.append(sep)

    for suite in suites:
        suite_models = [r for r in results if r["suite"] == suite]
        row = f"| {suite} | {len(suite_models)} |"
        for p in platform_ids:
            rs = [r[f"{p}_r"] for r in suite_models if r.get(f"{p}_r")]
            gm = geomean(rs)
            row += f" {gm:.3f} |" if gm else " — |"
        if cuda_plat in platform_ids and xpu_plats:
            for xp in xpu_plats:
                ratios = []
                for r in suite_models:
                    xt = r.get(f"{xp}_t2")
                    ct = r.get(f"{cuda_plat}_t2")
                    if xt and ct and ct > 0:
                        ratios.append(xt / ct)
                gm_ratio = geomean(ratios) if ratios else None
                row += f" {gm_ratio:.2f}x |" if gm_ratio else " — |"
        lines.append(row)

    # Total row
    row = f"| **Total** | **{n_models}** |"
    for p in platform_ids:
        gm = geomean(r_per_plat[p])
        row += f" **{gm:.3f}** |" if gm else " — |"
    if cuda_plat in platform_ids and xpu_plats:
        for xp in xpu_plats:
            ratios = []
            for r in results:
                xt = r.get(f"{xp}_t2")
                ct = r.get(f"{cuda_plat}_t2")
                if xt and ct and ct > 0:
                    ratios.append(xt / ct)
            gm_ratio = geomean(ratios) if ratios else None
            row += f" **{gm_ratio:.2f}x** |" if gm_ratio else " — |"
    lines.append(row)
    lines.append("")

    # -----------------------------------------------------------------------
    # Section 3. Model Scorecard
    # -----------------------------------------------------------------------
    lines.append("## 3. Model Scorecard\n")

    # Determine sort platform: prefer G31 (B70), fallback to first XPU
    sort_xpu = "G31" if "G31" in xpu_plats else (xpu_plats[0] if xpu_plats else primary_xpu)
    lines.append(f"Sorted by R_{sort_xpu}/R_{cuda_plat} ascending (worst first).\n")

    header = "| # | Model | BS | Suite |"
    sep = "|---|-------|----|-------|"
    for p in platform_ids:
        header += f" {p} R |"
        sep += "------:|"
    # R ratio columns (R_XPU / R_CUDA) for each XPU platform
    if cuda_plat in platform_ids and xpu_plats:
        for xp in xpu_plats:
            header += f" R_{xp}/R_{cuda_plat} |"
            sep += "------:|"
    for p in platform_ids:
        header += f" {p} T2 (ms) |"
        sep += "------:|"
    if cuda_plat in platform_ids and xpu_plats:
        for xp in xpu_plats:
            header += f" {xp}/4080S T2 |"
            sep += "------:|"
    header += " Top Gap Op |"
    sep += "------------|"
    lines.append(header)
    lines.append(sep)

    # Pre-compute R ratio for sorting
    def _r_ratio(r, xp):
        xr = r.get(f"{xp}_r")
        cr = r.get(f"{cuda_plat}_r")
        if xr and cr and cr > 0:
            return xr / cr
        return 999  # missing data sorts to end

    sorted_results = sorted(results, key=lambda r: _r_ratio(r, sort_xpu))

    for i, r in enumerate(sorted_results, 1):
        row = f"| {i} | {r['model']} | {r['bs']} | {r['suite']} |"
        for p in platform_ids:
            rv = r.get(f"{p}_r")
            row += f" {rv:.3f} |" if rv else " — |"
        # R ratio columns
        if cuda_plat in platform_ids and xpu_plats:
            for xp in xpu_plats:
                ratio = _r_ratio(r, xp)
                if ratio < 900:
                    row += f" {ratio:.3f} |"
                else:
                    row += " — |"
        for p in platform_ids:
            tv = r.get(f"{p}_t2")
            row += f" {tv:.1f} |" if tv else " — |"
        if cuda_plat in platform_ids and xpu_plats:
            for xp in xpu_plats:
                xt = r.get(f"{xp}_t2")
                ct = r.get(f"{cuda_plat}_t2")
                if xt and ct and ct > 0:
                    row += f" {xt / ct:.2f}x |"
                else:
                    row += " — |"
        row += f" {r.get('top_gap_op', '—')} |"
        lines.append(row)

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 4. Worst 10 Models
    # -----------------------------------------------------------------------
    lines.append("## 4. Worst 10 Models by R\n")
    lines.append(f"Models with lowest {primary_xpu} R value.\n")

    header = "| # | Model | Suite |"
    sep = "|---|-------|-------|"
    for p in platform_ids:
        header += f" {p} R |"
        sep += "------:|"
    header += " Top Gap Op | Gap (ms) | Likely Issue |"
    sep += "------------|------:|-------------|"
    lines.append(header)
    lines.append(sep)

    for i, r in enumerate(sorted_results[:10], 1):
        row = f"| {i} | {r['model']}_bs{r['bs']} | {r['suite']} |"
        for p in platform_ids:
            rv = r.get(f"{p}_r")
            row += f" {rv:.3f} |" if rv else " — |"

        gap_ms = r.get("top_gap_ms", 0)
        top_op = r.get("top_gap_op", "—")

        # Diagnose: if R is low on all platforms, likely projection issue
        r_vals = [r.get(f"{p}_r") for p in platform_ids
                  if r.get(f"{p}_r") is not None]
        all_low = all(v < 0.70 for v in r_vals) if r_vals else False
        if all_low:
            issue = "Projection inaccuracy (low R on all platforms)"
        elif r.get(f"{primary_xpu}_r", 1) < 0.70 and \
                r.get(f"{cuda_plat}_r", 0) >= 0.70:
            issue = f"XPU kernel inefficiency (4080S R={r.get(f'{cuda_plat}_r', 0):.3f})"
        else:
            issue = "Mixed — see per-model report"

        row += f" {top_op} | {gap_ms:+.1f} | {issue} |"
        lines.append(row)

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 5. Op Priority Ranking
    # -----------------------------------------------------------------------
    lines.append("## 5. Op Priority Ranking\n")
    lines.append("If we improve each op on XPU to match 4080S roofline efficiency "
                 "(R_op), how much does the fleet geomean R improve? "
                 "Ranked by geomean R delta.\n")

    if cuda_plat not in platform_ids:
        lines.append("*Requires 4080S data for comparison.*\n")
        return lines

    for xp in xpu_plats:
        lines.append(f"### {xp}\n")

        # Current geomean R for this XPU platform
        current_rs = [r[f"{xp}_r"] for r in results if r.get(f"{xp}_r")]
        current_gm = geomean(current_rs)
        if not current_gm:
            lines.append("Insufficient data.\n")
            continue

        # Collect per-op savings across all models
        # op_name -> list of (model_key, current_r, new_r_if_op_improved)
        op_savings = defaultdict(list)

        for r in results:
            xpu_r = r.get(f"{xp}_r")
            cuda_r = r.get(f"{cuda_plat}_r")
            if not xpu_r or not cuda_r:
                continue

            # Get plat_data for this model
            xpu_d = None
            cuda_d = None
            for d in r["plat_data"]:
                if d["platform"] == xp:
                    xpu_d = d
                elif d["platform"] == cuda_plat:
                    cuda_d = d
            if not xpu_d or not cuda_d:
                continue

            # For each op where XPU R_op < CUDA R_op
            common_ops = set(xpu_d["agg"].keys()) & set(cuda_d["agg"].keys())
            for op_name in common_ops:
                vx = xpu_d["agg"][op_name]
                vc = cuda_d["agg"][op_name]
                rx = vx.get("ratio")
                rc = vc.get("ratio")
                ax = vx.get("actual_time_ms", 0)

                if rx is None or rc is None or ax < 0.1:
                    continue
                if rx >= rc:
                    continue  # XPU already as good

                # If this op matched CUDA R_op, new actual time
                px = vx.get("proj_time_ms", 0)
                if rc > 0:
                    new_actual = px / rc
                else:
                    continue
                saving = ax - new_actual
                if saving <= 0:
                    continue

                # New T2 for this model
                new_t2 = xpu_d["t2"] - saving
                if new_t2 <= 0:
                    new_t2 = 0.001
                new_model_r = xpu_d["t1"] / new_t2

                op_savings[op_name].append({
                    "model": r["key"],
                    "old_r": xpu_r,
                    "new_r": new_model_r,
                    "saving_ms": saving,
                })

        # For each op, compute new fleet geomean R
        op_impact = []
        for op_name, savings_list in op_savings.items():
            # Build new R list: replace affected models' R with new_r
            affected_models = {s["model"]: s for s in savings_list}
            new_rs = []
            total_saving = 0
            n_affected = len(savings_list)
            for r in results:
                old_r = r.get(f"{xp}_r")
                if not old_r:
                    continue
                if r["key"] in affected_models:
                    new_rs.append(affected_models[r["key"]]["new_r"])
                    total_saving += affected_models[r["key"]]["saving_ms"]
                else:
                    new_rs.append(old_r)

            new_gm = geomean(new_rs)
            delta_gm = new_gm - current_gm if new_gm else 0

            op_impact.append({
                "op": op_name,
                "delta_gm": delta_gm,
                "new_gm": new_gm,
                "n_affected": n_affected,
                "total_saving_ms": total_saving,
            })

        op_impact.sort(key=lambda x: x["delta_gm"], reverse=True)

        lines.append(f"Current fleet geomean R ({xp}): **{current_gm:.3f}**\n")
        lines.append("| # | Op | New Geomean R | Delta | Models Affected | "
                     "Total Saving (ms) |")
        lines.append("|---|-----|--------:|------:|--------:|--------:|")

        for i, o in enumerate(op_impact[:20], 1):
            lines.append(
                f"| {i} | {o['op']} | {o['new_gm']:.3f} | "
                f"+{o['delta_gm']:.4f} | {o['n_affected']} | "
                f"{o['total_saving_ms']:.0f} |")

        lines.append("")

    # -----------------------------------------------------------------------
    # Section 6. Projection Accuracy
    # -----------------------------------------------------------------------
    lines.append("## 6. Projection Accuracy\n")
    lines.append("How well does our T1 roofline projection match actual GPU time? "
                 "Aggregated across all models per platform.\n")

    for plat in platform_ids:
        lines.append(f"### {plat}\n")

        # Collect per-op stats across all models for this platform
        # op_name -> list of {ratio, proj_ms, actual_ms, diff_ms, in_calc, in_actual, model}
        op_fleet = defaultdict(list)
        coverage_list = []  # (t1, t2_device) per model

        for r in results:
            plat_d = None
            for d in r["plat_data"]:
                if d["platform"] == plat:
                    plat_d = d
                    break
            if not plat_d:
                continue

            t1 = plat_d.get("t1", 0)
            t2d = plat_d.get("t2_device", 0)
            if t1 > 0 and t2d > 0:
                coverage_list.append((t1, t2d))

            for op_name, v in plat_d["agg"].items():
                op_fleet[op_name].append({
                    "ratio": v.get("ratio"),
                    "proj_ms": v.get("proj_time_ms", 0),
                    "actual_ms": v.get("actual_time_ms", 0),
                    "diff_ms": v.get("diff_ms", 0),
                    "in_calc": v.get("in_calc", False),
                    "in_actual": v.get("in_actual", False),
                    "model": r["key"],
                })

        # --- T1/T2_device coverage ---
        if coverage_list:
            coverages = [t1 / t2d for t1, t2d in coverage_list]
            avg_cov = sum(coverages) / len(coverages)
            med_cov = sorted(coverages)[len(coverages) // 2]
            lines.append(f"**T1/T2_device coverage**: median={med_cov:.1%}, "
                         f"mean={avg_cov:.1%} across {len(coverages)} models\n")

        # --- Overcounting (R_op > 1.05) ---
        lines.append("#### Overcounting (R_op > 1.05)\n")
        lines.append("Ops where projection consistently exceeds actual GPU time. "
                     "Possible causes: peak TFLOPS/BW too low in config, "
                     "FLOPs formula overestimates.\n")

        overcount = []
        for op_name, entries in op_fleet.items():
            oc_entries = [e for e in entries
                          if e["ratio"] is not None
                          and e["ratio"] != float("inf")
                          and e["ratio"] > 1.05
                          and e["actual_ms"] > 0.01]
            if len(oc_entries) < 2:  # need at least 2 models
                continue
            ratios = [e["ratio"] for e in oc_entries]
            total_excess = sum(e["diff_ms"] for e in oc_entries)
            median_r = sorted(ratios)[len(ratios) // 2]
            # top 3 models by excess
            top3 = sorted(oc_entries, key=lambda e: e["diff_ms"],
                          reverse=True)[:3]
            top3_names = [e["model"] for e in top3]
            overcount.append({
                "op": op_name,
                "n_models": len(oc_entries),
                "median_rop": median_r,
                "total_excess_ms": total_excess,
                "pct_models": len(oc_entries) / len(results) * 100,
                "top_models": top3_names,
            })

        overcount.sort(key=lambda x: x["total_excess_ms"], reverse=True)

        if overcount:
            lines.append("| # | Op | Models (%) | Median R_op | "
                         "Total Excess (ms) | Top Models |")
            lines.append("|---|-----|--------:|------:|--------:|------|")
            for i, o in enumerate(overcount[:15], 1):
                top_m = ", ".join(o["top_models"])
                lines.append(
                    f"| {i} | {o['op']} | {o['n_models']} "
                    f"({o['pct_models']:.0f}%) | {o['median_rop']:.2f} | "
                    f"+{o['total_excess_ms']:.0f} | {top_m} |")
        else:
            lines.append("No ops with systematic overcounting.\n")
        lines.append("")

        # --- Undercounting (R_op < 0.5) ---
        lines.append("#### Undercounting (R_op < 0.5)\n")
        lines.append("Ops where actual GPU time far exceeds projection. "
                     "Possible causes: missing FLOPs in projection, "
                     "memory accounting incomplete, kernel launch overhead.\n")

        undercount = []
        for op_name, entries in op_fleet.items():
            uc_entries = [e for e in entries
                          if e["ratio"] is not None
                          and e["ratio"] != float("inf")
                          and e["ratio"] < 0.5
                          and e["actual_ms"] > 0.1]
            if len(uc_entries) < 2:
                continue
            ratios = [e["ratio"] for e in uc_entries]
            total_deficit = sum(e["actual_ms"] - e["proj_ms"]
                               for e in uc_entries)
            median_r = sorted(ratios)[len(ratios) // 2]
            # top 3 models by deficit
            top3 = sorted(uc_entries,
                          key=lambda e: e["actual_ms"] - e["proj_ms"],
                          reverse=True)[:3]
            top3_names = [e["model"] for e in top3]
            undercount.append({
                "op": op_name,
                "n_models": len(uc_entries),
                "median_rop": median_r,
                "total_deficit_ms": total_deficit,
                "pct_models": len(uc_entries) / len(results) * 100,
                "top_models": top3_names,
            })

        undercount.sort(key=lambda x: x["total_deficit_ms"], reverse=True)

        if undercount:
            lines.append("| # | Op | Models (%) | Median R_op | "
                         "Total Deficit (ms) | Top Models |")
            lines.append("|---|-----|--------:|------:|--------:|------|")
            for i, u in enumerate(undercount[:15], 1):
                top_m = ", ".join(u["top_models"])
                lines.append(
                    f"| {i} | {u['op']} | {u['n_models']} "
                    f"({u['pct_models']:.0f}%) | {u['median_rop']:.2f} | "
                    f"-{u['total_deficit_ms']:.0f} | {top_m} |")
        else:
            lines.append("No ops with systematic undercounting.\n")
        lines.append("")

        # --- Uncovered ops (in actual but no projection) ---
        lines.append("#### Uncovered Ops\n")
        lines.append("Ops with actual GPU time but zero projection. "
                     "These represent gaps in context_func.py or "
                     "ops not modeled by the roofline framework.\n")
        lines.append("*Note: In training, optimizer ops (fill_, add_, "
                     "_foreach_add_, etc. from zero_grad/step) are expected "
                     "to be uncovered — projection only covers "
                     "model forward+backward.*\n")

        uncovered = []
        for op_name, entries in op_fleet.items():
            uc_entries = [e for e in entries
                          if not e["in_calc"]
                          and e["in_actual"]
                          and e["actual_ms"] > 0.01]
            if not uc_entries:
                continue
            total_actual = sum(e["actual_ms"] for e in uc_entries)
            avg_actual = total_actual / len(uc_entries)
            # top 3 models by actual time
            top3 = sorted(uc_entries, key=lambda e: e["actual_ms"],
                          reverse=True)[:3]
            top3_names = [e["model"] for e in top3]
            uncovered.append({
                "op": op_name,
                "n_models": len(uc_entries),
                "total_actual_ms": total_actual,
                "avg_actual_ms": avg_actual,
                "pct_models": len(uc_entries) / len(results) * 100,
                "top_models": top3_names,
            })

        uncovered.sort(key=lambda x: x["total_actual_ms"], reverse=True)

        if uncovered:
            lines.append("| # | Op | Models (%) | "
                         "Total Actual (ms) | Avg per Model (ms) | Top Models |")
            lines.append("|---|-----|--------:|--------:|--------:|------|")
            for i, u in enumerate(uncovered[:15], 1):
                top_m = ", ".join(u["top_models"])
                lines.append(
                    f"| {i} | {u['op']} | {u['n_models']} "
                    f"({u['pct_models']:.0f}%) | "
                    f"{u['total_actual_ms']:.1f} | "
                    f"{u['avg_actual_ms']:.2f} | {top_m} |")
        else:
            lines.append("No uncovered ops.\n")
        lines.append("")

    # -----------------------------------------------------------------------
    # Section 7. Graph Consistency
    # -----------------------------------------------------------------------
    lines.append("## 7. Graph Consistency\n")
    lines.append("Comparison of calcflops-based computational graphs between "
                 "CUDA and XPU. Differences indicate dispatch path divergence "
                 "(e.g., SDPA fusion, XPU-specific overrideable ops).\n")
    lines.append("*Only ops with FLOPs > 0 are compared (data-movement ops "
                 "like clone are excluded).*\n")

    if cuda_plat not in platform_ids or not xpu_plats:
        lines.append("*Requires both CUDA and XPU data for comparison.*\n")
        return lines

    # Run graph comparison for each model using calcflops paths from plat_data
    for xp in xpu_plats:
        lines.append(f"### CUDA vs {xp}\n")

        comparison_results = []
        for r in results:
            cuda_d = None
            xpu_d = None
            for d in r["plat_data"]:
                if d["platform"] == cuda_plat:
                    cuda_d = d
                elif d["platform"] == xp:
                    xpu_d = d

            if not cuda_d or not xpu_d:
                continue

            cuda_cf = cuda_d.get("calcflops_path")
            xpu_cf = xpu_d.get("calcflops_path")
            if not cuda_cf or not xpu_cf:
                continue
            if not os.path.isfile(cuda_cf) or not os.path.isfile(xpu_cf):
                continue

            try:
                cmp = compare_graphs_model(cuda_cf, xpu_cf)
                comparison_results.append((r["key"], cmp))
            except Exception:
                continue

        if not comparison_results:
            lines.append("No models with calcflops data on both platforms.\n")
            continue

        n_total = len(comparison_results)
        n_match = sum(1 for _, c in comparison_results if c["match"])
        n_diff = n_total - n_match

        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Models compared | {n_total} |")
        lines.append(f"| Identical graphs | {n_match} "
                     f"({n_match / n_total * 100:.0f}%) |")
        lines.append(f"| Different graphs | {n_diff} "
                     f"({n_diff / n_total * 100:.0f}%) |")
        lines.append("")

        if n_diff == 0:
            lines.append("All models have identical computational graphs.\n")
            continue

        # Categorize: SDPA-only, significant (>1% FLOPs or Memory diff), minor
        sdpa_only = []
        significant = []
        minor = []

        for key, c in comparison_results:
            if c["match"]:
                continue
            diff_names = set()
            for d in c["diff_ops"]:
                diff_names.add(d["name"])
            for o in c["only_a"]:
                diff_names.add(o["name"])
            for o in c["only_b"]:
                diff_names.add(o["name"])

            if diff_names <= {"aten::sdpa_forward"}:
                sdpa_only.append((key, c))
            elif c["max_diff_pct"] > 1.0:
                significant.append((key, c))
            else:
                minor.append((key, c))

        lines.append("#### Difference Categories\n")
        lines.append("| Category | Count | Description |")
        lines.append("|----------|------:|-------------|")
        lines.append(f"| SDPA-only | {len(sdpa_only)} | "
                     f"Only SDPA dispatch differs (flash vs fused) |")
        lines.append(f"| Significant | {len(significant)} | "
                     f"FLOPs or Memory diff > 1% |")
        lines.append(f"| Minor | {len(minor)} | "
                     f"Small numerical differences |")
        lines.append("")

        # Top models with largest diff
        if significant:
            lines.append("#### Significant Divergences\n")
            sig_sorted = sorted(significant,
                                key=lambda x: x[1]["max_diff_pct"],
                                reverse=True)
            lines.append("| # | Model | FLOPs Diff (%) | Mem Diff (%) | "
                         "CUDA-only Ops | XPU-only Ops | "
                         "Mismatched Ops |")
            lines.append("|---|-------|------:|------:|------:|------:|------:|")
            for i, (key, c) in enumerate(sig_sorted[:15], 1):
                lines.append(
                    f"| {i} | {key} | {c['flops_diff_pct']:.2f}% | "
                    f"{c['mem_diff_pct']:.2f}% | "
                    f"{len(c['only_a'])} | {len(c['only_b'])} | "
                    f"{len(c['diff_ops'])} |")
            lines.append("")

        # Fleet-level op differences
        lines.append("#### Op Differences Across Fleet\n")
        lines.append("Ops that differ between CUDA and XPU, "
                     "aggregated across all models.\n")

        op_diff_count = defaultdict(lambda: {"count": 0, "models": []})
        for key, c in comparison_results:
            if c["match"]:
                continue
            seen = set()
            for d in c["diff_ops"]:
                if d["name"] not in seen:
                    seen.add(d["name"])
                    op_diff_count[d["name"]]["count"] += 1
                    op_diff_count[d["name"]]["models"].append(key)
            for o in c["only_a"] + c["only_b"]:
                if o["name"] not in seen:
                    seen.add(o["name"])
                    op_diff_count[o["name"]]["count"] += 1
                    op_diff_count[o["name"]]["models"].append(key)

        if op_diff_count:
            sorted_ops = sorted(op_diff_count.items(),
                                key=lambda x: x[1]["count"], reverse=True)
            lines.append("| Op | # Models | Top Models |")
            lines.append("|-----|------:|------|")
            for op_name, info in sorted_ops:
                top3 = ", ".join(info["models"][:3])
                lines.append(f"| {op_name} | {info['count']} | {top3} |")
        else:
            lines.append("No op-level differences found.\n")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate fleet-level summary report")
    parser.add_argument("--b580-dir", help="B580 result directory")
    parser.add_argument("--4080s-dir", help="4080S result directory")
    parser.add_argument("--b70-dir", help="B70 result directory")
    parser.add_argument("--config", help="Path to hardware_specs.yaml")
    parser.add_argument("--suite-dir",
                        help="Optional directory with suite YAML files")
    parser.add_argument("-o", "--output", help="Output markdown file")
    parser.add_argument("--precision", default="fp16",
                        help="Precision tag (default: fp16)")
    parser.add_argument("--test", default="eval", dest="test_mode",
                        help="Test mode: eval or train (default: eval)")
    args = parser.parse_args()

    platform_dirs = []
    platform_ids = []
    if args.b580_dir:
        platform_dirs.append(("B580", args.b580_dir))
        platform_ids.append("B580")
    if getattr(args, "4080s_dir"):
        platform_dirs.append(("4080", getattr(args, "4080s_dir")))
        platform_ids.append("4080")
    if args.b70_dir:
        platform_dirs.append(("G31", args.b70_dir))
        platform_ids.append("G31")

    if not platform_dirs:
        print("ERROR: Provide at least one platform directory")
        sys.exit(1)

    suite_mode = "training" if args.test_mode == "train" else "inference"
    suite_map = load_suite_map(args.suite_dir, test_mode=suite_mode)
    if args.suite_dir:
        print(f"Loaded {len(suite_map)} models in suite map")
    else:
        print("No suite map provided; models will be reported as 'unknown'")

    mode = "compile" if args.test_mode == "compile" else "eager"
    results = collect_all_models(platform_dirs, args.config, suite_map, mode=mode)
    print(f"\nCollected data for {len(results)} models")

    report_lines = generate_fleet_report(results, platform_ids,
                                         precision=args.precision,
                                         test_mode=args.test_mode)
    report_text = "\n".join(report_lines) + "\n"

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"\nFleet report written to {args.output}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
