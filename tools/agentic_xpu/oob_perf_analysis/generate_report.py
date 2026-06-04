#!/usr/bin/env python3
# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
generate_report.py

Generate cross-platform T1/T2/R analysis report for a single model.
Reads raw data files (calcflops, trace.json, unitrace) and produces a
structured markdown report.

Reuses parsing functions from compare_projection_vs_actual.py.

Usage:
    python generate_report.py \
        --model nanogpt --bs 1024 --precision fp16 --test eval \
        --b580-calcflops b580_calcflops.txt \
        --b580-trace b580_trace.json \
        --b580-unitrace b580_unitrace.json \
        --b580-t2 389.293 \
        --4080s-calcflops 4080s_calcflops.txt \
        --4080s-trace 4080s_trace.json \
        --4080s-t2 290.930 \
        -o report.md
"""

import argparse
import os
import sys

# Import parsing functions from sibling module
from compare_projection_vs_actual import (
    load_platform_specs,
    parse_calc_flops,
    parse_trace_ops,
    parse_unitrace_ops,
    aggregate_comparison,
)
from compare_graphs import (
    parse_calcflops_raw,
    aggregate_ops,
    compare_model as compare_graphs_model,
    _fmt_flops,
    _fmt_mem,
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_platform_data(platform_key, spec, calcflops_path, trace_path,
                          unitrace_path=None, t2_ms=None):
    """Collect all analysis data for one platform.

    Returns dict with keys: platform, spec, t2, t1, t1_compute, t1_memory,
        calc_ops, actual_ops, agg, actual_source
    """
    calc_ops = parse_calc_flops(calcflops_path, spec)

    # Compute T1 breakdown
    t1_total = sum(op["proj_time_s"] for op in calc_ops) * 1000
    t1_compute = sum(op["proj_time_s"] for op in calc_ops
                     if op["bound"] == "compute") * 1000
    t1_memory = sum(op["proj_time_s"] for op in calc_ops
                    if op["bound"] == "memory") * 1000

    # Parse actual GPU times
    if unitrace_path:
        actual_ops = parse_unitrace_ops(trace_path, unitrace_path)
        actual_source = "unitrace"
    else:
        actual_ops = parse_trace_ops(trace_path)
        actual_source = "profiler"

    actual_total = sum(op["gpu_dur_us"] for op in actual_ops) / 1000

    # Aggregate comparison
    agg = aggregate_comparison(calc_ops, actual_ops)

    # R computation
    r_value = t1_total / t2_ms if t2_ms and t2_ms > 0 else None

    return {
        "platform": platform_key,
        "spec": spec,
        "t2": t2_ms,
        "t1": t1_total,
        "t1_compute": t1_compute,
        "t1_memory": t1_memory,
        "t2_device": actual_total,
        "r": r_value,
        "calc_ops": calc_ops,
        "actual_ops": actual_ops,
        "agg": agg,
        "actual_source": actual_source,
        "n_ops": len(calc_ops),
        "n_compute": sum(1 for op in calc_ops if op["bound"] == "compute"),
        "n_memory": sum(1 for op in calc_ops if op["bound"] == "memory"),
        "calcflops_path": calcflops_path,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f"):
    """Format a numeric value, returning '—' for None."""
    return f"{v:{fmt}}" if v is not None else "—"


def generate_summary_table(plat_data_list, model_info):
    """Generate the summary table section."""
    lines = []
    lines.append("## 1. Summary\n")
    lines.append(f"**Model**: {model_info['model']}  ")
    lines.append(f"**Batch size**: {model_info['bs']}  ")
    lines.append(f"**Precision**: {model_info['precision']}  ")
    lines.append(f"**Mode**: {model_info['test']}  ")
    lines.append(f"**Ops per iteration**: {plat_data_list[0]['n_ops']}  ")
    lines.append("")

    # Header
    header = "| Metric |"
    sep = "|--------|"
    for d in plat_data_list:
        header += f" {d['spec']['label']} |"
        sep += "--------:|"
    lines.append(header)
    lines.append(sep)

    # Rows
    rows = [
        ("T2 (wall clock)", lambda d: f"{_fmt(d['t2'])} ms"),
        ("T1 (projection)", lambda d: f"{_fmt(d['t1'])} ms"),
        ("T1_compute", lambda d: f"{_fmt(d['t1_compute'])} ms"),
        ("T1_memory", lambda d: f"{_fmt(d['t1_memory'])} ms"),
        ("T2_device (kernel sum)", lambda d: f"{_fmt(d['t2_device'])} ms"),
        ("R = T1/T2", lambda d: f"**{_fmt(d['r'])}**"),
        ("Actual source", lambda d: d['actual_source']),
        ("Compute-bound ops", lambda d: str(d['n_compute'])),
        ("Memory-bound ops", lambda d: str(d['n_memory'])),
    ]

    for label, fn in rows:
        row = f"| {label} |"
        for d in plat_data_list:
            row += f" {fn(d)} |"
        lines.append(row)

    lines.append("")

    # Cross-Platform R Ratio
    cuda_plat = _find_cuda_platform(plat_data_list)
    xpu_plats = _find_xpu_platforms(plat_data_list)
    if cuda_plat and xpu_plats:
        lines.append("### Cross-Platform R Ratio\n")
        lines.append("| Comparison | R Ratio | Interpretation |")
        lines.append("|-----------|--------:|----------------|")
        for xpu in xpu_plats:
            if xpu["r"] and cuda_plat["r"] and cuda_plat["r"] > 0:
                ratio = xpu["r"] / cuda_plat["r"]
                xpu_short = xpu["platform"]
                if ratio >= 1:
                    interp = (f"{xpu_short} has {(ratio-1)*100:.1f}% "
                              f"better roofline efficiency")
                else:
                    interp = (f"{xpu_short} has {(1-ratio)*100:.1f}% "
                              f"worse roofline efficiency")
                lines.append(f"| R_{xpu_short} / R_4080S | {ratio:.3f} | {interp} |")
        lines.append("")

    # HW specs for reference
    lines.append("### Hardware Specs\n")
    header = "| Spec |"
    sep = "|------|"
    for d in plat_data_list:
        header += f" {d['spec']['label']} |"
        sep += "--------:|"
    lines.append(header)
    lines.append(sep)

    hw_rows = [
        ("Peak FP16 (TFLOPS)", lambda d: f"{d['spec']['peak_tflops']/1e12:.2f}"),
        ("DRAM BW (GB/s)", lambda d: f"{d['spec']['bandwidth']/1e9:.1f}"),
        ("Ridge point (OPs/B)", lambda d: f"{d['spec']['roofline_ratio']:.1f}"),
    ]
    for label, fn in hw_rows:
        row = f"| {label} |"
        for d in plat_data_list:
            row += f" {fn(d)} |"
        lines.append(row)

    lines.append("")
    return lines


def generate_action_items(plat_data_list):
    """Generate Action Items table and overall assessment for Section 1.

    Auto-generated from per-op R analysis across platforms.
    Categories:
    - Optimize XPU kernel: R_op < 0.80 on XPU but >= 0.80 on CUDA
    - Investigate XPU kernel: R_op < 0.80 on one XPU but OK on other XPU
    - Fix projection (undercounting): R_op < 0.80 on ALL platforms
    - Fix projection (overcounting): R_op > 1.05 on ALL platforms
    """
    lines = []
    actions = []  # list of (priority_score, action_dict)

    # Build per-op R map across platforms
    op_r = {}  # op_name -> {platform: R_op}
    op_actual = {}  # op_name -> {platform: actual_ms}
    op_proj = {}  # op_name -> {platform: proj_ms}
    op_bound = {}
    for d in plat_data_list:
        plat = d["platform"]
        for name, row in d["agg"].items():
            if name not in op_r:
                op_r[name] = {}
                op_actual[name] = {}
                op_proj[name] = {}
            actual = row["actual_time_ms"]
            proj = row["proj_time_ms"]
            if actual > 0.001:
                op_r[name][plat] = proj / actual
            else:
                op_r[name][plat] = None
            op_actual[name][plat] = actual
            op_proj[name][plat] = proj
            # Determine bound
            if row["compute_count"] > row["memory_count"]:
                op_bound[name] = "compute"
            elif row["memory_count"] > 0:
                op_bound[name] = "memory"

    # Find XPU and CUDA platforms
    xpu_plats = [d for d in plat_data_list
                 if d["actual_source"] in ("unitrace",)]
    cuda_plats = [d for d in plat_data_list
                  if d["actual_source"] in ("profiler",)]
    cuda_key = cuda_plats[0]["platform"] if cuda_plats else None

    for op_name, r_map in op_r.items():
        r_values = {k: v for k, v in r_map.items() if v is not None}
        if not r_values:
            continue

        actual_map = op_actual.get(op_name, {})
        # Max actual time across XPU platforms for impact assessment
        xpu_actual_max = max(
            (actual_map.get(d["platform"], 0) for d in xpu_plats), default=0)
        t2_max = max((d["t2"] for d in xpu_plats), default=1)
        pct_t2 = xpu_actual_max / t2_max * 100 if t2_max > 0 else 0

        all_low = all(v < 0.80 for v in r_values.values() if v is not None)
        all_high = all(v > 1.05 for v in r_values.values() if v is not None)

        xpu_r = [r_values.get(d["platform"]) for d in xpu_plats
                 if r_values.get(d["platform"]) is not None]
        cuda_r = r_values.get(cuda_key) if cuda_key else None

        any_xpu_low = any(r < 0.80 for r in xpu_r) if xpu_r else False
        cuda_ok = cuda_r is not None and cuda_r >= 0.80

        if all_low:
            # Projection undercounting
            r_str = ", ".join(
                f"{d['platform']}={r_values[d['platform']]:.2f}"
                for d in plat_data_list
                if r_values.get(d["platform"]) is not None)
            saving_str = f"Low T2 impact ({pct_t2:.1f}%)" if pct_t2 < 2 else \
                f"{pct_t2:.1f}% of T2"
            actions.append((
                -pct_t2,  # lower priority for projection fixes
                {
                    "action": "Fix projection",
                    "target": "context_func.py",
                    "op": op_name,
                    "impact": f"R_op {r_str}. {saving_str}",
                    "priority": "Medium" if pct_t2 > 2 else "Low",
                }
            ))
        elif all_high:
            # Projection overcounting
            r_str = ", ".join(
                f"{d['platform']}={r_values[d['platform']]:.2f}"
                for d in plat_data_list
                if r_values.get(d["platform"]) is not None)
            actions.append((
                -0.1,  # low priority
                {
                    "action": "Fix projection",
                    "target": "context_func.py",
                    "op": op_name,
                    "impact": f"R_op {r_str} (overcounting). Negligible T2 impact",
                    "priority": "Low",
                }
            ))
        elif any_xpu_low and cuda_ok:
            # XPU kernel inefficiency
            target_r = cuda_r
            for xd in xpu_plats:
                xr = r_values.get(xd["platform"])
                if xr is not None and xr < 0.80:
                    xa = actual_map.get(xd["platform"], 0)
                    proj_ms = op_proj.get(op_name, {}).get(xd["platform"], 0)
                    if proj_ms > 0 and target_r > 0:
                        target_actual = proj_ms / target_r
                        saving = xa - target_actual
                    else:
                        saving = 0
                    xpu_t2 = xd["t2"]
                    pct = saving / xpu_t2 * 100 if xpu_t2 > 0 else 0
                    # Show all platforms' R_op values
                    r_all_str = ", ".join(
                        f"{d['platform']}={r_values[d['platform']]:.2f}"
                        for d in plat_data_list
                        if r_values.get(d["platform"]) is not None)
                    impact = (f"R_op {r_all_str}. "
                              f"Save {saving:.1f}ms on {xd['platform']} "
                              f"({pct:.1f}% T2)")
                    pri = "High" if pct > 1.5 else "Medium" if pct > 0.5 else "Low"
                    actions.append((
                        pct,  # higher % = higher priority
                        {
                            "action": "Optimize XPU kernel",
                            "target": "torch-xpu-ops",
                            "op": op_name,
                            "impact": impact,
                            "priority": pri,
                        }
                    ))

    # De-duplicate: keep highest priority action per op
    seen_ops = {}
    for score, act in sorted(actions, key=lambda x: -x[0]):
        op = act["op"]
        if op not in seen_ops:
            seen_ops[op] = (score, act)
        elif act["action"] == "Optimize XPU kernel" and \
                seen_ops[op][1]["action"] != "Optimize XPU kernel":
            seen_ops[op] = (score, act)

    sorted_actions = sorted(seen_ops.values(), key=lambda x: -x[0])

    if not sorted_actions:
        lines.append("### Action Items\n")
        lines.append("No action items identified. All ops are performing "
                      "at or near roofline.\n")
    else:
        lines.append("### Action Items\n")
        header = ("| # | Action | Target | Op | Shape | Stride "
                  "| Expected Impact | Priority |")
        sep = ("|---|--------|--------|-----|-------|--------"
               "|----------------|----------|")
        lines.append(header)
        lines.append(sep)
        for i, (_, act) in enumerate(sorted_actions, 1):
            shape, stride = _get_dominant_shape_stride(
                act["op"], plat_data_list)
            shape_str = _fmt_shape(shape)
            stride_str = _fmt_shape(stride)
            lines.append(
                f"| {i} | **{act['action']}** | {act['target']} "
                f"| {act['op']} | {shape_str} | {stride_str} "
                f"| {act['impact']} | {act['priority']} |")
        lines.append("")

    # Overall assessment
    r_values = {d["platform"]: d["r"] for d in plat_data_list}
    xpu_rs = [d["r"] for d in xpu_plats]
    min_xpu_r = min(xpu_rs) if xpu_rs else 0
    max_xpu_r = max(xpu_rs) if xpu_rs else 0

    # Count high-priority actions
    n_high = sum(1 for _, a in sorted_actions if a["priority"] == "High")
    n_kernel = sum(1 for _, a in sorted_actions
                   if a["action"] == "Optimize XPU kernel")
    n_proj = sum(1 for _, a in sorted_actions
                 if a["action"] == "Fix projection")

    if max_xpu_r >= 0.95:
        health = "excellent"
    elif max_xpu_r >= 0.85:
        health = "good"
    elif max_xpu_r >= 0.70:
        health = "fair"
    else:
        health = "poor"

    # Build assessment
    parts = []
    plat_r_str = ", ".join(
        f"{d['platform']} R={d['r']:.3f}" for d in plat_data_list)
    parts.append(f"**Overall assessment**: {health} ({plat_r_str}).")

    if n_high > 0:
        parts.append(f"{n_high} high-priority action(s).")
    if n_kernel > 0:
        parts.append(f"{n_kernel} kernel optimization target(s).")
    if n_proj > 0:
        parts.append(f"{n_proj} projection fix(es).")
    if n_high == 0 and n_kernel == 0:
        parts.append("No kernel optimization needed.")

    # Wall clock comparison
    cuda_d = next((d for d in plat_data_list
                   if d["actual_source"] == "profiler"), None)
    if cuda_d:
        for xd in xpu_plats:
            ratio = xd["t2"] / cuda_d["t2"] if cuda_d["t2"] > 0 else 0
            if ratio < 1.05:
                parts.append(
                    f"{xd['platform']} matches 4080S wall clock "
                    f"({xd['t2']:.0f} vs {cuda_d['t2']:.0f}ms).")
            else:
                parts.append(
                    f"{xd['platform']} is {ratio:.2f}x slower than 4080S "
                    f"({xd['t2']:.0f} vs {cuda_d['t2']:.0f}ms).")

    lines.append(" ".join(parts))
    lines.append("")
    return lines


def _get_op_bound(v):
    """Return bound classification string for an aggregated op."""
    cc, mc = v.get("compute_count", 0), v.get("memory_count", 0)
    if cc > 0 and mc == 0:
        return "compute"
    if mc > 0 and cc == 0:
        return "memory"
    if cc + mc > 0:
        return "mixed"
    return "—"


def _get_dominant_shape_stride(op_name, plat_data_list):
    """Get dominant shape and stride for an op across all platforms.

    Returns the shape/stride from the first platform that has non-empty data.
    Dominant shape = the shape instance contributing the most actual GPU time
    (already computed by aggregate_comparison).
    """
    for d in plat_data_list:
        v = d["agg"].get(op_name)
        if v and v.get("dominant_shape"):
            return v["dominant_shape"], v.get("dominant_stride", "")
    return "", ""


def _fmt_shape(dims_str, max_len=None):
    """Format a shape string for table display."""
    if not dims_str:
        return "—"
    s = str(dims_str).strip()
    return s if s else "—"


def _find_cuda_platform(plat_data_list):
    """Find the CUDA platform data dict (profiler source)."""
    for d in plat_data_list:
        if d["actual_source"] == "profiler":
            return d
    return None


def _find_xpu_platforms(plat_data_list):
    """Find all XPU platform data dicts."""
    return [d for d in plat_data_list if d["actual_source"] == "unitrace"]


def generate_section_projection_quality(plat_data_list):
    """Section 2: Projection Quality — ops where R_op deviates significantly.

    Combines old Projection Audit (R>1) and Low-R Diagnosis (R<0.8) into
    one unified table per platform.
    """
    lines = []
    lines.append("## 2. Projection Quality\n")
    lines.append("Ops where per-op R deviates from 1.0, sorted by T2 impact. "
                 "R_op = Projected / Actual. "
                 "R_op >> 1 means projection overcounts (fix projection). "
                 "R_op << 1 means projection undercounts OR kernel is slow.\n")

    cuda_plat = _find_cuda_platform(plat_data_list)

    for d in plat_data_list:
        label = d["spec"]["label"]
        flagged = []

        for name, v in d["agg"].items():
            actual = v["actual_time_ms"]
            proj = v["proj_time_ms"]
            if actual < 0.01 and proj < 0.01:
                continue
            r_op = v.get("ratio")
            if r_op is None:
                continue
            # Skip inf ratios (actual~0 but proj>0 — op attributed to
            # different inner op on this platform)
            if r_op == float("inf"):
                continue
            if r_op > 1.05 or r_op < 0.80:
                gap = proj - actual  # positive = overcounting
                bound = _get_op_bound(v)
                # Cross-ref with CUDA R for diagnosis
                cuda_r = None
                if cuda_plat and cuda_plat["platform"] != d["platform"]:
                    cv = cuda_plat["agg"].get(name)
                    if cv and cv.get("ratio"):
                        cuda_r = cv["ratio"]
                # Diagnosis
                if r_op > 1.05:
                    issue = "Overcounting"
                elif cuda_r is not None and cuda_r >= 0.80 and r_op < 0.80:
                    issue = "Kernel slow"
                elif cuda_r is not None and cuda_r < 0.80 and r_op < 0.80:
                    issue = "Projection undercounts"
                else:
                    issue = "Undercounts or slow"

                flagged.append({
                    "name": name, "r_op": r_op, "actual": actual,
                    "proj": proj, "gap": gap, "bound": bound,
                    "cuda_r": cuda_r, "issue": issue,
                    "pct_t2": actual / d["t2"] * 100 if d["t2"] > 0 else 0,
                    "dominant_shape": v.get("dominant_shape", ""),
                    "dominant_stride": v.get("dominant_stride", ""),
                    "actual_tflops": v.get("actual_tflops"),
                    "actual_bw_gbs": v.get("actual_bw_gbs"),
                })

        flagged.sort(key=lambda x: x["actual"], reverse=True)

        lines.append(f"### {label} (R = {_fmt(d['r'])})\n")
        if not flagged:
            lines.append("All ops within R_op 0.80–1.05. Projection is accurate.\n")
            continue

        lines.append("| Op | R_op | Actual (ms) | % T2 | Proj (ms) | "
                     "Gap (ms) | Perf | Shape | 4080S R_op | Issue |")
        lines.append("|----|-----:|------------:|-----:|----------:|"
                     "---------:|-----:|-------|----------:|-------|")
        for f in flagged:
            cuda_str = f"{f['cuda_r']:.2f}" if f['cuda_r'] is not None else "—"
            shape_str = _fmt_shape(f["dominant_shape"])
            # Show actual TFLOPS and/or BW — display whatever is available
            perf_parts = []
            if f["actual_tflops"] is not None:
                perf_parts.append(f"{f['actual_tflops']:.1f}TFLOPS")
            if f["actual_bw_gbs"] is not None:
                perf_parts.append(f"{f['actual_bw_gbs']:.0f}GB/s")
            perf_str = "/".join(perf_parts) if perf_parts else "—"
            lines.append(
                f"| {f['name']} | {f['r_op']:.2f} | {f['actual']:.1f} | "
                f"{f['pct_t2']:.1f}% | {f['proj']:.1f} | "
                f"{f['gap']:+.1f} | {perf_str} | {shape_str} | {cuda_str} | {f['issue']} |")
        lines.append("")

    # Model-level analysis: T2 ops not captured by T1
    lines.append("### T2 Coverage by T1\n")
    lines.append("Checks whether all ops in the actual trace (T2) are captured "
                 "by the projection model (T1/calcflops). "
                 "Uncaptured ops mean T1 underestimates total time.\n")

    has_uncaptured = False
    for d in plat_data_list:
        label = d["spec"]["label"]
        uncaptured = []
        for name, v in d["agg"].items():
            if v["in_actual"] and not v["in_calc"] and v["actual_time_ms"] > 0.01:
                uncaptured.append({
                    "name": name,
                    "actual_ms": v["actual_time_ms"],
                    "pct_t2": v["actual_time_ms"] / d["t2"] * 100 if d["t2"] > 0 else 0,
                    "count": v["count_actual"],
                })
        if uncaptured:
            has_uncaptured = True
            uncaptured.sort(key=lambda x: x["actual_ms"], reverse=True)
            total_uncaptured = sum(u["actual_ms"] for u in uncaptured)
            total_pct = total_uncaptured / d["t2"] * 100 if d["t2"] > 0 else 0
            lines.append(f"**{label}**: {len(uncaptured)} ops in T2 not captured "
                         f"by T1 ({total_uncaptured:.1f} ms, {total_pct:.1f}% of T2)\n")
            lines.append("| Op | Actual (ms) | % T2 | Count |")
            lines.append("|----|------------:|-----:|------:|")
            for u in uncaptured:
                lines.append(
                    f"| {u['name']} | {u['actual_ms']:.3f} | "
                    f"{u['pct_t2']:.1f}% | {u['count']} |")
            lines.append("")

    if not has_uncaptured:
        lines.append("All ops in T2 are captured by T1 on all platforms. "
                     "Projection coverage is complete.\n")

    return lines


def generate_section_vs_4080s(plat_data_list):
    """Section 3: XPU vs 4080S Per-Op Efficiency Comparison.

    For each XPU platform, show all ops sorted by R_diff = R_xpu - R_4080S.
    Top = XPU strengths, Bottom = XPU weaknesses.
    """
    lines = []
    cuda_plat = _find_cuda_platform(plat_data_list)
    if not cuda_plat:
        return lines

    xpu_plats = _find_xpu_platforms(plat_data_list)
    if not xpu_plats:
        return lines

    lines.append("## 4. XPU vs 4080S: Per-Op Efficiency\n")
    lines.append("R_op = Projected / Actual (per-op roofline efficiency). "
                 "R_diff = R_xpu − R_4080S. "
                 "Positive = XPU is more efficient for this op.\n")

    for xpu in xpu_plats:
        label = xpu["spec"]["label"]
        common = set(xpu["agg"].keys()) & set(cuda_plat["agg"].keys())

        rows = []
        for name in common:
            vx = xpu["agg"][name]
            vc = cuda_plat["agg"][name]
            ax = vx.get("actual_time_ms", 0)
            ac = vc.get("actual_time_ms", 0)
            if ax < 0.01 and ac < 0.01:
                continue
            rx = vx.get("ratio")
            rc = vc.get("ratio")
            if rx is None or rc is None:
                continue
            # Skip inf ratios (op attributed to different inner op)
            if rx == float("inf") or rc == float("inf"):
                continue
            rows.append({
                "name": name, "r_xpu": rx, "r_4080s": rc,
                "r_diff": rx - rc, "xpu_ms": ax, "cuda_ms": ac,
                "bound": _get_op_bound(vx),
                "pct_t2": ax / xpu["t2"] * 100 if xpu["t2"] > 0 else 0,
            })

        # Sort by %T2 descending (most impactful ops first)
        rows.sort(key=lambda x: x["pct_t2"], reverse=True)

        lines.append(f"### {label} vs 4080S\n")
        lines.append("| Op | R_xpu | R_4080S | R_diff | XPU (ms) | "
                     "4080S (ms) | % T2 | Verdict |")
        lines.append("|----|------:|--------:|-------:|---------:|"
                     "----------:|-----:|---------|")
        for r in rows:
            if abs(r["r_diff"]) < 0.03 and r["pct_t2"] < 1:
                continue  # skip tiny differences on tiny ops
            if r["r_diff"] > 0.05:
                verdict = "**XPU wins**"
            elif r["r_diff"] < -0.05:
                verdict = "XPU behind"
            else:
                verdict = "~tie"
            lines.append(
                f"| {r['name']} | {r['r_xpu']:.2f} | {r['r_4080s']:.2f} | "
                f"{r['r_diff']:+.2f} | {r['xpu_ms']:.1f} | {r['cuda_ms']:.1f} | "
                f"{r['pct_t2']:.1f}% | {verdict} |")
        lines.append("")

    return lines


def generate_section_optimization_targets(plat_data_list):
    """Section 4: Top Optimization Targets.

    For each XPU platform, rank ops by potential T2 saving if R_op matches
    4080S. Only includes ops where XPU R_op < 4080S R_op (room to improve).
    """
    lines = []
    cuda_plat = _find_cuda_platform(plat_data_list)
    if not cuda_plat:
        return lines

    xpu_plats = _find_xpu_platforms(plat_data_list)
    if not xpu_plats:
        return lines

    lines.append("## 5. Optimization Targets: Improve R by Matching 4080S\n")
    lines.append("If XPU kernel matched 4080S roofline efficiency for each op, "
                 "how much wall-clock time would we save? "
                 "Ranked by potential T2 saving.\n")

    for xpu in xpu_plats:
        label = xpu["spec"]["label"]
        common = set(xpu["agg"].keys()) & set(cuda_plat["agg"].keys())

        targets = []
        for name in common:
            vx = xpu["agg"][name]
            vc = cuda_plat["agg"][name]
            ax = vx.get("actual_time_ms", 0)
            px = vx.get("proj_time_ms", 0)
            rx = vx.get("ratio")
            rc = vc.get("ratio")

            if rx is None or rc is None or ax < 0.1:
                continue
            if rx >= rc:
                continue  # XPU already >= 4080S efficiency

            # If we match 4080S R_op: new_actual = proj / R_target
            if rc > 0:
                target_actual = px / rc
            else:
                continue
            saving = ax - target_actual
            if saving <= 0:
                continue

            pct_t2 = saving / xpu["t2"] * 100 if xpu["t2"] > 0 else 0
            bound = _get_op_bound(vx)

            # Classify action
            # Check if ALL platforms have low R (projection issue)
            all_low = True
            for d in plat_data_list:
                dv = d["agg"].get(name)
                if dv and dv.get("ratio") is not None and dv["ratio"] >= 0.80:
                    all_low = False
                    break

            if all_low and rx < 0.80:
                action = "Fix projection"
            else:
                action = "Optimize kernel"

            targets.append({
                "name": name, "r_xpu": rx, "r_4080s": rc,
                "actual": ax, "target": target_actual, "saving": saving,
                "pct_t2": pct_t2, "bound": bound, "action": action,
            })

        targets.sort(key=lambda x: x["saving"], reverse=True)

        lines.append(f"### {label} (T2 = {xpu['t2']:.1f} ms)\n")
        if not targets:
            lines.append("No ops where XPU trails 4080S. All ops at or above 4080S efficiency.\n")
            continue

        lines.append("| # | Op | R_xpu | R_4080S | Actual (ms) | Target (ms) | "
                     "Saving (ms) | % T2 | Action |")
        lines.append("|---|-----|------:|--------:|------------:|------------:|"
                     "------------:|-----:|--------|")
        for i, t in enumerate(targets[:15], 1):
            lines.append(
                f"| {i} | {t['name']} | {t['r_xpu']:.2f} | {t['r_4080s']:.2f} | "
                f"{t['actual']:.1f} | {t['target']:.1f} | "
                f"{t['saving']:.1f} | {t['pct_t2']:.1f}% | {t['action']} |")

        total_saving = sum(t["saving"] for t in targets)
        total_pct = total_saving / xpu["t2"] * 100 if xpu["t2"] > 0 else 0
        lines.append(f"| | **TOTAL** | | | | | "
                     f"**{total_saving:.1f}** | **{total_pct:.1f}%** | |")
        lines.append("")

        # If all savings come from projection fixes, note it
        n_kernel = sum(1 for t in targets if t["action"] == "Optimize kernel")
        n_proj = sum(1 for t in targets if t["action"] == "Fix projection")
        if n_kernel == 0 and n_proj > 0:
            lines.append(f"All gaps are projection issues — no kernel optimization needed. "
                         f"Fixing projection in context_func.py would improve R accuracy "
                         f"but not actual performance.\n")
        elif n_kernel > 0:
            kernel_saving = sum(t["saving"] for t in targets
                                if t["action"] == "Optimize kernel")
            kernel_pct = kernel_saving / xpu["t2"] * 100 if xpu["t2"] > 0 else 0
            lines.append(f"Kernel optimization potential: **{kernel_saving:.1f} ms "
                         f"({kernel_pct:.1f}% of T2)** across {n_kernel} op(s).\n")

    return lines


def generate_section_xpu_cuda_consistency(plat_data_list):
    """Model-level analysis: XPU vs CUDA op list and shape consistency.

    Checks whether XPU and CUDA run the same set of ops and whether
    the dominant shapes match.  Differences may indicate fusion/dispatch
    differences that affect projection accuracy.
    """
    lines = []
    cuda_plat = _find_cuda_platform(plat_data_list)
    xpu_plats = _find_xpu_platforms(plat_data_list)
    if not cuda_plat or not xpu_plats:
        return lines

    lines.append("## 3. XPU vs CUDA Consistency\n")
    lines.append("Checks whether XPU and CUDA run the same ops with the same "
                 "shapes. Differences indicate fusion or dispatch divergence.\n")

    cuda_label = cuda_plat["spec"]["label"]

    # --- 3a. Graph consistency (calcflops-based) ---
    cuda_cf_path = cuda_plat.get("calcflops_path")
    for xpu in xpu_plats:
        xpu_label = xpu["spec"]["label"]
        xpu_cf_path = xpu.get("calcflops_path")

        if cuda_cf_path and xpu_cf_path:
            try:
                gc = compare_graphs_model(cuda_cf_path, xpu_cf_path)
                lines.append(f"### Graph Consistency ({xpu_label} vs {cuda_label})\n")
                lines.append("Compares calcflops output: both devices compute T1 for "
                             "all platforms, so identical graphs produce identical "
                             "FLOPs/memory. Differences reveal dispatch divergence.\n")

                status = "MATCH" if gc["match"] else "DIFF"
                lines.append(f"| Metric | Value |")
                lines.append(f"|--------|------:|")
                lines.append(f"| Status | **{status}** |")
                lines.append(f"| Total FLOPs ({cuda_label}) | "
                             f"{_fmt_flops(gc['total_flops_a'])} |")
                lines.append(f"| Total FLOPs ({xpu_label}) | "
                             f"{_fmt_flops(gc['total_flops_b'])} |")
                lines.append(f"| FLOPs diff | {gc['flops_diff_pct']:.2f}% |")
                lines.append(f"| Total Memory ({cuda_label}) | "
                             f"{_fmt_mem(gc['total_mem_a'])} |")
                lines.append(f"| Total Memory ({xpu_label}) | "
                             f"{_fmt_mem(gc['total_mem_b'])} |")
                lines.append(f"| Memory diff | {gc['mem_diff_pct']:.2f}% |")
                lines.append(f"| Common ops | {gc['n_common']} "
                             f"({gc['n_matching']} matching) |")
                lines.append(f"| {cuda_label}-only ops | {len(gc['only_a'])} |")
                lines.append(f"| {xpu_label}-only ops | {len(gc['only_b'])} |")
                lines.append("")

                if gc["only_a"]:
                    lines.append(f"**{cuda_label}-only ops:**\n")
                    lines.append("| Op | FLOPs | Memory | Count |")
                    lines.append("|----|------:|------:|------:|")
                    for o in gc["only_a"][:10]:
                        raw = ", ".join(sorted(o["raw_names"]))
                        lines.append(f"| {raw} | {_fmt_flops(o['flops'])} | "
                                     f"{_fmt_mem(o['memory'])} | {o['count']} |")
                    lines.append("")

                if gc["only_b"]:
                    lines.append(f"**{xpu_label}-only ops:**\n")
                    lines.append("| Op | FLOPs | Memory | Count |")
                    lines.append("|----|------:|------:|------:|")
                    for o in gc["only_b"][:10]:
                        raw = ", ".join(sorted(o["raw_names"]))
                        lines.append(f"| {raw} | {_fmt_flops(o['flops'])} | "
                                     f"{_fmt_mem(o['memory'])} | {o['count']} |")
                    lines.append("")

                if gc["diff_ops"]:
                    lines.append("**Ops with different FLOPs/memory:**\n")
                    lines.append(f"| Op | {cuda_label} FLOPs | {xpu_label} FLOPs | "
                                 f"FLOPs Diff | {cuda_label} Mem | {xpu_label} Mem |")
                    lines.append("|-----|------:|------:|------:|------:|------:|")
                    for d in gc["diff_ops"][:10]:
                        fdiff = d["flops_b"] - d["flops_a"]
                        sign = "+" if fdiff >= 0 else ""
                        lines.append(
                            f"| {d['name']} | {_fmt_flops(d['flops_a'])} | "
                            f"{_fmt_flops(d['flops_b'])} | "
                            f"{sign}{_fmt_flops(fdiff)} | "
                            f"{_fmt_mem(d['memory_a'])} | "
                            f"{_fmt_mem(d['memory_b'])} |")
                    lines.append("")

            except Exception as e:
                lines.append(f"### Graph Consistency ({xpu_label} vs {cuda_label})\n")
                lines.append(f"*Error computing graph consistency: {e}*\n")

    # --- 3b. Trace-based op/shape comparison ---
    cuda_ops = cuda_plat["agg"]

    for xpu in xpu_plats:
        xpu_label = xpu["spec"]["label"]
        xpu_ops = xpu["agg"]

        cuda_only = set(cuda_ops.keys()) - set(xpu_ops.keys())
        xpu_only = set(xpu_ops.keys()) - set(cuda_ops.keys())
        common = set(cuda_ops.keys()) & set(xpu_ops.keys())

        lines.append(f"### Trace Comparison: {xpu_label} vs {cuda_label}\n")
        lines.append(f"- Common ops: {len(common)}")
        lines.append(f"- {cuda_label}-only ops: {len(cuda_only)}")
        lines.append(f"- {xpu_label}-only ops: {len(xpu_only)}")
        lines.append("")

        # Ops only on one platform (significant ones)
        platform_only = []
        for name in cuda_only:
            v = cuda_ops[name]
            if v["actual_time_ms"] > 0.01 or v["proj_time_ms"] > 0.01:
                platform_only.append((name, cuda_label, v["actual_time_ms"],
                                      v["proj_time_ms"]))
        for name in xpu_only:
            v = xpu_ops[name]
            if v["actual_time_ms"] > 0.01 or v["proj_time_ms"] > 0.01:
                platform_only.append((name, xpu_label, v["actual_time_ms"],
                                      v["proj_time_ms"]))

        if platform_only:
            platform_only.sort(key=lambda x: x[2], reverse=True)
            lines.append("**Platform-specific ops:**\n")
            lines.append("| Op | Only on | Actual (ms) | Proj (ms) |")
            lines.append("|----|---------|------------:|----------:|")
            for name, plat, actual, proj in platform_only:
                lines.append(f"| {name} | {plat} | {actual:.3f} | {proj:.3f} |")
            lines.append("")

        # Shape set comparison on common ops — only for compute ops (have flops).
        # Pure data-movement ops (clone/copy_) have dispatch-path shape
        # differences that are not meaningful for model consistency.
        shape_diffs = []
        for name in common:
            vc = cuda_ops[name]
            vx = xpu_ops[name]
            # Skip pure data-movement ops (no flops)
            if vc.get("flops", 0) == 0 and vx.get("flops", 0) == 0:
                continue
            cuda_shapes = vc.get("all_shapes", set())
            xpu_shapes = vx.get("all_shapes", set())
            if not cuda_shapes and not xpu_shapes:
                continue
            cuda_only_shapes = cuda_shapes - xpu_shapes
            xpu_only_shapes = xpu_shapes - cuda_shapes
            if cuda_only_shapes or xpu_only_shapes:
                max_actual = max(vc["actual_time_ms"], vx["actual_time_ms"])
                if max_actual > 0.1:
                    shape_diffs.append({
                        "name": name,
                        "cuda_only": cuda_only_shapes,
                        "xpu_only": xpu_only_shapes,
                        "common_shapes": cuda_shapes & xpu_shapes,
                        "cuda_actual": vc["actual_time_ms"],
                        "xpu_actual": vx["actual_time_ms"],
                    })

        if shape_diffs:
            shape_diffs.sort(key=lambda x: max(x["cuda_actual"],
                                                x["xpu_actual"]),
                             reverse=True)
            lines.append("**Shape set differences** (ops where XPU and CUDA "
                         "see different input shapes):\n")
            for sd in shape_diffs[:10]:
                lines.append(f"**{sd['name']}** "
                             f"({cuda_label}={sd['cuda_actual']:.1f}ms, "
                             f"{xpu_label}={sd['xpu_actual']:.1f}ms)")
                if sd["common_shapes"]:
                    lines.append(f"- Common shapes: "
                                 f"{', '.join(sorted(sd['common_shapes']))}")
                if sd["cuda_only"]:
                    lines.append(f"- {cuda_label} only: "
                                 f"{', '.join(sorted(sd['cuda_only']))}")
                if sd["xpu_only"]:
                    lines.append(f"- {xpu_label} only: "
                                 f"{', '.join(sorted(sd['xpu_only']))}")
                lines.append("")
        else:
            lines.append("All common ops have identical shape sets across platforms.\n")

    return lines


def generate_report(plat_data_list, model_info):
    """Generate the 5-section per-model report.

    Sections:
    1. Summary — R/T2 metrics + Action Items + Overall Assessment
    2. Projection Quality — ops where R_op deviates (>1.05 or <0.80)
       + T2 coverage by T1
    3. XPU vs CUDA Consistency — graph consistency (calcflops) + trace
       op list and shape differences
    4. XPU vs 4080S — per-op efficiency comparison
    5. Optimization Targets — rank ops by T2 saving if matching 4080S
    """
    lines = []
    model_str = (f"{model_info['model']} bs={model_info['bs']} "
                 f"{model_info['precision']} {model_info['test']}")
    lines.append(f"# T1/T2/R Analysis: {model_str}\n")

    # Section 1: Summary
    lines.extend(generate_summary_table(plat_data_list, model_info))
    lines.extend(generate_action_items(plat_data_list))

    # Section 2: Projection Quality (includes T2 coverage by T1)
    lines.extend(generate_section_projection_quality(plat_data_list))

    # Section 3: XPU vs CUDA Consistency
    lines.extend(generate_section_xpu_cuda_consistency(plat_data_list))

    # Section 4: XPU vs 4080S Efficiency
    lines.extend(generate_section_vs_4080s(plat_data_list))

    # Section 5: Optimization Targets
    lines.extend(generate_section_optimization_targets(plat_data_list))

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-platform T1/T2/R analysis report")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--bs", required=True, help="Batch size")
    parser.add_argument("--precision", default="fp16", help="Precision (default: fp16)")
    parser.add_argument("--test", default="eval", help="Test mode (default: eval)")

    # B580
    parser.add_argument("--b580-calcflops", help="B580 calcflops file")
    parser.add_argument("--b580-trace", help="B580 trace.json")
    parser.add_argument("--b580-unitrace", help="B580 unitrace JSON")
    parser.add_argument("--b580-t2", type=float, help="B580 T2 (ms)")

    # 4080S
    parser.add_argument("--4080s-calcflops", help="4080S calcflops file")
    parser.add_argument("--4080s-trace", help="4080S trace.json")
    parser.add_argument("--4080s-unitrace", help="4080S unitrace JSON (if available)")
    parser.add_argument("--4080s-t2", type=float, help="4080S T2 (ms)")

    # B70
    parser.add_argument("--b70-calcflops", help="B70 calcflops file")
    parser.add_argument("--b70-trace", help="B70 trace.json")
    parser.add_argument("--b70-unitrace", help="B70 unitrace JSON")
    parser.add_argument("--b70-t2", type=float, help="B70 T2 (ms)")

    parser.add_argument("-o", "--output", help="Output markdown file path")
    parser.add_argument("--config", help="Path to hardware_specs.yaml")

    args = parser.parse_args()

    model_info = {
        "model": args.model, "bs": args.bs,
        "precision": args.precision, "test": args.test,
    }

    # Collect platform data
    plat_data_list = []

    platform_args = [
        ("B580", args.b580_calcflops, args.b580_trace,
         args.b580_unitrace, args.b580_t2),
        ("4080", getattr(args, "4080s_calcflops"),
         getattr(args, "4080s_trace"),
         getattr(args, "4080s_unitrace"),
         getattr(args, "4080s_t2")),
        ("G31", args.b70_calcflops, args.b70_trace,
         args.b70_unitrace, args.b70_t2),
    ]

    for plat_key, calcflops, trace, unitrace, t2 in platform_args:
        if not calcflops or not trace:
            continue

        print(f"Processing {plat_key}...")
        spec = load_platform_specs(plat_key, config_path=args.config)
        data = collect_platform_data(
            plat_key, spec, calcflops, trace, unitrace, t2)
        plat_data_list.append(data)
        print(f"  T1={data['t1']:.3f}ms, T2={_fmt(data['t2'])}ms, "
              f"R={_fmt(data['r'])}, T2_device={data['t2_device']:.3f}ms "
              f"({data['actual_source']})")

    if not plat_data_list:
        print("ERROR: No platform data provided. Use --b580-calcflops etc.")
        sys.exit(1)

    # Generate report
    report_lines = generate_report(plat_data_list, model_info)
    report_text = "\n".join(report_lines) + "\n"

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"\nReport written to {args.output}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
