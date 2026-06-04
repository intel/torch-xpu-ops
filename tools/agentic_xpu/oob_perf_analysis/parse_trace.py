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
parse_trace.py

Parse torch profiler trace.json and print per-op summary table.
Shows GPU time, CPU time, kernel count, and kernel names for each aten:: op.

Usage:
    python parse_trace.py timeline/trace.json --top 30 --sort-by gpu_time
    python parse_trace.py timeline/trace.json --detail
"""

import json
import argparse
from collections import defaultdict


def parse_trace(path):
    """Parse trace.json and return per-op events with GPU/CPU times and kernel info.
    
    Returns list of dicts: name, cpu_dur_us, gpu_dur_us, ext_id, input_dims, kernel_names
    """
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"]

    cpu_ops = []
    device_events = []

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "cpu_op":
            cpu_ops.append(e)
        elif cat in ("kernel", "gpu_memcpy"):
            device_events.append(e)

    # Build external id -> (total device time, kernel names)
    ext_map = defaultdict(lambda: {"dur": 0, "kernels": []})
    for d in device_events:
        ext_id = d.get("args", {}).get("External id")
        if ext_id is not None:
            ext_map[ext_id]["dur"] += d.get("dur", 0)
            ext_map[ext_id]["kernels"].append(d.get("name", ""))

    cpu_ops.sort(key=lambda e: e["ts"])

    # Filter to aten:: ops, identify top-level ops (not nested inside another aten:: op)
    aten_ops = []
    for op in cpu_ops:
        name = op.get("name", "")
        if not name.startswith("aten::"):
            continue
        ts = op.get("ts", 0)
        dur = op.get("dur", 0)
        ext_id = op.get("args", {}).get("External id")
        dev = ext_map.get(ext_id, {"dur": 0, "kernels": []}) if ext_id else {"dur": 0, "kernels": []}
        input_dims = op.get("args", {}).get("Input Dims", "")
        aten_ops.append({
            "name": name,
            "ts": ts,
            "end": ts + dur,
            "cpu_dur_us": dur,
            "gpu_dur_us": dev["dur"],
            "ext_id": ext_id,
            "input_dims": str(input_dims),
            "kernel_names": dev["kernels"],
        })

    # Merge nested ops into parent (top-level) ops
    result = []
    parent_stack = []  # (end_ts, result_index)

    for op in aten_ops:
        while parent_stack and parent_stack[-1][0] <= op["ts"]:
            parent_stack.pop()

        if parent_stack:
            # Nested — merge into parent
            parent_idx = parent_stack[-1][1]
            result[parent_idx]["gpu_dur_us"] += op["gpu_dur_us"]
            result[parent_idx]["kernel_names"].extend(op["kernel_names"])
            parent_stack.append((op["end"], parent_idx))
        else:
            # Top-level op
            idx = len(result)
            result.append({
                "name": op["name"],
                "cpu_dur_us": op["cpu_dur_us"],
                "gpu_dur_us": op["gpu_dur_us"],
                "ext_id": op["ext_id"],
                "input_dims": op["input_dims"],
                "kernel_names": list(op["kernel_names"]),
            })
            parent_stack.append((op["end"], idx))

    return result


def aggregate_ops(ops):
    """Aggregate per-invocation ops into per-op-name summary."""
    agg = defaultdict(lambda: {
        "gpu_us": 0, "cpu_us": 0, "count": 0,
        "kernel_count": 0, "kernel_names": set(),
    })
    for op in ops:
        name = op["name"]
        agg[name]["gpu_us"] += op["gpu_dur_us"]
        agg[name]["cpu_us"] += op["cpu_dur_us"]
        agg[name]["count"] += 1
        agg[name]["kernel_count"] += len(op["kernel_names"])
        agg[name]["kernel_names"].update(op["kernel_names"])
    return agg


def print_summary(agg, sort_by="gpu_time", top_n=30):
    """Print aggregate summary table."""
    if sort_by == "gpu_time":
        items = sorted(agg.items(), key=lambda x: x[1]["gpu_us"], reverse=True)
    elif sort_by == "cpu_time":
        items = sorted(agg.items(), key=lambda x: x[1]["cpu_us"], reverse=True)
    elif sort_by == "kernel_count":
        items = sorted(agg.items(), key=lambda x: x[1]["kernel_count"], reverse=True)
    else:
        items = sorted(agg.items(), key=lambda x: x[1]["gpu_us"], reverse=True)

    total_gpu = sum(v["gpu_us"] for v in agg.values())
    total_cpu = sum(v["cpu_us"] for v in agg.values())

    print(f"\n{'='*130}")
    print(f"{'OP NAME':<45} {'GPU(ms)':>10} {'GPU%':>7} {'CPU(ms)':>10} {'Calls':>6} {'Kernels':>8}  {'Kernel Names'}")
    print(f"{'-'*130}")

    for name, v in items[:top_n]:
        gpu_pct = v["gpu_us"] / total_gpu * 100 if total_gpu > 0 else 0
        kernel_str = ", ".join(sorted(v["kernel_names"]))
        if len(kernel_str) > 50:
            kernel_str = kernel_str[:47] + "..."
        print(f"{name:<45} {v['gpu_us']/1000:>10.3f} {gpu_pct:>6.1f}% {v['cpu_us']/1000:>10.3f} {v['count']:>6} {v['kernel_count']:>8}  {kernel_str}")

    print(f"{'='*130}")
    print(f"{'TOTAL':<45} {total_gpu/1000:>10.3f} {'100.0%':>7} {total_cpu/1000:>10.3f}")
    print()


def print_detail(ops, top_names, max_invocations=10):
    """Print per-invocation detail for top ops."""
    for name in top_names:
        invocations = [op for op in ops if op["name"] == name]
        print(f"\n--- {name} ({len(invocations)} invocations) ---")
        print(f"  {'#':>4} {'GPU(ms)':>10} {'CPU(ms)':>10} {'Kernels':>8}  {'Input Dims'}")
        for i, op in enumerate(invocations[:max_invocations]):
            print(f"  {i+1:>4} {op['gpu_dur_us']/1000:>10.3f} {op['cpu_dur_us']/1000:>10.3f} {len(op['kernel_names']):>8}  {op['input_dims']}")
        if len(invocations) > max_invocations:
            print(f"  ... ({len(invocations) - max_invocations} more)")


def main():
    parser = argparse.ArgumentParser(description="Parse torch profiler trace.json")
    parser.add_argument("trace_file", help="Path to trace.json")
    parser.add_argument("--top", type=int, default=30, help="Show top N ops (default: 30)")
    parser.add_argument("--sort-by", choices=["gpu_time", "cpu_time", "kernel_count"],
                        default="gpu_time", help="Sort metric (default: gpu_time)")
    parser.add_argument("--detail", action="store_true",
                        help="Show per-invocation detail for top ops")
    args = parser.parse_args()

    print(f"Loading trace from {args.trace_file}...")
    ops = parse_trace(args.trace_file)
    print(f"Found {len(ops)} top-level aten:: ops")

    total_gpu = sum(op["gpu_dur_us"] for op in ops) / 1000
    print(f"Total GPU time (sum of kernels): {total_gpu:.3f} ms")

    agg = aggregate_ops(ops)
    print_summary(agg, sort_by=args.sort_by, top_n=args.top)

    if args.detail:
        # Get top N op names by GPU time
        sorted_names = sorted(agg.keys(), key=lambda n: agg[n]["gpu_us"], reverse=True)
        print_detail(ops, sorted_names[:args.top])


if __name__ == "__main__":
    main()
