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
Create flat symlink directories from session-based layout for generate_fleet_summary.py.

Session layout:
    raw_logs/<session>/<model>/
        t1/rcpi1-ins0.log          → calcflops
        xpu_t2/rcpi1-ins0.log      → baseline (B70)
        cuda_t2/rcpi1-ins0.log     → baseline (4080S)
        xpu_profiler/timeline/trace.json → trace (B70)
        cuda_profiler/timeline/trace.json → trace (4080S)
        unitrace/python.*.json     → unitrace (B70)

Flat layout expected by generate_all_reports.py / generate_fleet_summary.py:
    <result_dir>/<model>_bs<N>_baseline.txt
    <result_dir>/<model>_bs<N>_calcflops.txt
    <result_dir>/<model>_bs<N>_trace.json
    <result_dir>/<model>_bs<N>_unitrace.json
"""

import json
import os
import sys

def main():
    session_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_logs/b70_vs_4080s_fp16_eager"
    output_base = sys.argv[2] if len(sys.argv) > 2 else "flat_views"

    # Load metadata for batch sizes
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    model_info = {}
    for m in metadata["models"]:
        model_info[m["short_name"]] = m

    # Create flat view dirs
    b70_dir = os.path.join(output_base, "b70")
    cuda_dir = os.path.join(output_base, "4080s")
    os.makedirs(b70_dir, exist_ok=True)
    os.makedirs(cuda_dir, exist_ok=True)

    session_abs = os.path.abspath(session_dir)

    for model_name in sorted(os.listdir(session_dir)):
        model_dir = os.path.join(session_abs, model_name)
        if not os.path.isdir(model_dir) or model_name == ".git":
            continue
        if model_name not in model_info:
            continue

        bs = model_info[model_name]["batch_size"]
        key = f"{model_name}_bs{bs}"

        # T1 (calcflops) - shared between platforms
        t1_file = os.path.join(model_dir, "t1", "rcpi1-ins0.log")
        if not os.path.exists(t1_file):
            print(f"  SKIP {model_name}: no T1")
            continue

        # B70 files
        xpu_t2 = os.path.join(model_dir, "xpu_t2", "rcpi1-ins0.log")
        xpu_trace = os.path.join(model_dir, "xpu_profiler", "timeline", "trace.json")
        unitrace_dir = os.path.join(model_dir, "unitrace")
        unitrace_file = None
        if os.path.isdir(unitrace_dir):
            for f in os.listdir(unitrace_dir):
                if f.startswith("python.") and f.endswith(".json"):
                    unitrace_file = os.path.join(unitrace_dir, f)
                    break

        if os.path.exists(xpu_t2) and os.path.exists(xpu_trace):
            _symlink(t1_file, os.path.join(b70_dir, f"{key}_calcflops.txt"))
            _symlink(xpu_t2, os.path.join(b70_dir, f"{key}_baseline.txt"))
            _symlink(xpu_trace, os.path.join(b70_dir, f"{key}_trace.json"))
            if unitrace_file:
                _symlink(unitrace_file, os.path.join(b70_dir, f"{key}_unitrace.json"))
            print(f"  B70:  {key} OK")
        else:
            print(f"  B70:  {key} SKIP (missing trace or T2)")

        # 4080S files
        cuda_t2 = os.path.join(model_dir, "cuda_t2", "rcpi1-ins0.log")
        cuda_trace = os.path.join(model_dir, "cuda_profiler", "timeline", "trace.json")

        if os.path.exists(cuda_t2) and os.path.exists(cuda_trace):
            _symlink(t1_file, os.path.join(cuda_dir, f"{key}_calcflops.txt"))
            _symlink(cuda_t2, os.path.join(cuda_dir, f"{key}_baseline.txt"))
            _symlink(cuda_trace, os.path.join(cuda_dir, f"{key}_trace.json"))
            print(f"  4080: {key} OK")
        else:
            print(f"  4080: {key} SKIP (missing trace or T2)")

    print("\nFlat views created:")
    print(f"  B70:  {b70_dir}/ ({len(os.listdir(b70_dir))} files)")
    print(f"  4080: {cuda_dir}/ ({len(os.listdir(cuda_dir))} files)")


def _symlink(src, dst):
    """Create symlink, removing existing."""
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)


if __name__ == "__main__":
    main()
