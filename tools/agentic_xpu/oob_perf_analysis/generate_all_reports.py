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
generate_all_reports.py

Batch generate cross-platform T1/T2/R markdown reports for all models
that have data on multiple platforms.

Scans result directories, pairs up matching models, extracts T2 from
baseline files, and calls generate_report logic for each.

Usage:
    python generate_all_reports.py \
        --b580-dir /path/to/b580/results \
        --4080s-dir /path/to/4080s/results \
        --output-dir reports/oob300/per_model \
        --config config/hardware_specs.yaml

    # Single platform:
    python generate_all_reports.py \
        --b580-dir /path/to/b580/results \
        --output-dir reports/oob300/per_model

    # Filter to specific models:
    python generate_all_reports.py \
        --b580-dir ... --4080s-dir ... \
        --models resnet18,resnet152,pytorch_unet \
        --output-dir reports/oob300/per_model
"""

import argparse
import os
import re
import sys
import traceback

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from compare_projection_vs_actual import load_platform_specs
from generate_report import collect_platform_data, generate_report


# ---------------------------------------------------------------------------
# Discover models from result directory
# ---------------------------------------------------------------------------

def discover_models(result_dir):
    """Scan a result directory and return dict of {model_key: {model, bs, files}}.

    model_key is e.g. "resnet18_bs512".
    """
    models = {}
    if not result_dir or not os.path.isdir(result_dir):
        return models

    # Find all baseline files: <model>_bs<bs>_baseline.txt
    pattern = re.compile(r"^(.+)_bs(\d+)_baseline\.txt$")
    for fname in os.listdir(result_dir):
        m = pattern.match(fname)
        if not m:
            continue
        model_name = m.group(1)
        bs = m.group(2)
        key = f"{model_name}_bs{bs}"

        baseline_path = os.path.join(result_dir, fname)
        calcflops_path = os.path.join(result_dir, f"{key}_calcflops.txt")
        trace_path = os.path.join(result_dir, f"{key}_trace.json")
        unitrace_path = os.path.join(result_dir, f"{key}_unitrace.json")

        # Must have at least baseline + calcflops + trace
        if not os.path.isfile(calcflops_path) or not os.path.isfile(trace_path):
            continue

        # Extract T2 from baseline
        t2 = _extract_t2(baseline_path)
        if t2 is None:
            continue

        models[key] = {
            "model": model_name,
            "bs": bs,
            "t2": t2,
            "calcflops": calcflops_path,
            "trace": trace_path,
            "unitrace": unitrace_path if os.path.isfile(unitrace_path) else None,
        }

    return models


def _extract_t2(baseline_path):
    """Extract T2 (ms) from baseline file.

    Looks for 'GPU Time per batch: NNN.NNN milliseconds'.
    Returns float or None if not found / FAILED.
    """
    try:
        with open(baseline_path) as f:
            text = f.read()
        if "FAILED" in text:
            return None
        m = re.search(r"GPU Time per batch:\s+([\d.]+)", text)
        if m:
            return float(m.group(1))
        # Also try "CPU Wall Time per batch" as fallback
        m = re.search(r"CPU Wall Time per batch:\s+([\d.]+)", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Batch report generation
# ---------------------------------------------------------------------------

def generate_model_report(key, platform_data, config_path, output_dir,
                          precision="fp16", test_mode="eval"):
    """Generate a single model's cross-platform report.

    platform_data: list of (platform_id, model_info_dict) tuples
    Returns (key, success, error_msg)
    """
    model_info_ref = platform_data[0][1]
    model_info = {
        "model": model_info_ref["model"],
        "bs": model_info_ref["bs"],
        "precision": precision,
        "test": test_mode,
    }

    plat_data_list = []
    for plat_id, info in platform_data:
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
            return (key, False, f"{plat_id}: {e}")

    if not plat_data_list:
        return (key, False, "No platform data collected")

    report_lines = generate_report(plat_data_list, model_info)
    report_text = "\n".join(report_lines) + "\n"

    out_path = os.path.join(
        output_dir,
        f"{key}_{precision}_{test_mode}.md",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)

    # Summary line
    r_strs = []
    for d in plat_data_list:
        r_val = f"{d['r']:.3f}" if d['r'] else "N/A"
        r_strs.append(f"{d['platform']}:R={r_val}")
    summary = ", ".join(r_strs)

    return (key, True, summary)


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate cross-platform T1/T2/R reports")
    parser.add_argument("--b580-dir", help="B580 result directory")
    parser.add_argument("--4080s-dir", help="4080S result directory")
    parser.add_argument("--b70-dir", help="B70 result directory")
    parser.add_argument("--output-dir", default="reports/oob300/per_model",
                        help="Output directory for reports")
    parser.add_argument("--config", help="Path to hardware_specs.yaml")
    parser.add_argument("--models", help="Comma-separated model filter (e.g. resnet18,resnet152)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have a report")
    parser.add_argument("--precision", default="fp16",
                        help="Precision tag (default: fp16)")
    parser.add_argument("--test", default="eval", dest="test_mode",
                        help="Test mode tag: eval or train (default: eval)")
    args = parser.parse_args()

    # Discover models per platform
    platform_dirs = []
    if args.b580_dir:
        platform_dirs.append(("B580", args.b580_dir))
    if getattr(args, "4080s_dir"):
        platform_dirs.append(("4080", getattr(args, "4080s_dir")))
    if args.b70_dir:
        platform_dirs.append(("G31", args.b70_dir))

    if not platform_dirs:
        print("ERROR: Provide at least one platform directory (--b580-dir, --4080s-dir, --b70-dir)")
        sys.exit(1)

    all_models = {}  # platform_id -> {model_key: info}
    for plat_id, result_dir in platform_dirs:
        models = discover_models(result_dir)
        all_models[plat_id] = models
        print(f"[{plat_id}] Found {len(models)} models in {result_dir}")

    # Find models present on ALL platforms
    all_keys = None
    for plat_id, models in all_models.items():
        keys = set(models.keys())
        if all_keys is None:
            all_keys = keys
        else:
            all_keys &= keys

    if not all_keys:
        # Fall back: models on ANY platform
        all_keys = set()
        for models in all_models.values():
            all_keys |= set(models.keys())
        print(f"No models on all platforms; generating single-platform reports for {len(all_keys)} models")

    # Apply model filter
    if args.models:
        filter_names = set(args.models.split(","))
        all_keys = {k for k in all_keys
                    if any(k.startswith(n + "_bs") or k == n for n in filter_names)}

    all_keys = sorted(all_keys)
    print(f"\nGenerating reports for {len(all_keys)} models...\n")

    success = 0
    fail = 0
    for i, key in enumerate(all_keys, 1):
        # Check skip
        if args.skip_existing:
            out_path = os.path.join(args.output_dir,
                                    f"{key}_{args.precision}_{args.test_mode}.md")
            if os.path.isfile(out_path):
                print(f"[{i}/{len(all_keys)}] SKIP (exists): {key}")
                success += 1
                continue

        # Collect platform data for this model
        platform_data = []
        for plat_id, models in all_models.items():
            if key in models:
                platform_data.append((plat_id, models[key]))

        try:
            k, ok, msg = generate_model_report(
                key, platform_data, args.config, args.output_dir,
                precision=args.precision, test_mode=args.test_mode)
            if ok:
                print(f"[{i}/{len(all_keys)}] OK: {key} — {msg}")
                success += 1
            else:
                print(f"[{i}/{len(all_keys)}] FAIL: {key} — {msg}")
                fail += 1
        except Exception as e:
            print(f"[{i}/{len(all_keys)}] ERROR: {key} — {e}")
            traceback.print_exc()
            fail += 1

    print(f"\n{'='*60}")
    print(f"DONE: {success} success, {fail} fail out of {len(all_keys)}")
    print(f"Reports in: {args.output_dir}")


if __name__ == "__main__":
    main()
