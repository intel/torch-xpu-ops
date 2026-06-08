#!/usr/bin/env python3
"""
Generate eager-mode T1/T2/R reports for all models in a session.

Usage:
    python scripts/generate_all_eager_reports.py \
        --session raw_logs/b70_vs_4080s_fp16_eager \
        --output reports/b70_vs_4080s_fp16_eager \
        --config config/hardware_specs.yaml
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_t2(log_path):
    """Extract T2 (GPU Time per batch) from rcpi1-ins0.log."""
    with open(log_path) as f:
        for line in f:
            match = re.search(r"GPU Time per batch:\s+([\d.]+)\s+milliseconds", line)
            if match:
                return float(match.group(1))
    return None


def find_unitrace_json(unitrace_dir):
    """Find the python.*.json file in the unitrace directory."""
    for f in os.listdir(unitrace_dir):
        if f.startswith("python.") and f.endswith(".json"):
            return os.path.join(unitrace_dir, f)
    return None


def get_model_info(session_dir):
    """Get model info from metadata.json."""
    metadata_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate eager-mode T1/T2/R reports for all models"
    )
    parser.add_argument("--session", type=str, required=True,
                        help="Session directory containing model subdirs (e.g. raw_logs/b70_vs_4080s_fp16_eager)")
    parser.add_argument("--output", type=str,
                        help="Output directory for reports (default: reports/<session_name>)")
    parser.add_argument("--config", type=str, default="config/hardware_specs.yaml",
                        help="Path to hardware_specs.yaml")
    parser.add_argument("--scripts-dir", type=str, default="scripts",
                        help="Directory containing generate_report.py")
    args = parser.parse_args()

    session_dir = args.session
    output_dir = args.output or f"reports/{os.path.basename(session_dir)}/models"
    precision = "fp16"
    test_mode = "eval"

    os.makedirs(output_dir, exist_ok=True)

    # Get metadata
    metadata = get_model_info(session_dir)

    # Discover models by looking at directories
    model_dirs = []
    for entry in sorted(os.listdir(session_dir)):
        entry_path = os.path.join(session_dir, entry)
        if os.path.isdir(entry_path) and entry != ".git":
            # Check if it has at least t1 and one profiler
            if os.path.isdir(os.path.join(entry_path, "t1")):
                model_dirs.append(entry)

    print(f"Found {len(model_dirs)} models with T1 data")
    print(f"Output directory: {output_dir}")
    print(f"Config: {args.config}")
    print("=" * 60)

    results = []

    for model_name in model_dirs:
        model_dir = os.path.join(session_dir, model_name)
        print(f"\n--- {model_name} ---")

        # Check required files
        t1_file = os.path.join(model_dir, "t1", "rcpi1-ins0.log")
        if not os.path.exists(t1_file):
            print(f"  SKIP: No T1 file")
            continue

        # Build command
        cmd = [
            sys.executable, os.path.join(args.scripts_dir, "generate_report.py"),
            "--model", model_name,
            "--precision", "fp16",
            "--test", "eval",
            "--config", args.config,
            "-o", os.path.join(output_dir, f"{model_name}_report.md"),
        ]

        # Get batch size from metadata
        bs = "1"
        if metadata:
            for m in metadata.get("models", []):
                if m.get("short_name") == model_name:
                    bs = str(m.get("batch_size", 1))
                    break
        cmd.extend(["--bs", bs])

        # B70 (G31) data
        b70_trace = os.path.join(model_dir, "xpu_profiler", "timeline", "trace.json")
        b70_t2_log = os.path.join(model_dir, "xpu_t2", "rcpi1-ins0.log")
        unitrace_dir = os.path.join(model_dir, "unitrace")

        if os.path.exists(b70_trace) and os.path.exists(b70_t2_log):
            b70_t2 = extract_t2(b70_t2_log)
            if b70_t2:
                cmd.extend([
                    "--b70-calcflops", t1_file,
                    "--b70-trace", b70_trace,
                    "--b70-t2", str(b70_t2),
                ])
                # Add unitrace if available
                if os.path.isdir(unitrace_dir):
                    unitrace_file = find_unitrace_json(unitrace_dir)
                    if unitrace_file:
                        cmd.extend(["--b70-unitrace", unitrace_file])
                print(f"  B70: T2={b70_t2:.3f}ms, trace=OK, unitrace={'OK' if '--b70-unitrace' in cmd else 'N/A'}")
            else:
                print(f"  B70: SKIP (no T2 value in log)")
        else:
            print(f"  B70: SKIP (missing trace or T2 log)")

        # 4080S data
        cuda_trace = os.path.join(model_dir, "cuda_profiler", "timeline", "trace.json")
        cuda_t2_log = os.path.join(model_dir, "cuda_2", "rcpi1-ins0.log")
        # Fix: might be cuda_t2 not cuda_2
        if not os.path.exists(cuda_t2_log):
            cuda_t2_log = os.path.join(model_dir, "cuda_t2", "rcpi1-ins0.log")

        if os.path.exists(cuda_trace) and os.path.exists(cuda_t2_log):
            cuda_t2 = extract_t2(cuda_t2_log)
            if cuda_t2:
                cmd.extend([
                    "--4080s-calcflops", t1_file,
                    "--4080s-trace", cuda_trace,
                    "--4080s-t2", str(cuda_t2),
                ])
                print(f"  4080S: T2={cuda_t2:.3f}ms, trace=OK")
            else:
                print(f"  4080S: SKIP (no T2 value in log)")
        else:
            print(f"  4080S: SKIP (missing trace or T2 log)")

        # Run generate_report.py
        print(f"  Running report generation...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse R values from output
            r_values = {}
            for line in result.stdout.split("\n"):
                if "T1=" in line and "R=" in line:
                    r_match = re.search(r"R=([\d.]+)", line)
                    platform_match = re.search(r"Processing (\w+)", prev_line) if 'prev_line' in dir() else None
                    if r_match:
                        r_val = float(r_match.group(1))
                        if "4080" in line or (prev_line and "4080" in prev_line):
                            r_values["4080"] = r_val
                        elif "G31" in line or (prev_line and "G31" in prev_line):
                            r_values["G31"] = r_val
                prev_line = line

            print(f"  OK: {os.path.join(output_dir, model_name + '_report.md')}")
            if result.stdout.strip():
                # Print key lines from output
                for line in result.stdout.strip().split("\n"):
                    if "T1=" in line or "Processing" in line:
                        print(f"    {line.strip()}")
            results.append({"model": model_name, "status": "OK", "stdout": result.stdout})
        else:
            print(f"  FAILED: {result.stderr[:200]}")
            results.append({"model": model_name, "status": "FAILED", "stderr": result.stderr})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAILED")
    print(f"  Reports generated: {ok}")
    print(f"  Failed: {fail}")
    if fail > 0:
        print(f"  Failed models: {[r['model'] for r in results if r['status'] == 'FAILED']}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
