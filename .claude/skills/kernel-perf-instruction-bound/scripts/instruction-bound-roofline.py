#!/usr/bin/env python3
"""
Compute per-pipe instruction roofline from unitrace CSVs.

Usage:
    python instruction-bound-roofline.py \
        --compute-basic cb.csv \
        --vector-engine ve.csv \
        --kernel AvgPool2dScalarKernel \
        --xves 160 \
        --peak-bw 456
"""

import argparse
import csv
import sys
from collections import defaultdict


def parse_unitrace_csv(path: str, target_kernel: str):
    """Return a dict {metric_name: [values]} for rows matching target_kernel."""
    table = defaultdict(list)
    with open(path, newline="") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    header = None
    header_raw = None
    for ln in lines:
        if ln.startswith("Kernel,GlobalInstanceId"):
            header_raw = ln
            header = [col.split("[")[0] for col in ln.split(",")]
            continue
        if header is None:
            continue
        if not ln.startswith(f'"{target_kernel}['):
            continue
        parts = list(csv.reader([ln]))[0]
        if len(parts) != len(header):
            continue
        for name, val in zip(header, parts):
            if name in ("Kernel", "GlobalInstanceId"):
                continue
            try:
                table[name].append(float(val))
            except ValueError:
                pass
    return table, header_raw


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="XPU instruction-bound roofline from unitrace")
    parser.add_argument("--compute-basic", required=True, help="ComputeBasic CSV")
    parser.add_argument("--vector-engine", required=True, help="VectorEngineProfile CSV")
    parser.add_argument("--kernel", required=True, help="Exact kernel functor name")
    parser.add_argument("--xves", type=int, required=True, help="Number of XVEs on the device")
    parser.add_argument("--peak-bw", type=float, required=True, help="Peak DRAM BW in GB/s")
    args = parser.parse_args()

    cb, cb_header = parse_unitrace_csv(args.compute_basic, args.kernel)
    ve, ve_header = parse_unitrace_csv(args.vector_engine, args.kernel)

    if not cb:
        print(f"Kernel {args.kernel} not found in {args.compute_basic}", file=sys.stderr)
        sys.exit(1)
    if not ve:
        print(f"Kernel {args.kernel} not found in {args.vector_engine}", file=sys.stderr)
        sys.exit(1)

    def get_one(table, *names):
        for name in names:
            if name in table and table[name]:
                return avg(table[name])
        return None

    gpu_time_ns = get_one(cb, "GpuTime")
    mem_read = get_one(cb, "GPU_MEMORY_BYTE_READ")
    mem_write = get_one(cb, "GPU_MEMORY_BYTE_WRITE")
    freq_mhz = get_one(cb, "AvgGpuCoreFrequencyMHz") or get_one(ve, "AvgGpuCoreFrequencyMHz")

    alu0_all = get_one(ve, "XVE_INST_EXECUTED_ALU0_ALL")
    alu1_all = get_one(ve, "XVE_INST_EXECUTED_ALU1_ALL")
    alu2_all = get_one(ve, "XVE_INST_EXECUTED_ALU2_ALL")
    math = get_one(ve, "XVE_INST_EXECUTED_MATH")
    send_all = get_one(ve, "XVE_INST_EXECUTED_SEND_ALL")
    ctrl_all = get_one(ve, "XVE_INST_EXECUTED_CONTROL_ALL")
    int64 = get_one(ve, "XVE_INST_EXECUTED_INT64")
    bitconv = get_one(ve, "XVE_INST_EXECUTED_BITCONV")

    if gpu_time_ns is None or freq_mhz is None or mem_read is None:
        print("Missing required metrics; check column names.", file=sys.stderr)
        sys.exit(1)

    peak_slots_per_sec = args.xves * freq_mhz * 1e6

    def slot_time_ms(total_events):
        if total_events is None:
            return None
        return total_events / peak_slots_per_sec * 1e3

    T_actual = gpu_time_ns / 1e6  # ms
    T_mem = (mem_read + mem_write) / (args.peak_bw * 1e9) * 1e3  # ms

    T_alu0 = slot_time_ms(alu0_all)
    T_alu1_serial = slot_time_ms((alu1_all or 0) + (math or 0))
    T_alu1_par = slot_time_ms(max(alu1_all or 0, math or 0))
    T_send = slot_time_ms(send_all)
    T_ctrl = slot_time_ms(ctrl_all)

    compute_candidates = [
        ("ALU0", T_alu0),
        ("ALU1 (serial)", T_alu1_serial),
        ("SEND", T_send),
        ("CONTROL", T_ctrl),
    ]
    if alu2_all:
        compute_candidates.append(("ALU2", slot_time_ms(alu2_all)))

    T_compute_val = max(v for _, v in compute_candidates if v is not None)
    T_compute_name = [n for n, v in compute_candidates if v == T_compute_val][0]
    T_lower = max(T_mem, T_compute_val)

    print(f"Kernel        : {args.kernel}")
    print(f"XVEs          : {args.xves}")
    print(f"Freq (MHz)    : {freq_mhz:.0f}")
    print(f"Peak BW (GB/s): {args.peak_bw}")
    print()
    print(f"T_actual (ms)            : {T_actual:.3f}")
    print(f"T_mem    (ms)            : {T_mem:.3f}")
    print(f"T_compute (ms)           : {T_compute_val:.3f}  ({T_compute_name})")
    print(f"T_ALU0 (ms)              : {T_alu0:.3f}")
    print(f"T_ALU1_serial (ms)       : {T_alu1_serial:.3f}")
    print(f"T_ALU1_parallel (ms)     : {T_alu1_par:.3f}")
    print(f"T_SEND (ms)              : {T_send:.3f}")
    print(f"T_CONTROL (ms)           : {T_ctrl:.3f}")
    print(f"T_lower_bound (ms)       : {T_lower:.3f}")
    print(f"T_actual / T_lower_bound : {T_actual / T_lower:.2f}x")
    print()
    # Real kernels do not perfectly overlap compute with memory. Use a
    # conservative 0.8x threshold: if compute is within 80% of memory time,
    # instruction reduction can still help.
    OVERLAP_THRESHOLD = 0.8
    is_compute_bound = T_compute_val >= OVERLAP_THRESHOLD * T_mem

    if not is_compute_bound:
        bound = "memory-bound"
    elif T_compute_name.startswith("ALU0"):
        bound = "ALU0 instruction-bound"
    elif T_compute_name.startswith("ALU1"):
        bound = "ALU1 instruction-bound"
    else:
        bound = f"{T_compute_name.split()[0]}-bound"
    print(f"Verdict: {bound} (overlap threshold = {OVERLAP_THRESHOLD})")

    print("\nPer-element slots (approx):")
    print(f"  ALU0_ALL : {alu0_all or 0:.0f}")
    print(f"  ALU1_ALL : {alu1_all or 0:.0f}")
    print(f"  MATH     : {math or 0:.0f}")
    print(f"  SEND_ALL : {send_all or 0:.0f}")
    print(f"  INT64    : {int64 or 0:.0f}")
    print(f"  BITCONV  : {bitconv or 0:.0f}")


if __name__ == "__main__":
    main()
