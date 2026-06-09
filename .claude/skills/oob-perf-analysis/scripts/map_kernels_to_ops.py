#!/usr/bin/env python3
"""
map_kernels_to_ops.py

Map unitrace GPU kernel durations to top-level aten:: ops using profiler
trace.json as the bridge.  This enables per-op "projection vs actual" analysis.

Approach
--------
1. trace.json kernel events (cat="kernel") each carry an "External id" that
   links to the cpu_op that launched them.
2. We build a top-level-op index so every kernel can be attributed to the
   outermost aten:: op in the call tree.
3. unitrace GPU kernels are matched 1:1 by execution order (both lists sorted
   by timestamp).  Kernel-name verification (strip unitrace's [SIMDxx ...]
   suffix) catches ordering mismatches early.
4. Per-op aggregation sums unitrace durations, giving the "actual GPU time"
   each top-level op consumed.

NOTE: gpu_memcpy events in trace.json are excluded because unitrace only
captures compute kernels.

Usage
-----
    python map_kernels_to_ops.py trace.json unitrace.json [--top 30]
    python map_kernels_to_ops.py trace.json unitrace.json --detail
    python map_kernels_to_ops.py trace.json unitrace.json --csv output.csv
"""

import json
import re
import argparse
import csv
import sys


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _load_trace_kernels_and_ops(trace_path):
    """Extract kernel events and cpu_op events from trace.json.

    Returns
    -------
    kernels : list[dict]
        Kernel events (cat="kernel") sorted by ts.  Each has keys:
        name, ts, dur_us, ext_id
    cpu_ops : list[dict]
        cpu_op events sorted by ts.  Each has keys:
        name, ts, end, dur_us, ext_id, input_dims, input_strides
    """
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"]

    kernels = []
    cpu_ops = []

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "kernel":
            kernels.append({
                "name": e.get("name", ""),
                "ts": e["ts"],
                "dur_us": e.get("dur", 0),
                "ext_id": e.get("args", {}).get("External id"),
            })
        elif cat == "cpu_op":
            name = e.get("name", "")
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            cpu_ops.append({
                "name": name,
                "ts": ts,
                "end": ts + dur,
                "dur_us": dur,
                "ext_id": e.get("args", {}).get("External id"),
                "input_dims": str(e.get("args", {}).get("Input Dims", "")),
                "input_strides": str(e.get("args", {}).get("Input Strides", "")),
            })

    kernels.sort(key=lambda e: e["ts"])
    cpu_ops.sort(key=lambda e: e["ts"])
    return kernels, cpu_ops


def _build_toplevel_op_index(cpu_ops):
    """Build index of all aten:: ops and map ext_id to owning op.

    Each op gets its own entry.  ext_id maps to the op that owns it (the
    innermost dispatch op), NOT to an outermost parent.  GPU kernels reference
    the innermost op's ext_id, so this gives correct per-op attribution.

    Wrapper ops (aten::linear, aten::matmul, etc.) will accumulate 0 kernel
    time because no GPU kernels reference their ext_id directly.

    Returns
    -------
    ext_id_to_op : dict[int, dict]
        Maps External id -> the aten:: op that owns this ext_id.
    all_ops : list[dict]
        All aten:: ops in execution order.
    """
    aten_ops = [op for op in cpu_ops if op["name"].startswith("aten::")]
    aten_ops.sort(key=lambda e: e["ts"])

    all_ops = []
    ext_id_to_op = {}

    for i, op in enumerate(aten_ops):
        entry = {
            "name": op["name"],
            "input_dims": op["input_dims"],
            "input_strides": op["input_strides"],
            "ts": op["ts"],
            "end": op["end"],
            "cpu_dur_us": op["dur_us"],
            "seq_idx": i,
        }
        all_ops.append(entry)
        if op["ext_id"] is not None:
            ext_id_to_op[op["ext_id"]] = entry

    return ext_id_to_op, all_ops


def _load_unitrace_kernels(unitrace_path):
    """Load unitrace GPU kernels (excluding ze* runtime events), sorted by ts.

    Returns list of dicts: name, dur_us, ts_us
    """
    with open(unitrace_path) as f:
        data = json.load(f)

    kernels = []
    for e in data["traceEvents"]:
        if not isinstance(e, dict) or e.get("ph") != "X" or "dur" not in e:
            continue
        name = e.get("name", "")
        if name.startswith("ze"):
            continue
        kernels.append({
            "name": name,
            "dur_us": e["dur"],
            "ts_us": e["ts"],
        })
    kernels.sort(key=lambda e: e["ts_us"])
    return kernels


_SIMD_SUFFIX = re.compile(r"\[SIMD\d+\s+\{[^}]*\}\s+\{[^}]*\}\]$")


def _strip_simd_suffix(name):
    """Remove unitrace's [SIMDxx {a; b; c} {d; e; f}] suffix."""
    return _SIMD_SUFFIX.sub("", name).strip()


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

def map_kernels_to_ops(trace_path, unitrace_path, strict=False):
    """Map unitrace kernel durations to top-level aten:: ops.

    Parameters
    ----------
    trace_path : str
        Path to torch profiler trace.json.
    unitrace_path : str
        Path to unitrace JSON (python.<pid>.json).
    strict : bool
        If True, raise on kernel count or name mismatch.  Default: warn.

    Returns
    -------
    mapped : list[dict]
        One entry per matched kernel pair.  Keys:
        trace_kernel_name, unitrace_kernel_name, unitrace_dur_us,
        trace_dur_us, op_name, op_input_dims, op_seq_idx
    toplevel_ops : list[dict]
        Top-level ops enriched with 'unitrace_dur_us' (sum of mapped kernels).
    mismatches : list[dict]
        Entries where kernel names did not match (for diagnostics).
    """
    trace_kernels, cpu_ops = _load_trace_kernels_and_ops(trace_path)
    ext_id_to_op, all_ops = _build_toplevel_op_index(cpu_ops)
    unitrace_kernels = _load_unitrace_kernels(unitrace_path)

    # --- Count check ---
    n_trace = len(trace_kernels)
    n_uni = len(unitrace_kernels)
    if n_trace != n_uni:
        msg = (f"Kernel count mismatch: trace.json has {n_trace} kernel events, "
               f"unitrace has {n_uni} GPU kernels")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    n_match = min(n_trace, n_uni)

    # Initialize unitrace accumulator on each op
    for op in all_ops:
        op["unitrace_dur_us"] = 0
        op["unitrace_kernel_count"] = 0
        op["unitrace_kernel_names"] = []

    mapped = []
    mismatches = []

    for i in range(n_match):
        tk = trace_kernels[i]
        uk = unitrace_kernels[i]

        # Verify kernel name match
        uk_base = _strip_simd_suffix(uk["name"])
        name_ok = (tk["name"] == uk_base)

        # Find owning op via External id
        tl_op = ext_id_to_op.get(tk["ext_id"])
        op_name = tl_op["name"] if tl_op else "UNKNOWN"
        op_dims = tl_op["input_dims"] if tl_op else ""
        op_strides = tl_op["input_strides"] if tl_op else ""
        op_idx = tl_op["seq_idx"] if tl_op else -1

        entry = {
            "idx": i,
            "trace_kernel_name": tk["name"],
            "unitrace_kernel_name": uk["name"],
            "unitrace_dur_us": uk["dur_us"],
            "trace_dur_us": tk["dur_us"],
            "op_name": op_name,
            "op_input_dims": op_dims,
            "op_input_strides": op_strides,
            "op_seq_idx": op_idx,
            "name_match": name_ok,
        }
        mapped.append(entry)

        if not name_ok:
            mismatches.append(entry)

        # Accumulate on top-level op
        if tl_op is not None:
            tl_op["unitrace_dur_us"] += uk["dur_us"]
            tl_op["unitrace_kernel_count"] += 1
            tl_op["unitrace_kernel_names"].append(uk["name"])

    return mapped, all_ops, mismatches


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_per_op_summary(toplevel_ops, top_n=30):
    """Print per-top-level-op summary of unitrace GPU time."""
    total_ut = sum(op["unitrace_dur_us"] for op in toplevel_ops)

    # Sort by unitrace GPU time descending
    ranked = sorted(toplevel_ops, key=lambda o: o["unitrace_dur_us"], reverse=True)

    print(f"\n{'=' * 140}")
    print(f"{'#':>4} {'OP NAME':<40} {'UNITRACE(ms)':>12} {'UT%':>7} "
          f"{'CPU(ms)':>10} {'Kernels':>8}  {'Input Dims'}")
    print(f"{'-' * 140}")

    for i, op in enumerate(ranked[:top_n]):
        if op["unitrace_dur_us"] == 0:
            continue
        pct = op["unitrace_dur_us"] / total_ut * 100 if total_ut > 0 else 0
        dims = op["input_dims"][:50]
        print(f"{op['seq_idx']:>4} {op['name']:<40} "
              f"{op['unitrace_dur_us'] / 1000:>12.3f} {pct:>6.1f}% "
              f"{op['cpu_dur_us'] / 1000:>10.3f} {op['unitrace_kernel_count']:>8}  {dims}")

    print(f"{'=' * 140}")
    n_with_kernels = sum(1 for op in toplevel_ops if op["unitrace_dur_us"] > 0)
    print(f"Total unitrace GPU time: {total_ut / 1000:.3f} ms  "
          f"({n_with_kernels} ops with kernels out of {len(toplevel_ops)} top-level ops)")
    print()


def print_per_op_detail(toplevel_ops):
    """Print every top-level op in execution order with its unitrace kernels."""
    print(f"\n{'=' * 140}")
    print("Per-op detail (execution order)")
    print(f"{'=' * 140}")

    for op in toplevel_ops:
        if op["unitrace_kernel_count"] == 0:
            print(f"\n[{op['seq_idx']:>3}] {op['name']:<40} "
                  f"(no GPU kernels)  dims={op['input_dims']}")
        else:
            print(f"\n[{op['seq_idx']:>3}] {op['name']:<40} "
                  f"unitrace={op['unitrace_dur_us'] / 1000:.3f}ms  "
                  f"kernels={op['unitrace_kernel_count']}  "
                  f"dims={op['input_dims']}")
            for kn in op["unitrace_kernel_names"]:
                short = kn[:100] + "..." if len(kn) > 100 else kn
                print(f"        {short}")


def write_csv(mapped, toplevel_ops, csv_path):
    """Write per-kernel mapping to CSV."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_idx", "trace_kernel_name", "unitrace_kernel_name",
                    "unitrace_dur_us", "trace_dur_us",
                    "op_name", "op_input_dims", "op_input_strides",
                    "op_seq_idx", "name_match"])
        for m in mapped:
            w.writerow([m["idx"], m["trace_kernel_name"], m["unitrace_kernel_name"],
                        m["unitrace_dur_us"], m["trace_dur_us"],
                        m["op_name"], m["op_input_dims"], m["op_input_strides"],
                        m["op_seq_idx"], m["name_match"]])

    # Also write per-op summary CSV
    op_csv = csv_path.replace(".csv", "_per_op.csv")
    with open(op_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_seq_idx", "op_name", "input_dims", "input_strides",
                    "unitrace_dur_us", "cpu_dur_us", "kernel_count"])
        for op in toplevel_ops:
            w.writerow([op["seq_idx"], op["name"], op["input_dims"],
                        op["input_strides"],
                        op["unitrace_dur_us"], op["cpu_dur_us"],
                        op["unitrace_kernel_count"]])

    print(f"Wrote {csv_path} ({len(mapped)} rows)")
    print(f"Wrote {op_csv} ({len(toplevel_ops)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map unitrace kernel durations to top-level aten:: ops")
    parser.add_argument("trace_file", help="Path to torch profiler trace.json")
    parser.add_argument("unitrace_file", help="Path to unitrace JSON (python.<pid>.json)")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N ops (default: 30)")
    parser.add_argument("--detail", action="store_true",
                        help="Show per-op detail with kernel names")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write per-kernel mapping to CSV")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on kernel count or name mismatch")
    args = parser.parse_args()

    print(f"Loading trace from {args.trace_file} ...")
    print(f"Loading unitrace from {args.unitrace_file} ...")

    mapped, toplevel_ops, mismatches = map_kernels_to_ops(
        args.trace_file, args.unitrace_file, strict=args.strict)

    print(f"\nMatched {len(mapped)} kernel pairs")
    if mismatches:
        print(f"WARNING: {len(mismatches)} kernel name mismatches:")
        for m in mismatches[:5]:
            print(f"  [{m['idx']}] trace: {m['trace_kernel_name'][:60]}")
            print(f"       unitrace: {m['unitrace_kernel_name'][:60]}")
        if len(mismatches) > 5:
            print(f"  ... ({len(mismatches) - 5} more)")

    print_per_op_summary(toplevel_ops, top_n=args.top)

    if args.detail:
        print_per_op_detail(toplevel_ops)

    if args.csv:
        write_csv(mapped, toplevel_ops, args.csv)


if __name__ == "__main__":
    main()
