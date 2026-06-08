#!/usr/bin/env python3
"""
parse_unitrace.py

Parse unitrace chrome-kernel-logging JSON output and print per-kernel summary.
Filters out Level Zero runtime events (ze*) to show only GPU compute kernels.

Usage:
    python parse_unitrace.py python.1234.json --top 20 --sort-by gpu_time
"""

import json
import argparse
from collections import defaultdict


def parse_unitrace(path):
    """Parse unitrace JSON and return GPU kernel events (excluding ze* runtime events).
    
    Returns list of dicts: name, dur_us, ts_us
    """
    with open(path) as f:
        data = json.load(f)

    events = [e for e in data["traceEvents"]
              if isinstance(e, dict) and e.get("ph") == "X" and "dur" in e]

    # Split into GPU kernels vs runtime events
    gpu_kernels = []
    for e in events:
        name = e.get("name", "")
        if name.startswith("ze"):
            continue  # Skip Level Zero runtime calls
        gpu_kernels.append({
            "name": name,
            "dur_us": e["dur"],
            "ts_us": e["ts"],
        })

    gpu_kernels.sort(key=lambda e: e["ts_us"])
    return gpu_kernels


def aggregate_kernels(kernels):
    """Aggregate by kernel name."""
    agg = defaultdict(lambda: {"dur": 0, "count": 0})
    for k in kernels:
        agg[k["name"]]["dur"] += k["dur_us"]
        agg[k["name"]]["count"] += 1
    return agg


def print_summary(agg, total_gpu_us, sort_by="gpu_time", top_n=20):
    """Print aggregate summary table."""
    if sort_by == "gpu_time":
        items = sorted(agg.items(), key=lambda x: x[1]["dur"], reverse=True)
    elif sort_by == "count":
        items = sorted(agg.items(), key=lambda x: x[1]["count"], reverse=True)
    else:
        items = sorted(agg.items(), key=lambda x: x[1]["dur"], reverse=True)

    print()
    print("%-70s %10s %7s %6s" % ("KERNEL NAME", "DUR(ms)", "DUR%", "COUNT"))
    print("-" * 100)
    for name, v in items[:top_n]:
        pct = v["dur"] / total_gpu_us * 100 if total_gpu_us > 0 else 0
        short = name[:67] + "..." if len(name) > 70 else name
        print("%-70s %10.3f %6.1f%% %6d" % (short, v["dur"] / 1000, pct, v["count"]))
    print("-" * 100)
    total_count = sum(v["count"] for v in agg.values())
    print("%-70s %10.3f %7s %6d" % ("TOTAL", total_gpu_us / 1000, "100%", total_count))


def print_timeline(kernels, limit=50):
    """Print kernels in time order."""
    print()
    print("%-6s %-70s %10s %15s" % ("#", "KERNEL NAME", "DUR(us)", "TS(us)"))
    print("-" * 110)
    for i, k in enumerate(kernels[:limit]):
        short = k["name"][:67] + "..." if len(k["name"]) > 70 else k["name"]
        print("%-6d %-70s %10.1f %15.1f" % (i + 1, short, k["dur_us"], k["ts_us"]))
    if len(kernels) > limit:
        print("... (%d more)" % (len(kernels) - limit))


def main():
    parser = argparse.ArgumentParser(description="Parse unitrace chrome-kernel-logging JSON")
    parser.add_argument("unitrace_file", help="Path to unitrace JSON (python.<pid>.json)")
    parser.add_argument("--top", type=int, default=20, help="Show top N kernels (default: 20)")
    parser.add_argument("--sort-by", choices=["gpu_time", "count"],
                        default="gpu_time", help="Sort metric (default: gpu_time)")
    parser.add_argument("--timeline", action="store_true",
                        help="Print kernels in time order")
    parser.add_argument("--timeline-limit", type=int, default=50,
                        help="Max kernels to show in timeline (default: 50)")
    args = parser.parse_args()

    print("Loading unitrace from %s..." % args.unitrace_file)
    kernels = parse_unitrace(args.unitrace_file)

    total_gpu_us = sum(k["dur_us"] for k in kernels)
    if len(kernels) > 0:
        span = (kernels[-1]["ts_us"] + kernels[-1]["dur_us"] - kernels[0]["ts_us"]) / 1000
    else:
        span = 0

    print("GPU kernels: %d" % len(kernels))
    print("GPU kernel sum: %.3f ms" % (total_gpu_us / 1000))
    print("Time span (first to last): %.3f ms" % span)

    agg = aggregate_kernels(kernels)
    print_summary(agg, total_gpu_us, sort_by=args.sort_by, top_n=args.top)

    if args.timeline:
        print_timeline(kernels, limit=args.timeline_limit)


if __name__ == "__main__":
    main()
