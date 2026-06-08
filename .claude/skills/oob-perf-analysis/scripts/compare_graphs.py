#!/usr/bin/env python3
"""Compare calcflops outputs from two devices to detect graph differences.

context_func.py outputs per-op cumulative FLOPs and memory for ALL platforms
regardless of which device runs the model. If the model's computational
graph is identical on both devices, columns 1 (cum_flops) and 2 (cum_memory)
will match exactly. Differences indicate dispatch path divergence
(e.g., SDPA fusion, XPU-specific overrideable ops, decomposition differences).

Usage:
    python scripts/oob300/compare_graphs.py \\
        --dir-a /home2/jianyizh/results_cl/4080s \\
        --dir-b /home2/jianyizh/results_cl/b580 \\
        --label-a CUDA --label-b XPU \\
        [-o output.md]

    # Training mode
    python scripts/oob300/compare_graphs.py \\
        --dir-a /home2/jianyizh/results_training/4080s \\
        --dir-b /home2/jianyizh/results_training/b580 \\
        --label-a CUDA --label-b XPU \\
        --precision bf16 --test train \\
        [-o output.md]
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Op name normalization (subset — keep in sync with compare_projection_vs_actual.py)
# ---------------------------------------------------------------------------

_VIEW_NOOP_OPS = {"aten::reshape", "aten::contiguous", "aten::unbind"}

_NORMALIZE_MAP = {
    "aten::copy_": "aten::clone",
    "aten::convolution_overrideable": "aten::convolution",
    "aten::convolution_backward_overrideable": "aten::convolution_backward",
}

# SDPA variants — normalize all to a common name for comparison
_SDPA_VARIANTS = {
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_fused_attention_overrideable",
    "aten::_scaled_dot_product_math",
}

_SDPA_NORMALIZED = "aten::sdpa_forward"


def normalize_op_name(name):
    """Normalize op name for cross-device comparison."""
    if name in _VIEW_NOOP_OPS:
        return "__view_noop__"
    if name in _SDPA_VARIANTS:
        return _SDPA_NORMALIZED
    return _NORMALIZE_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Calcflops parser (lightweight, device-independent columns only)
# ---------------------------------------------------------------------------

def parse_calcflops_raw(path, iteration=-1):
    """Parse calcflops output, return per-op list with device-independent data.

    Returns list of dicts with keys:
        name_raw, name, flops, memory, args
    """
    all_ops = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("aten::"):
            continue

        parts = line.split("|")
        if len(parts) < 24:
            continue

        name_raw = parts[0]
        cum_flops = int(float(parts[1]))
        cum_mem = int(float(parts[2]))

        args_str = ""
        if len(parts) > 23 and parts[23].startswith("args:"):
            args_str = parts[23].replace("args:", "")

        all_ops.append({
            "name_raw": name_raw,
            "name": normalize_op_name(name_raw),
            "cum_flops": cum_flops,
            "cum_mem": cum_mem,
            "args": args_str,
        })

    if not all_ops:
        return []

    # Detect benchmark iterations by cumulative value resets
    iter_boundaries = [0]
    for i in range(1, len(all_ops)):
        if all_ops[i]["cum_mem"] < all_ops[i - 1]["cum_mem"] * 0.5:
            iter_boundaries.append(i)

    num_iters = len(iter_boundaries)
    iter_boundaries.append(len(all_ops))

    if iteration == -1:
        iteration = num_iters - 1
    if iteration >= num_iters:
        iteration = num_iters - 1

    start = iter_boundaries[iteration]
    end = iter_boundaries[iteration + 1]
    ops = all_ops[start:end]

    # Compute per-op deltas
    result = []
    prev_flops = 0
    prev_mem = 0

    for op in ops:
        delta_flops = op["cum_flops"] - prev_flops
        delta_mem = op["cum_mem"] - prev_mem

        result.append({
            "name_raw": op["name_raw"],
            "name": op["name"],
            "flops": max(0, delta_flops),
            "memory": max(0, delta_mem),
            "args": op["args"],
        })

        prev_flops = op["cum_flops"]
        prev_mem = op["cum_mem"]

    return result


def aggregate_ops(ops):
    """Aggregate per-op data by normalized name.

    Returns dict: name -> {flops, memory, count, raw_names, args_set}
    """
    agg = defaultdict(lambda: {
        "flops": 0, "memory": 0, "count": 0,
        "raw_names": set(), "args_set": set(),
    })
    for op in ops:
        if op["name"] == "__view_noop__":
            continue
        d = agg[op["name"]]
        d["flops"] += op["flops"]
        d["memory"] += op["memory"]
        d["count"] += 1
        d["raw_names"].add(op["name_raw"])
        if op["args"]:
            d["args_set"].add(op["args"])

    return dict(agg)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(data_dir, precision="fp16", test="eval"):
    """Find models in a data directory by matching calcflops files.

    Returns dict: model_key -> calcflops_path
    """
    models = {}
    suffix = "_calcflops.txt"

    if not os.path.isdir(data_dir):
        return models

    for fname in os.listdir(data_dir):
        if not fname.endswith(suffix):
            continue
        key = fname[:-len(suffix)]  # e.g., "nanogpt_bs1024"
        models[key] = os.path.join(data_dir, fname)

    return models


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_model(path_a, path_b):
    """Compare calcflops from two devices for the same model.

    Returns dict with:
        match: bool — True if graphs are functionally identical
        total_flops_a, total_flops_b: int
        total_mem_a, total_mem_b: int
        flops_diff_pct: float — % difference in total FLOPs
        common_ops: list of dicts for ops present in both
        only_a: list of dicts for ops only in A
        only_b: list of dicts for ops only in B
        diff_ops: list of dicts for common ops with significant differences
    """
    ops_a = parse_calcflops_raw(path_a)
    ops_b = parse_calcflops_raw(path_b)

    agg_a = aggregate_ops(ops_a)
    agg_b = aggregate_ops(ops_b)

    all_ops = set(agg_a.keys()) | set(agg_b.keys())

    total_flops_a = sum(v["flops"] for v in agg_a.values())
    total_flops_b = sum(v["flops"] for v in agg_b.values())
    total_mem_a = sum(v["memory"] for v in agg_a.values())
    total_mem_b = sum(v["memory"] for v in agg_b.values())

    if total_flops_a > 0:
        flops_diff_pct = abs(total_flops_a - total_flops_b) / total_flops_a * 100
    else:
        flops_diff_pct = 0.0

    if total_mem_a > 0:
        mem_diff_pct = abs(total_mem_a - total_mem_b) / total_mem_a * 100
    else:
        mem_diff_pct = 0.0

    only_a = []
    only_b = []
    common_ops = []
    diff_ops = []

    for op in sorted(all_ops):
        in_a = op in agg_a
        in_b = op in agg_b

        if in_a and not in_b:
            only_a.append({
                "name": op,
                "flops": agg_a[op]["flops"],
                "memory": agg_a[op]["memory"],
                "count": agg_a[op]["count"],
                "raw_names": agg_a[op]["raw_names"],
            })
        elif in_b and not in_a:
            only_b.append({
                "name": op,
                "flops": agg_b[op]["flops"],
                "memory": agg_b[op]["memory"],
                "count": agg_b[op]["count"],
                "raw_names": agg_b[op]["raw_names"],
            })
        else:
            da = agg_a[op]
            db = agg_b[op]
            flops_match = da["flops"] == db["flops"]
            mem_match = da["memory"] == db["memory"]
            count_match = da["count"] == db["count"]

            entry = {
                "name": op,
                "flops_a": da["flops"],
                "flops_b": db["flops"],
                "memory_a": da["memory"],
                "memory_b": db["memory"],
                "count_a": da["count"],
                "count_b": db["count"],
                "match": flops_match and mem_match and count_match,
            }
            common_ops.append(entry)

            if not entry["match"]:
                # Compute diff details
                if da["flops"] > 0:
                    flops_pct = (db["flops"] - da["flops"]) / da["flops"] * 100
                elif db["flops"] > 0:
                    flops_pct = 100.0
                else:
                    flops_pct = 0.0

                if da["memory"] > 0:
                    mem_pct = (db["memory"] - da["memory"]) / da["memory"] * 100
                elif db["memory"] > 0:
                    mem_pct = 100.0
                else:
                    mem_pct = 0.0

                entry["flops_diff_pct"] = flops_pct
                entry["mem_diff_pct"] = mem_pct
                entry["args_a"] = da["args_set"]
                entry["args_b"] = db["args_set"]
                diff_ops.append(entry)

    # Sort only_a/only_b by flops descending
    only_a.sort(key=lambda x: x["flops"], reverse=True)
    only_b.sort(key=lambda x: x["flops"], reverse=True)
    diff_ops.sort(key=lambda x: abs(x["flops_a"] - x["flops_b"]), reverse=True)

    is_match = len(only_a) == 0 and len(only_b) == 0 and len(diff_ops) == 0

    return {
        "match": is_match,
        "total_flops_a": total_flops_a,
        "total_flops_b": total_flops_b,
        "total_mem_a": total_mem_a,
        "total_mem_b": total_mem_b,
        "flops_diff_pct": flops_diff_pct,
        "mem_diff_pct": mem_diff_pct,
        "max_diff_pct": max(flops_diff_pct, mem_diff_pct),
        "n_common": len(common_ops),
        "n_matching": sum(1 for c in common_ops if c["match"]),
        "only_a": only_a,
        "only_b": only_b,
        "diff_ops": diff_ops,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_flops(f):
    """Format FLOPs to human-readable."""
    if f >= 1e12:
        return f"{f / 1e12:.2f}T"
    if f >= 1e9:
        return f"{f / 1e9:.2f}G"
    if f >= 1e6:
        return f"{f / 1e6:.2f}M"
    return str(f)


def _fmt_mem(m):
    """Format memory bytes to human-readable."""
    if m >= 1e9:
        return f"{m / 1e9:.2f}GB"
    if m >= 1e6:
        return f"{m / 1e6:.2f}MB"
    if m >= 1e3:
        return f"{m / 1e3:.1f}KB"
    return f"{m}B"


def generate_report(results, label_a, label_b, precision, test):
    """Generate markdown report from comparison results.

    Parameters
    ----------
    results : list of (model_key, comparison_dict) tuples
    label_a, label_b : str — labels for the two data sources
    precision, test : str — e.g., "fp16", "eval"

    Returns
    -------
    str — markdown report content
    """
    lines = []
    lines.append(f"# Graph Consistency: {label_a} vs {label_b}")
    lines.append(f"\n**Precision**: {precision} | **Mode**: {test} | "
                 f"**Models**: {len(results)}\n")

    # Fleet summary
    n_match = sum(1 for _, r in results if r["match"])
    n_diff = len(results) - n_match

    lines.append("## Fleet Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Total models | {len(results)} |")
    lines.append(f"| Identical graphs | {n_match} ({n_match/len(results)*100:.0f}%) |")
    lines.append(f"| Different graphs | {n_diff} ({n_diff/len(results)*100:.0f}%) |")
    lines.append("")

    if n_diff == 0:
        lines.append(f"All {len(results)} models have identical computational graphs "
                     f"on {label_a} and {label_b}.\n")
        return "\n".join(lines)

    # Categorize differences
    sdpa_only = []  # Only SDPA differs
    significant = []  # Multiple ops or large FLOPs/memory diff
    minor = []  # Small differences

    for key, r in results:
        if r["match"]:
            continue

        # Check if difference is SDPA-only
        diff_names = set()
        for d in r["diff_ops"]:
            diff_names.add(d["name"])
        for o in r["only_a"]:
            diff_names.add(o["name"])
        for o in r["only_b"]:
            diff_names.add(o["name"])

        if diff_names == {_SDPA_NORMALIZED} or diff_names == {"aten::sdpa_forward"}:
            sdpa_only.append((key, r))
        elif r["max_diff_pct"] > 1.0:
            significant.append((key, r))
        else:
            minor.append((key, r))

    lines.append("### Difference Categories\n")
    lines.append(f"| Category | Count | Description |")
    lines.append(f"|----------|------:|-------------|")
    lines.append(f"| SDPA-only | {len(sdpa_only)} | "
                 f"Only SDPA dispatch differs ({label_a} flash vs {label_b} fused) |")
    lines.append(f"| Significant | {len(significant)} | "
                 f"FLOPs or Memory diff > 1% |")
    lines.append(f"| Minor | {len(minor)} | "
                 f"Small numerical differences |")
    lines.append("")

    # Models with different graphs — scorecard
    lines.append("## Model Scorecard\n")
    diff_results = [(k, r) for k, r in results if not r["match"]]
    diff_results.sort(key=lambda x: x[1]["max_diff_pct"], reverse=True)

    lines.append(f"| Model | FLOPs Diff (%) | Mem Diff (%) | "
                 f"{label_a}-only Ops | {label_b}-only Ops | "
                 f"Mismatched Ops | Category |")
    lines.append(f"|-------|------:|------:|------:|------:|------:|----------|")

    for key, r in diff_results:
        diff_names = set()
        for d in r["diff_ops"]:
            diff_names.add(d["name"])
        for o in r["only_a"]:
            diff_names.add(o["name"])
        for o in r["only_b"]:
            diff_names.add(o["name"])

        if diff_names == {_SDPA_NORMALIZED} or diff_names == {"aten::sdpa_forward"}:
            cat = "SDPA-only"
        elif r["max_diff_pct"] > 1.0:
            cat = "Significant"
        else:
            cat = "Minor"

        lines.append(
            f"| {key} | {r['flops_diff_pct']:.2f}% | {r['mem_diff_pct']:.2f}% | "
            f"{len(r['only_a'])} | {len(r['only_b'])} | "
            f"{len(r['diff_ops'])} | {cat} |")
    lines.append("")

    # Detailed per-op diff for significant models
    sig_models = significant  # Show details for significant diffs
    if sig_models:
        lines.append("## Significant Differences — Per-Op Detail\n")
        for key, r in sig_models:
            lines.append(f"### {key}\n")
            lines.append(f"Total FLOPs: {label_a}={_fmt_flops(r['total_flops_a'])}, "
                         f"{label_b}={_fmt_flops(r['total_flops_b'])} "
                         f"(diff={r['flops_diff_pct']:.2f}%)")
            lines.append(f"Total Memory: {label_a}={_fmt_mem(r['total_mem_a'])}, "
                         f"{label_b}={_fmt_mem(r['total_mem_b'])} "
                         f"(diff={r['mem_diff_pct']:.2f}%)\n")

            if r["only_a"]:
                lines.append(f"**{label_a}-only ops:**\n")
                lines.append(f"| Op | FLOPs | Memory | Count |")
                lines.append(f"|-----|------:|------:|------:|")
                for o in r["only_a"]:
                    raw = ", ".join(sorted(o["raw_names"]))
                    lines.append(f"| {raw} | {_fmt_flops(o['flops'])} | "
                                 f"{_fmt_mem(o['memory'])} | {o['count']} |")
                lines.append("")

            if r["only_b"]:
                lines.append(f"**{label_b}-only ops:**\n")
                lines.append(f"| Op | FLOPs | Memory | Count |")
                lines.append(f"|-----|------:|------:|------:|")
                for o in r["only_b"]:
                    raw = ", ".join(sorted(o["raw_names"]))
                    lines.append(f"| {raw} | {_fmt_flops(o['flops'])} | "
                                 f"{_fmt_mem(o['memory'])} | {o['count']} |")
                lines.append("")

            if r["diff_ops"]:
                lines.append(f"**Ops with different FLOPs/memory:**\n")
                lines.append(f"| Op | {label_a} FLOPs | {label_b} FLOPs | "
                             f"FLOPs Diff | {label_a} Mem | {label_b} Mem |")
                lines.append(f"|-----|------:|------:|------:|------:|------:|")
                for d in r["diff_ops"]:
                    fdiff = d["flops_b"] - d["flops_a"]
                    sign = "+" if fdiff >= 0 else ""
                    lines.append(
                        f"| {d['name']} | {_fmt_flops(d['flops_a'])} | "
                        f"{_fmt_flops(d['flops_b'])} | "
                        f"{sign}{_fmt_flops(fdiff)} | "
                        f"{_fmt_mem(d['memory_a'])} | {_fmt_mem(d['memory_b'])} |")
                lines.append("")

    # Fleet-level op differences summary
    lines.append("## Fleet Op Difference Summary\n")
    lines.append("Ops that differ across models (aggregated).\n")

    op_diff_count = defaultdict(lambda: {"count": 0, "models": []})
    for key, r in results:
        if r["match"]:
            continue
        seen = set()
        for d in r["diff_ops"]:
            if d["name"] not in seen:
                seen.add(d["name"])
                op_diff_count[d["name"]]["count"] += 1
                op_diff_count[d["name"]]["models"].append(key)
        for o in r["only_a"] + r["only_b"]:
            if o["name"] not in seen:
                seen.add(o["name"])
                op_diff_count[o["name"]]["count"] += 1
                op_diff_count[o["name"]]["models"].append(key)

    if op_diff_count:
        sorted_ops = sorted(op_diff_count.items(),
                            key=lambda x: x[1]["count"], reverse=True)
        lines.append(f"| Op | # Models | Top Models |")
        lines.append(f"|-----|------:|------|")
        for op_name, info in sorted_ops:
            top3 = ", ".join(info["models"][:3])
            lines.append(f"| {op_name} | {info['count']} | {top3} |")
    else:
        lines.append("No op-level differences found.\n")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare calcflops outputs from two devices to detect "
                    "graph differences.")
    parser.add_argument("--dir-a", required=True,
                        help="First data directory (e.g., CUDA results)")
    parser.add_argument("--dir-b", required=True,
                        help="Second data directory (e.g., XPU results)")
    parser.add_argument("--label-a", default="A",
                        help="Label for first directory (default: A)")
    parser.add_argument("--label-b", default="B",
                        help="Label for second directory (default: B)")
    parser.add_argument("--precision", default="fp16",
                        help="Precision filter (default: fp16)")
    parser.add_argument("--test", default="eval",
                        help="Test mode filter (default: eval)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: all common)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    # Discover models
    models_a = discover_models(args.dir_a, args.precision, args.test)
    models_b = discover_models(args.dir_b, args.precision, args.test)

    common_keys = sorted(set(models_a.keys()) & set(models_b.keys()))

    if args.models:
        filter_set = set(args.models.split(","))
        common_keys = [k for k in common_keys if k in filter_set]

    if not common_keys:
        print("ERROR: No common models found between the two directories.")
        print(f"  {args.label_a} ({args.dir_a}): {len(models_a)} models")
        print(f"  {args.label_b} ({args.dir_b}): {len(models_b)} models")
        sys.exit(1)

    print(f"Comparing {len(common_keys)} models: {args.label_a} vs {args.label_b}\n")

    # Compare each model
    results = []
    for i, key in enumerate(common_keys, 1):
        try:
            r = compare_model(models_a[key], models_b[key])
            status = "MATCH" if r["match"] else f"DIFF (F:{r['flops_diff_pct']:.2f}% M:{r['mem_diff_pct']:.2f}%)"
            print(f"  [{i}/{len(common_keys)}] {key}: {status}")
            results.append((key, r))
        except Exception as e:
            print(f"  [{i}/{len(common_keys)}] {key}: ERROR {e}")

    # Generate report
    report = generate_report(results, args.label_a, args.label_b,
                             args.precision, args.test)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
