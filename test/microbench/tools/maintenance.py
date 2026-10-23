# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Maintenance tools (report only, never fail a job).

``coverage``  XPU dispatch op coverage;
``srcmap``    kernel files missing from op_source_map.yaml;
``noise``     A/A calibration into history.json, nightly noise and case costs.
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from .. import CONFIG_DIR, judge, load_history, load_json, load_yaml, REPO_ROOT
from ..case import expected_names, functional_name, manifest
from ..selection import match_any

# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
# XPU dispatch ops are the functional names of ``native_functions.yaml``
# entries with an XPU, SparseXPU, SparseCsrXPU or NestedTensorXPU kernel,
# plus non-aten ops this repository registers for XPU itself (torchvision
# nms, roi_align, ...). In-place, ``out`` and structured-delegate variants
# fold onto the functional op. Covered = executed by a case (validator
# ``dispatched_ops``/``kernel_ops`` with ``--observed``, else the declared
# ``expected_aten_ops``).

XPU_KEYS = {"XPU", "SparseXPU", "SparseCsrXPU", "NestedTensorXPU"}
RE_LIB_IMPL = re.compile(r"TORCH_LIBRARY_IMPL\s*\(\s*(\w+)\s*,\s*XPU\s*,\s*(\w+)\s*\)")
RE_IMPL = re.compile(
    r'\.impl\s*\(\s*(?:TORCH_SELECTIVE_NAME\s*\(\s*)?"(?:(\w+)::)?([\w.]+)"'
)
RE_IDENT = re.compile(r"[A-Za-z_]\w*")


def _src_files(repo, exts):
    for root, _, files in os.walk(os.path.join(repo, "src")):
        for name in files:
            if name.endswith(exts):
                path = os.path.join(root, name)
                with open(path, errors="ignore") as f:
                    yield os.path.relpath(path, repo), f.read()


def xpu_dispatch_ops(native_functions):
    """{functional_name: {"keys", "kernels", "overloads", "view"}}."""
    import yaml

    with open(native_functions) as f:
        entries = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    out = {}
    for e in entries:
        if not isinstance(e, dict) or "func" not in e:
            continue
        disp = {
            k.strip(): str(v)
            for ks, v in (e.get("dispatch") or {}).items()
            for k in str(ks).split(",")
        }
        keys = XPU_KEYS & set(disp)
        if not keys:
            continue
        name, ret = e["func"].split("(", 1)[0], e["func"].rsplit("->", 1)[-1]
        ent = out.setdefault(
            functional_name(name),
            {"keys": set(), "kernels": set(), "overloads": set(), "view": True},
        )
        ent["keys"] |= keys
        ent["kernels"] |= {disp[k] for k in keys}
        ent["overloads"].add(name)
        # a functional op is a view only if every XPU overload returns an alias
        ent["view"] = ent["view"] and "(a)" in ret.replace(" ", "") and "!" not in ret
    return out


def repo_registered_ops(repo=None):
    """Non-aten ops this repo registers for XPU (TORCH_LIBRARY_IMPL(ns, XPU, m)
    or the lazy torchvision registration in XPUFallback.cpp)."""
    repo = repo or REPO_ROOT
    out = {}
    for path, text in _src_files(repo, (".cpp",)):
        starts = [m.end() for m in RE_LIB_IMPL.finditer(text) if m.group(1) != "aten"]
        if "DispatchKey::XPU" in text and "Library::IMPL" in text:
            starts.append(0)
        for start in starts:
            for im in RE_IMPL.finditer(text[start:]):
                if im.group(1) not in (None, "aten"):
                    op = im.group(2).split(".", 1)[0]
                    out[op] = {
                        "keys": {"XPU"},
                        "namespace": im.group(1),
                        "file": path,
                        "kernels": set(),
                    }
    return out


def build_report(
    native_functions,
    manifest,
    observed=None,
    exclude=None,
    repo=None,
    prev_native_functions=None,
):
    repo = repo or REPO_ROOT
    ops = xpu_dispatch_ops(native_functions)
    repo_ops = repo_registered_ops(repo)
    for op, ent in repo_ops.items():
        ops.setdefault(op, ent)
    if exclude is None:
        data = load_yaml(CONFIG_DIR / "coverage_exclude.yaml", {})
        exclude = data.get("excluded") or {}, data.get("deferred") or {}
    excluded, deferred = exclude
    cov = defaultdict(set)
    for r in observed or []:
        for op in (r.get("kernel_ops") or []) + (r.get("dispatched_ops") or []):
            cov[functional_name(op)].add(r.get("op"))
    for m in manifest if not observed else []:
        for name in expected_names(m["expected_aten_ops"]):
            cov[name].add(m["op"])
    # kernel function name -> files mentioning it (one pass over src)
    kernels = {k for e in ops.values() for k in e["kernels"]}
    kfiles = defaultdict(set)
    for path, text in _src_files(repo, (".cpp", ".h")):
        for k in kernels & set(RE_IDENT.findall(text)):
            kfiles[k].add(path)
    rows = []
    for op, ent in sorted(ops.items()):
        files = set().union(*(kfiles.get(k, set()) for k in ent["kernels"]))
        if ent.get("file"):
            files.add(ent["file"])
        if op in cov:
            status, note = "covered", ", ".join(sorted(cov[op]))
        elif op in excluded:
            status, note = "excluded", excluded[op]
        elif op in deferred:
            status, note = "deferred", deferred[op]
        else:
            status, note = (
                "unregistered",
                "not covered and not in coverage_exclude.yaml",
            )
        first = sorted(files, key=lambda p: ("/sycl/" not in p, p))[:1]
        family = (
            re.sub(r"Kernels?$", "", os.path.splitext(os.path.basename(first[0]))[0])
            if first
            else "unknown"
        )
        rows.append(
            {
                "op": op,
                "status": status,
                "note": note,
                "keys": sorted(ent["keys"]),
                "family": family,
                "files": sorted(files),
            }
        )
    new_in_yaml = []
    if prev_native_functions:
        new_in_yaml = sorted(
            set(ops) - set(xpu_dispatch_ops(prev_native_functions)) - set(repo_ops)
        )
    return {
        "rows": rows,
        "new_in_native_functions": new_in_yaml,
        "stale_exclusions": sorted((set(excluded) | set(deferred)) & set(cov)),
    }


def render(rep):
    rows = rep["rows"]
    counts = Counter(r["status"] for r in rows)
    fam = defaultdict(Counter)
    for r in rows:
        fam[r["family"]][r["status"]] += 1
    lines = [
        "## Microbench coverage of XPU dispatch ops",
        "",
        f"Total {len(rows)}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| Family | Covered | Excluded | Deferred | Unregistered |",
        "|---|---|---|---|---|",
    ]
    for f, c in sorted(fam.items()):
        lines.append(
            f"| {f} | {c['covered']} | {c['excluded']} | {c['deferred']} | {c['unregistered']} |"
        )
    unreg = [r for r in rows if r["status"] == "unregistered"]
    if unreg:
        lines += [
            "",
            "### Not covered and not registered",
            "",
            "| Op | Keys | Files |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{r['op']}` | {','.join(r['keys'])} | {', '.join(r['files'][:3])} |"
            for r in unreg
        ]
    if rep["new_in_native_functions"]:
        lines += [
            "",
            "### New XPU entries in native_functions.yaml since the previous PyTorch commit",
            "",
        ]
        lines += [f"- `{o}`" for o in rep["new_in_native_functions"]]
    if rep["stale_exclusions"]:
        lines += ["", "### Covered ops still listed in coverage_exclude.yaml", ""]
        lines += [f"- `{o}`" for o in rep["stale_exclusions"]]
    for status in ("covered", "excluded", "deferred"):
        sub = [r for r in rows if r["status"] == status]
        lines += ["", f"<details><summary>{status} ({len(sub)})</summary>", ""]
        lines += [f"- `{r['op']}` ({r['family']}): {r['note']}" for r in sub]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def run_coverage(args):
    observed = None
    if args.observed:
        observed = [json.loads(line) for line in open(args.observed) if line.strip()]
    rep = build_report(
        args.native_functions,
        manifest(),
        observed=observed,
        prev_native_functions=args.prev_native_functions,
    )
    md = render(rep)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=1)
    for path in filter(None, (args.out_md, args.summary)):
        with open(path, "a" if path == args.summary else "w") as f:
            f.write(md)
    if not (args.out_md or args.summary):
        print(md)
    return 0


# ---------------------------------------------------------------------------
# op_source_map.yaml completeness (report only)
# ---------------------------------------------------------------------------

KERNEL_DIRS = (
    "src/ATen/native/xpu/",
    "src/ATen/native/nested/xpu/",
    "src/ATen/native/sparse/xpu/",
    "src/ATen/native/quantized/",
    "src/ATen/native/transformers/",
)


def unmapped_kernel_files(repo=None, srcmap=None):
    """Kernel files that are neither mapped to an op nor marked no_bench."""
    repo = repo or REPO_ROOT
    srcmap = srcmap or load_yaml(CONFIG_DIR / "op_source_map.yaml", {})
    globs = list(srcmap.get("no_bench") or []) + list(srcmap.get("global") or [])
    for section in ("common_groups", "ops"):
        for g in (srcmap.get(section) or {}).values():
            globs += g
    files = [
        p
        for p, _ in _src_files(repo, (".cpp", ".h", ".hpp"))
        if p.startswith(KERNEL_DIRS)
    ]
    return sorted(f for f in files if not match_any(f, globs))


def run_srcmap(args):
    unmapped = unmapped_kernel_files(args.repo)
    lines = [
        "## op_source_map.yaml completeness (report only)",
        "",
        f"{len(unmapped)} kernel file(s) are neither mapped to an op nor marked no_bench.",
    ]
    text = "\n".join(lines + [f"- `{f}`" for f in unmapped]) + "\n"
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    return 0


# ---------------------------------------------------------------------------
# Noise calibration and case costs
# ---------------------------------------------------------------------------
# ``calibrate``  A/A runs (same wheel on both sides, e.g. 50+ on-demand A/A
#                runs) -> per-case delta, CI width and false-positive count.
#                A case is ``calibrated`` (allowed to block PRs) when it has at
#                least ``--min-runs`` runs, no false positive, and a p95 CI
#                width within ``max_ci_width``. Written into history.json.
# ``summarize``  any runs -> per-case between-process CV and CI width, plus
#                measured seconds per worker process (history.json case_cost).


def _run_groups(run_dir, metric):
    recs = judge.load_records(os.path.join(run_dir, "raw"))
    return recs, judge.group_samples(recs, metric)


def calibrate(run_dirs, metric="device_us", min_runs=20, min_delta=0.02, gating=None):
    gating = gating or judge.load_gating()
    per_case = defaultdict(lambda: {"log_r": [], "ci_width": [], "fp": 0})
    for d in run_dirs:
        mode = (load_json(os.path.join(d, "run_meta.json"), {}) or {}).get("mode")
        if mode != "aa":
            print(f"skip {d}: not an A/A run (mode={mode})", file=sys.stderr)
            continue
        for (cid, stage), g in _run_groups(d, metric)[1].items():
            res = (
                judge.compare(g["A"], g["B"], n_boot=500) if stage == "screen" else None
            )
            if res is None:
                continue
            ent = per_case[cid]
            ent["log_r"].append(math.log(res["ratio"]))
            ent["ci_width"].append(res["ci_width"])
            verdict, _ = judge.classify(
                res,
                threshold=gating["threshold"],
                delta=gating["default_delta"]["normal"],
                max_ci_width=gating["max_ci_width"],
            )
            ent["fp"] += verdict in ("regression", "improvement")
    out = {}
    for cid, ent in per_case.items():
        q95 = float(np.quantile(np.abs(ent["log_r"]), 0.95))
        ciw = float(np.quantile(ent["ci_width"], 0.95))
        out[cid] = {
            "runs": len(ent["log_r"]),
            "delta": round(max(min_delta, math.expm1(q95)), 4),
            "ci_width_p95": round(ciw, 4),
            "false_positives": ent["fp"],
            "calibrated": len(ent["log_r"]) >= min_runs
            and not ent["fp"]
            and ciw <= gating["max_ci_width"],
        }
    return out


def summarize(run_dirs, metric="device_us"):
    cv, ciw, secs = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in run_dirs:
        recs, groups = _run_groups(d, metric)
        for r in recs:
            if (
                r.get("type") == "result"
                and r.get("status") == "ok"
                and r.get("seconds")
            ):
                secs[r["case_id"]].append(float(r["seconds"]))
        for (cid, stage), g in groups.items():
            if stage != "screen":
                continue
            for side in ("A", "B"):
                if len(g[side]) >= 2:
                    meds = [float(np.median(x)) for x in g[side]]
                    cv[cid].append(float(np.std(meds, ddof=1) / np.mean(meds)))
            res = (
                judge.compare(g["A"], g["B"], n_boot=300) if g["A"] and g["B"] else None
            )
            if res:
                ciw[cid].append(res["ci_width"])
    noise = {
        cid: {
            "between_process_cv": round(float(np.median(v)), 4),
            "ci_width": round(float(np.median(ciw.get(cid, [0]))), 4),
        }
        for cid, v in cv.items()
    }
    return noise, {cid: round(float(np.median(v)), 2) for cid, v in secs.items()}


def run_noise(args):
    hist = load_history(args.history)
    if args.action == "calibrate":
        out = calibrate(args.runs, metric=args.metric, min_runs=args.min_runs)
        hist["noise_stats"].update(out)
        with open(args.out, "w") as f:
            json.dump(hist, f, indent=1, sort_keys=True)
        n_cal = sum(v["calibrated"] for v in out.values())
        print(
            f"{len(out)} case(s) from {len(args.runs)} run dir(s); {n_cal} calibrated"
        )
        return 0
    noise, costs = summarize(args.runs, metric=args.metric)
    with open(args.out, "w") as f:
        json.dump(noise, f, indent=1, sort_keys=True)
    if args.history_out:
        hist["case_cost"].update(costs)
        with open(args.history_out, "w") as f:
            json.dump(hist, f, indent=1, sort_keys=True)
    print(f"noise for {len(noise)} case(s), costs for {len(costs)} case(s)")
    return 0


def add_parsers(sub):
    p = sub.add_parser("coverage", help="XPU op coverage report (report only)")
    p.add_argument(
        "--native-functions", required=True, help="path to native_functions.yaml"
    )
    p.add_argument(
        "--prev-native-functions", default=None, help="yaml of the previous nightly"
    )
    p.add_argument(
        "--observed", default=None, help="validator JSONL with dispatched/kernel ops"
    )
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--summary", default=None)
    p.set_defaults(func=lambda a: sys.exit(run_coverage(a)))

    p = sub.add_parser(
        "srcmap", help="report kernel files missing from op_source_map.yaml"
    )
    p.add_argument("--repo", default=None)
    p.add_argument("--summary", default=None)
    p.set_defaults(func=lambda a: sys.exit(run_srcmap(a)))

    p = sub.add_parser("noise", help="noise calibration / run statistics / case costs")
    p.add_argument("action", choices=["calibrate", "summarize"])
    p.add_argument(
        "runs", nargs="+", help="scheduler output dirs (run_meta.json + raw/)"
    )
    p.add_argument("--metric", default="device_us", choices=["device_us", "e2e_us"])
    p.add_argument("--min-runs", type=int, default=20)
    p.add_argument("--history", default=None, help="default: config/history.json")
    p.add_argument(
        "--history-out",
        default=None,
        help="summarize: history.json with updated case_cost",
    )
    p.add_argument(
        "--out",
        required=True,
        help="calibrate: updated history.json; summarize: noise JSON",
    )
    p.set_defaults(func=lambda a: sys.exit(run_noise(a)))
