# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Verdicts: A/B statistics, per-case judgement, English report, check job
(the on-demand report and the nightly check)."""

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time

import numpy as np

from . import CONFIG_DIR, load_history, load_json, load_yaml, read_jsonl, reference
from .case import case_log_dir, load_cases

# ---------------------------------------------------------------------------
# A/B statistics
# ---------------------------------------------------------------------------
# * Point estimate: ``r = median(B) / median(A)`` over pooled samples.
# * Uncertainty: hierarchical bootstrap. Processes are resampled first, then
#   iterations within each resampled process: iterations of one process share
#   allocation, alignment and clock state, so treating them as independent
#   would make the interval too narrow.
# * p-values: one-sided bootstrap tests. For regressions the null is "B is at
#   most delta slower than A"; Holm's step-down correction controls the
#   family-wise error over all cases of a run.
#
# Defaults; config/gating.yaml overrides them.
DEFAULT_GATING = {
    "threshold": 0.10,
    "alpha": 0.05,
    "max_ci_width": 0.10,
    "default_delta": {
        "normal": 0.03,
        "atomic": 0.08,
        "random": 0.08,
        "host_bound": 0.05,
    },
    "e2e_blocking_noise_classes": [],
    "e2e_threshold": 0.20,
    "n_boot": 2000,
    "suspicious_frac": 0.7,
}
G = DEFAULT_GATING


def boot_medians(groups, n_boot, rng):
    """Bootstrap distribution of the pooled median (processes, then samples)."""
    k, m = len(groups), max(len(g) for g in groups)
    arr = np.full((k, m), np.nan)
    lens = np.array([len(g) for g in groups])
    for i, g in enumerate(groups):
        arr[i, : len(g)] = g
    pick = rng.integers(0, k, size=(n_boot, k))  # resampled processes
    n = lens[pick]
    idx = np.floor(rng.random((n_boot, k, m)) * n[:, :, None]).astype(np.int64)
    vals = arr[pick[:, :, None], idx]
    vals = np.where(np.arange(m)[None, None, :] < n[:, :, None], vals, np.nan)
    return np.nanmedian(vals.reshape(n_boot, k * m), axis=1)


def _between_cv(groups):
    meds = [float(np.median(g)) for g in groups]
    if len(meds) < 2 or np.mean(meds) <= 0:
        return None
    return float(np.std(meds, ddof=1) / np.mean(meds))


def compare(
    groups_a, groups_b, delta=G["default_delta"]["normal"], n_boot=G["n_boot"], seed=0
):
    """Compare two sides given per-process sample lists."""
    groups_a = [np.asarray(g, dtype=float) for g in groups_a if len(g)]
    groups_b = [np.asarray(g, dtype=float) for g in groups_b if len(g)]
    if not groups_a or not groups_b:
        return None
    pa, pb = np.concatenate(groups_a), np.concatenate(groups_b)
    med_a, med_b = float(np.median(pa)), float(np.median(pb))
    if med_a <= 0 or med_b <= 0:
        return None
    rng = np.random.default_rng(seed)
    logr = np.log(boot_medians(groups_b, n_boot, rng)) - np.log(
        boot_medians(groups_a, n_boot, rng)
    )
    lo, hi = np.exp(np.percentile(logr, [2.5, 97.5]))
    return {
        "median_a": med_a,
        "median_b": med_b,
        "ratio": med_b / med_a,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_width": float(hi / lo - 1.0),
        "p_reg": float((np.sum(logr <= math.log1p(delta)) + 1) / (n_boot + 1)),
        "p_imp": float((np.sum(logr >= math.log1p(-delta)) + 1) / (n_boot + 1)),
        "procs_a": len(groups_a),
        "procs_b": len(groups_b),
        "samples_a": int(pa.size),
        "samples_b": int(pb.size),
        "cv_between_a": _between_cv(groups_a),
        "cv_between_b": _between_cv(groups_b),
    }


def classify(
    res,
    threshold=G["threshold"],
    delta=G["default_delta"]["normal"],
    max_ci_width=G["max_ci_width"],
    p_reg_adj=None,
    p_imp_adj=None,
    alpha=G["alpha"],
):
    """Verdict for one comparison: regression / improvement / inconclusive / no_change."""
    if res is None:
        return "inconclusive", "no data"
    r = res["ratio"]
    if (
        r > 1 + threshold
        and res["ci_lo"] > 1 + delta
        and (p_reg_adj is None or p_reg_adj < alpha)
    ):
        return "regression", ""
    if (
        r < 1 / (1 + threshold)
        and res["ci_hi"] < 1 - delta
        and (p_imp_adj is None or p_imp_adj < alpha)
    ):
        return "improvement", ""
    if res["ci_width"] > max_ci_width:
        return "inconclusive", f"CI width {res['ci_width']:.3f} > {max_ci_width:.3f}"
    if r > 1 + threshold or r < 1 / (1 + threshold):
        return "inconclusive", "beyond threshold but not significant"
    return "no_change", ""


def suspicious(res, threshold=G["threshold"], frac=G["suspicious_frac"]):
    """Screening filter: anything that might turn out a regression/improvement."""
    thr = threshold * frac
    return res is not None and (res["ratio"] > 1 + thr or res["ratio"] < 1 / (1 + thr))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def group_samples(records, metric="device_us"):
    """{(case_id, stage): {side: [[...], ...]}} from worker records ("A" and
    "B" are always present)."""
    out = {}
    for r in records:
        if r.get("type") != "result" or r.get("status") != "ok":
            continue
        if metric == "device_us" and r.get("device_status") not in ("ok", None):
            continue
        vals = [v for v in r.get(metric) or [] if v is not None]
        if vals:
            key = (r["case_id"], r.get("stage", "screen"))
            out.setdefault(key, {"A": [], "B": []}).setdefault(r["side"], []).append(
                vals
            )
    return out


# ---------------------------------------------------------------------------
# Per-case verdicts
# ---------------------------------------------------------------------------


def load_gating(path=None):
    """gating.yaml over DEFAULT_GATING; ``quarantine`` is split off."""
    data = load_yaml(path or CONFIG_DIR / "gating.yaml", {})
    q = data.pop("quarantine", None) or {}
    cfg = dict(DEFAULT_GATING, **data)
    cfg["quarantine"] = (
        {e["case_id"]: e.get("reason", "") for e in q.get("cases") or []},
        {e["op"]: e.get("reason", "") for e in q.get("ops") or []},
    )
    return cfg


def load_records(raw_dir):
    recs = []
    for root, _, files in os.walk(raw_dir):
        for name in sorted(files):
            if name.endswith(".jsonl"):
                recs.extend(read_jsonl(os.path.join(root, name)))
    return recs


def report_only_reasons(man, gating):
    """Why a confirmed regression of a case is reported but files no issue;
    empty = it files one."""
    q_ids, q_ops = gating.get("quarantine", ({}, {}))
    return [
        reason
        for reason, cond in (
            ("op excluded from gating", not man.get("gating", True)),
            ("quarantined", man.get("case_id") in q_ids or man.get("op") in q_ops),
        )
        if cond
    ]


def case_delta(cid, noise_class, gating, noise):
    """Noise margin of a case: the default of its noise class, raised to the
    case's measured A/A delta when that is larger (never lowered: the nightly
    A/A gives a handful of runs per case)."""
    default = gating["default_delta"].get(
        noise_class, gating["default_delta"]["normal"]
    )
    return max(default, (noise.get(cid) or {}).get("delta") or 0.0)


def merge_noise(history_noise, state_noise):
    """Per-case noise: history.json, with the delta of the nightly A/A in the
    reference state taking over where it has one."""
    out = {cid: dict(v) for cid, v in (history_noise or {}).items()}
    for cid, st in (state_noise or {}).items():
        if (st or {}).get("delta") is not None:
            out.setdefault(cid, {})["delta"] = st["delta"]
    return out


def _final(screen, confirm, gating, delta, p_adj=None):
    """Two-stage decision. Returns (verdict, reason, result_used)."""
    kw = dict(
        threshold=gating["threshold"],
        delta=delta,
        max_ci_width=gating["max_ci_width"],
        alpha=gating["alpha"],
    )
    flagged = ("regression", "improvement")
    p_reg, p_imp = p_adj or (None, None)
    v_s, why_s = classify(screen, **kw)
    if confirm is None:
        if v_s in flagged:
            return "inconclusive", "flagged in screening, no confirmation data", screen
        return v_s, why_s, screen
    v_c, why_c = classify(confirm, p_reg_adj=p_reg, p_imp_adj=p_imp, **kw)
    if v_s == v_c and v_s in flagged:
        return v_c, "", confirm
    if v_c in flagged or v_s in flagged:
        return "inconclusive", f"screening said {v_s}, confirmation said {v_c}", confirm
    return v_c, why_c, confirm


def _row(cid, man, delta):
    keys = ("case_id", "op", "pass", "label", "tier", "noise_class", "gating", "repro")
    row = {k: man.get(k) for k in keys}
    row.update(case_id=cid, metrics={}, delta=delta)
    return row


def judge(
    raw_dir,
    cases_manifest,
    meta,
    selection=None,
    gating=None,
    noise=None,
):
    """Compute verdicts. ``cases_manifest`` maps case_id -> manifest dict."""
    gating = gating or load_gating()
    noise = noise if noise is not None else load_history()["noise_stats"]
    recs = load_records(raw_dir)
    mode = meta.get("mode", "ab")

    status, errors = {}, {}
    for r in recs:
        if r.get("type") != "result":
            continue
        status.setdefault(r["case_id"], {"A": set(), "B": set()}).setdefault(
            r["side"], set()
        ).add(r.get("status"))
        if r.get("status") != "ok":
            errors.setdefault(
                (r["case_id"], r["side"]),
                (r.get("error") or r.get("status") or "")[:200],
            )

    metrics = {"device": "device_us", "e2e": "e2e_us"}
    groups = {m: group_samples(recs, key) for m, key in metrics.items()}
    unknown = {
        "op": "?",
        "pass": "?",
        "tier": "nightly",
        "noise_class": "normal",
        "gating": False,
    }
    rows = []
    for cid in (selection or {}).get("cases") or sorted(status):
        man = cases_manifest.get(cid) or dict(unknown, label=cid)
        nc = man.get("noise_class", "normal")
        row = _row(cid, man, case_delta(cid, nc, gating, noise))
        rows.append(row)
        st = status.get(cid, {"A": set(), "B": set()})
        if "ok" not in st["B"]:
            row["status"] = "error" if st["B"] else "not_run"
            row["detail"] = errors.get((cid, "B")) or "no data"
        elif "ok" not in st["A"]:
            row["status"] = "new_no_baseline" if st["A"] else "not_run"
            row["detail"] = (
                f"case failed on the base wheel: {errors.get((cid, 'A'), '')}"
                if st["A"]
                else "no base data"
            )
            g = groups["device"].get((cid, "screen")) or groups["e2e"].get(
                (cid, "screen")
            )
            if g and g["B"]:
                row["b_median_us"] = statistics.median(v for grp in g["B"] for v in grp)
        else:
            row["status"] = "ok"
            for m in metrics:
                stages = {s: groups[m].get((cid, s)) for s in ("screen", "confirm")}
                if m == "device" and not any(stages.values()):
                    continue  # no device timing at all (e.g. CPU smoke run)
                row["metrics"][m] = {
                    s: compare(g["A"], g["B"], row["delta"], gating["n_boot"], seed=i)
                    if g and g["A"] and g["B"]
                    else None
                    for i, (s, g) in enumerate(stages.items())
                }

    # Holm over the final-stage p-values of the device metric.
    def has_data(m):
        return bool(m and (m.get("screen") or m.get("confirm")))

    ok_rows = [
        r for r in rows if r["status"] == "ok" and has_data(r["metrics"].get("device"))
    ]
    final = [
        r["metrics"]["device"]["confirm"] or r["metrics"]["device"]["screen"]
        for r in ok_rows
    ]
    adj = dict(
        zip(
            (r["case_id"] for r in ok_rows),
            zip(holm([f["p_reg"] for f in final]), holm([f["p_imp"] for f in final])),
        )
    )

    for r in rows:
        if r["status"] != "ok":
            r["verdict"] = None
            continue
        e2e_gates = r["noise_class"] in gating["e2e_blocking_noise_classes"]
        for m, mres in r["metrics"].items():
            if not has_data(mres):
                mres["verdict"], mres["reason"] = "inconclusive", "no valid samples"
                continue
            thr = (
                gating["e2e_threshold"]
                if m == "e2e" and e2e_gates
                else gating["threshold"]
            )
            p = adj.get(r["case_id"]) if m == "device" else None
            v, why, used = _final(
                mres["screen"],
                mres["confirm"],
                dict(gating, threshold=thr),
                r["delta"],
                p,
            )
            mres.update(verdict=v, reason=why, used=used)
            if p:
                mres["p_holm"] = p[1] if v == "improvement" else p[0]
        r["verdict"] = (
            r["metrics"].get("device") or r["metrics"].get("e2e") or {}
        ).get("verdict")
    return summarize_rows({"mode": mode, "rows": rows, "gating": gating})


def summarize_rows(result):
    rows = result["rows"]

    def with_verdict(v):
        return [
            r["case_id"]
            for r in rows
            if any(m.get("verdict") == v for m in r["metrics"].values())
        ]

    result.update(
        regressions=with_verdict("regression"),
        improvements=with_verdict("improvement"),
    )
    return result


# ---------------------------------------------------------------------------
# On-demand report
# ---------------------------------------------------------------------------


def _fmt_us(x):
    return "-" if x is None else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _ratio_ci(res):
    if not res:
        return ["-", "-"]
    return [f"{res['ratio']:.3f}", f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]"]


def _entries(rows, verdict):
    """(row, metric) pairs with ``verdict``, by magnitude."""
    out = [
        (r, m)
        for r in rows
        for m, mm in r["metrics"].items()
        if mm.get("verdict") == verdict
    ]

    def key(item):
        r, m = item
        return -abs((r["metrics"][m].get("used") or {}).get("ratio", 1.0) - 1.0)

    return sorted(out, key=key)


def _table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def render(result, ctx):
    """Markdown report of an on-demand A/B or A/A run. ``ctx`` keys:
    mode_label, env, selection, run_url, flow."""
    rows = result["rows"]
    regs, imps = _entries(rows, "regression"), _entries(rows, "improvement")
    counts = f"{len({r['case_id'] for r, _ in regs})} regression(s), {len({r['case_id'] for r, _ in imps})} improvement(s)"
    if result["mode"] != "ab":
        status = f"A/A run, report only - {counts}"
    elif regs or imps:
        status = counts
    else:
        status = "no significant change in this run"
    lines = ["## Microbench Performance Report", ""]
    if ctx.get("mode_label"):
        lines += [f"**MODE: {ctx['mode_label']}**", ""]
    thr = int(round(result["gating"]["threshold"] * 100))
    note = " ".join(
        filter(None, [f"Threshold: {thr}% on device time.", ctx.get("flow")])
    )
    lines += [f"**Result:** {status}", f"_{note}_"]

    def cells(r, m):
        res = r["metrics"][m].get("used") or {}
        base = [f"`{r['op']}`", r["label"], r["pass"], m]
        return base + [
            _fmt_us(res.get("median_a")),
            _fmt_us(res.get("median_b")),
            *_ratio_ci(res),
            _log_cell(r),
        ]

    header = [
        "Op",
        "Case",
        "Pass",
        "Metric",
        "Base (us)",
        "B (us)",
        "Ratio",
        "95% CI",
        "Log",
    ]
    if regs:
        lines += ["", "### Regressions"]
        lines += _table(header, [cells(r, m) for r, m in regs])
    if imps:
        lines += ["", "### Improvements"]
        lines += _table(header, [cells(r, m) for r, m in imps])

    inconc = _entries(rows, "inconclusive")
    others = [r for r in rows if r["status"] != "ok"]
    if inconc or others:
        lines += [
            "",
            "<details><summary>Inconclusive cases and cases without a result</summary>",
            "",
        ]
        if inconc:
            lines += _table(
                ["Op", "Case", "Pass", "Metric", "Ratio", "95% CI", "Reason"],
                [
                    [
                        f"`{r['op']}`",
                        r["label"],
                        r["pass"],
                        m,
                        *_ratio_ci(r["metrics"][m].get("used")),
                        r["metrics"][m]["reason"],
                    ]
                    for r, m in inconc
                ],
            )
        if others:
            lines += [""] + _table(
                ["Op", "Case", "Pass", "Status", "Detail"],
                [
                    [
                        f"`{r['op']}`",
                        r["label"],
                        r["pass"],
                        "new (no baseline)"
                        if r["status"] == "new_no_baseline"
                        else r["status"],
                        f"B median {_fmt_us(r['b_median_us'])} us"
                        if r.get("b_median_us")
                        else r.get("detail", ""),
                    ]
                    for r in others
                ],
            )
        lines += ["", "</details>"]

    sel = ctx.get("selection") or {}
    if sel:
        lines += ["", "<details><summary>Case selection</summary>", ""]
        lines += [f"- {reason}" for reason in sel.get("reasons") or []]
        ops = ", ".join(sel.get("ops") or [])
        lines += [
            "",
            f"Selected {len(sel.get('cases') or [])} case(s) over ops: {ops}",
            "",
            "</details>",
        ]

    env = {k: v for k, v in (ctx.get("env") or {}).items() if v}
    if env:
        lines += [
            "",
            "**Environment:** " + ", ".join(f"{k} `{v}`" for k, v in env.items()),
        ]
    if ctx.get("run_url"):
        lines.append(f"**Artifacts:** {ctx['run_url']}")
    lines.append(
        f"**All {len(rows)} case(s):** `all_cases.md` / `all_cases.csv` in the `microbench-report-*` artifact; "
        "per-case logs (`logs/<op>/<pass>_<case_id>/`, index in `logs/index.csv`) in the `microbench-data-*` artifacts"
    )
    reps = regs or imps
    if reps:
        lines.append(f"**Reproduce:** `{reps[0][0].get('repro')}`")
    return "\n".join(lines) + "\n"


def _log_cell(r):
    return f"`logs/{case_log_dir(r['op'], r['pass'], r['case_id'])}/`"


ALL_CASES_HEADER = [
    "Op",
    "Case",
    "Pass",
    "Metric",
    "Base (us)",
    "B (us)",
    "Ratio",
    "95% CI",
    "Verdict",
    "Reason",
    "Case id",
]


def all_cases(result):
    """One row per (case, metric) of every selected case; cases without A/B
    data get one row with their status."""
    out = []
    for r in result["rows"]:
        head = [r["op"], r["label"], r["pass"]]
        if r["status"] != "ok":
            b = _fmt_us(r["b_median_us"]) if r.get("b_median_us") else "-"
            out.append(
                head
                + ["-", "-", b, "-", "-"]
                + [r["status"], r.get("detail", ""), r["case_id"]]
            )
            continue
        for metric, m in r["metrics"].items():
            res = m.get("used") or {}
            base, b = _fmt_us(res.get("median_a")), _fmt_us(res.get("median_b"))
            out.append(
                head
                + [metric, base, b, *_ratio_ci(res)]
                + [m["verdict"], m.get("reason", ""), r["case_id"]]
            )
    return out


def write_all_cases(result, out_dir):
    """all_cases.md and all_cases.csv: the full result table (artifact only)."""
    rows = all_cases(result)
    with open(os.path.join(out_dir, "all_cases.csv"), "w", newline="") as f:
        csv.writer(f).writerows([ALL_CASES_HEADER] + rows)
    md = ["# Microbench results: all cases", ""]
    md += _table(
        ALL_CASES_HEADER, [[str(c).replace("|", "\\|") for c in row] for row in rows]
    )
    with open(os.path.join(out_dir, "all_cases.md"), "w") as f:
        f.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Nightly check
# ---------------------------------------------------------------------------
# ``python -m microbench check --scope nightly``: the nightly microbench-check.
#
# Merges the ``nightly-vs-reference`` shards (detection, recheck, candidate
# commits) and the optional ``nightly-xpu-ops-vs-pytorch`` run, attributes each
# confirmed regression, appends the night's measurements and
# confirmed jumps to the reference state, writes the nightly
# report and the issue evidence.
#
# Attribution from the three-wheel run (P' = previous xpu-ops built with this
# nightly's PyTorch):
#
# | r_xpu = N / P'          | r_pt = P' / P              | attribution  |
# |-------------------------|----------------------------|--------------|
# | significantly slower    | significantly slower       | mixed        |
# | regression (>= 10%)     | no change                  | xpu-ops      |
# | no change               | regression or slower       | pytorch      |
# | anything else           |                            | inconclusive |
#
# A confirmed regression without candidate commits (no related xpu-ops file
# changed between P and N) is attributed to PyTorch without the three-wheel
# run ("inferred").

ISSUE_LABELS = {
    "xpu-ops": ["microbench-regression"],
    "pytorch": ["microbench-regression-upstream"],
    "mixed": ["microbench-regression", "microbench-regression-upstream"],
}


def load_shards(nightly_dir):
    """[(dir, nightly.json)] of every shard below ``nightly_dir``."""
    out = []
    for root, _, files in sorted(os.walk(nightly_dir or "")):
        if "nightly.json" in files:
            out.append((root, load_json(os.path.join(root, "nightly.json"), {})))
    return out


def attribute(r_xpu, r_pt, gating, delta):
    kw = dict(
        threshold=gating["threshold"], delta=delta, max_ci_width=gating["max_ci_width"]
    )
    vx, _ = classify(r_xpu, **kw)
    vp, _ = classify(r_pt, **kw)

    def slower(r):
        return r is not None and r["ci_lo"] > 1 + delta

    if slower(r_xpu) and slower(r_pt):
        return "mixed"
    if vx == "regression" and vp == "no_change":
        return "xpu-ops"
    if vx == "no_change" and (vp == "regression" or slower(r_pt)):
        return "pytorch"
    return "inconclusive"


def attribution_results(attr_dir, gating, noise, metric):
    """{case_id: {"attribution", "r_xpu", "r_pt"}} from the P / P' / N run."""
    raw = os.path.join(attr_dir or "", "raw")
    if not attr_dir or not os.path.isdir(raw):
        return {}
    groups = group_samples(load_records(raw), metric)
    out = {}
    for (cid, stage), g in groups.items():
        if stage != "attribution" or not all(g.get(s) for s in ("P", "Pp", "N")):
            continue
        delta = case_delta(cid, "normal", gating, noise)
        r_xpu = compare(g["Pp"], g["N"], delta, gating["n_boot"])
        r_pt = compare(g["P"], g["Pp"], delta, gating["n_boot"], seed=1)
        out[cid] = {
            "attribution": attribute(r_xpu, r_pt, gating, delta),
            "r_xpu": r_xpu,
            "r_pt": r_pt,
        }
    return out


def _key_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def issue_key(cat, candidates, prev_pt, cur_pt):
    shas = ",".join(sorted(c["sha"] for c in candidates or []))
    rng = f"{(prev_pt or '?')[:12]}..{(cur_pt or '?')[:12]}"
    if cat == "xpu-ops":
        return f"xpu-ops:{_key_hash(shas)}"
    if cat == "pytorch":
        return f"pytorch:{rng}"
    return f"mixed:{_key_hash(shas)}:{rng}"


def suggestions(state, screening, man):
    """Quarantine suggestions from the nightly A/A noise statistics."""
    stats = reference.noise_stats(state, screening["default_noise"])
    out = {"quarantine": []}
    for cid, st in sorted(stats.items()):
        if cid not in man:
            continue
        if st.get("delta") is not None and st["delta"] > 0.08:
            out["quarantine"].append(
                {
                    "case_id": cid,
                    "reason": f"A/A delta {st['delta']:.3f} over {st['aa_runs']} runs",
                }
            )
    return out


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _case_entry(cid, case, det, rc, attr, kernels, shard_dir):
    ra, rp = (attr or {}).get("r_xpu"), (attr or {}).get("r_pt")
    log = f"logs/{case_log_dir(case.op, case.pass_, cid) if case else cid}/"
    return {
        "case_id": cid,
        "op": case.op if case else None,
        "pass": case.pass_ if case else None,
        "label": case.label if case else cid,
        "params": case.params if case else None,
        "tier": case.tier if case else None,
        "before_us": rc.get("before_us"),
        "after_us": rc.get("after_us"),
        "ratio": rc.get("ratio"),
        "ci": rc.get("ci"),
        "detect_ratio": det.get("ratio"),
        "detect_reference_runs": det.get("reference_runs"),
        "r_xpu": {k: ra[k] for k in ("ratio", "ci_lo", "ci_hi")} if ra else None,
        "r_pt": {k: rp[k] for k in ("ratio", "ci_lo", "ci_hi")} if rp else None,
        "kernels": kernels or [],
        "log": f"{log} and recheck/{log} in {os.path.basename(shard_dir)}",
        "repro": case.repro() if case else None,
    }


def run_nightly_check(args):
    os.makedirs(args.out_dir, exist_ok=True)
    gating = load_gating(args.gating)
    hist = load_history(args.history)
    screening = reference.load_screening()
    man = load_cases()
    shards = load_shards(args.nightly_dir)
    if not shards:
        print("no nightly-vs-reference results")
        return 1
    metric = shards[0][1].get("metric", "device_us")
    cur, prev = {}, {}
    for _, d in shards:
        # a failed shard has no PyTorch SHA; take every value any shard has
        cur.update({k: v for k, v in (d.get("cur") or {}).items() if v})
        prev.update({k: v for k, v in (d.get("prev") or {}).items() if v})
    now = time.time()
    state = reference.load_state(args.state)
    noise = merge_noise(
        hist["noise_stats"], reference.noise_stats(state, screening["default_noise"])
    )

    detect, recheck, candidates, kernels, shard_of = {}, {}, {}, {}, {}
    new_entries = []
    for sdir, d in shards:
        for cid in d.get("detect") or {}:
            shard_of[cid] = sdir
        detect.update(d.get("detect") or {})
        recheck.update(d.get("recheck") or {})
        candidates.update(d.get("candidates") or {})
        kernels.update(d.get("kernels") or {})
        recs = read_jsonl(os.path.join(sdir, "records.jsonl")) + read_jsonl(
            os.path.join(sdir, "aa", "records.jsonl")
        )
        new_entries += reference.measure_entries(
            recs, args.run_id or cur.get("run_id"), now, metric=metric
        )
    attr = attribution_results(args.attribution_dir, gating, noise, metric)

    # classify confirmed regressions
    confirmed = {c: v for c, v in recheck.items() if v.get("verdict") == "regression"}
    improvements = {
        c: v for c, v in recheck.items() if v.get("verdict") == "improvement"
    }
    cat, inferred = {}, set()
    for cid in confirmed:
        cands = candidates.get(cid)
        if cands is None:
            cat[cid] = "unknown"
        elif not cands:
            cat[cid] = "pytorch"
            inferred.add(cid)
        else:
            cat[cid] = (attr.get(cid) or {}).get("attribution", "inconclusive")
    false_alarms = [
        c
        for c, r in detect.items()
        if r.get("status") in ("suspect_regression", "suspect_improvement")
        and (recheck.get(c) or {}).get("verdict")
        not in ("regression", "improvement", None)
    ]

    # jumps: confirmed regressions with an attribution, and improvements
    jumps = []
    for cid, c in cat.items():
        if c in ISSUE_LABELS:
            jumps.append(
                reference.jump_entry(
                    cid,
                    prev,
                    cur,
                    c,
                    [x["sha"] for x in candidates.get(cid) or []],
                    now,
                    inferred=cid in inferred,
                )
            )
    for cid in improvements:
        jumps.append(reference.jump_entry(cid, prev, cur, "improvement", (), now))
    new_state = reference.roll(state, new_entries + jumps, now)
    if args.state_out:
        reference.save_state(args.state_out, new_state)

    # issue groups: every regression the recheck A/B confirmed, except
    # quarantined cases and ops kept out of gating
    groups = {}
    report_only = []
    for cid, c in sorted(cat.items()):
        case = man.get(cid)
        why = report_only_reasons(case.manifest() if case else {"case_id": cid}, gating)
        if c not in ISSUE_LABELS or why:
            report_only.append((cid, c, ", ".join(why) or c))
            continue
        key = issue_key(
            c, candidates.get(cid), prev.get("pytorch_sha"), cur.get("pytorch_sha")
        )
        g = groups.setdefault(
            key,
            {
                "key": key,
                "attribution": c,
                "inferred": cid in inferred,
                "labels": ISSUE_LABELS[c],
                "candidates": candidates.get(cid) or [],
                "pytorch_range": [prev.get("pytorch_sha"), cur.get("pytorch_sha")],
                "compare_url": f"https://github.com/pytorch/pytorch/compare/{prev.get('pytorch_sha')}...{cur.get('pytorch_sha')}"
                if c in ("pytorch", "mixed")
                else None,
                "cases": [],
            },
        )
        g["cases"].append(
            _case_entry(
                cid,
                case,
                detect.get(cid) or {},
                confirmed[cid],
                attr.get(cid),
                kernels.get(cid),
                shard_of.get(cid, ""),
            )
        )
    evidence = {
        "run_id": args.run_id,
        "run_url": args.run_url,
        "test_type": args.test_type,
        "date": time.strftime("%Y-%m-%d", time.gmtime(now)),
        "prev": prev,
        "cur": cur,
        "threshold": gating["threshold"],
        "groups": list(groups.values()),
        "report_only": [
            {"case_id": c, "attribution": a, "reason": w} for c, a, w in report_only
        ],
    }
    with open(os.path.join(args.out_dir, "issue_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=1, default=str)

    sugg = suggestions(new_state, screening, man)
    counts = {}
    for r in detect.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [
        "## Microbench nightly report",
        "",
        f"Run `{args.run_id}` ({args.test_type or 'nightly'}); this nightly: xpu-ops `{cur.get('xpu_ops_sha')}`, "
        f"PyTorch `{cur.get('pytorch_sha')}`; previous nightly (P): run `{prev.get('run_id')}`, xpu-ops "
        f"`{prev.get('xpu_ops_sha')}`, PyTorch `{prev.get('pytorch_sha')}`.",
        f"Reference state: run `{args.state_run or 'none'}` ({len(state)} entries) -> "
        f"{len(new_state)} entries ({len(new_entries)} measurements, {len(jumps)} jumps added)"
        + ("" if args.state_out else " (not written)"),
        "",
        "### Measurement",
        f"- {len(detect)} case(s) over {len(shards)} shard(s)",
        "- detection: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        f"- recheck (P vs N, same card): {len(recheck)} case(s), {len(confirmed)} regression(s), "
        f"{len(improvements)} improvement(s), {len(false_alarms)} detection false alarm(s)",
    ]
    for _, d in shards:
        if d.get("recheck_skipped"):
            lines.append(
                f"- shard {d.get('shard')}: recheck skipped ({d['recheck_skipped']})"
            )
        if d.get("no_reference_not_rechecked"):
            lines.append(
                f"- shard {d.get('shard')}: {len(d['no_reference_not_rechecked'])} nightly-tier case(s) without "
                "reference not rechecked (too many)"
            )
    if confirmed:
        lines += [
            "",
            "### Confirmed regressions",
            "",
            "| Op | Case | Pass | Before (us) | After (us) | Ratio | 95% CI | N/P' | P'/P | Attribution | Candidates | Log |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for cid, v in sorted(
            confirmed.items(), key=lambda kv: -(kv[1].get("ratio") or 0)
        ):
            case, a = man.get(cid), attr.get(cid) or {}
            ci = v.get("ci") or [None, None]
            cands = candidates.get(cid)
            lines.append(
                f"| `{case.op if case else '?'}` | {case.label if case else cid} | {case.pass_ if case else '?'} | "
                f"{_fmt(v.get('before_us'))} | {_fmt(v.get('after_us'))} | {_fmt(v.get('ratio'))} | "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt((a.get('r_xpu') or {}).get('ratio'))} | "
                f"{_fmt((a.get('r_pt') or {}).get('ratio'))} | {cat[cid]}{' (inferred)' if cid in inferred else ''} | "
                f"{'unavailable' if cands is None else ', '.join(c['sha'][:10] for c in cands) or '-'} | "
                f"`logs/{case_log_dir(case.op, case.pass_, cid) if case else cid}/` |"
            )
    if improvements:
        lines += ["", f"### Confirmed improvements ({len(improvements)})", ""]
        lines += [
            f"- `{man[c].op if c in man else '?'}` {man[c].label if c in man else c}: ratio {_fmt(v.get('ratio'))}"
            for c, v in sorted(improvements.items())
        ]
    lines += ["", f"### Issues: {len(groups)} group(s) for the auto-issue job"]
    for g in groups.values():
        lines.append(
            f"- `{g['key']}` {g['attribution']}{' (inferred)' if g['inferred'] else ''}: {len(g['cases'])} case(s), "
            f"{len(g['candidates'])} candidate commit(s)"
        )
    if report_only:
        lines += [
            "",
            f"Report only ({len(report_only)}): "
            + ", ".join(f"`{c}` ({w})" for c, _, w in report_only[:40]),
        ]
    if false_alarms:
        lines += [
            "",
            f"Detection false alarms ({len(false_alarms)}): "
            + ", ".join(f"`{c}`" for c in false_alarms[:60]),
        ]
    if sugg["quarantine"]:
        lines += [
            "",
            "### Suggestions for config/gating.yaml (maintainer review)",
            "",
        ]
        lines += [
            f"- quarantine: `{s['case_id']}` ({s['reason']})"
            for s in sugg["quarantine"][:50]
        ]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "nightly_report.md"), "w") as f:
        f.write(md)
    summary = {
        "detect": detect,
        "recheck": recheck,
        "attribution": {c: {"attribution": a["attribution"]} for c, a in attr.items()},
        "categories": cat,
        "false_alarms": false_alarms,
        "suggestions": sugg,
    }
    with open(os.path.join(args.out_dir, "nightly_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md)
    print(md)
    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"has_regressions={str(bool(groups)).lower()}\n")
    return 0


# ---------------------------------------------------------------------------
# Check job (scope ondemand; nightly: run_nightly_check above)
# ---------------------------------------------------------------------------
# The on-demand report never fails: it is for a person to read.


def _freq(meta):
    vals = {
        str(d["min"]) if d["min"] == d["max"] else f"{d['min']}-{d['max']}"
        for card in (meta.get("env_start") or {}).values()
        for d in card.get("freq") or []
        if d.get("min") and d.get("max")
    }
    return ",".join(sorted(vals)) or None


def flow_text(selection, ab_doc):
    path = (ab_doc or {}).get("path") or selection.get("path")
    if path == "aa":
        return "Flow: A/A, the same wheel on both sides (report only)."
    if path == "vs-base":
        n_ab = len((ab_doc or {}).get("cases") or [])
        return f"Flow: direct A/B (requested on demand): {n_ab} case(s) A/B-tested against the base build."
    return ""


def _dir_file(d, name):
    return (
        os.path.join(d, name) if d and os.path.exists(os.path.join(d, name)) else None
    )


def run_check(args):
    if args.scope == "nightly":
        return run_nightly_check(args)
    os.makedirs(args.out_dir, exist_ok=True)
    ab_dir = (
        args.data_dir
        if args.data_dir and os.path.isdir(os.path.join(args.data_dir, "raw"))
        else None
    )
    ab_doc = load_json(_dir_file(ab_dir, "ab_cases.json"), None)
    selection = load_json(args.selection, None) or load_json(
        _dir_file(ab_dir, "selection.json"), {}
    )
    meta = load_json(_dir_file(ab_dir, "run_meta.json"), {})
    gating = load_gating(args.gating)
    manifest = {cid: c.manifest() for cid, c in load_cases().items()}
    if ab_dir:
        cases = (ab_doc or {}).get("cases") or selection.get("cases")
        result = judge(
            os.path.join(ab_dir, "raw"),
            manifest,
            meta,
            selection={"cases": cases},
            gating=gating,
            noise=load_history(args.history)["noise_stats"],
        )
    else:
        result = summarize_rows({"mode": "ab", "rows": [], "gating": gating})

    shas = meta.get("torch_git_version") or {}
    ctx = {
        "mode_label": args.mode_label,
        "env": {
            "runner": meta.get("machine_id") or meta.get("host"),
            "cards": ",".join(
                f"{c.get('index')} ({c.get('card_id')})" for c in meta.get("cards", [])
            ),
            "GPU freq (MHz)": _freq(meta),
            "base": args.base_sha or selection.get("base"),
            "B": args.head_sha or selection.get("head"),
            "PyTorch": shas.get("B") or selection.get("pytorch_sha"),
        },
        "selection": selection,
        "run_url": args.run_url,
        "flow": flow_text(selection, ab_doc),
    }
    md = render(result, ctx)
    with open(os.path.join(args.out_dir, "verdicts.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(md)
    write_all_cases(result, args.out_dir)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md)
    print(md)
    return 0


def add_check_parser(sub):
    p = sub.add_parser(
        "check",
        help="final verdicts: on-demand A/B / A/A report, or the nightly report",
    )
    p.add_argument("--scope", choices=["ondemand", "nightly"], default="ondemand")
    p.add_argument(
        "--data-dir",
        "--ab-dir",
        dest="data_dir",
        default=None,
        help="A/B run output (run_meta.json, raw/, ab_cases.json)",
    )
    p.add_argument("--selection", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gating", default=None)
    p.add_argument("--history", default=None, help="default: config/history.json")
    p.add_argument("--head-sha", default=None)
    p.add_argument("--base-sha", default=None)
    p.add_argument("--run-url", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--mode-label", default=None)
    # nightly
    p.add_argument(
        "--nightly-dir", default=None, help="nightly: shard outputs (nightly.json)"
    )
    p.add_argument("--attribution-dir", default=None, help="nightly: P/P'/N run output")
    p.add_argument("--state", default=None, help="nightly: previous reference.jsonl")
    p.add_argument("--state-run", default=None, help="nightly: run id of that state")
    p.add_argument(
        "--state-out", default=None, help="nightly: write the new state here"
    )
    p.add_argument(
        "--test-type",
        default=None,
        help="nightly: build-nightly / build-weekly / ondemand",
    )
    p.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=lambda a: sys.exit(run_check(a)))
