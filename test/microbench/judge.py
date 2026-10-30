# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Verdicts: A/B statistics, per-case judgement, English report, check job.

``python -m microbench check`` is the microbench-check CI job: it fails only
for a blocking regression without a valid maintainer override.
"""

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
from .github import OVERRIDE_LABEL, REPORT_MARKER as COMMENT_MARKER

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


def not_blocking_reasons(man, gating, quarantine=None):
    """Why a case (manifest or verdict row) cannot block a PR; empty = it can."""
    q_ids, q_ops = quarantine or gating.get("quarantine", ({}, {}))
    return [
        reason
        for reason, cond in (
            ("nightly tier", man.get("tier") != "pr"),
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
    if screen is None:
        # confirm-only: the direction came from screening against reference
        # data; combine_pr checks that the two agree
        v_c, why_c = classify(confirm, p_reg_adj=p_reg, p_imp_adj=p_imp, **kw)
        return v_c, why_c, confirm
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
    quarantine=None,
):
    """Compute verdicts. ``cases_manifest`` maps case_id -> manifest dict."""
    gating = gating or load_gating()
    noise = noise if noise is not None else load_history()["noise_stats"]
    q_ids, q_ops = (
        quarantine if quarantine is not None else gating.get("quarantine", ({}, {}))
    )
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
                row["pr_median_us"] = statistics.median(
                    v for grp in g["B"] for v in grp
                )
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
        set_blocking(r, gating, (q_ids, q_ops), mode)

    return summarize_rows({"mode": mode, "rows": rows, "gating": gating})


def set_blocking(r, gating, quarantine, mode):
    """Blocking flags of one verdict row: only device-time regressions (E2E
    for noise classes listed in gating.yaml) of cases that can block."""
    e2e_gates = r.get("noise_class") in gating["e2e_blocking_noise_classes"]
    reasons = (["A/A run"] if mode != "ab" else []) + not_blocking_reasons(
        r, gating, quarantine
    )
    r["blocking_eligible"] = not reasons
    r["not_blocking_reasons"] = reasons
    r["blocking_metrics"] = [
        m
        for m in ("device", "e2e")
        if not reasons
        and (m == "device" or e2e_gates)
        and (r["metrics"].get(m) or {}).get("verdict") == "regression"
    ]
    r["blocking"] = bool(r["blocking_metrics"])


def summarize_rows(result):
    rows = result["rows"]

    def with_verdict(v):
        return [
            r["case_id"]
            for r in rows
            if any(m.get("verdict") == v for m in r["metrics"].values())
        ]

    result.update(
        blocking=[r["case_id"] for r in rows if r.get("blocking")],
        regressions=with_verdict("regression"),
        improvements=with_verdict("improvement"),
        unconfirmed_improvements=[
            r["case_id"] for r in rows if r.get("verdict") == "unconfirmed_improvement"
        ],
    )
    return result


# ---------------------------------------------------------------------------
# PR conclusion matrix
# ---------------------------------------------------------------------------
# | screening                  | A/B                        | conclusion          |
# |----------------------------|----------------------------|---------------------|
# | not suspicious             | -                          | no change (screened)|
# | suspect_regression         | regression                 | regression          |
# | suspect_regression         | no change                  | no change (screened |
# |                            |                            | but not confirmed)  |
# | suspect_regression         | inconclusive / improvement | inconclusive        |
# | undirected / direct A/B    | two-stage A/B              | its verdict         |
# | suspect_improvement        | improvement                | improvement         |
# | suspect_improvement        | not A/B-tested             | unconfirmed improv. |
# | suspicious, not A/B-tested | -                          | inconclusive        |


def _screened_row(cid, man, gating, noise, s, truncated):
    nc = man.get("noise_class", "normal")
    row = _row(cid, man, case_delta(cid, nc, gating, noise))
    row.update(status="screened", screen=s)
    st = s.get("status")
    if st == "error":
        row.update(
            status="error",
            verdict=None,
            detail=s.get("reason") or "error on the PR wheel",
        )
    elif st == "no_change":
        row.update(verdict="no_change_screened", reason="")
    elif st == "suspect_improvement" and cid not in truncated:
        row.update(verdict="unconfirmed_improvement", reason="vs reference, no A/B")
    elif cid in truncated:
        row.update(
            verdict="inconclusive",
            reason="suspicious in screening; A/B truncated by the time budget",
        )
    else:
        why = s.get("reason") or st
        row.update(
            verdict="inconclusive",
            reason=f"suspicious in screening ({why}); not A/B-tested (report only)",
        )
    if row.get("verdict") == "inconclusive" and st in (
        "suspect_regression",
        "suspect_improvement",
    ):
        row["reason"] += f", screen ratio {s.get('ratio', 0):.3f}"
    return row


def combine_pr(result, screen_doc, ab_doc, manifest, gating, noise, ab_possible=True):
    """Merge screening (pr-vs-reference) and A/B (pr-vs-base) (final verdict table above)."""
    screen = (screen_doc or {}).get("cases") or {}
    truncated = set((ab_doc or {}).get("truncated") or [])
    rows = {r["case_id"]: r for r in result["rows"]}
    q = gating.get("quarantine", ({}, {}))
    unknown = {
        "op": "?",
        "pass": "?",
        "tier": "nightly",
        "noise_class": "normal",
        "gating": False,
    }
    for cid, s in screen.items():
        r = rows.get(cid)
        if r is None:
            r = _screened_row(
                cid,
                manifest.get(cid) or dict(unknown, label=cid),
                gating,
                noise,
                s,
                truncated,
            )
            result["rows"].append(r)
            rows[cid] = r
            set_blocking(r, gating, q, "ab" if ab_possible else "aa")
            continue
        r["screen"] = s
        st, dev = s.get("status"), r["metrics"].get("device") or r["metrics"].get("e2e")
        if not dev or r["status"] != "ok":
            continue
        v = dev.get("verdict")
        if st == "suspect_regression" and v == "no_change":
            r["screened_not_confirmed"] = True
        elif st == "suspect_regression" and v != "regression":
            dev.update(
                verdict="inconclusive",
                reason=f"screening said regression, A/B said {v}",
            )
        elif st == "suspect_improvement" and v != "improvement" and v != "no_change":
            dev.update(
                verdict="inconclusive",
                reason=f"screening said improvement, A/B said {v}",
            )
        r["verdict"] = dev.get("verdict")
        set_blocking(r, gating, q, result["mode"])
    if screen and not ab_possible:
        result["mode"] = "aa"
    return summarize_rows(result)


# ---------------------------------------------------------------------------
# Report and PR comment
# ---------------------------------------------------------------------------


def _fmt_us(x):
    return "-" if x is None else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _ratio_ci(res):
    if not res:
        return ["-", "-"]
    return [f"{res['ratio']:.3f}", f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]"]


def _entries(rows, verdict):
    """(row, metric) pairs with ``verdict``: blocking first, then by magnitude."""
    out = [
        (r, m)
        for r in rows
        for m, mm in r["metrics"].items()
        if mm.get("verdict") == verdict
    ]

    def key(item):
        r, m = item
        ratio = (r["metrics"][m].get("used") or {}).get("ratio", 1.0)
        return (m not in r.get("blocking_metrics", []), -abs(ratio - 1.0))

    return sorted(out, key=key)


def _table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def _screen_ratio(r):
    s = r.get("screen") or {}
    return f"{s['ratio']:.3f}" if s.get("ratio") is not None else "-"


def _ref_runs(s):
    runs = s.get("reference_runs") or []
    return (
        ", ".join(f"run {rid} {d} (`{(sha or '?')[:10]}`)" for rid, d, sha in runs)
        or "-"
    )


def render(result, ctx):
    """Markdown report. ``ctx`` keys: mode_label, override, env, selection,
    new_ops, run_url, flow, notes."""
    rows = result["rows"]
    regs, imps = _entries(rows, "regression"), _entries(rows, "improvement")
    unconf = [r for r in rows if r.get("verdict") == "unconfirmed_improvement"]
    blocking = [r for r in rows if r.get("blocking")]
    override = ctx.get("override") or {}
    counts = f"{len({r['case_id'] for r, _ in regs})} regression(s), {len({r['case_id'] for r, _ in imps})} improvement(s)"
    if unconf:
        counts += f", {len(unconf)} unconfirmed improvement(s)"
    if result["mode"] != "ab":
        status = f"PASS (A/A run, report only) - {counts}"
    elif blocking and override.get("valid"):
        status = (
            f"PASS (regression accepted by @{override.get('approver')} at {override.get('time')} "
            f"for commit {override.get('sha', '')[:12]}) - {counts}"
        )
    elif blocking:
        status = f"FAIL - blocking regression detected ({len(blocking)} blocking case(s); {counts})"
    elif regs or imps or unconf:
        status = f"PASS - no blocking regression ({counts})"
    else:
        status = "PASS - no significant change in this run"
    lines = [COMMENT_MARKER, "## Microbench Performance Report", ""]
    if ctx.get("mode_label"):
        lines += [f"**MODE: {ctx['mode_label']}**", ""]
    thr = int(round(result["gating"]["threshold"] * 100))
    lines += [
        f"**Result:** {status}",
        f"_Threshold: {thr}% on device time (E2E is report-only). "
        + (f"{ctx['flow']} " if ctx.get("flow") else "")
        + f"Maintainers can add label `{OVERRIDE_LABEL}` to accept a regression for the current commit._",
    ]
    if override.get("ignored"):
        lines += ["", f"_Override label ignored: {override['ignored']}_"]
    if ctx.get("notes"):
        lines += ["", *(f"**Note:** {n}" for n in ctx["notes"])]

    def cells(r, m):
        res = r["metrics"][m].get("used") or {}
        base = [f"`{r['op']}`", r["label"], r["pass"], m]
        return base + [
            _fmt_us(res.get("median_a")),
            _fmt_us(res.get("median_b")),
            *_ratio_ci(res),
            _screen_ratio(r),
        ]

    header = [
        "Op",
        "Case",
        "Pass",
        "Metric",
        "Base (us)",
        "PR (us)",
        "Ratio",
        "95% CI",
        "Screen ratio",
    ]
    if regs:
        lines += ["", "### Regressions"]
        lines += _table(
            header + ["Blocking", "Log"],
            [cells(r, m) + [_blocking_cell(r, m), _log_cell(r)] for r, m in regs],
        )
    if imps:
        lines += ["", "### Improvements"] + _table(
            header + ["Log"],
            [cells(r, m) + [_log_cell(r)] for r, m in imps],
        )
    if unconf:
        lines += ["", "### Unconfirmed improvements (vs reference, no A/B)"]
        lines += _table(
            [
                "Op",
                "Case",
                "Pass",
                "Reference (us)",
                "PR (us)",
                "Ratio",
                "Reference nightlies",
            ],
            [
                [
                    f"`{r['op']}`",
                    r["label"],
                    r["pass"],
                    _fmt_us(r["screen"].get("reference_us")),
                    _fmt_us(r["screen"].get("new_us")),
                    _screen_ratio(r),
                    _ref_runs(r["screen"]),
                ]
                for r in unconf
            ],
        )

    inconc = _entries(rows, "inconclusive")
    inconc_screen = [
        r
        for r in rows
        if r.get("status") == "screened" and r.get("verdict") == "inconclusive"
    ]
    not_confirmed = [r for r in rows if r.get("screened_not_confirmed")]
    others = [r for r in rows if r["status"] not in ("ok", "screened")]
    sel = ctx.get("selection") or {}
    truncated = sel.get("truncated") or []
    if inconc or inconc_screen or not_confirmed or others or truncated:
        lines += [
            "",
            "<details><summary>Inconclusive / screened-but-not-confirmed / truncated cases</summary>",
            "",
        ]
        if inconc or inconc_screen:
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
                ]
                + [
                    [
                        f"`{r['op']}`",
                        r["label"],
                        r["pass"],
                        "device",
                        _screen_ratio(r),
                        "-",
                        r["reason"],
                    ]
                    for r in inconc_screen
                ],
            )
        if not_confirmed:
            lines += [
                "",
                f"Screened as suspicious but no change in the A/B ({len(not_confirmed)}): "
                + ", ".join(
                    f"`{r['op']}` {r['label']} (screen {_screen_ratio(r)})"
                    for r in not_confirmed[:30]
                ),
            ]
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
                        f"PR median {_fmt_us(r['pr_median_us'])} us"
                        if r.get("pr_median_us")
                        else r.get("detail", ""),
                    ]
                    for r in others
                ],
            )
        if truncated:
            lines += [
                "",
                f"Truncated by the time budget ({len(truncated)} case(s)); run an on-demand A/B "
                "(`nightly_ondemand.yml`, `ut=microbench`, `microbench_base=merge-base`) for the full list: "
                + ", ".join(f"`{t}`" for t in truncated[:50]),
            ]
        lines += ["", "</details>"]

    if sel:
        lines += [
            "",
            "<details><summary>Case selection (changed files -> ops)</summary>",
            "",
        ]
        lines += [f"- {reason}" for reason in sel.get("reasons") or []]
        if sel.get("file_map"):
            lines += [""] + _table(
                ["Changed file", "Ops"],
                [
                    [f"`{f}`", ", ".join(ops) or "-"]
                    for f, ops in sorted(sel["file_map"].items())
                ],
            )
        ops = ", ".join(sel.get("ops") or [])
        lines += [
            "",
            f"Selected {len(sel.get('cases') or [])} case(s) over ops: {ops}",
            "",
            "</details>",
        ]

    screened = [r for r in rows if r.get("screen")]
    if screened:
        by_status = {}
        for r in screened:
            by_status.setdefault(r["screen"].get("status"), []).append(r)
        lines += [
            "",
            "<details><summary>Reference data used</summary>",
            "",
            f"Reference state of nightly run `{sel.get('reference_run') or 'none'}`. Screening: "
            + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_status.items())),
            "",
        ]
        flagged = [r for r in screened if r["screen"].get("status") != "no_change"][:50]
        if flagged:
            lines += _table(
                [
                    "Op",
                    "Case",
                    "Pass",
                    "Screening",
                    "Screen ratio",
                    "Threshold",
                    "Reference nightlies",
                ],
                [
                    [
                        f"`{r['op']}`",
                        r["label"],
                        r["pass"],
                        r["screen"].get("status"),
                        _screen_ratio(r),
                        f"{r['screen']['threshold']:.3f}"
                        if r["screen"].get("threshold") is not None
                        else "-",
                        _ref_runs(r["screen"]),
                    ]
                    for r in flagged
                ],
            )
        lines += ["", "</details>"]

    if ctx.get("new_ops"):
        lines += [
            "",
            "<details><summary>New XPU ops detected (informational)</summary>",
            "",
        ]
        lines += _table(
            ["Op", "Where", "Status"],
            [[f"`{n['op']}`", f"`{n['where']}`", n["status"]] for n in ctx["new_ops"]],
        )
        lines += ["", "</details>"]

    env = {k: v for k, v in (ctx.get("env") or {}).items() if v}
    if env:
        lines += [
            "",
            "**Environment:** " + ", ".join(f"{k} `{v}`" for k, v in env.items()),
        ]
    if ctx.get("run_url"):
        lines.append(f"**Artifacts:** {ctx['run_url']}")
    lines.append(
        f"**All {len(rows)} case(s):** `all_cases.md` / `all_cases.csv` in the `microbench-comment-*` artifact; "
        "per-case logs (`logs/<op>/<pass>_<case_id>/`, index in `logs/index.csv`) in the `microbench-data-*` artifacts"
    )
    reps = regs or imps
    if reps:
        lines.append(f"**Reproduce:** `{reps[0][0].get('repro')}`")
    return "\n".join(lines) + "\n"


def _log_cell(r):
    return f"`logs/{case_log_dir(r['op'], r['pass'], r['case_id'])}/`"


def _blocking_cell(r, metric):
    if metric in r.get("blocking_metrics", []):
        return "Yes"
    if metric == "e2e":
        return "No (E2E is report-only)"
    return "No (" + ", ".join(r.get("not_blocking_reasons") or ["report-only"]) + ")"


ALL_CASES_HEADER = [
    "Op",
    "Case",
    "Pass",
    "Metric",
    "Base (us)",
    "PR (us)",
    "Ratio",
    "95% CI",
    "Screening",
    "Screen ratio",
    "Verdict",
    "Reason",
    "Case id",
]


def all_cases(result):
    """One row per (case, metric) of every selected case; cases without A/B
    data get one row with their status (screened cases: reference vs PR)."""
    out = []
    for r in result["rows"]:
        head = [r["op"], r["label"], r["pass"]]
        scr = [(r.get("screen") or {}).get("status", "-"), _screen_ratio(r)]
        if r["status"] == "screened":
            s = r["screen"]
            out.append(
                head
                + [
                    "device",
                    _fmt_us(s.get("reference_us")),
                    _fmt_us(s.get("new_us")),
                    _screen_ratio(r),
                    "-",
                ]
                + scr
                + [r.get("verdict") or "-", r.get("reason", ""), r["case_id"]]
            )
            continue
        if r["status"] != "ok":
            pr = _fmt_us(r["pr_median_us"]) if r.get("pr_median_us") else "-"
            out.append(
                head
                + ["-", "-", pr, "-", "-"]
                + scr
                + [r["status"], r.get("detail", ""), r["case_id"]]
            )
            continue
        for metric, m in r["metrics"].items():
            res = m.get("used") or {}
            base, pr = _fmt_us(res.get("median_a")), _fmt_us(res.get("median_b"))
            out.append(
                head
                + [metric, base, pr, *_ratio_ci(res)]
                + scr
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


def should_comment(result, ctx):
    """Post when there is something to say; otherwise only refresh an existing comment."""
    if (
        result.get("regressions")
        or result.get("improvements")
        or result.get("unconfirmed_improvements")
        or ctx.get("new_ops")
    ):
        return "post"
    return "update_only"


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
    """screen_mode / quarantine suggestions from the noise statistics."""
    stats = reference.noise_stats(state, screening["default_noise"])
    cfg = screening["pr_vs_reference"]
    always = {e["case_id"] for e in cfg.get("always_ab") or []}
    out = {"always_ab": [], "quarantine": []}
    for cid, st in sorted(stats.items()):
        m = man.get(cid)
        if not m:
            continue
        if m.tier == "pr" and st["runs"] >= 7 and cid not in always:
            sig = reference.sigmas(st, screening["default_noise"])
            fake_ref = [
                {"run_id": i, "procs": [1.0] * 3} for i in range(cfg["max_records"])
            ]
            t, _ = reference.threshold(sig, cfg["processes"], fake_ref, True, cfg)
            if t < cfg["t_min"]:
                out["always_ab"].append(
                    {
                        "case_id": cid,
                        "reason": f"sigma_run {sig['run']:.3f}, t_s {t:.3f} ({st['runs']} nightlies)",
                    }
                )
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
    first = shards[0][1]
    metric = first.get("metric", "device_us")
    cur = dict(first.get("cur") or {})
    prev = {}
    for _, d in shards:
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
        why = [
            r
            for r in not_blocking_reasons(
                case.manifest() if case else {"case_id": cid}, gating
            )
            if r != "nightly tier"
        ]
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
    if sugg["always_ab"] or sugg["quarantine"]:
        lines += [
            "",
            "### Suggestions for config/gating.yaml (maintainer review)",
            "",
        ]
        lines += [
            f"- always_ab: `{s['case_id']}` ({s['reason']})"
            for s in sugg["always_ab"][:50]
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
# Check job (scope pr / ondemand; nightly: run_nightly_check above)
# ---------------------------------------------------------------------------
# Exit 1 only for a blocking regression without a valid override. Everything
# else (A/A runs, inconclusive cases, environment anomalies) passes.


def _freq(meta):
    vals = {
        str(d["min"]) if d["min"] == d["max"] else f"{d['min']}-{d['max']}"
        for card in (meta.get("env_start") or {}).values()
        for d in card.get("freq") or []
        if d.get("min") and d.get("max")
    }
    return ",".join(sorted(vals)) or None


def flow_text(selection, screen_doc, ab_doc):
    path = (ab_doc or {}).get("path") or selection.get("path")
    n_ab = len((ab_doc or {}).get("cases") or [])
    if path == "aa":
        return "Flow: A/A, the same wheel on both sides (report only)."
    if path == "vs-base":
        return f"Flow: direct A/B (requested on demand): {n_ab} case(s) A/B-tested against the base build."
    if not screen_doc:
        return ""
    n = len(screen_doc.get("cases") or {})
    if not selection.get("ab_possible", True):
        return f"Flow: A/A (only microbench code changed): {n} case(s) of the PR wheel compared with reference data (report only)."
    if ab_doc:
        return (
            f"Flow: screened {n} case(s) against reference data from the last nightlies; "
            f"{len(screen_doc.get('ab_cases') or [])} suspicious case(s), {n_ab} confirmed by A/B against the merge-base build."
        )
    return f"Flow: screened {n} case(s) against reference data from the last nightlies; none needed an A/B."


def report_notes(selection, screen_doc):
    notes = []
    if selection.get("protocol_note"):
        notes.append(selection["protocol_note"])
    cases = (screen_doc or {}).get("cases") or {}
    if cases:
        sus = [
            c for c, s in cases.items() if s.get("status") not in ("no_change", "error")
        ]
        frac = len(sus) / len(cases)
        if frac > 0.5:
            notes.append(
                f"{len(sus)}/{len(cases)} cases were suspicious in screening: check the environment, the "
                "reference data and the PyTorch commit difference."
            )
        if not selection.get("ab_possible", True) and frac > 0.3:
            notes.append(
                "Many cases moved against the reference data while only microbench code changed: if the "
                "framework change affects measured values, bump PROTOCOL_VERSION."
            )
    return notes


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
    screen_doc = load_json(_dir_file(args.screen_dir, "screen.json"), None)
    ab_doc = load_json(_dir_file(ab_dir, "ab_cases.json"), None)
    selection = load_json(args.selection, None) or load_json(
        _dir_file(args.screen_dir, "selection.json")
        or _dir_file(ab_dir, "selection.json"),
        {},
    )
    meta = load_json(
        _dir_file(ab_dir, "run_meta.json")
        or _dir_file(args.screen_dir, "run_meta.json"),
        {},
    )
    gating = load_gating(args.gating)
    # the select job matched the nightly reference state: its A/A deltas
    noise = merge_noise(
        load_history(args.history)["noise_stats"], selection.get("noise")
    )
    manifest = {cid: c.manifest() for cid, c in load_cases().items()}
    if ab_dir:
        cases = (ab_doc or {}).get("cases") or selection.get("cases")
        result = judge(
            os.path.join(ab_dir, "raw"),
            manifest,
            meta,
            selection={"cases": cases},
            gating=gating,
            noise=noise,
        )
    else:
        result = summarize_rows({"mode": "ab", "rows": [], "gating": gating})
    result = combine_pr(
        result,
        screen_doc,
        ab_doc,
        manifest,
        gating,
        noise,
        selection.get("ab_possible", True),
    )

    override = {"valid": False}
    if args.override_json:
        override = load_json(args.override_json)
    elif result["blocking"] and args.pr and args.gh_repo and args.head_sha:
        from . import github

        try:
            override = github.fetch_override(args.gh_repo, args.pr, args.head_sha)
        except RuntimeError as e:
            override = {"valid": False, "ignored": f"could not read labels: {e}"}

    shas = meta.get("torch_git_version") or {}
    ctx = {
        "mode_label": args.mode_label,
        "override": override,
        "env": {
            "runner": meta.get("machine_id") or meta.get("host"),
            "cards": ",".join(
                f"{c.get('index')} ({c.get('card_id')})" for c in meta.get("cards", [])
            ),
            "GPU freq (MHz)": _freq(meta),
            "base": args.base_sha or selection.get("base"),
            "PR": args.head_sha or selection.get("head"),
            "PyTorch": shas.get("B")
            or shas.get("single")
            or selection.get("pytorch_sha"),
        },
        "selection": selection,
        "new_ops": selection.get("new_ops"),
        "run_url": args.run_url,
        "flow": flow_text(selection, screen_doc, ab_doc),
        "notes": report_notes(selection, screen_doc),
    }
    md = render(result, ctx)
    failed = (
        bool(result["blocking"])
        and result["mode"] == "ab"
        and not override.get("valid")
    )
    result.update(override=override, result="fail" if failed else "pass")
    with open(os.path.join(args.out_dir, "verdicts.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(args.out_dir, "comment.md"), "w") as f:
        f.write(md)
    write_all_cases(result, args.out_dir)
    comment = {
        "pr": args.pr,
        "head_sha": args.head_sha,
        "action": should_comment(result, ctx),
        "marker": COMMENT_MARKER,
        "result": result["result"],
    }
    with open(os.path.join(args.out_dir, "comment.json"), "w") as f:
        json.dump(comment, f)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md)
    print(md)
    return 1 if failed else 0


def add_check_parser(sub):
    p = sub.add_parser(
        "check", help="final verdicts: PR / on-demand report, or the nightly report"
    )
    p.add_argument("--scope", choices=["pr", "ondemand", "nightly"], default="pr")
    p.add_argument(
        "--data-dir",
        "--ab-dir",
        dest="data_dir",
        default=None,
        help="A/B run output (run_meta.json, raw/, ab_cases.json)",
    )
    p.add_argument(
        "--screen-dir", default=None, help="pr-vs-reference output (screen.json)"
    )
    p.add_argument("--selection", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gating", default=None)
    p.add_argument("--history", default=None, help="default: config/history.json")
    p.add_argument("--pr", default=None)
    p.add_argument("--gh-repo", default=None)
    p.add_argument("--head-sha", default=None)
    p.add_argument("--base-sha", default=None)
    p.add_argument(
        "--override-json", default=None, help="precomputed override status (tests)"
    )
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
