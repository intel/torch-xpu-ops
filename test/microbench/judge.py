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

import json
import math
import os
import statistics
import sys

import numpy as np

from . import CANARY_ID, CONFIG_DIR, load_history, load_json, load_yaml, read_jsonl
from .case import load_cases
from .github import OVERRIDE_LABEL, REPORT_MARKER as COMMENT_MARKER

# ---------------------------------------------------------------------------
# A/B statistics (plan 7)
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
    "require_calibration": True,
    "e2e_blocking_noise_classes": [],
    "e2e_threshold": 0.20,
    "canary_drift": 0.03,
    "canary_history_tol": 0.05,
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
    """{(case_id, stage): {"A": [[...], ...], "B": [[...], ...]}} from worker records."""
    out = {}
    for r in records:
        if r.get("type") != "result" or r.get("status") != "ok":
            continue
        if metric == "device_us" and r.get("device_status") not in ("ok", None):
            continue
        vals = [v for v in r.get(metric) or [] if v is not None]
        if vals:
            key = (r["case_id"], r.get("stage", "screen"))
            out.setdefault(key, {"A": [], "B": []})[r["side"]].append(vals)
    return out


# ---------------------------------------------------------------------------
# Per-case verdicts (plan 7)
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


def evaluate_canary(recs, meta, gating, history):
    """Return (ok, details). Uses device time when present, else E2E."""
    per = {}
    for r in recs:
        if (
            r.get("type") == "result"
            and r.get("case_id") == CANARY_ID
            and r.get("status") == "ok"
        ):
            vals = [
                v
                for v in (r.get("device_us") or r.get("e2e_us") or [])
                if v is not None
            ]
            if vals:
                per.setdefault(str(r.get("card")), {})[r.get("stage")] = (
                    statistics.median(vals)
                )
    details, ok = [], True
    host_hist = (history or {}).get(meta.get("host") or "", {})
    for card, st in sorted(per.items()):
        s, e = st.get("canary_start"), st.get("canary_end")
        d = {"card": card, "start_us": s, "end_us": e}
        if s and e:
            d["drift"] = e / s - 1
            if abs(d["drift"]) > gating["canary_drift"]:
                ok = False
                d["problem"] = f"canary drifted {d['drift']:+.1%} during the run"
        h = host_hist.get(card)
        if h and s:
            d["history_us"], d["vs_history"] = h, s / h - 1
            if abs(d["vs_history"]) > gating["canary_history_tol"]:
                ok = False
                d["problem"] = f"canary {d['vs_history']:+.1%} off this card's history"
        details.append(d)
    return ok, details


def _final(screen, confirm, gating, delta, p_adj=None):
    """Two-stage decision. Returns (verdict, reason, result_used)."""
    kw = dict(
        threshold=gating["threshold"],
        delta=delta,
        max_ci_width=gating["max_ci_width"],
        alpha=gating["alpha"],
    )
    flagged = ("regression", "improvement")
    v_s, why_s = classify(screen, **kw)
    if confirm is None:
        if v_s in flagged:
            return "inconclusive", "flagged in screening, no confirmation data", screen
        return v_s, why_s, screen
    p_reg, p_imp = p_adj or (None, None)
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
    canary_history=None,
):
    """Compute verdicts. ``cases_manifest`` maps case_id -> manifest dict."""
    gating = gating or load_gating()
    noise = noise if noise is not None else load_history()["noise_stats"]
    q_ids, q_ops = (
        quarantine if quarantine is not None else gating.get("quarantine", ({}, {}))
    )
    recs = load_records(raw_dir)
    mode = meta.get("mode", "ab")
    canary_ok, canary = evaluate_canary(recs, meta, gating, canary_history)

    status, errors = {}, {}
    for r in recs:
        if r.get("type") != "result" or r.get("case_id") == CANARY_ID:
            continue
        status.setdefault(r["case_id"], {"A": set(), "B": set()})[r["side"]].add(
            r.get("status")
        )
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
        row = _row(
            cid,
            man,
            (noise.get(cid) or {}).get("delta")
            or gating["default_delta"].get(nc, 0.03),
        )
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
    ok_rows = [
        r
        for r in rows
        if r["status"] == "ok" and (r["metrics"].get("device") or {}).get("screen")
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
            if not mres.get("screen"):
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
            if v in ("regression", "improvement") and not canary_ok:
                v, why = "inconclusive", "environment anomaly (canary)"
            mres.update(verdict=v, reason=why, used=used)
            if p:
                mres["p_holm"] = p[1] if v == "improvement" else p[0]
        r["verdict"] = (
            r["metrics"].get("device") or r["metrics"].get("e2e") or {}
        ).get("verdict")
        reasons = [
            reason
            for reason, cond in (
                ("A/A run", mode != "ab"),
                ("nightly tier", r["tier"] != "pr"),
                ("op excluded from gating", not r.get("gating", True)),
                ("quarantined", r["case_id"] in q_ids or r["op"] in q_ops),
                (
                    "not A/A calibrated yet",
                    gating["require_calibration"]
                    and not (noise.get(r["case_id"]) or {}).get("calibrated"),
                ),
            )
            if cond
        ]
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

    def with_verdict(v):
        return [
            r["case_id"]
            for r in rows
            if any(m.get("verdict") == v for m in r["metrics"].values())
        ]

    return {
        "mode": mode,
        "canary_ok": canary_ok,
        "canary": canary,
        "rows": rows,
        "blocking": [r["case_id"] for r in rows if r.get("blocking")],
        "regressions": with_verdict("regression"),
        "improvements": with_verdict("improvement"),
        "gating": gating,
    }


# ---------------------------------------------------------------------------
# Report and PR comment (plan 8.1)
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


def render(result, ctx):
    """Markdown report. ``ctx`` keys: mode_label, override, env, selection, new_ops, run_url."""
    rows = result["rows"]
    regs, imps = _entries(rows, "regression"), _entries(rows, "improvement")
    blocking = [r for r in rows if r.get("blocking")]
    override = ctx.get("override") or {}
    counts = f"{len({r['case_id'] for r, _ in regs})} regression(s), {len({r['case_id'] for r, _ in imps})} improvement(s)"
    if result["mode"] != "ab":
        status = f"PASS (A/A run, report only) - {counts}"
    elif blocking and override.get("valid"):
        status = (
            f"PASS (regression accepted by @{override.get('approver')} at {override.get('time')} "
            f"for commit {override.get('sha', '')[:12]}) - {counts}"
        )
    elif blocking:
        status = f"FAIL - blocking regression detected ({len(blocking)} blocking case(s); {counts})"
    elif regs or imps:
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
        f"Maintainers can add label `{OVERRIDE_LABEL}` to accept a regression for the current commit._",
    ]
    if not result.get("canary_ok", True):
        lines += [
            "",
            "**Environment anomaly:** the canary kernel deviated, so regressions/improvements "
            "of this run are reported as inconclusive.",
        ]
    if override.get("ignored"):
        lines += ["", f"_Override label ignored: {override['ignored']}_"]

    def cells(r, m):
        res = r["metrics"][m].get("used") or {}
        base = [f"`{r['op']}`", r["label"], r["pass"], m]
        return base + [
            _fmt_us(res.get("median_a")),
            _fmt_us(res.get("median_b")),
            *_ratio_ci(res),
        ]

    header = ["Op", "Case", "Pass", "Metric", "Base (us)", "PR (us)", "Ratio", "95% CI"]
    if regs:
        lines += ["", "### Regressions"]
        lines += _table(
            header + ["Blocking"],
            [cells(r, m) + [_blocking_cell(r, m)] for r, m in regs],
        )
    if imps:
        lines += ["", "### Improvements"] + _table(
            header, [cells(r, m) for r, m in imps]
        )

    inconc = _entries(rows, "inconclusive")
    others = [r for r in rows if r["status"] != "ok"]
    sel = ctx.get("selection") or {}
    truncated = sel.get("truncated") or []
    if inconc or others or truncated:
        lines += ["", "<details><summary>Inconclusive / skipped cases</summary>", ""]
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
                f"Truncated by the time budget ({len(truncated)} case(s)): "
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
    reps = regs or imps
    if reps:
        lines.append(f"**Reproduce:** `{reps[0][0].get('repro')}`")
    return "\n".join(lines) + "\n"


def _blocking_cell(r, metric):
    if metric in r.get("blocking_metrics", []):
        return "Yes"
    if metric == "e2e":
        return "No (E2E is report-only)"
    return "No (" + ", ".join(r.get("not_blocking_reasons") or ["report-only"]) + ")"


def should_comment(result, ctx):
    """Post when there is something to say; otherwise only refresh an existing comment."""
    if result.get("regressions") or result.get("improvements") or ctx.get("new_ops"):
        return "post"
    return "update_only"


# ---------------------------------------------------------------------------
# Check job
# ---------------------------------------------------------------------------
# Exit 1 only for a blocking regression without a valid override. Everything
# else (A/A runs, inconclusive cases, environment anomalies) passes.


def _freq(meta):
    vals = {
        str(d["min"])
        for card in (meta.get("env_start") or {}).values()
        for d in card.get("freq") or []
        if d.get("min") and d.get("min") == d.get("max")
    }
    return ",".join(sorted(vals)) or None


def run_check(args):
    os.makedirs(args.out_dir, exist_ok=True)
    meta = load_json(os.path.join(args.data_dir, "run_meta.json"), {})
    selection = load_json(args.selection, {})
    gating = load_gating(args.gating)
    if args.require_calibration is not None:
        gating["require_calibration"] = args.require_calibration == "true"
    hist = load_history(args.history)
    result = judge(
        os.path.join(args.data_dir, "raw"),
        {cid: c.manifest() for cid, c in load_cases().items()},
        meta,
        selection=selection,
        gating=gating,
        noise=hist["noise_stats"],
        canary_history=hist["canary"],
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

    ctx = {
        "mode_label": args.mode_label,
        "override": override,
        "env": {
            "runner": meta.get("host"),
            "cards": ",".join(str(c.get("index")) for c in meta.get("cards", [])),
            "GPU freq (MHz)": _freq(meta),
            "base": args.base_sha or selection.get("base"),
            "PR": args.head_sha or selection.get("head"),
            "PyTorch": (meta.get("torch_git_version") or {}).get("B"),
        },
        "selection": selection,
        "new_ops": selection.get("new_ops"),
        "run_url": args.run_url,
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
    p = sub.add_parser("check", help="judge raw A/B data, render the report")
    p.add_argument(
        "--data-dir", required=True, help="scheduler output (run_meta.json, raw/)"
    )
    p.add_argument("--selection", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gating", default=None)
    p.add_argument("--history", default=None, help="default: config/history.json")
    p.add_argument("--require-calibration", choices=["true", "false"], default=None)
    p.add_argument("--pr", default=None)
    p.add_argument("--gh-repo", default=None)
    p.add_argument("--head-sha", default=None)
    p.add_argument("--base-sha", default=None)
    p.add_argument(
        "--override-json", default=None, help="precomputed override status (tests)"
    )
    p.add_argument("--run-url", default=None)
    p.add_argument("--mode-label", default=None)
    p.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    p.set_defaults(func=lambda a: sys.exit(run_check(a)))
