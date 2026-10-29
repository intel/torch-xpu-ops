# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Reference data: rolling state, noise statistics and nightly detection
against it. No torch, no numpy.

The state is one JSONL file (``reference.jsonl``) in the artifact
``microbench-reference-state`` that every scheduled nightly re-uploads with
its own measurements appended (30 days kept). Two kinds of lines:

* ``measure``: one per (nightly, case, side) with the per-process medians,
  case_code_hash, protocol_version, commits, card and environment key;
  ``role`` is ``nightly_vs_reference`` (the nightly measurement) or
  ``nightly_aa`` (A/A rotation, sides A and B).
* ``jump``: a confirmed step of a case (recheck of a nightly). Reference
  values and noise statistics only use measurements from the night of the
  last jump of a case on.

Everything else (reference values, noise statistics) is computed on read.
Only runs of nightly_ondemand.yml on main that were scheduled (or a
maintainer's rebuild dispatch) upload ``microbench-reference-state``; other
runs upload ``microbench-reference-state-ondemand``, which is only read when
a run id is given explicitly (on-demand tests).
"""

import json
import math
import os
import statistics
import subprocess
import sys
import time

from . import CONFIG_DIR, load_yaml, PROTOCOL_VERSION, read_jsonl

STATE_ARTIFACT = "microbench-reference-state"
ONDEMAND_STATE_ARTIFACT = "microbench-reference-state-ondemand"
STATE_FILE = "reference.jsonl"
KEEP_DAYS = 30
DAY = 86400.0
NIGHTLY_ROLES = ("nightly_vs_reference",)
# A/A runs a case needs for its delta: the nightly A/A takes a seventh of the
# PR-tier cases, so 30 days of state hold 3-4 A/A runs per case
MIN_AA_RUNS = 3
# night-to-night ratios a case needs for its own sigma_night
MIN_NIGHT_RATIOS = 3

DEFAULT_SCREENING = {
    "nightly_detect": {
        "z": 2.33,
        "t_max": 0.05,
        "t_min": 0.02,
        "max_age_days": 7,
        "processes": 3,
        "aa_fraction": 0.14,
        "max_no_reference_recheck": 300,
    },
    "default_noise": {"night": 0.01},
}


def load_screening(path=None):
    data = load_yaml(path or CONFIG_DIR / "gating.yaml", {}) or {}
    out = {}
    for k, v in DEFAULT_SCREENING.items():
        out[k] = (
            dict(v, **(data.get(k) or {})) if isinstance(v, dict) else data.get(k, v)
        )
    return out


# ---------------------------------------------------------------------------
# State storage
# ---------------------------------------------------------------------------


def _gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def fetch_state(repo, out_dir, run_id=None, exclude_run=None, limit=60):
    """Download the newest valid state. Returns (path, run_id), or
    (None, None) when there is none (first run, or more than 90 days
    without a nightly that produced one)."""
    if run_id:
        candidates, names = [str(run_id)], (STATE_ARTIFACT, ONDEMAND_STATE_ARTIFACT)
    else:
        out = _gh(
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            "nightly_ondemand.yml",
            "--branch",
            "main",
            "--limit",
            str(limit),
            "--json",
            "databaseId,event",
        )
        if out.returncode != 0:
            print(f"cannot list nightly runs: {out.stderr.strip()}", file=sys.stderr)
            return None, None
        candidates = [
            str(r["databaseId"])
            for r in json.loads(out.stdout or "[]")
            if r.get("event") in ("schedule", "workflow_dispatch")
            and str(r["databaseId"]) != str(exclude_run)
        ]
        names = (STATE_ARTIFACT,)
    for rid in candidates:
        for name in names:
            dest = os.path.join(out_dir, rid)
            if (
                _gh(
                    "run", "download", rid, "--repo", repo, "-n", name, "-D", dest
                ).returncode
                == 0
            ):
                path = os.path.join(dest, STATE_FILE)
                if os.path.exists(path):
                    return path, rid
    return None, None


def load_state(path):
    return read_jsonl(path) if path else []


def save_state(path, entries):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for e in sorted(
            entries, key=lambda e: (e.get("time", 0), e.get("case_id", ""))
        ):
            f.write(json.dumps(e, sort_keys=True) + "\n")


def measure_entries(records, run_id, now=None, metric="device_us"):
    """State ``measure`` lines from records.jsonl rows of a nightly."""
    now = now or time.time()
    out = []
    for r in records:
        if r.get("role") not in ("nightly_vs_reference", "nightly_aa"):
            continue
        procs = [p for p in r.get("per_process") or [] if p.get(metric) is not None]
        if not procs:
            continue
        out.append(
            {
                "kind": "measure",
                "time": now,
                "date": time.strftime("%Y-%m-%d", time.gmtime(now)),
                "run_id": str(run_id),
                "role": r["role"],
                "side": r.get("side"),
                "case_id": r["case_id"],
                "case_code_hash": r.get("case_code_hash"),
                "protocol_version": r.get("protocol_version"),
                "xpu_ops_sha": r.get("xpu_ops_sha"),
                "pytorch_sha": r.get("pytorch_sha"),
                "machine_id": r.get("machine_id"),
                "card_id": r.get("card_id"),
                "env_key": r.get("env_key"),
                "metric": metric,
                "procs": [round(p[metric], 3) for p in procs],
                "e2e": [p.get("e2e_us") for p in procs],
                "n": [
                    p.get("n_device" if metric == "device_us" else "n_e2e")
                    for p in procs
                ],
            }
        )
    return out


def jump_entry(case_id, prev, cur, cause, candidates=(), now=None, **extra):
    now = now or time.time()
    e = {
        "kind": "jump",
        "time": now,
        "date": time.strftime("%Y-%m-%d", time.gmtime(now)),
        "case_id": case_id,
        "prev": prev,
        "cur": cur,
        "cause": cause,
        "candidates": list(candidates),
    }
    e.update(extra)
    return e


def roll(state, new, now=None, keep_days=KEEP_DAYS):
    now = now or time.time()
    return [e for e in state if e.get("time", 0) >= now - keep_days * DAY] + list(new)


def last_jumps(state):
    """{case_id: time of its last confirmed jump}."""
    out = {}
    for e in state:
        if e.get("kind") == "jump":
            out[e["case_id"]] = max(out.get(e["case_id"], 0), e.get("time", 0))
    return out


def last_aa_case(state):
    """A case_id of the latest nightly A/A in the state (None: no A/A yet)."""
    aa = [
        e for e in state if e.get("kind") == "measure" and e.get("role") == "nightly_aa"
    ]
    return max(aa, key=lambda e: e.get("time", 0))["case_id"] if aa else None


def nightly_entries(state, case_id=None):
    return [
        e
        for e in state
        if e.get("kind") == "measure"
        and e.get("role") in NIGHTLY_ROLES
        and (case_id is None or e["case_id"] == case_id)
    ]


def by_case(entries):
    out = {}
    for e in entries:
        out.setdefault(e["case_id"], []).append(e)
    return out


# ---------------------------------------------------------------------------
# Noise statistics, computed on read
# ---------------------------------------------------------------------------
# sigma_night  robust spread of log(median / previous median) over the
#              consecutive nightly records of a case (since its last jump),
#              whatever machine each night ran on; with fewer than
#              MIN_NIGHT_RATIOS ratios, gating.yaml default_noise.night;
# delta        A/A rotation: the largest max(r, 1/r) - 1 over the A/A runs
#              (the confirmation margin).


def _spread(xs):
    """Robust std of xs (MAD; plain std below 3 values); None below 2."""
    if len(xs) < 2:
        return None
    if len(xs) < 3:
        return statistics.stdev(xs)
    m = statistics.median(xs)
    return 1.4826 * statistics.median(abs(x - m) for x in xs)


def entry_median(e):
    return statistics.median(e["procs"])


def noise_stats(state, defaults, jumps=None):
    """{case_id: {"night", "nights", "delta", "runs", "aa_runs"}}."""
    jumps = last_jumps(state) if jumps is None else jumps
    out = {}
    measures = [e for e in state if e.get("kind") == "measure"]
    for cid, es in by_case(measures).items():
        after = jumps.get(cid, 0)
        es = [e for e in es if e.get("time", 0) >= after]
        nightly = sorted(
            (e for e in es if e.get("role") in NIGHTLY_ROLES and entry_median(e) > 0),
            key=lambda e: e.get("time", 0),
        )
        night = [
            math.log(entry_median(b) / entry_median(a))
            for a, b in zip(nightly, nightly[1:])
        ]
        aa = {}
        for e in es:
            if e.get("role") == "nightly_aa":
                aa.setdefault(e.get("run_id"), {})[e.get("side")] = entry_median(e)
        logr = [
            math.log(s["B"] / s["A"])
            for s in aa.values()
            if s.get("A", 0) > 0 and s.get("B", 0) > 0
        ]
        out[cid] = {
            "night": _spread(night) if len(night) >= MIN_NIGHT_RATIOS else None,
            "nights": len(night),
            # the largest A/A deviation (a few runs per case)
            "delta": math.expm1(max(map(abs, logr)))
            if len(logr) >= MIN_AA_RUNS
            else None,
            "runs": len({e.get("run_id") for e in nightly}),
            "aa_runs": len(logr),
        }
    return out


def sigma_night(stats, defaults):
    """sigma_night of one case, default_noise.night without enough nights."""
    night = (stats or {}).get("night")
    return night if night is not None else defaults["night"]


# ---------------------------------------------------------------------------
# Comparing one measurement with the previous nightly
# ---------------------------------------------------------------------------
# The reference is the most recent nightly record of the case (normally last
# night's), on whatever machine it ran, if it is at most max_age_days old.
# r = median(new process medians) / median(reference process medians);
# threshold t = min(t_max, 0.10 - z * sigma_night), so a real 10% step is
# flagged with probability >= Phi(z). A case whose t falls below t_min is too
# noisy to detect and goes to the recheck A/B directly.


def pick_reference(entries, now, cfg):
    """The most recent record within max_age_days, or None."""
    recent = [e for e in entries if e.get("time", 0) >= now - cfg["max_age_days"] * DAY]
    return max(recent, key=lambda e: e.get("time", 0)) if recent else None


def threshold(sigma, cfg):
    return min(cfg["t_max"], 0.10 - cfg["z"] * sigma)


def compare_to_reference(procs, ref, sigma, cfg, card_id):
    """Detection verdict of one case against its reference record."""
    ref_us = entry_median(ref)
    new_us = statistics.median(procs)
    t = threshold(sigma, cfg)
    logr, bound = math.log(new_us / ref_us), math.log1p(max(t, 0.0))
    res = {
        "ratio": new_us / ref_us,
        "reference_us": ref_us,
        "new_us": new_us,
        "threshold": t,
        "sigma_night": sigma,
        "same_card": ref.get("card_id") == card_id,
        "reference_runs": [
            (ref.get("run_id"), ref.get("date"), ref.get("xpu_ops_sha"))
        ],
        "reference_cards": [str(ref.get("card_id"))],
    }
    if t < cfg["t_min"]:
        res.update(
            status="screen_invalid",
            reason=f"too noisy to detect (t={t:.3f} < t_min={cfg['t_min']})",
        )
    elif logr > bound:
        res.update(status="suspect_regression", reason="")
    elif logr < -bound:
        res.update(status="suspect_improvement", reason="")
    else:
        res.update(status="no_change", reason="")
    return res


def classify(record, ref_entries, stats, screening, metric="device_us", now=None):
    """Nightly detection of one case. ``ref_entries`` are the candidate
    reference records (already filtered for code, protocol and jumps)."""
    cfg = screening["nightly_detect"]
    procs = [
        p[metric]
        for p in record.get("per_process") or []
        if p.get(metric) is not None and p[metric] > 0
    ]
    res = {
        "case_id": record["case_id"],
        "card_id": record.get("card_id"),
        "processes": len(procs),
    }
    if not procs:
        return dict(
            res, status="screen_invalid", reason="no valid samples on this side"
        )
    ref = pick_reference(ref_entries or [], now or time.time(), cfg)
    if ref:
        sigma = sigma_night(stats, screening["default_noise"])
        res.update(compare_to_reference(procs, ref, sigma, cfg, record.get("card_id")))
    else:
        res.update(
            status="no_reference",
            reason=f"no record of this case code in the last {cfg['max_age_days']} days",
            new_us=statistics.median(procs),
        )
    return res


def eligible_entries(entries, case_id, code_hash, jumps):
    """Reference candidates of one case: same case code and protocol, from
    its last jump on (the night of a jump measured the new level)."""
    after = jumps.get(case_id, 0)
    return [
        e
        for e in entries
        if e["case_id"] == case_id
        and e.get("case_code_hash") == code_hash
        and e.get("protocol_version") == PROTOCOL_VERSION
        and e.get("time", 0) >= after
    ]


# ---------------------------------------------------------------------------
# CLI: python -m microbench reference fetch|show
# ---------------------------------------------------------------------------


def run_reference(args):
    if args.action == "fetch":
        path, rid = fetch_state(args.repo, args.out_dir, args.run_id, args.exclude_run)
        info = {"path": path, "run_id": rid}
        print(json.dumps(info))
        if args.github_output:
            with open(args.github_output, "a") as f:
                f.write(f"state={path or ''}\nstate_run={rid or ''}\n")
        return 0
    state = load_state(args.state)
    nightly = nightly_entries(state)
    runs = sorted({(e.get("date"), e.get("run_id")) for e in nightly})
    print(
        f"{len(state)} entries, {len(nightly)} nightly measurements, {len(last_jumps(state))} case(s) with jumps"
    )
    for d, r in runs:
        print(f"  {d} run {r}")
    return 0


def add_reference_parser(sub):
    p = sub.add_parser(
        "reference", help="reference state: fetch the latest, show a summary"
    )
    p.add_argument("action", choices=["fetch", "show"])
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    p.add_argument("--out-dir", default="mb_reference")
    p.add_argument(
        "--run-id", default=None, help="read this run's state (on-demand tests only)"
    )
    p.add_argument("--exclude-run", default=None)
    p.add_argument("--state", default=None, help="show: reference.jsonl")
    p.add_argument("--github-output", default=None)
    p.set_defaults(func=lambda a: sys.exit(run_reference(a)))
