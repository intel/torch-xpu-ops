#!/usr/bin/env python3
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Collects the evidence for microbench regression issues.

The nightly microbench-check job already grouped the confirmed regressions
(issue_evidence.json in the microbench-nightly-<run_id> artifact). This
script adds what needs the GitHub API and states facts only:

* for every candidate commit: its PR (number, title, description, head), the
  patch of the related files, whether the PR accepted a microbench regression
  (label ``perf_regression_accepted`` plus the override workflow's approval
  record), and the PR's microbench verdicts for these cases;
* the open microbench issues: their dedup markers and the cases they list;
* one prepared Markdown row per case, which drafts must copy verbatim.

It groups nothing new and files nothing. Standard library only.

    python3 .github/scripts/microbench_collect_evidence.py --run-id 123 \\
        --work-dir work --evidence-dir report/evidence
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY") or "intel/torch-xpu-ops"
SERVER = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
ISSUE_LABELS = ("microbench-regression", "microbench-regression-upstream")
OVERRIDE_LABEL = "perf_regression_accepted"
OVERRIDE_RE = re.compile(r"<!-- microbench-override sha=(\w+) approver=(\S+) time=(\S+) -->")
MARKER_RE = re.compile(r"<!-- microbench-regression: (\S+) -->")
CASE_RE = re.compile(r"`logs/([0-9a-f]{12})/`")
MAX_PATCH = 6000
MAX_BODY = 3000


def run(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def gh_json(path: str, default=None):
    try:
        out = run(["gh", "api", path])
    except RuntimeError as e:
        print(f"::warning::{e}")
        return default
    return json.loads(out) if out.strip() else default


def marker(key: str) -> str:
    return f"<!-- microbench-regression: {key} -->"


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def case_row(c: dict) -> str:
    """The one Markdown row of a case; drafts must copy it verbatim."""
    ci = c.get("ci") or [None, None]
    label = str(c.get("label") or c["case_id"]).replace("|", "\\|")
    return (
        f"| `{c.get('op')}` | {label} | {c.get('pass')} | {fmt(c.get('before_us'))} | "
        f"{fmt(c.get('after_us'))} | {fmt(c.get('ratio'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
        f"`logs/{c['case_id']}/` |"
    )


def commit_pr(sha: str) -> dict | None:
    prs = gh_json(f"repos/{REPO}/commits/{sha}/pulls", []) or []
    merged = [p for p in prs if p.get("merged_at")] or prs
    if not merged:
        return None
    p = merged[0]
    return {
        "number": p["number"],
        "title": p.get("title"),
        "body": (p.get("body") or "")[:MAX_BODY],
        "url": p.get("html_url"),
        "head_sha": (p.get("head") or {}).get("sha"),
        "labels": [lbl["name"] for lbl in p.get("labels") or []],
    }


def commit_patch(sha: str, files: list[str]) -> str:
    data = gh_json(f"repos/{REPO}/commits/{sha}", {}) or {}
    out = []
    for f in data.get("files") or []:
        if f.get("filename") in files and f.get("patch"):
            out.append(f"--- {f['filename']}\n{f['patch']}")
    text = "\n".join(out)
    return text if len(text) <= MAX_PATCH else text[:MAX_PATCH] + "\n... (truncated)"


def override_record(pr: dict) -> dict | None:
    """The approval record of the override workflow, if the label is on the PR."""
    if OVERRIDE_LABEL not in pr.get("labels", []):
        return None
    comments = gh_json(f"repos/{REPO}/issues/{pr['number']}/comments?per_page=100", []) or []
    rec = None
    for c in comments:
        m = OVERRIDE_RE.search(c.get("body") or "")
        if m and (c.get("user") or {}).get("login") == "github-actions[bot]":
            rec = dict(zip(("sha", "approver", "time"), m.groups()))
    return rec


def pr_microbench(pr: dict, case_ids: list[str], work: Path) -> dict:
    """This PR's microbench verdicts for the cases (latest pull.yml run of its head)."""
    if not pr.get("head_sha"):
        return {}
    runs = gh_json(
        f"repos/{REPO}/actions/workflows/pull.yml/runs?head_sha={pr['head_sha']}&per_page=5", {}
    ) or {}
    for r in runs.get("workflow_runs") or []:
        dest = work / f"pr{pr['number']}_{r['id']}"
        name = f"microbench-comment-{pr['number']}"
        run(["gh", "run", "download", str(r["id"]), "--repo", REPO, "-n", name, "-D", str(dest)], check=False)
        verdicts = dest / "verdicts.json"
        if verdicts.exists():
            data = json.loads(verdicts.read_text())
            out = {}
            for row in data.get("rows") or []:
                if row.get("case_id") in case_ids:
                    m = (row.get("metrics") or {}).get("device") or {}
                    used = m.get("used") or {}
                    out[row["case_id"]] = {
                        "selected": True,
                        "screen_status": (row.get("screen") or {}).get("status"),
                        "screen_ratio": (row.get("screen") or {}).get("ratio"),
                        "ab_ratio": used.get("ratio"),
                        "verdict": row.get("verdict"),
                    }
            for cid in case_ids:
                out.setdefault(cid, {"selected": False})
            return {"run_id": r["id"], "cases": out}
    return {}


def open_issues() -> list[dict]:
    out = []
    for label in ISSUE_LABELS:
        try:
            data = json.loads(
                run(["gh", "issue", "list", "--repo", REPO, "--state", "open", "--label", label,
                     "--limit", "200", "--json", "number,title,body,url"])
            )
        except RuntimeError as e:
            print(f"::warning::{e}")
            continue
        for i in data:
            if any(o["number"] == i["number"] for o in out):
                continue
            body = i.get("body") or ""
            out.append(
                {
                    "number": i["number"],
                    "url": i.get("url"),
                    "title": i.get("title"),
                    "markers": MARKER_RE.findall(body),
                    "cases": sorted(set(CASE_RE.findall(body))),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--nightly-dir", default=None, help="already downloaded microbench-nightly artifact")
    args = ap.parse_args()
    work, out_dir = Path(args.work_dir), Path(args.evidence_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.nightly_dir) if args.nightly_dir else work / "nightly"
    if not args.nightly_dir:
        run(["gh", "run", "download", args.run_id, "--repo", REPO, "-n", f"microbench-nightly-{args.run_id}", "-D", str(src)])
    nightly = json.loads((src / "issue_evidence.json").read_text())

    issues = open_issues()
    reported = {c: i["number"] for i in issues for c in i["cases"]}
    groups = []
    for g in nightly.get("groups") or []:
        case_ids = [c["case_id"] for c in g["cases"]]
        for c in g["cases"]:
            c["row"] = case_row(c)
            c["already_reported_in"] = reported.get(c["case_id"])
        cands = []
        for cand in g.get("candidates") or []:
            pr = commit_pr(cand["sha"])
            entry = dict(cand, pr=pr)
            if pr:
                entry["patch"] = commit_patch(cand["sha"], cand.get("files") or [])
                entry["override"] = override_record(pr)
                entry["pr_microbench"] = pr_microbench(pr, case_ids, work)
            cands.append(entry)
        g["candidates"] = cands
        g["marker"] = marker(g["key"])
        g["existing_issue"] = next((i["number"] for i in issues if g["key"] in i["markers"]), None)
        # a single candidate whose PR accepted the regression: not filed
        g["accepted"] = len(cands) == 1 and bool(cands[0].get("override"))
        g["new_cases"] = [c["case_id"] for c in g["cases"] if not c["already_reported_in"]]
        groups.append(g)
    evidence = dict(nightly, repo=REPO, run_url=f"{SERVER}/{REPO}/actions/runs/{args.run_id}", groups=groups,
                    open_issues=issues)
    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
    if (src / "nightly_report.md").exists():
        (out_dir / "nightly_report.md").write_text((src / "nightly_report.md").read_text())
    print(f"{len(groups)} group(s), {sum(len(g['new_cases']) for g in groups)} new case(s), "
          f"{sum(g['accepted'] for g in groups)} accepted, {len(issues)} open microbench issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
