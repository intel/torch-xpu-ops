#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Bot review gate: move a PR from `in progress` to `ready for review`.

Like pytorch's Dr.CI promotion, all of these must hold on the PR's head commit:

  * the bot review passed (`bot-review` commit status is success);
  * no maintainer commented after that review started;
  * the `pull` workflow finished, and every failure in it is explained by main:
    a UT case or E2E model that fails here also fails in the latest
    build-from-source nightly (BROKEN_TRUNK). A failure that passes on the
    nightly is NEW, one the nightly did not run is UNKNOWN; both block, as does
    any other failed job (lint, build, a crashed or cancelled test job --
    rerun those).

Run when the review is posted and when `pull` completes; whichever comes last
decides. Usage:

    bot_review_promote.py --repo intel/torch-xpu-ops --pr 123 [--report-ci]

`--report-ci` comments the reasons once per head commit when CI blocks;
without it a blocked PR is left as is (CI may still be running).
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

IN_PROGRESS, READY, OPT_OUT = "in progress", "ready for review", "no automated review"
STATUS_CONTEXT = "bot-review"
MAINTAINER_ROLES = {"triage", "write", "maintain", "admin"}
BOT_LOGINS = {"torchxpubot"}
OK = {"success", "skipped", "neutral"}
NIGHTLY_TITLE = "Build-from-source"

# `linux-ut (basic) / check`, `linux-distributed (xpu_distributed) / check`,
# `windows-ut (basic) / check`: one UT group per matrix entry.
UT_JOB = re.compile(r"^(linux|windows)-(?:ut|distributed) \(([^)]+)\) / (.+)$")
E2E_SUMMARY = "linux-e2e-summary"
DERIVED = {"ut-report"}  # mirrors the UT check jobs


# --------------------------------------------------------------------------- #
# Pure decisions (unit tested)
# --------------------------------------------------------------------------- #

def parse_started(description):
    """Review start time recorded in the `bot-review` status description."""
    m = re.search(r"review started (\S+Z)", description or "")
    return m.group(1) if m else None


def classify(lines, nightly_failed, nightly_passed):
    """PR failure line -> 'broken_trunk' | 'new' | 'unknown' against the nightly."""
    out = {}
    for line in lines:
        if line in nightly_failed:
            out[line] = "broken_trunk"
        elif line in nightly_passed:
            out[line] = "new"
        else:
            out[line] = "unknown"
    return out


def ut_baseline(root):
    """(failed, passed) case lines `category,class,test` from UT-Check case logs."""
    failed, passed = set(), set()
    for path in Path(root).rglob("failures_*.log"):
        failed |= _lines(path)
    for path in Path(root).rglob("passed_*.log"):
        passed |= _lines(path)
    return failed, passed


def ut_group_blockers(check_root, nightly_failed, nightly_passed, name="UT"):
    """Why one UT check artifact blocks; also the broken-trunk lines it tolerated."""
    blockers, tolerated = [], []
    verdicts = list(Path(check_root).rglob("verdict.json"))
    if not verdicts:
        return [f"{name} check results are missing"], []
    for verdict_path in verdicts:
        job_dir = verdict_path.parent
        verdict = json.loads(verdict_path.read_text())
        unhealthy = {
            h["category"] for h in _jsonl(job_dir / "run_health.jsonl") if not h.get("healthy", True)
        }
        new_lines = sorted(_lines(job_dir / "new_failures.txt"))
        for cat in verdict.get("categories", []):
            if cat.get("passed", True):
                continue
            name = cat["category"]
            if name in unhealthy:
                blockers.append(f"UT `{name}` did not run completely")
                continue
            lines = [ln for ln in new_lines if ln.split(",", 1)[0] == name]
            if not lines:
                blockers.append(f"UT `{name}` failed without a new-failure list")
                continue
            for line, kind in classify(lines, nightly_failed, nightly_passed).items():
                if kind == "broken_trunk":
                    tolerated.append(line)
                else:
                    blockers.append(f"UT `{line}`: {'passes' if kind == 'new' else 'not run'} on main")
    return blockers, tolerated


def e2e_accuracy(root):
    """(suite, dtype, mode, model) -> accuracy, from every *-accuracy.csv under root."""
    out = {}
    for path in Path(root).rglob("*-accuracy.csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("name"):
                    out[row["suite"], row["dtype"], row["mode"], row["name"]] = row.get("accuracy", "")
    return out


def e2e_blockers(pr_acc, nightly_acc):
    """Models that do not pass on the PR but pass on (or are missing from) main."""
    blockers, tolerated = [], []
    for key, acc in sorted(pr_acc.items()):
        if "pass" in acc:
            continue
        label = "/".join(key)
        base = nightly_acc.get(key)
        if base is not None and "pass" not in base:
            tolerated.append(label)
        else:
            blockers.append(f"E2E `{label}` is `{acc}`: {'not run' if base is None else base} on main")
    return blockers, tolerated


def ci_verdict(jobs, ut_group, e2e):
    """Decide from the `pull` run's jobs.

    `ut_group(platform, ut)` and `e2e()` return (blockers, tolerated) and are
    only called for a failed check / summary. Returns (pending, blockers, tolerated).
    """
    pending, blockers, tolerated = False, [], []
    groups = defaultdict(list)
    summary = None
    for job in jobs:
        if job["status"] != "completed":
            pending = True
            continue
        m = UT_JOB.match(job["name"])
        if m:
            groups[m.group(1), m.group(2)].append((m.group(3), job["conclusion"]))
        elif job["name"].startswith(E2E_SUMMARY):
            summary = job["conclusion"]
        elif job["name"] not in DERIVED and job["conclusion"] not in OK:
            blockers.append(f"job `{job['name']}` {job['conclusion']}")
    if pending:
        return True, [], []
    for (platform, ut), members in sorted(groups.items()):
        check = dict(members).get("check")
        if check == "success":
            continue  # the check already judged the test jobs' results
        if check == "failure":
            b, t = ut_group(platform, ut)
            blockers += b
            tolerated += t
        elif any(c not in OK for _, c in members):
            blockers.append(f"UT `{platform}-{ut}` did not finish")
    if summary == "failure":
        b, t = e2e()
        blockers += b
        tolerated += t
    elif summary not in (None, *OK):
        blockers.append(f"job `{E2E_SUMMARY}` {summary}")
    return False, blockers, tolerated


def maintainer_logins_since(activity, since, author, role_of):
    """Maintainers (not the author or a bot) who commented/reviewed after `since`."""
    since_t = _ts(since)
    logins = {
        a["login"] for a in activity
        if a.get("at") and _ts(a["at"]) > since_t and a["login"] != author
        and a["login"] not in BOT_LOGINS and a.get("type") != "Bot"
    }
    return sorted(u for u in logins if role_of(u) in MAINTAINER_ROLES)


# --------------------------------------------------------------------------- #
# GitHub I/O
# --------------------------------------------------------------------------- #

def gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def api(path):
    return json.loads(gh("api", path))


def api_items(path, jq=".[]"):
    out = gh("api", "--paginate", path, "--jq", jq)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


class Repo:
    def __init__(self, name):
        self.name = name

    def role(self, login):
        r = subprocess.run(["gh", "api", f"repos/{self.name}/collaborators/{login}/permission",
                            "--jq", ".role_name"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    def activity(self, pr, since):
        base = f"repos/{self.name}"
        items = [{"login": c["user"]["login"], "type": c["user"]["type"], "at": c["created_at"]}
                 for c in api_items(f"{base}/issues/{pr}/comments?since={since}&per_page=100")]
        items += [{"login": c["user"]["login"], "type": c["user"]["type"], "at": c["created_at"]}
                  for c in api_items(f"{base}/pulls/{pr}/comments?since={since}&per_page=100")]
        items += [{"login": r["user"]["login"], "type": r["user"]["type"], "at": r["submitted_at"]}
                  for r in api_items(f"{base}/pulls/{pr}/reviews?per_page=100")
                  if r.get("user") and (r["state"] != "APPROVED" or r.get("body"))]
        return items

    def pull_run(self, sha):
        runs = api_items(f"repos/{self.name}/actions/runs?head_sha={sha}&per_page=100",
                         ".workflow_runs[]")
        runs = [r for r in runs if r["path"].endswith("workflows/pull.yml")]
        return max(runs, key=lambda r: r["run_number"], default=None)

    def nightly_run(self):
        runs = api_items(f"repos/{self.name}/actions/workflows/nightly_ondemand.yml/runs"
                         "?event=schedule&branch=main&status=completed&per_page=20",
                         ".workflow_runs[]")
        return next((r for r in runs if NIGHTLY_TITLE in r.get("display_title", "")), None)

    def download(self, run_id, pattern, dest):
        gh("run", "download", str(run_id), "-R", self.name, "-p", pattern, "-D", str(dest),
           check=False)
        return dest


def evaluate_ci(repo, sha, workdir):
    run = repo.pull_run(sha)
    if run is None or run["status"] != "completed":
        return True, [], []
    jobs = api_items(f"repos/{repo.name}/actions/runs/{run['id']}/jobs?per_page=100", ".jobs[]")
    nightly = {}

    def nightly_dir(pattern):
        if pattern not in nightly:
            night = repo.nightly_run()
            dest = Path(workdir, "nightly", str(len(nightly)))
            nightly[pattern] = repo.download(night["id"], pattern, dest) if night else dest
        return nightly[pattern]

    def ut_group(platform, ut):
        pr_dir = repo.download(run["id"], f"UT-Check-{platform}-{ut}-*", Path(workdir, "pr", platform, ut))
        failed, passed = ut_baseline(nightly_dir(f"UT-Check-{platform}-{ut}-*"))
        return ut_group_blockers(pr_dir, failed, passed, f"UT `{platform}-{ut}`")

    def e2e():
        pr_dir = repo.download(run["id"], "Inductor-*-XPU-E2E-Data-*-accuracy-*", Path(workdir, "pr", "e2e"))
        pr_acc = e2e_accuracy(pr_dir)
        if not pr_acc:
            return ["E2E accuracy results are missing"], []
        return e2e_blockers(pr_acc, e2e_accuracy(nightly_dir("Inductor-*-XPU-E2E-Data-*-accuracy-*")))

    return ci_verdict(jobs, ut_group, e2e)


def comment_once(repo, pr, sha, kind, body):
    marker = f"<!-- bot-review-promote:{kind}:{sha} -->"
    existing = api_items(f"repos/{repo.name}/issues/{pr}/comments?per_page=100")
    if any(marker in (c.get("body") or "") for c in existing):
        return
    gh("api", f"repos/{repo.name}/issues/{pr}/comments", "-f", f"body={marker}\n{body}")


def promote(repo, pr_number, report_ci):
    pr = api(f"repos/{repo.name}/pulls/{pr_number}")
    labels = {label["name"] for label in pr["labels"]}
    if pr["state"] != "open" or pr["draft"] or IN_PROGRESS not in labels or labels & {READY, OPT_OUT}:
        print("not an `in progress` PR in the gate; nothing to do")
        return
    sha = pr["head"]["sha"]
    status = next((s for s in api(f"repos/{repo.name}/commits/{sha}/status")["statuses"]
                   if s["context"] == STATUS_CONTEXT), None)
    if not status or status["state"] != "success":
        print(f"bot review has not passed on {sha}")
        return
    since = parse_started(status.get("description"))
    if since:
        maintainers = maintainer_logins_since(repo.activity(pr_number, since), since,
                                              pr["user"]["login"], repo.role)
        if maintainers:
            comment_once(repo, pr_number, sha, "maintainer",
                         f"{', '.join('@' + u for u in maintainers)} commented after the bot review "
                         "started. Address those comments, then push or comment "
                         "`@torchxpubot review` to re-run the review.")
            return
    with tempfile.TemporaryDirectory() as workdir:
        pending, blockers, tolerated = evaluate_ci(repo, sha, workdir)
    if pending:
        print("CI has not finished; promotion waits for it")
        return
    if blockers:
        print("CI blocks promotion:\n  " + "\n  ".join(blockers))
        if report_ci:
            shown = "\n".join(f"- {b}" for b in blockers[:30])
            more = f"\n- ... and {len(blockers) - 30} more" if len(blockers) > 30 else ""
            comment_once(repo, pr_number, sha, "ci",
                         "Bot review passed, but CI blocks `ready for review`:\n\n"
                         f"{shown}{more}\n\nFix new failures and push. Rerun jobs that failed for "
                         "infrastructure reasons (`@torchxpubot rerun failed`).")
        return
    gh("api", f"repos/{repo.name}/issues/{pr_number}/labels", "-f", f"labels[]={READY}")
    gh("api", "-X", "DELETE",
       f"repos/{repo.name}/issues/{pr_number}/labels/{urllib.parse.quote(IN_PROGRESS)}", check=False)
    note = ""
    if tolerated:
        note = (f" {len(tolerated)} CI failure(s) also fail on main and were not counted: "
                + ", ".join(f"`{t}`" for t in tolerated[:10]) + (" ..." if len(tolerated) > 10 else ""))
    gh("api", f"repos/{repo.name}/issues/{pr_number}/comments", "-f",
       f"body=Bot review passed and CI is clean: moved to `{READY}` for maintainers.{note}")
    print("promoted")


# --------------------------------------------------------------------------- #

def _lines(path):
    path = Path(path)
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()} if path.is_file() else set()


def _jsonl(path):
    path = Path(path)
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []


def _ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--report-ci", action="store_true",
                        help="comment why CI blocks promotion (once per head commit)")
    args = parser.parse_args()
    promote(Repo(args.repo), args.pr, args.report_ci)


if __name__ == "__main__":
    main()
