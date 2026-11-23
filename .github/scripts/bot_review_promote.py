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

    bot_review_promote.py --repo intel/torch-xpu-ops --pr 123

When finished CI blocks, the reasons are commented once per distinct set of
reasons on the head commit; while CI is running nothing is said.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

IN_PROGRESS, READY, OPT_OUT = "in progress", "ready for review", "no automated review"
STATUS_CONTEXT = "bot-review"
MAINTAINER_ROLES = {"triage", "write", "maintain", "admin"}
UNKNOWN_ROLE = "unknown"  # permission lookup failed
BOT_LOGINS = {"torchxpubot"}
BOT_COMMAND = "@torchxpubot"
MAX_REVIEWERS = 5  # from the description; the author picks them, keep it to a few
OK = {"success", "skipped", "neutral"}
NIGHTLY_TITLE = "Build-from-source"
# A nightly that broke before testing uploads nothing, so older ones are tried;
# 5 days covers a weekend plus a broken night; older, main has moved too far.
NIGHTLY_MAX_AGE = timedelta(days=5)

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


def latest_attempts(root):
    """Artifact dirs under `root`, keeping only the highest `-<attempt>` of each.

    A rerun uploads a new artifact per attempt; older attempts are stale.
    """
    best = {}
    for d in sorted(p for p in Path(root).glob("*") if p.is_dir()):
        m = re.fullmatch(r"(.+)-(\d+)", d.name)
        key, attempt = (m.group(1), int(m.group(2))) if m else (d.name, 0)
        if key not in best or attempt > best[key][0]:
            best[key] = (attempt, d)
    return [d for _, d in best.values()]


def ut_baseline(root):
    """(failed, passed) case lines `category,class,test` from UT-Check case logs."""
    failed, passed = set(), set()
    for art in latest_attempts(root):
        for path in art.rglob("failures_*.log"):
            failed |= _lines(path)
        for path in art.rglob("passed_*.log"):
            passed |= _lines(path)
    return failed, passed


def ut_group_blockers(check_root, nightly_failed, nightly_passed, name="UT"):
    """Why one UT check artifact blocks; also the broken-trunk lines it tolerated."""
    blockers, tolerated = [], []
    verdicts = [v for art in latest_attempts(check_root) for v in art.rglob("verdict.json")]
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
    for path in (p for art in latest_attempts(root) for p in art.rglob("*-accuracy.csv")):
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


def reviewers_to_request(body, reviews, author):
    """Logins to ask once the PR is ready: the description's `Reviewers:` line
    plus anyone whose latest review still requests changes (to re-review)."""
    wanted = []
    for m in re.finditer(r"^\s*Reviewers?:(.*)$", body or "", re.I | re.M):
        for token in re.split(r"[\s,]+", m.group(1)):
            token = token.strip(".;")
            if re.fullmatch(r"@?[A-Za-z0-9][A-Za-z0-9-]*", token):  # teams (org/team) are skipped
                wanted.append(token.lstrip("@"))
    latest = {}
    for r in reviews:
        if r.get("user") and r["state"] != "COMMENTED":
            latest[r["user"]["login"]] = r["state"]
    wanted += [u for u, state in latest.items() if state == "CHANGES_REQUESTED"]
    out = []
    for u in wanted:
        if u != author and u not in BOT_LOGINS and u.lower() not in {o.lower() for o in out}:
            out.append(u)
    return out


def recent_nightlies(runs, now):
    """Build-from-source nightlies younger than NIGHTLY_MAX_AGE, newest first."""
    return sorted((r for r in runs if NIGHTLY_TITLE in r.get("display_title", "")
                   and now - _ts(r["created_at"]) <= NIGHTLY_MAX_AGE),
                  key=lambda r: r["created_at"], reverse=True)


def maintainer_logins_since(activity, since, author, role_of):
    """Maintainers (not the author or a bot) who commented/reviewed after `since`.

    A failed permission lookup counts as a maintainer: fail closed, no promotion.
    """
    since_t = _ts(since)
    logins = {
        a["login"] for a in activity
        if a.get("at") and _ts(a["at"]) > since_t and a["login"] != author
        and a["login"] not in BOT_LOGINS and a.get("type") != "Bot"
        and not (a.get("body") or "").lstrip().startswith(BOT_COMMAND)  # e.g. `rerun failed`
    }
    return sorted(u for u in logins if role_of(u) in MAINTAINER_ROLES | {UNKNOWN_ROLE})


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
        return r.stdout.strip() if r.returncode == 0 else UNKNOWN_ROLE

    def activity(self, pr, since):
        base = f"repos/{self.name}"
        def item(c, at):
            return {"login": c["user"]["login"], "type": c["user"]["type"], "at": c[at],
                    "body": c.get("body")}

        items = [item(c, "created_at")
                 for c in api_items(f"{base}/issues/{pr}/comments?since={since}&per_page=100")]
        items += [item(c, "created_at")
                  for c in api_items(f"{base}/pulls/{pr}/comments?since={since}&per_page=100")]
        items += [item(r, "submitted_at")
                  for r in api_items(f"{base}/pulls/{pr}/reviews?per_page=100")
                  if r.get("user") and (r["state"] != "APPROVED" or r.get("body"))]
        return items

    def pull_run(self, sha):
        runs = api_items(f"repos/{self.name}/actions/runs?head_sha={sha}&per_page=100",
                         ".workflow_runs[]")
        runs = [r for r in runs if r["path"].endswith("workflows/pull.yml")]
        return max(runs, key=lambda r: r["run_number"], default=None)

    def nightly_runs(self):
        now = datetime.now(timezone.utc)
        since = (now - NIGHTLY_MAX_AGE).strftime("%Y-%m-%d")
        runs = api_items(f"repos/{self.name}/actions/workflows/nightly_ondemand.yml/runs"
                         f"?event=schedule&branch=main&status=completed&created=>={since}&per_page=100",
                         ".workflow_runs[]")
        return recent_nightlies(runs, now)

    def download(self, run_id, pattern, dest):
        gh("run", "download", str(run_id), "-R", self.name, "-p", pattern, "-D", str(dest),
           check=False)
        return dest


def evaluate_ci(repo, sha, workdir):
    run = repo.pull_run(sha)
    if run is None:
        # `pull` did not run for this commit (e.g. a stacked PR whose base is
        # not main). Like Dr.CI with no jobs, there is nothing to wait for; a
        # run that was only queued would be listed here.
        return False, [], []
    if run["status"] != "completed":
        return True, [], []
    jobs = api_items(f"repos/{repo.name}/actions/runs/{run['id']}/jobs?per_page=100", ".jobs[]")
    nightly = {}

    def nightly_dir(pattern):
        """The newest nightly that uploaded `pattern`; empty dir if none did."""
        if pattern not in nightly:
            dest = Path(workdir, "nightly", str(len(nightly)))
            for night in repo.nightly_runs():
                repo.download(night["id"], pattern, dest)
                if any(dest.glob("*")):
                    break
            nightly[pattern] = dest
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


def request_reviewers(repo, pr):
    """Ask reviewers now that the PR is ready; only collaborators who can review."""
    reviews = api_items(f"repos/{repo.name}/pulls/{pr['number']}/reviews?per_page=100")
    candidates = reviewers_to_request(pr.get("body"), reviews, pr["user"]["login"])
    logins = [u for u in candidates[:MAX_REVIEWERS] if repo.role(u) in MAINTAINER_ROLES]
    if logins:
        gh("api", f"repos/{repo.name}/pulls/{pr['number']}/requested_reviewers",
           *[arg for u in logins for arg in ("-f", f"reviewers[]={u}")], check=False)
    return logins


def comment_once(repo, pr, sha, kind, body):
    marker = f"<!-- bot-review-promote:{kind}:{sha} -->"
    existing = api_items(f"repos/{repo.name}/issues/{pr}/comments?per_page=100")
    if any(marker in (c.get("body") or "") for c in existing):
        return
    gh("api", f"repos/{repo.name}/issues/{pr}/comments", "-f", f"body={marker}\n{body}")


def promote(repo, pr_number):
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
                         "started (or their access could not be checked). Address those comments, "
                         "then push, or ask a maintainer to comment `@torchxpubot review`, to "
                         "re-run the review.")
            return
    with tempfile.TemporaryDirectory() as workdir:
        pending, blockers, tolerated = evaluate_ci(repo, sha, workdir)
    if pending:
        print("CI has not finished; promotion waits for it")
        return
    if blockers:
        print("CI blocks promotion:\n  " + "\n  ".join(blockers))
        shown = "\n".join(f"- {b}" for b in blockers[:30])
        more = f"\n- ... and {len(blockers) - 30} more" if len(blockers) > 30 else ""
        # Keyed by the reasons too, so a rerun that changes them is reported again.
        digest = hashlib.sha1("\n".join(blockers).encode()).hexdigest()[:8]
        comment_once(repo, pr_number, sha, f"ci:{digest}",
                     "Bot review passed, but CI blocks `ready for review`:\n\n"
                     f"{shown}{more}\n\n\"On main\" means the latest build-from-source nightly. Fix new "
                     "failures and push; if your branch is far behind main, rebase first, since a "
                     "fix already on main shows up here as a new failure. Rerun jobs that failed "
                     "for infrastructure reasons (`@torchxpubot rerun failed`).")
        return
    gh("api", f"repos/{repo.name}/issues/{pr_number}/labels", "-f", f"labels[]={READY}")
    gh("api", "-X", "DELETE",
       f"repos/{repo.name}/issues/{pr_number}/labels/{urllib.parse.quote(IN_PROGRESS)}", check=False)
    asked = request_reviewers(repo, pr)
    note = f" Requested review from {', '.join('@' + u for u in asked)}." if asked else ""
    if tolerated:
        note += (f" {len(tolerated)} CI failure(s) also fail on main and were not counted: "
                + ", ".join(f"`{t}`" for t in tolerated[:10]) + (" ..." if len(tolerated) > 10 else ""))
    comment_once(repo, pr_number, sha, "promoted",
                 f"Bot review passed and CI is clean: moved to `{READY}` for maintainers.{note}")
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
    parser.add_argument("--request-reviewers", action="store_true",
                        help="only request reviewers (for PRs promoted by the opt-out label)")
    args = parser.parse_args()
    repo = Repo(args.repo)
    if args.request_reviewers:
        print(request_reviewers(repo, api(f"repos/{repo.name}/pulls/{args.pr}")))
    else:
        promote(repo, args.pr)


if __name__ == "__main__":
    main()
