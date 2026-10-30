# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GitHub side of the gate: labels, override record, PR comment.

``python -m microbench.github comment|labeled|synchronize`` runs in the
privileged workflows (``workflow_run`` / ``pull_request_target``) that never
check out PR code. Standard library only.

* A label only counts if the user who last added it is a maintainer: a member
  of ``MICROBENCH_MAINTAINER_TEAM`` (``org/team-slug``) or, without a team, a
  user with ``admin``/``maintain`` permission on the repository.
* ``perf_regression_accepted`` is also bound to one commit: the override
  workflow checks the labeler and writes ``<!-- microbench-override sha=... -->``
  into a bot comment; the check job only honours the label when that record
  exists for the head SHA under test. New pushes remove the label.
* On fork PRs the select job's token cannot look up maintainers, so a
  ``disable_microbench`` label there is reported as ignored (fail-safe).
"""

import argparse
import datetime
import functools
import json
import os
import re
import subprocess
import sys

SKIP_LABEL = "disable_microbench"
OVERRIDE_LABEL = "perf_regression_accepted"
OVERRIDE_MARKER = "<!-- microbench-override"
REPORT_MARKER = "<!-- microbench-report -->"
CHECK_JOB_SUFFIX = "microbench-check"
BOT_LOGINS = {"github-actions[bot]"}


def gh_api(path, paginate=False, method="GET", fields=None):
    cmd = ["gh", "api", "-X", method, path]
    if paginate:
        cmd += ["--paginate", "--jq", ".[]"]  # one JSON element per line
    for k, v in (fields or {}).items():
        cmd += ["-f", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {path}: {out.stderr.strip()}")
    text = out.stdout.strip()
    if paginate:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text) if text else None


def _comments(repo, pr):
    return gh_api(f"repos/{repo}/issues/{pr}/comments?per_page=100", paginate=True)


def _post(repo, pr, body):
    gh_api(f"repos/{repo}/issues/{pr}/comments", method="POST", fields={"body": body})


def label_adders(repo, pr):
    """{label: (login, created_at)} of the latest 'labeled' event per label."""
    events = gh_api(f"repos/{repo}/issues/{pr}/events?per_page=100", paginate=True)
    return {
        e["label"]["name"]: ((e.get("actor") or {}).get("login"), e.get("created_at"))
        for e in events
        if e.get("event") == "labeled" and e.get("label")
    }


def current_labels(repo, pr):
    return [
        lbl["name"]
        for lbl in gh_api(f"repos/{repo}/issues/{pr}/labels?per_page=100") or []
    ]


@functools.cache
def is_maintainer(repo, user):
    if not user or user.endswith("[bot]"):
        return False
    team = os.environ.get("MICROBENCH_MAINTAINER_TEAM", "")
    try:
        if "/" in team:
            org, slug = team.split("/", 1)
            m = gh_api(f"orgs/{org}/teams/{slug}/memberships/{user}") or {}
            return m.get("state") == "active"
        p = gh_api(f"repos/{repo}/collaborators/{user}/permission") or {}
        return (p.get("role_name") or p.get("permission")) in ("admin", "maintain")
    except RuntimeError:
        return False


def label_status(labels, adders, label, maintainer_fn):
    """Validity of one label: dict(present, valid, by[, time | ignored])."""
    if label not in labels:
        return {"present": False, "valid": False}
    by, when = adders.get(label, (None, None))
    if by and maintainer_fn(by):
        return {"present": True, "valid": True, "by": by, "time": when}
    who = f"@{by}" if by else "an unknown user"
    reason = f"label '{label}' was added by {who}, who is not a maintainer"
    return {"present": True, "valid": False, "by": by, "ignored": reason}


def override_status(labels, adders, comments, head_sha):
    """Valid when the label is present and a bot comment of the override
    workflow (written only after it verified the labeler) approves exactly
    this head SHA. The check job thus needs no maintainer lookup, which fork
    PR tokens cannot do.
    """
    if OVERRIDE_LABEL not in labels:
        return {"valid": False}
    pat = re.compile(
        r"<!-- microbench-override sha=(\w+) approver=(\S+) time=(\S+) -->"
    )
    rec = None
    for c in comments:
        m = pat.search(c.get("body") or "")
        if m and (c.get("user") or {}).get("login") in BOT_LOGINS:
            rec = dict(zip(("sha", "approver", "time"), m.groups()))
    if not rec:
        by = (adders.get(OVERRIDE_LABEL) or (None,))[0]
        who = f" (added by @{by})" if by else ""
        return {
            "valid": False,
            "ignored": f"label '{OVERRIDE_LABEL}'{who} has no maintainer approval record",
        }
    if rec["sha"] != head_sha:
        return {
            "valid": False,
            "ignored": f"override was granted for {rec['sha'][:12]}, not the current head {head_sha[:12]}",
        }
    return dict(rec, valid=True)


def fetch_override(repo, pr, head_sha):
    labels = current_labels(repo, pr)
    if OVERRIDE_LABEL not in labels:
        return {"valid": False}
    return override_status(
        labels, label_adders(repo, pr), _comments(repo, pr), head_sha
    )


def fetch_skip(repo, pr):
    labels = current_labels(repo, pr)
    if SKIP_LABEL not in labels:
        return {"present": False, "valid": False}
    return label_status(
        labels, label_adders(repo, pr), SKIP_LABEL, lambda u: is_maintainer(repo, u)
    )


def upsert_comment(repo, pr, body, marker, create=True):
    for c in _comments(repo, pr):
        if (c.get("user") or {}).get("login") in BOT_LOGINS and marker in (
            c.get("body") or ""
        ):
            gh_api(
                f"repos/{repo}/issues/comments/{c['id']}",
                method="PATCH",
                fields={"body": body},
            )
            return "updated"
    if not create:
        return "skipped"
    _post(repo, pr, body)
    return "created"


def cmd_comment(args):
    """Post/update the single report comment from the check job's artifact."""
    meta = json.load(open(os.path.join(args.dir, "comment.json")))
    body = open(os.path.join(args.dir, "comment.md")).read()
    pr = str(meta.get("pr") or "")
    if not pr:
        print("no PR in the artifact (nightly/ondemand run); nothing to post")
        return 0
    # the artifact comes from PR code: only trust an open PR whose head is this run's head
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        args.repo,
        "--state",
        "open",
        "--search",
        args.head_sha,
    ]
    out = subprocess.run(
        cmd + ["--json", "number,headRefOid"], capture_output=True, text=True
    )
    found = json.loads(out.stdout or "[]") if out.returncode == 0 else []
    allowed = [str(p["number"]) for p in found if p.get("headRefOid") == args.head_sha]
    if pr not in allowed or (meta.get("head_sha") or args.head_sha) != args.head_sha:
        print(
            f"refusing to comment: artifact PR #{pr} / head {meta.get('head_sha')} do not match run head {args.head_sha} (PRs: {allowed})"
        )
        return 1
    result = upsert_comment(
        args.repo, pr, body, REPORT_MARKER, create=meta.get("action") == "post"
    )
    print(f"PR #{pr}: comment {result}")
    return 0


def latest_check_job(repo, sha):
    runs = (
        gh_api(
            f"repos/{repo}/actions/workflows/pull.yml/runs?head_sha={sha}&per_page=20"
        )
        or {}
    )
    for run in runs.get("workflow_runs", []):
        jobs = (
            gh_api(
                f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100&filter=latest"
            )
            or {}
        )
        for job in jobs.get("jobs", []):
            if job["name"].endswith(CHECK_JOB_SUFFIX):
                return job
    return None


def cmd_labeled(args):
    """A maintainer accepts the regression of the current head: record the SHA
    and rerun only the check job."""
    if args.label != OVERRIDE_LABEL:
        return 0
    if not is_maintainer(args.repo, args.sender):
        body = (
            f"{OVERRIDE_MARKER} ignored -->\nLabel `{OVERRIDE_LABEL}` added by @{args.sender} is ignored: "
            "only maintainers can accept a microbench regression."
        )
        upsert_comment(args.repo, args.pr, body, f"{OVERRIDE_MARKER} ignored")
        return 0
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _post(
        args.repo,
        args.pr,
        f"{OVERRIDE_MARKER} sha={args.head_sha} approver={args.sender} time={now} -->\n"
        f"Microbench regression accepted by @{args.sender} at {now} for commit `{args.head_sha[:12]}`. "
        "This approval only covers that commit; new pushes remove the label and are measured again.",
    )
    job = latest_check_job(args.repo, args.head_sha)
    if not job or job.get("conclusion") == "skipped":
        print(
            "no microbench-check job with data for this head; the next check run reads the label"
        )
        return 0
    gh_api(f"repos/{args.repo}/actions/jobs/{job['id']}/rerun", method="POST")
    print(f"reran {job['name']} (job {job['id']})")
    return 0


def cmd_synchronize(args):
    """New commits: drop the override label, it covered the old head only."""
    if OVERRIDE_LABEL not in current_labels(args.repo, args.pr):
        return 0
    gh_api(
        f"repos/{args.repo}/issues/{args.pr}/labels/{OVERRIDE_LABEL}", method="DELETE"
    )
    _post(
        args.repo,
        args.pr,
        f"New commits were pushed, so label `{OVERRIDE_LABEL}` was removed: it only covered the previous head. "
        "Microbench runs again on the new code; if a regression remains, a maintainer has to add the label again.",
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m microbench.github")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, extra in (
        ("comment", cmd_comment, ["dir", "head-sha"]),
        ("labeled", cmd_labeled, ["pr", "label", "sender", "head-sha"]),
        ("synchronize", cmd_synchronize, ["pr"]),
    ):
        sp = sub.add_parser(name, help=fn.__doc__)
        for arg in ["repo", *extra]:
            sp.add_argument(f"--{arg}", required=True)
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
