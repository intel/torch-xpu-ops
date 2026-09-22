#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Queue upstream DISABLED-test batches onto the tracking issue as `@torchxpubot fix`.

Usage:
    GH_TOKEN=<read on SOURCE_REPO, write on TARGET_REPO> \
        python ci_disabled_queue.py [--dry-run]
    python ci_disabled_queue.py --self-test

The XPU CI report issue (SOURCE) grows a comment per failing `xpu.yml` commit.
The ones that matter here carry a `Disable issues:` list of `pytorch/pytorch`
DISABLED issues, and the same issue is re-listed on every later failing commit.
The tracking issue (TARGET) is the bot's work queue: one `@torchxpubot fix`
comment per batch, which is what a human posts by hand today.

The rules, all derived from GitHub state so there is no local database to keep
in sync:

- START: batches older than it are ignored. Whether a DISABLED test already has
  a PR is not recorded anywhere GitHub can be asked, so the pre-START backlog is
  triaged and triggered by hand; the queue only owns what arrives after it.
- closed upstream: nothing to fix, never queued.
- dedup: an issue that has stayed open since the last `@torchxpubot fix` comment
  naming it on TARGET is not mirrored again, however many times SOURCE repeats
  it. Dedup is per open episode, not forever: an issue that was closed and then
  reopened is blocking CI for a fresh reason, so a later SOURCE report of it is
  queued again.
- serialization: nothing is posted while a `fix` job is queued or running, so
  the previous trigger has always answered before the next one lands.

One batch per invocation, oldest batch first.
"""

import json
import os
import re
import sys
import urllib.request

SOURCE_REPO, SOURCE_ISSUE = "chuanqi129/pytorch-xpu-ci", 364
TARGET_REPO, TARGET_ISSUE = "intel/torch-xpu-ops", 5272
# bot.yml serves every @torchxpubot command; only its `fix` job is exclusive.
BOT_WORKFLOW, FIX_JOB = "bot.yml", "fix"
# Where the queue takes over from the humans. SOURCE batches before this are the
# hand-triaged backlog -- several of those issues already have a PR, and that is
# tracked in a spreadsheet, not in anything GitHub can be asked for. Move it back
# to hand the queue more history; it only ever needs to move once.
START = "2026-09-22"

ISSUE_RE = re.compile(r"https://github\.com/pytorch/pytorch/issues/(\d+)")


def api(path, body=None):
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def paged(path):
    out, page = [], 1
    sep = "&" if "?" in path else "?"
    while True:
        chunk = api(f"{path}{sep}per_page=100&page={page}")
        out += chunk
        if len(chunk) < 100:
            return out
        page += 1


def disabled_issues(body):
    """pytorch issue numbers listed under a comment's `Disable issues:` header.

    Comments without that header (the far more common "rerunning" notices) yield
    nothing, and the run URLs above the header are never mistaken for issues.
    """
    _, _, tail = body.partition("Disable issues:")
    return list(dict.fromkeys(ISSUE_RE.findall(tail)))


def last_mirrored():
    """Newest `@torchxpubot fix` timestamp on TARGET per pytorch issue it names."""
    seen = {}
    for c in paged(f"/repos/{TARGET_REPO}/issues/{TARGET_ISSUE}/comments"):
        if "@torchxpubot fix" not in c["body"]:
            continue
        for num in ISSUE_RE.findall(c["body"]):
            seen[num] = max(seen.get(num, ""), c["created_at"])
    return seen


def fix_in_flight():
    """URL of a bot run whose `fix` job has not finished, else None."""
    runs = api(
        f"/repos/{TARGET_REPO}/actions/workflows/{BOT_WORKFLOW}/runs?per_page=30"
    )["workflow_runs"]
    for run in runs:
        if run["status"] == "completed":
            continue
        jobs = api(f"/repos/{TARGET_REPO}/actions/runs/{run['id']}/jobs?per_page=100")[
            "jobs"
        ]
        # A run whose jobs are not enumerated yet could still turn into a `fix`;
        # wait for the next invocation rather than race it.
        if not jobs or any(
            j["name"] == FIX_JOB and j["status"] != "completed" for j in jobs
        ):
            return run["html_url"]
    return None


def is_open_upstream(num):
    return api(f"/repos/pytorch/pytorch/issues/{num}")["state"] == "open"


def closed_since(num, when):
    """Was this issue closed after `when`?

    Called only for issues that are open now, so a close after the last mirror
    means it was reopened since -- a new breakage rather than SOURCE repeating
    itself. `closed_at` cannot answer this: reopening clears it.
    """
    events = paged(f"/repos/pytorch/pytorch/issues/{num}/events")
    return any(e["event"] == "closed" and e["created_at"] > when for e in events)


def pending(comment, mirrored):
    """Issues in this SOURCE batch that the bot should be asked to fix now."""
    out = []
    for num in disabled_issues(comment["body"]):
        since = mirrored.get(num)
        # A batch at or before the last mirror IS what was already mirrored, or
        # predates it. Only a later report can open a new episode.
        if since is not None and comment["created_at"] <= since:
            continue
        if is_open_upstream(num) and (since is None or closed_since(num, since)):
            out.append(num)
    return out


def next_batch(mirrored):
    """Oldest SOURCE comment since START that still has something to queue."""
    for comment in paged(f"/repos/{SOURCE_REPO}/issues/{SOURCE_ISSUE}/comments"):
        if comment["created_at"] < START:
            continue
        new = pending(comment, mirrored)
        if new:
            return comment, new
    return None, []


def trigger_body(comment, issues):
    # The `Commit ... xpu.yml run` line carries the failing commit the batch came
    # from; the agent reads only this comment, so it has to travel with it.
    commit_line = [
        line for line in comment["body"].splitlines() if line.startswith("Commit ")
    ][:1]
    return "\n".join(
        [
            "@torchxpubot fix",
            "",
            f"Copied from [{SOURCE_REPO}#{SOURCE_ISSUE}]({comment['html_url']}) "
            f"({comment['created_at'][:10]}). These tests are DISABLED upstream.",
            "",
            "---",
            "",
            *commit_line,
            "Disable issues:",
            *(f"- https://github.com/pytorch/pytorch/issues/{n}" for n in issues),
            "",
        ]
    )


def self_test():
    rerun = (
        "Commit [`abc`](https://hud.pytorch.org/pytorch/pytorch/commit/abc) - run\n"
        "Test job(s) failed but no test cases could be parsed; rerunning:\n"
        "- linux-noble-xpu-n-py3.10 / test: "
        "https://github.com/pytorch/pytorch/actions/runs/1/job/2\n"
    )
    assert disabled_issues(rerun) == [], "a rerun notice is not a batch"

    batch = rerun + (
        "Disable issues:\n"
        "- https://github.com/pytorch/pytorch/issues/197334\n"
        "- https://github.com/pytorch/pytorch/issues/197335\n"
        "- https://github.com/pytorch/pytorch/issues/197334\n"
    )
    assert disabled_issues(batch) == ["197334", "197335"], "listed once, in order"

    body = trigger_body(
        {
            "body": batch,
            "html_url": "https://example.com/c",
            "created_at": "2026-09-16T22:36:24Z",
        },
        ["197334"],
    )
    assert body.startswith("@torchxpubot fix\n"), "the command must be the first line"
    assert "Commit [`abc`]" in body and "(2026-09-16)" in body
    assert "197335" not in body, "only the issues this run mirrors"
    assert disabled_issues(body) == ["197334"], "re-readable by the dedup scan"

    # Per-episode dedup. Stub the two upstream lookups: 197334 was closed once,
    # after the first mirror; 197335 has never been closed.
    global is_open_upstream, closed_since
    is_open_upstream = lambda num: True  # noqa: E731
    closed_since = lambda num, when: num == "197334" and when < "2026-09-30"  # noqa: E731
    old, new = (
        {"created_at": "2026-09-23T00:00:00Z"},
        {"created_at": "2026-10-02T00:00:00Z"},
    )
    assert old["created_at"] > START, "the dedup cases must sit after the cutoff"
    mirrored = {"197334": "2026-09-23T22:36:24Z", "197335": "2026-09-23T22:36:24Z"}

    assert pending(dict(old, body=batch), {}) == ["197334", "197335"], "never mirrored"
    assert pending(dict(old, body=batch), mirrored) == [], "the batch already mirrored"
    assert pending(dict(new, body=batch), mirrored) == ["197334"], (
        "re-reported after a reopen is queued again; still-open is deduped"
    )
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        return self_test()

    busy = fix_in_flight()
    if busy:
        print(f"a fix job is still running ({busy}); nothing queued this round")
        return

    mirrored = last_mirrored()
    comment, issues = next_batch(mirrored)
    if not comment:
        print(f"nothing to queue since {START}; {len(mirrored)} issues mirrored so far")
        return

    body = trigger_body(comment, issues)
    if "--dry-run" in sys.argv:
        print(body)
        return

    posted = api(f"/repos/{TARGET_REPO}/issues/{TARGET_ISSUE}/comments", {"body": body})
    print(f"queued {', '.join(issues)}: {posted['html_url']}")


if __name__ == "__main__":
    main()
