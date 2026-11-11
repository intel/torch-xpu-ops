#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Post `@torchxpubot fix` on the tracking issue for each new batch of DISABLED tests.

    GH_TOKEN=<read SOURCE, write TARGET> python ci_disabled_queue.py  # print
    GH_TOKEN=<...> DRY_RUN=false        python ci_disabled_queue.py  # post
    python ci_disabled_queue.py --self-test

SOURCE grows one comment per failing `xpu.yml` run; the ones that matter list the
`pytorch/pytorch` DISABLED issues it produced, and re-list them every time it
fails again. TARGET is the bot's work queue: one `@torchxpubot fix` comment per
batch, batches kept whole.

One batch per invocation, oldest first, never while a `fix` job is still running.
No local state -- every rule reads its answer back out of GitHub. No by-hand mode:
a human who wants one test fixed types the command on TARGET.
"""

import functools
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Two issues in this repo: #5490 is the CI report ("XPU Periodic Run Auto Skip &
# Rerun"), #5272 is the bot's work queue.
SOURCE_REPO, SOURCE_ISSUE = "intel/torch-xpu-ops", 5490
TARGET_REPO, TARGET_ISSUE = "intel/torch-xpu-ops", 5272
# bot.yml serves every @torchxpubot command; only its `fix` job is exclusive.
BOT_WORKFLOW, FIX_JOB = "bot.yml", "fix"
# The day SOURCE moved into this repo. Earlier batches are the hand-triaged
# backlog: whether one of those already has a PR lives in a spreadsheet, not in
# anything GitHub can be asked. Move START back only to hand over more history.
START = "2026-09-22"

ISSUE_RE = re.compile(r"https://github\.com/pytorch/pytorch/issues/(\d+)")
# What bot.yml itself accepts as the command (`startsWith(comment.body,
# '@torchxpubot')` plus `/^@torchxpubot\s+(\S+)/i`): start-anchored, so a comment
# that merely quotes the command is not a trigger.
FIX_CMD_RE = re.compile(r"@torchxpubot\s+fix\b", re.I)


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
    while True:
        chunk = api(f"{path}?per_page=100&page={page}")
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
    """Newest `@torchxpubot fix` timestamp on TARGET per pytorch issue it names.

    Ceiling: a comment that reads as a trigger but never ran one (its author
    failed the permission gate) still counts. The comment is the record; pairing
    comments with runs by timestamp would be guesswork.
    """
    seen = {}
    for c in paged(f"/repos/{TARGET_REPO}/issues/{TARGET_ISSUE}/comments"):
        if not FIX_CMD_RE.match(c["body"]):
            continue
        for num in ISSUE_RE.findall(c["body"]):
            seen[num] = max(seen.get(num, ""), c["created_at"])
    return seen


def fix_in_flight():
    """URL of a bot run whose `fix` job has not finished, else None.

    By status, not by reading the newest N runs: 30 runs of bot.yml span about
    three hours while a `fix` takes one to five, so a long one slides out of any
    window and the queue starts a second GPU job on top of it.
    """
    runs = [
        run
        for status in ("queued", "in_progress")
        for run in api(
            f"/repos/{TARGET_REPO}/actions/workflows/{BOT_WORKFLOW}/runs"
            f"?status={status}&per_page=100"
        )["workflow_runs"]
    ]
    for run in runs:
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


@functools.cache
def upstream(num):
    """The upstream issue. Cached because a SOURCE comment can list the same
    issue as an earlier one still being walked."""
    return api(f"/repos/pytorch/pytorch/issues/{num}")


def reopened_since(num, when):
    """Was this issue reopened after `when`? That starts a new episode -- the test
    is blocking CI again for a reason the last fix did not settle.

    From the events, because reopening clears `closed_at` (pytorch#194562: closed
    09-03, reopened 09-16, `closed_at` null today).
    """
    events = paged(f"/repos/pytorch/pytorch/issues/{num}/events")
    return any(e["event"] == "reopened" and e["created_at"] > when for e in events)


def pending(comment, mirrored):
    """Issues in this SOURCE batch that the bot should be asked to fix now."""
    out = []
    for num in disabled_issues(comment["body"]):
        since = mirrored.get(num)
        # A batch at or before the last mirror IS what was already mirrored, or
        # predates it. Only a later report can open a new episode.
        if since is not None and comment["created_at"] <= since:
            continue
        if upstream(num)["state"] == "open" and (
            since is None or reopened_since(num, since)
        ):
            out.append(num)
    return out


def next_batch(mirrored):
    """Oldest SOURCE comment since START that still has something to queue.

    First in, first out. Dedup already rules out repeats, so jumping the queue
    buys nothing, while newest-first would starve the tail if SOURCE ever
    reported faster than the queue drains.
    """
    try:
        comments = paged(f"/repos/{SOURCE_REPO}/issues/{SOURCE_ISSUE}/comments")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        sys.exit(
            f"cannot read {SOURCE_REPO}#{SOURCE_ISSUE} (404) -- deleted, renumbered, "
            f"or SOURCE_ISSUE is stale. Nothing can be queued until it points at the "
            f"CI report issue again."
        )
    for comment in comments:
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
    # Backticks on purpose: a bare URL makes GitHub file a cross-reference, so
    # every trigger would leave a backlink on pytorch's tracker. Code spans are
    # not scanned for references, and ISSUE_RE still matches inside them.
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
            *(f"- `https://github.com/pytorch/pytorch/issues/{n}`" for n in issues),
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
    assert FIX_CMD_RE.match(body), "the trigger this script posts is a trigger"
    assert not FIX_CMD_RE.match(
        "<!-- agent:session -->\n\nRunning `@torchxpubot fix` for "
        "https://github.com/pytorch/pytorch/issues/196748"
    ), "a session comment quoting the command is not a trigger"

    # Per-episode dedup. 197334 was reopened once, after the first mirror;
    # everything stays open.
    global upstream, reopened_since
    upstream = lambda num: {"state": "open"}  # noqa: E731
    reopened_since = lambda num, when: num == "197334" and when < "2026-09-30"  # noqa: E731
    old, new = (
        {"created_at": "2026-09-23T00:00:00Z"},
        {"created_at": "2026-10-02T00:00:00Z"},
    )
    assert old["created_at"] > START, "the dedup cases must sit after the cutoff"
    mirrored = {"197334": "2026-09-23T22:36:24Z", "197335": "2026-09-23T22:36:24Z"}

    assert pending(dict(old, body=batch), {}) == ["197334", "197335"], "never mirrored"
    assert pending(dict(old, body=batch), mirrored) == [], "the batch already mirrored"
    assert pending(dict(new, body=batch), mirrored) == ["197334"], (
        "reopened since its trigger is queued again; still-open is deduped"
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
        print(f"nothing to queue since {START}; {len(mirrored)} mirrored so far")
        return
    body = trigger_body(comment, issues)

    # Compared against the literal, so a typo or an unset value prints.
    if os.environ.get("DRY_RUN", "true").strip().lower() != "false":
        print(f"[dry run]\n\n{body}")
        return

    posted = api(f"/repos/{TARGET_REPO}/issues/{TARGET_ISSUE}/comments", {"body": body})
    print(f"queued {', '.join(issues)}: {posted['html_url']}")


if __name__ == "__main__":
    main()
