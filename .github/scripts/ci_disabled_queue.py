#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Queue upstream DISABLED-test batches onto the tracking issue as `@torchxpubot fix`.

Usage:
    GH_TOKEN=<read on SOURCE_REPO, write on TARGET_REPO> \
        python ci_disabled_queue.py                   # print the trigger, post nothing
    GH_TOKEN=<...> DRY_RUN=false \
        python ci_disabled_queue.py                   # actually comment
    python ci_disabled_queue.py --self-test

Printing is the default and `DRY_RUN=false` is the only value that posts, since
the live side comments on a tracking issue and starts a multi-hour GPU job.

There is no by-hand mode. A human who wants a specific test fixed now says
`@torchxpubot fix` on the tracking issue from their own account, which is fewer
steps than dispatching a workflow. This script exists for the runs nobody
triggers.

The XPU CI report issue (SOURCE) grows a comment per failing `xpu.yml` commit.
The ones that matter here carry a `Disable issues:` list of `pytorch/pytorch`
DISABLED issues, and the same issue is re-listed on every later failing commit.
The tracking issue (TARGET) is the bot's work queue: one `@torchxpubot fix`
comment per batch, which is what a human posts by hand today.

Every rule below is derived from GitHub state, so there is no local database to
keep in sync: START (the cutoff), dedup, grouping and serialization each live in
the function that applies them. One batch per invocation, oldest first.

A batch is one SOURCE comment, which is one failing xpu.yml run: its issues come
from the same commit, so they are likelier to share a cause than any grouping
this script could infer from test names. Splitting them would also cost a build
each -- /issue-handler fixes the first entry and re-runs the rest against the
staged fix, listing what that covers.
"""

import functools
import json
import os
import re
import sys
import urllib.error
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

    Matched the way bot.yml matches it. Merely containing the command counted the
    bot's own session comments as triggers, which both moved a timestamp later
    than the real trigger and would have silenced any issue a session comment
    happened to link but nobody had queued.

    Ceiling: a comment that reads as a trigger but never ran one -- posted by
    someone the permission gate rejects -- still counts. Asking "did a fix run
    for this issue" instead means matching comments to runs by timestamp, which
    is guesswork; the comment is the record.
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

    Asked by status, not by reading the newest N runs: 30 runs of bot.yml span
    about three hours and a `fix` takes one to five, so a long one slides out of
    any fixed window and the queue would start a second GPU job on top of it.
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
    """Was this issue reopened after `when`?

    That is what starts a new episode: the test is blocking CI again, for a
    reason the last fix did not settle, so it gets queued again. Read from the
    events -- `closed_at` cannot answer it, since reopening clears the field
    (verified on pytorch#194562: closed 09-03, reopened 09-16, `closed_at` null).
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

    First in, first out. Dedup already makes a repeat impossible, so taking the
    newest batch first would buy nothing and could starve the tail whenever
    SOURCE reports faster than the queue drains; oldest-first cannot starve
    anything, and the batch that has waited longest goes next.
    """
    try:
        comments = paged(f"/repos/{SOURCE_REPO}/issues/{SOURCE_ISSUE}/comments")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # The expected failure today, and a traceback would bury the one thing
        # the operator can act on.
        sys.exit(
            f"cannot read {SOURCE_REPO}#{SOURCE_ISSUE} (404), so discovery is not "
            f"available: MERGE_TOKEN is a fine-grained PAT and those cannot reach a "
            f"repo owned by another personal account. Until the report issue moves "
            f"into {TARGET_REPO}, a human posts `@torchxpubot fix` by hand."
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
    # The issue URLs are wrapped in backticks on purpose. A bare URL makes GitHub
    # file a cross-reference on the upstream issue, so every trigger would leave
    # an `intel/torch-xpu-ops#5272` backlink on pytorch's tracker -- the nine
    # already posted by hand each did. Code spans are not scanned for references,
    # and ISSUE_RE still matches inside them, so dedup is unaffected.
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

    # Posting takes the exact string "false" and nothing else, so an unset or
    # misspelled DRY_RUN prints instead of commenting on a live issue.
    if os.environ.get("DRY_RUN", "true").strip().lower() != "false":
        print(f"[dry run]\n\n{body}")
        return

    posted = api(f"/repos/{TARGET_REPO}/issues/{TARGET_ISSUE}/comments", {"body": body})
    print(f"queued {', '.join(issues)}: {posted['html_url']}")


if __name__ == "__main__":
    main()
