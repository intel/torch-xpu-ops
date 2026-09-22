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
the function that applies them. One problem per invocation, oldest first.
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
# `DISABLED test_foo_xpu_float32 (__main__.TestBarXPU)` -- the shape every
# upstream DISABLED issue's title has.
TITLE_RE = re.compile(r"^DISABLED\s+(\S+)\s+\(__main__\.(\w+)\)")


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


@functools.cache
def upstream(num):
    """The upstream issue. Cached: `pending` reads its state, `one_problem` its
    title, and a batch is small enough that one process-lifetime cache is the
    whole story."""
    return api(f"/repos/pytorch/pytorch/issues/{num}")


def problem_key(title):
    """What makes two DISABLED issues the same problem: class + test name.

    The name keeps its parametrization out of it, so the four dtype variants of
    `test_redispatch_scatter` share a key. The class alone would not do: the
    grid_sample and the scatter families live in the same
    `TestTorchFunctionRedispatchOpsDeviceXPU` and are unrelated failures. A title
    that does not parse gets a key of its own and is never grouped.
    """
    m = TITLE_RE.match(title)
    if not m:
        return (title,)
    test, cls = m.groups()
    # Cut at the LAST device token, not a regex anchored on the first one: the
    # parametrisation is always the tail, and `test_copy_xpu_to_cuda` /
    # `test_copy_cpu_to_xpu` are two tests that cutting at the first would merge.
    cut = max(test.rfind(d) for d in ("_xpu", "_cpu", "_cuda"))
    return cls, test[:cut] if cut > 0 else test


def one_problem(nums):
    """The oldest problem's issues, so a run has one root cause to find and its
    patch is one thing to review. The rest waits for a later round.

    Batching, not a verdict on what shares a cause: /issue-handler settles that
    by evidence (fix one entry, re-run the others against the staged fix, list
    what passes in `covers`). The title only has to put the entries that
    mechanism pays off on -- one test's dtype variants, one build -- in the same
    run. Do not replace this with a model: at queue time it would guess from
    these same titles, before anything has been run.
    """
    keys = {n: problem_key(upstream(n)["title"]) for n in nums}
    return [n for n in nums if keys[n] == keys[nums[0]]]


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
        if upstream(num)["state"] == "open" and (
            since is None or closed_since(num, since)
        ):
            out.append(num)
    return out


def next_batch(mirrored):
    """Oldest SOURCE comment since START that still has something to queue."""
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
            return comment, one_problem(new)
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

    # One trigger per problem. The two families share a class, so the class alone
    # would have merged two unrelated failures into one run.
    titles = {
        "197521": "DISABLED test_redispatch_nn_functional_grid_sample_xpu_bfloat16 "
        "(__main__.TestTorchFunctionRedispatchOpsDeviceXPU)",
        "197523": "DISABLED test_redispatch_nn_functional_grid_sample_xpu_float32 "
        "(__main__.TestTorchFunctionRedispatchOpsDeviceXPU)",
        "197334": "DISABLED test_redispatch_scatter_xpu_float8_e4m3fn "
        "(__main__.TestTorchFunctionRedispatchOpsDeviceXPU)",
        "196247": "DISABLED test_1mb_allocation_uses_small_block (__main__.TestXpu)",
        "196308": "DISABLED test_graph_checkpoint_preserve_rng_state (__main__.TestXpu)",
        "1": "Something that is not a DISABLED title",
    }
    assert problem_key(titles["197521"]) == problem_key(titles["197523"]), (
        "dtype variants of one test are one problem"
    )
    assert problem_key(titles["197521"]) != problem_key(titles["197334"]), (
        "same class, different test: two problems"
    )
    assert problem_key(titles["196247"]) != problem_key(titles["196308"]), (
        "unrelated tests in one class stay apart"
    )
    assert problem_key(titles["1"]) == (titles["1"],), "an unparseable title is its own"
    assert problem_key(
        "DISABLED test_copy_xpu_to_cuda (__main__.TestXpu)"
    ) != problem_key("DISABLED test_copy_cpu_to_xpu (__main__.TestXpu)"), (
        "the device token in the tail is the parametrisation, not one mid-name"
    )

    # One stub for both readers of the upstream issue: everything stays open, and
    # 197334 was closed once, after the first mirror.
    global upstream, closed_since
    upstream = lambda num: {"state": "open", "title": titles.get(num, "")}  # noqa: E731
    closed_since = lambda num, when: num == "197334" and when < "2026-09-30"  # noqa: E731

    assert one_problem(["197521", "197334", "197523", "196247"]) == [
        "197521",
        "197523",
    ], "the oldest problem only, however the batch is ordered"
    assert one_problem(["1", "197521"]) == ["1"], "unparseable goes alone"

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
