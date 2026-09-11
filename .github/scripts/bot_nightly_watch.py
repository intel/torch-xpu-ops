#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Copy new DISABLED-test announcements from chuanqi129/pytorch-xpu-ci#364 into
intel/torch-xpu-ops#5272, with `@torchxpubot fix` on the first line.

The disabled tests are filed as issues in pytorch/pytorch and announced in #364.
`bot.yml` sees neither -- it listens to `issue_comment` on this repo only.
Copying the announcement here gives it something to act on. Nothing in this
script fixes anything.

#364 is a log, not a queue: most of its comments are reruns for unparseable or
timed-out jobs, and only some carry `Disable issues:`. The cursor is a marker in
#5272's body, the same trick #364 uses for its own "Last Processed Run", so a
re-run copies nothing until there is something new.

    python3 bot_nightly_watch.py --dry-run
"""

import argparse
import json
import re
import subprocess
import sys

CI_REPO = "chuanqi129/pytorch-xpu-ci"
CI_ISSUE = 364
TRACKING = "repos/intel/torch-xpu-ops/issues/5272"
MARKER_RE = re.compile(r"<!--\s*last-364-comment:\s*(\d+)\s*-->")
ISSUE_RE = re.compile(r"pytorch/pytorch/issues/(\d+)")


def gh(path, *args):
    out = subprocess.run(["gh", "api", path, *args], capture_output=True)
    if out.returncode != 0:
        sys.exit(f"gh api {path}: {out.stderr.decode()[:400]}")
    return out.stdout.decode()


def announcements(since_id):
    """Disable announcements after `since_id`, oldest first, and the id of the
    newest comment seen.

    Oldest first because `--limit` cuts the tail. The second value counts
    reruns too, which announce nothing but must still be passed or they are
    re-read every run.
    """
    raw = gh(f"repos/{CI_REPO}/issues/{CI_ISSUE}/comments?per_page=100",
             "--paginate", "-q", ".[]")
    seen = sorted((json.loads(l) for l in raw.splitlines() if l.strip()),
                  key=lambda c: c["id"])
    return ([c for c in seen
             if c["id"] > since_id and "Disable issues:" in (c["body"] or "")],
            seen[-1]["id"] if seen else since_id)


def copied_issues():
    """Issues already copied here. The tracking issue's own comments are the
    record -- `command_comment` quotes the announcement verbatim, links and
    all -- so there is no second copy of this state to keep in sync."""
    raw = gh(f"{TRACKING}/comments?per_page=100", "--paginate", "-q", ".[]")
    return set(ISSUE_RE.findall(raw))


def unseen(found, done):
    """Announcements carrying at least one issue not already copied.

    #364 re-announces a test every night it stays disabled: over the first 24
    days, 36 announcements carried 46 links but only 24 distinct issues, and 16
    of the 36 were pure repeats. Without this each repeat starts another fix
    job for an issue already being worked.

    `done` grows as we go, so a repeat inside one run is caught too.
    """
    out = []
    done = set(done)
    for c in found:
        ids = set(ISSUE_RE.findall(c["body"] or ""))
        # ponytail: an announcement mixing new and already-copied issues is
        # copied whole, re-fixing the old ones (3 of 36). Split it only if that
        # gets common.
        if ids - done:
            out.append(c)
            done |= ids
    return out


def command_comment(c):
    """The announcement, copied under the command.

    The command must be the first line: bot.yml anchors on
    `^@torchxpubot <cmd>` and treats the rest of the body as free text.
    """
    return (f"@torchxpubot fix\n\n"
            f"Copied from [{CI_REPO}#{CI_ISSUE}](https://github.com/{CI_REPO}"
            f"/issues/{CI_ISSUE}#issuecomment-{c['id']}) "
            f"({c['created_at'][:10]}). These tests are DISABLED upstream.\n\n"
            f"---\n\n{c['body'] or ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1,
                    help="cap the copies made in one run; each one starts a "
                         "fix job that holds a GPU runner")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    # `0` reads like "no cap" in the dispatch form but slices to nothing, and a
    # negative one silently drops the newest announcement.
    if a.limit < 1:
        ap.error("--limit must be at least 1")

    body = json.loads(gh(TRACKING))["body"] or ""
    m = MARKER_RE.search(body)
    since = int(m.group(1)) if m else 0

    found, newest = announcements(since)
    seen = unseen(found, copied_issues())
    todo = seen[:a.limit]
    print(f"{len(found)} new announcement(s) since comment {since}, "
          f"{len(found) - len(seen)} already copied, copying {len(todo)}")
    if a.dry_run:
        for c in todo:
            print("--- would post:\n" + command_comment(c) + "\n")
        return 0

    for c in todo:
        gh(f"{TRACKING}/comments", "--method", "POST",
           "-f", f"body={command_comment(c)}")
        print(f"copied #364 comment {c['id']} and asked for a fix")

    # A capped run advances only past what it copied; the rest waits.
    cursor = todo[-1]["id"] if len(todo) < len(seen) else newest
    gh(TRACKING, "--method", "PATCH", "-f",
       f"body={MARKER_RE.sub('', body).rstrip()}"
       f"\n\n<!-- last-364-comment: {cursor} -->\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
