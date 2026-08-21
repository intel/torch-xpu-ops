#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""File one reviewed alignment candidate from the standing triage issue.

Usage:
    python bot_alignment_file.py --repo owner/repo --triage-issue 5018 \
        --requested-on 5018 --unit-id candidate-1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alignment_triage import (
    UNIT_ID_RE,
    create_issue,
    fail,
    filed_body,
    find_draft,
    list_comments,
    parse_draft,
    update_comment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="File one reviewed alignment candidate")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--triage-issue", type=int, required=True)
    parser.add_argument("--requested-on", type=int, required=True)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args()

    # The drafts only exist on the standing triage issue, so refuse anywhere else.
    if args.requested_on != args.triage_issue:
        fail(f"`file` only works on the alignment triage issue #{args.triage_issue}.")
    if not UNIT_ID_RE.fullmatch(args.unit_id):
        fail(f"`{args.unit_id}` is not a valid unit id.")

    comment = find_draft(list_comments(args.repo, args.triage_issue), args.unit_id)
    title, body = parse_draft(comment["body"], args.unit_id)
    issue_url = create_issue(args.repo, title, body)
    update_comment(args.repo, comment["id"], filed_body(comment["body"], args.unit_id, issue_url))

    print(f"Filed {args.unit_id} as {issue_url}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"issue_url={issue_url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
