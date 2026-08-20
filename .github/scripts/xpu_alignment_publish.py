#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Publish the alignment gate's decision to the standing triage issue.

`file-one` files one issue and records it; `triage` posts one draft comment per
reviewed candidate for a human to approve with `@torchxpubot file <unit-id>`.

Usage:
    python xpu_alignment_publish.py --repo owner/repo --triage-issue 5018 \
        --decision alignment-artifacts/filing_decision.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment_triage import (
    create_issue,
    fail,
    has_unit,
    list_comments,
    post_comment,
    render_draft,
    render_filed_note,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the alignment gate decision")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--triage-issue", type=int, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    verdict = decision["decision"]
    if verdict in {"blocked", "none"}:
        print(f"Nothing to publish (decision: {verdict}).")
        return 0

    run_id = str(decision.get("run_id", ""))
    scan_date = str(decision.get("scan_date", ""))
    payloads = decision["payloads"]
    if verdict == "file-one" and len(payloads) != 1:
        fail(f"decision file-one carries {len(payloads)} payloads")

    # Re-running a day must not repost drafts or file a second copy.
    existing = list_comments(args.repo, args.triage_issue)

    for payload in payloads:
        unit_id = payload["unit_id"]
        if has_unit(existing, unit_id):
            print(f"Skipping {unit_id}: already present on #{args.triage_issue}.")
            continue
        if verdict == "file-one":
            issue_url = create_issue(args.repo, payload["title"], payload["body"])
            post_comment(
                args.repo,
                args.triage_issue,
                render_filed_note(unit_id, issue_url, run_id, scan_date),
            )
            print(f"Filed {unit_id} as {issue_url}")
        else:
            post_comment(
                args.repo,
                args.triage_issue,
                render_draft(unit_id, payload["title"], payload["body"], run_id, scan_date),
            )
            print(f"Queued {unit_id} for triage on #{args.triage_issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
