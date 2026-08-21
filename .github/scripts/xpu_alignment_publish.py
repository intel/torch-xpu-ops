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
    has_run_note,
    has_unit,
    list_comments,
    post_comment,
    render_draft,
    render_filed_note,
    render_run_note,
)


def run_note(decision: dict, payloads: list[dict]) -> tuple[str, list[str]] | None:
    """What a human must be told about this run, if anything."""
    lines: list[str] = []
    if decision["decision"] == "triage" and len(payloads) > 1:
        lines.append(f"{len(payloads)} reviewed candidates are waiting for approval:")
        lines.append("")
        lines += [
            f"- `{payload['unit_id']}` \u2014 {payload['title']}" for payload in payloads
        ]
        lines.append("")
        lines.append("Approve them one at a time with `@torchxpubot file <unit-id>`.")

    if decision.get("needs_attention"):
        reasons: list[str] = []
        pending = decision.get("pending_units") or []
        if pending:
            reasons.append(f"{len(pending)} candidate(s) never ran, so the day is unfinished")
        unpublishable = decision.get("unpublishable_units") or []
        if unpublishable:
            reasons.append(f"{len(unpublishable)} reviewed candidate(s) have no usable payload")
        blockers = decision.get("blockers") or []
        if blockers:
            shown = ", ".join(f"`{blocker}`" for blocker in blockers[:5])
            more = "" if len(blockers) <= 5 else f" (+{len(blockers) - 5} more)"
            reasons.append(f"{len(blockers)} blocker(s): {shown}{more}")
        if lines:
            lines.append("")
        lines.append("This run did not finish cleanly:")
        lines.append("")
        lines += [f"- {reason}" for reason in reasons]
        lines.append("")
        lines.append("Nothing was published for the affected part; see the gate artifact.")

    if not lines:
        return None
    headline = (
        "XPU alignment run needs attention"
        if decision.get("needs_attention")
        else f"{len(payloads)} XPU alignment candidates need triage"
    )
    return headline, lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the alignment gate decision")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--triage-issue", type=int, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--notify", default="")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    verdict = decision["decision"]
    run_id = str(decision.get("run_id", ""))
    scan_date = str(decision.get("scan_date", ""))
    # A blocked gate carries no payloads, so this loop is what keeps its
    # unreviewed verdicts off GitHub while the note below still reaches a human.
    payloads = decision["payloads"]
    if verdict == "file-one" and len(payloads) != 1:
        fail(f"decision file-one carries {len(payloads)} payloads")

    note = run_note(decision, payloads)
    if not payloads and not note:
        print(f"Nothing to publish (decision: {verdict}).")
        return 0

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

    if note and not has_run_note(existing, run_id):
        headline, lines = note
        post_comment(
            args.repo,
            args.triage_issue,
            render_run_note(run_id, scan_date, headline, lines, args.notify),
        )
        print(f"Notified #{args.triage_issue}: {headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
