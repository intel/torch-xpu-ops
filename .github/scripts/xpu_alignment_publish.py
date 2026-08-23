#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Publish the alignment gate's decision to the standing triage issue.

Every reviewed candidate is posted as a draft comment first. `triage` stops
there and waits for `@torchxpubot file <unit-id>`; `file-one` goes on to open
the issue and marks its own draft as filed.

Usage:
    python xpu_alignment_publish.py --repo owner/repo --triage-issue 5018 \
        --decision alignment-artifacts/filing_decision.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment_triage import (
    FILED_MARKER_RE,
    create_issue,
    fail,
    filed_body,
    find_draft,
    has_run_note,
    has_unit,
    list_comments,
    post_comment,
    render_draft,
    render_run_note,
    parse_draft,
    update_comment,
)


def run_note(
    decision: dict, payloads: list[dict], filed: list[tuple[str, str]]
) -> tuple[str, list[str], bool]:
    """What a human must be told about this run, if anything."""
    lines: list[str] = []
    mode = decision.get("mode", "schedule")
    if mode == "dry-run":
        would = decision.get("would_decision", "blocked")
        lines.append(
            f"Dry run completed with `{would}` and {len(payloads)} review-approved candidate(s)."
        )
        if payloads:
            lines.append("")
            lines += [
                f"- `{payload['unit_id']}` — {payload['title']}" for payload in payloads
            ]
    elif filed:
        lines.append("Filed without human approval, so it is worth a second look:")
        lines.append("")
        lines += [f"- `{unit_id}` \u2014 {url}" for unit_id, url in filed]
    elif decision["decision"] == "triage" and len(payloads) > 1:
        lines.append(f"{len(payloads)} reviewed candidates are waiting for approval:")
        lines.append("")
        lines += [
            f"- `{payload['unit_id']}` \u2014 {payload['title']}" for payload in payloads
        ]
        lines.append("")
        lines.append("Approve them one at a time with `@torchxpubot file <unit-id>`.")

    if decision.get("needs_attention"):
        reasons: list[str] = []
        attention = decision.get("attention_reasons") or []
        if attention:
            reasons.append("scan state: " + ", ".join(f"`{item}`" for item in attention))
        pending = decision.get("pending_units") or []
        if pending:
            reasons.append(f"{len(pending)} candidate(s) never ran, so the day is unfinished")
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
        lines.append("The scheduled scan completed with 0 review-approved candidates.")
    if mode == "dry-run":
        headline = "[DRY RUN] XPU alignment run"
    elif decision.get("needs_attention"):
        headline = "XPU alignment run needs attention"
    elif filed:
        headline = f"{len(filed)} XPU alignment issue(s) filed automatically"
    elif decision.get("decision") == "none":
        headline = "XPU alignment run complete"
    else:
        headline = f"{len(payloads)} XPU alignment candidates need triage"
    should_notify = mode == "schedule" and (
        bool(filed)
        or decision.get("decision") == "triage"
        or bool(decision.get("needs_attention"))
    )
    return headline, lines, should_notify


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the alignment gate decision")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--triage-issue", type=int, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--notify", default="")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if not isinstance(decision, dict) or decision.get("schema_version") != 1:
        fail("The publishing decision is not an XPU alignment v1 artifact.")
    verdict = decision["decision"]
    if verdict not in {"none", "file-one", "triage", "blocked", "dry-run"}:
        fail(f"Unknown publishing decision: {verdict}")
    mode = decision.get("mode", "schedule")
    if mode not in {"schedule", "dry-run"}:
        fail(f"Unknown publishing mode: {mode}")
    if (mode == "dry-run") != (verdict == "dry-run"):
        fail("Dry-run mode and decision do not agree.")
    run_id = str(decision.get("run_id", ""))
    scan_date = str(decision.get("scan_date", ""))
    # A blocked gate carries no payloads, so this loop is what keeps its
    # unreviewed verdicts off GitHub while the note below still reaches a human.
    payloads = decision["payloads"]
    if verdict == "file-one" and len(payloads) != 1:
        fail(f"decision file-one carries {len(payloads)} payloads")
    if verdict in {"none", "blocked"} and payloads:
        fail(f"decision {verdict} must not carry payloads")
    if mode == "schedule" and verdict == "triage" and len(payloads) < 2:
        fail("scheduled triage requires at least two payloads")

    existing = list_comments(args.repo, args.triage_issue)
    filed: list[tuple[str, str]] = []
    for payload in payloads:
        unit_id = payload["unit_id"]
        if mode == "schedule" and has_unit(existing, unit_id):
            if verdict == "file-one":
                comment = find_draft(existing, unit_id)
                already_filed = FILED_MARKER_RE.search(comment["body"])
                if already_filed:
                    issue_url = f"https://github.com/{args.repo}/issues/{already_filed.group(1)}"
                else:
                    title, body = parse_draft(comment["body"], unit_id)
                    if title != payload["title"] or body != payload["body"].strip():
                        fail(f"Existing draft for `{unit_id}` does not match the gate payload.")
                    issue_url = create_issue(args.repo, title, body, unit_id)
                    update_comment(
                        args.repo,
                        comment["id"],
                        filed_body(comment["body"], unit_id, issue_url),
                    )
                filed.append((unit_id, issue_url))
                print(f"Recovered filed unit {unit_id} as {issue_url}")
            else:
                print(f"Skipping {unit_id}: already present on #{args.triage_issue}.")
            continue
        # The draft goes up before the issue does. Crashing after this point
        # leaves a candidate a human can still file by hand; crashing after the
        # issue instead would leave no record and re-file it on the next run.
        draft = render_draft(
            unit_id,
            payload["title"],
            payload["body"],
            run_id,
            scan_date,
            dry_run=mode == "dry-run",
        )
        comment_id = post_comment(args.repo, args.triage_issue, draft)
        if mode == "dry-run":
            print(f"Posted dry-run draft for {unit_id} on #{args.triage_issue}")
            continue
        if verdict != "file-one":
            print(f"Queued {unit_id} for triage on #{args.triage_issue}")
            continue
        issue_url = create_issue(args.repo, payload["title"], payload["body"], unit_id)
        update_comment(args.repo, comment_id, filed_body(draft, unit_id, issue_url))
        filed.append((unit_id, issue_url))
        print(f"Filed {unit_id} as {issue_url}")

    headline, lines, should_notify = run_note(decision, payloads, filed)
    if mode == "dry-run" or not has_run_note(existing, run_id):
        post_comment(
            args.repo,
            args.triage_issue,
            render_run_note(
                run_id,
                scan_date,
                headline,
                lines,
                args.notify if should_notify else "",
            ),
        )
        print(f"Posted run summary on #{args.triage_issue}: {headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
