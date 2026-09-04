#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Publish the alignment gate's decision to the standing triage issue.

Every reviewed candidate is posted or refreshed as a draft comment first.
`auto-file` opens up to three issues and marks their drafts as filed; larger
batches stop at drafts for manual handling.

Usage:
    python xpu_alignment_publish.py --repo owner/repo --triage-issue 5018 \
        --run-url https://github.com/owner/repo/actions/runs/123 \
        --decision alignment-artifacts/filing_decision.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from alignment_triage import (
    AUTO_FILE_LIMIT,
    FILED_MARKER_RE,
    create_issue,
    fail,
    filed_body,
    find_run_note,
    find_unit_comments,
    list_comments,
    post_comment,
    render_draft,
    render_run_note,
    parse_draft,
    update_comment,
)


RUN_STATES = {"complete", "complete-with-warnings", "partial", "failed"}
UNIT_BLOCKER_PREFIX = "scan-blocked-result:"


def _excluded_units(
    decision: dict,
) -> tuple[list[tuple[str, str]], list[str]]:
    excluded: dict[str, str] = {}
    unrecognized: list[str] = []
    for blocker in map(str, decision.get("unit_blockers") or []):
        if blocker.startswith(UNIT_BLOCKER_PREFIX):
            unit_id, _, result = blocker[len(UNIT_BLOCKER_PREFIX) :].rpartition(":")
            if unit_id:
                excluded[unit_id] = result
                continue
        unrecognized.append(blocker)
    for unit_id, verdict in (decision.get("unit_verdicts") or {}).items():
        if verdict == "verification-gap":
            excluded[str(unit_id)] = "verification-gap"
    return sorted(excluded.items()), sorted(set(unrecognized))


def _safe(value: object, limit: int = 200) -> str:
    message = " ".join(str(value).split()).replace("`", "'")
    return message[: limit - 1].rstrip() + "…" if len(message) > limit else message


def _blocker(blocker: object) -> str:
    code, separator, detail = str(blocker).partition(":")
    if separator:
        return f"`{_safe(code)}`: `{_safe(detail)}`"
    return f"`{_safe(code)}`"


def _partial_progress(decision: dict) -> list[str]:
    affected = [
        item
        for item in decision.get("collection_progress") or []
        if item.get("status") == "partial" or item.get("error")
    ]
    if not affected:
        return ["  - No partial source progress was recorded."]
    lines = []
    for item in affected:
        error = item.get("error") or {}
        lines.append(
            f"  - `{item.get('source', 'unknown')}`: "
            f"{item.get('pages_completed', 0)} page(s), "
            f"{item.get('items_fetched', 0)} item(s), status "
            f"`{item.get('status', 'unknown')}`, last cursor "
            f"`{item.get('last_cursor') or 'none'}`, rate reset "
            f"`{item.get('rate_reset_at') or 'unknown'}`, error "
            f"`{error.get('kind') or 'unknown'}`: "
            f"`{_safe(error.get('message') or 'no message recorded')}`"
        )
    return lines


def run_note(
    decision: dict,
    payloads: list[dict],
    filed: list[tuple[str, str]],
    publication_failures: list[str],
) -> tuple[str, list[str], bool]:
    """Render the outcome snapshot for one completed gate decision."""
    state = decision.get("run_state")
    if state not in RUN_STATES:
        fail(f"Unknown alignment run state: {state}")
    mode = decision.get("mode", "schedule")
    verdicts = decision.get("unit_verdicts") or {}
    counts = Counter(map(str, verdicts.values()))
    reviewed = f"- Reviewed units: {len(verdicts)}"
    if counts:
        detail = ", ".join(f"`{name}`: {counts[name]}" for name in sorted(counts))
        reviewed += f" ({detail})"
    lines = [
        f"- Scan date: `{decision.get('scan_date', '')}`",
        f"- Collection: {decision.get('collection_status') or 'unknown'}",
        reviewed,
        f"- Review-approved tracker candidates: {len(payloads)}",
        "",
    ]
    effective_decision = decision.get("would_decision")
    if filed:
        lines.append("Automatically filed:")
        lines += [f"- `{unit_id}` — {url}" for unit_id, url in filed]
    if mode == "dry-run" and payloads:
        if filed:
            lines.append("")
        lines.append("Dry-run drafts:")
        lines += [f"- `{item['unit_id']}` — {item['title']}" for item in payloads]
        lines += ["", "Dry-run drafts cannot be filed."]
    elif effective_decision == "triage" and payloads:
        if filed:
            lines.append("")
        lines.append("Formal candidate drafts:")
        lines += [f"- `{item['unit_id']}` — {item['title']}" for item in payloads]
        lines += [
            "",
            f"Automatic filing is limited to {AUTO_FILE_LIMIT} candidates; "
            "review these drafts and create trackers manually.",
        ]
    elif not filed and payloads and not publication_failures:
        lines.append("No new XPU tracker was filed.")
    elif not payloads:
        lines.append("No new XPU tracker was filed or drafted.")

    if publication_failures:
        lines += ["", f"- Publication failures: {len(publication_failures)}"]
        lines += [f"  - `{unit_id}` — issue publication failed; see the workflow log" for unit_id in publication_failures]

    excluded, unrecognized = _excluded_units(decision)
    if excluded or unrecognized:
        lines += ["", "- Excluded units:"]
        lines += [f"  - `{unit_id}` — `{reason}`" for unit_id, reason in excluded]
        lines += [f"  - {_blocker(blocker)}" for blocker in unrecognized]

    if decision.get("collection_status") == "partial":
        lines += ["", "- Incomplete collection progress:"]
        lines += _partial_progress(decision)
    if state == "failed":
        blockers = decision.get("global_blockers") or []
        distinct = list(dict.fromkeys(_blocker(blocker) for blocker in blockers))
        count = f"- Global blockers: {len(blockers)}"
        if len(distinct) != len(blockers):
            count += f" ({len(distinct)} distinct)"
        lines += ["", count]
        lines += [f"  - {blocker}" for blocker in distinct[:5]]
        if len(distinct) > 5:
            lines.append(f"  - {len(distinct) - 5} additional blocker(s) omitted")

    headlines = {
        "complete": "XPU alignment run complete",
        "complete-with-warnings": "XPU alignment run completed with warnings",
        "partial": "XPU alignment run completed with partial collection",
        "failed": "XPU alignment run failed",
    }
    headline = "XPU alignment run failed" if publication_failures else headlines[state]
    if mode == "dry-run":
        headline = f"[DRY RUN] {headline}"
    should_notify = mode == "schedule" and (
        bool(payloads) or state != "complete" or bool(publication_failures)
    )
    return headline, lines, should_notify


def _file_candidate(
    repo: str,
    payload: dict,
    draft: str,
    comment_id: int,
) -> str:
    unit_id = payload["unit_id"]
    title, body = parse_draft(draft, unit_id)
    issue_url = create_issue(repo, title, body, unit_id)
    update_comment(repo, comment_id, filed_body(draft, unit_id, issue_url))
    return issue_url


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the alignment gate decision")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--triage-issue", type=int, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--notify", default="")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if not isinstance(decision, dict) or decision.get("schema_version") != 1:
        fail("The publishing decision is not an XPU alignment artifact.")
    verdict = decision["decision"]
    outcomes = {"none", "auto-file", "triage", "blocked"}
    if verdict not in outcomes | {"dry-run"}:
        fail(f"Unknown publishing decision: {verdict}")
    would_decision = decision.get("would_decision")
    if would_decision not in outcomes:
        fail(f"Unknown effective publishing decision: {would_decision}")
    mode = decision.get("mode", "schedule")
    if mode not in {"schedule", "dry-run"}:
        fail(f"Unknown publishing mode: {mode}")
    if (mode == "dry-run") != (verdict == "dry-run"):
        fail("Dry-run mode and decision do not agree.")
    if mode == "schedule" and verdict != would_decision:
        fail("Scheduled decision and effective decision do not agree.")
    if decision.get("run_state") not in RUN_STATES:
        fail(f"Unknown alignment run state: {decision.get('run_state')}")
    run_id = str(decision.get("run_id", ""))
    scan_date = str(decision.get("scan_date", ""))
    payloads = decision["payloads"]
    if verdict == "auto-file" and not (1 <= len(payloads) <= AUTO_FILE_LIMIT):
        fail(f"decision auto-file carries {len(payloads)} payloads")
    if verdict in {"none", "blocked"} and payloads:
        fail(f"decision {verdict} must not carry payloads")
    if mode == "schedule" and verdict == "triage" and len(payloads) <= AUTO_FILE_LIMIT:
        fail(f"scheduled triage requires more than {AUTO_FILE_LIMIT} payloads")

    existing = list_comments(args.repo, args.triage_issue)
    existing_run_note = find_run_note(existing, run_id) if mode == "schedule" else None
    filed: list[tuple[str, str]] = []
    publication_failures: list[str] = []
    for payload in payloads:
        unit_id = payload["unit_id"]
        draft = render_draft(
            unit_id,
            payload["title"],
            payload["body"],
            run_id,
            scan_date,
            args.run_url,
            dry_run=mode == "dry-run",
        )
        comments = find_unit_comments(existing, unit_id) if mode == "schedule" else []
        filed_comment = next(
            (comment for comment in reversed(comments) if FILED_MARKER_RE.search(comment["body"])),
            None,
        )
        filed_issue_url = None
        if filed_comment is not None:
            issue_number = FILED_MARKER_RE.search(filed_comment["body"]).group(1)
            filed_issue_url = f"https://github.com/{args.repo}/issues/{issue_number}"
            if int(filed_comment["id"]) == int(comments[-1]["id"]):
                if verdict == "auto-file":
                    filed.append((unit_id, filed_issue_url))
                print(f"Skipping {unit_id}: already filed as {filed_issue_url}.")
                continue

        try:
            if comments:
                comment_id = int(comments[-1]["id"])
                update_comment(args.repo, comment_id, draft)
                print(f"Updated the latest draft for {unit_id} on #{args.triage_issue}")
            else:
                comment_id = post_comment(args.repo, args.triage_issue, draft)

            if mode == "dry-run":
                print(f"Posted dry-run draft for {unit_id} on #{args.triage_issue}")
                continue
            if filed_issue_url is not None:
                update_comment(
                    args.repo,
                    comment_id,
                    filed_body(draft, unit_id, filed_issue_url),
                )
                if verdict == "auto-file":
                    filed.append((unit_id, filed_issue_url))
                print(f"Skipping {unit_id}: already filed as {filed_issue_url}.")
                continue
            if verdict == "triage":
                print(f"Queued {unit_id} for manual triage on #{args.triage_issue}")
                continue

            issue_url = _file_candidate(args.repo, payload, draft, comment_id)
            filed.append((unit_id, issue_url))
            print(f"Filed {unit_id} as {issue_url}")
        except SystemExit:
            publication_failures.append(unit_id)
            print(f"::warning::Continuing after publication failed for `{unit_id}`.")

    headline, lines, should_notify = run_note(
        decision, payloads, filed, publication_failures
    )
    summary = render_run_note(
        run_id,
        args.run_url,
        headline,
        lines,
        args.notify if should_notify else "",
        dry_run=mode == "dry-run",
    )
    if existing_run_note:
        update_comment(args.repo, existing_run_note["id"], summary)
        print(f"Updated run summary on #{args.triage_issue}: {headline}")
    else:
        post_comment(args.repo, args.triage_issue, summary)
        print(f"Posted run summary on #{args.triage_issue}: {headline}")
    return 1 if publication_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
