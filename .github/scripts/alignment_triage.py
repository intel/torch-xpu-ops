#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""The comment protocol of the standing XPU alignment triage issue.

Drafts live as bot comments carrying a unit marker. Publishing preserves the
reviewed title and visible body and adds only a hidden stable-unit marker for
idempotency. No agent runs between approval and filing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

UNIT_MARKER = "<!-- alignment-unit: {unit_id} -->"
DRY_RUN_UNIT_MARKER = "<!-- alignment-dry-run-unit: {run_id}:{unit_id} -->"
DIAGNOSTIC_UNIT_MARKER = "<!-- alignment-diagnostic-unit: {scan_date}:{unit_id} -->"
DRY_RUN_DIAGNOSTIC_UNIT_MARKER = (
    "<!-- alignment-dry-run-diagnostic-unit: {run_id}:{unit_id} -->"
)
# Provenance is visible text, not an HTML comment: a triager reading a draft
# needs the run that produced it in order to re-read the underlying evidence.
PROVENANCE_LINE = "<sub>alignment scan `{scan_date}`, run `{run_id}`</sub>"
FILED_MARKER = "<!-- alignment-unit-filed: #{number} -->"
FILED_MARKER_RE = re.compile(r"<!-- alignment-unit-filed: #(\d+) -->")
PUBLISHED_UNIT_MARKER = "<!-- alignment-published-unit: {unit_id} -->"
# One notification per run, so a re-run does not ping anyone twice.
RUN_NOTE_MARKER = "<!-- alignment-run-note: {run_id} -->"
TITLE_LINE_RE = re.compile(r"^### (.+)$", re.MULTILINE)

ISSUE_TITLE_PREFIX = "[xpu-alignment]"
ISSUE_LABELS = ["ai_generated"]
# Unit ids become comment markers, file names and glob fragments, so they are
# restricted to one plain token with no separator or metacharacter.
UNIT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def gh(args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, input=stdin, check=False
    )
    if result.returncode != 0:
        fail(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def list_comments(repo: str, issue: int) -> list[dict]:
    comments: list[dict] = []
    page = 1
    while True:
        payload = json.loads(
            gh(["api", f"repos/{repo}/issues/{issue}/comments?per_page=100&page={page}"])
        )
        comments.extend(payload)
        if len(payload) < 100:
            return comments
        page += 1


def render_draft(
    unit_id: str,
    title: str,
    body: str,
    run_id: str,
    scan_date: str,
    *,
    dry_run: bool = False,
) -> str:
    marker = (
        DRY_RUN_UNIT_MARKER.format(run_id=run_id, unit_id=unit_id)
        if dry_run
        else UNIT_MARKER.format(unit_id=unit_id)
    )
    prefix = "[DRY RUN] " if dry_run else ""
    return (
        f"{marker}\n"
        f"{PROVENANCE_LINE.format(run_id=run_id, scan_date=scan_date)}\n"
        f"### {prefix}{title}\n\n{body}\n"
    )


def render_diagnostic_draft(
    unit_id: str,
    title: str,
    body: str,
    run_id: str,
    scan_date: str,
    *,
    dry_run: bool = False,
) -> str:
    """Render evidence from a partial scan without a fileable unit marker."""
    marker = (
        DRY_RUN_DIAGNOSTIC_UNIT_MARKER.format(run_id=run_id, unit_id=unit_id)
        if dry_run
        else DIAGNOSTIC_UNIT_MARKER.format(scan_date=scan_date, unit_id=unit_id)
    )
    prefix = "[DRY RUN][INCOMPLETE SCAN]" if dry_run else "[INCOMPLETE SCAN]"
    return (
        f"{marker}\n"
        f"{PROVENANCE_LINE.format(run_id=run_id, scan_date=scan_date)}\n"
        f"> This diagnostic draft came from a partial collection and cannot be filed.\n\n"
        f"### {prefix} {title}\n\n{body}\n"
    )


def post_comment(repo: str, issue: int, body: str) -> int:
    created = json.loads(
        gh(
            ["api", "-X", "POST", f"repos/{repo}/issues/{issue}/comments", "--input", "-"],
            stdin=json.dumps({"body": body}),
        )
    )
    return int(created["id"])


def has_unit(comments: list[dict], unit_id: str) -> bool:
    marker = UNIT_MARKER.format(unit_id=unit_id)
    return any(marker in (comment.get("body") or "") for comment in comments)


def has_diagnostic_unit(comments: list[dict], scan_date: str, unit_id: str) -> bool:
    marker = DIAGNOSTIC_UNIT_MARKER.format(scan_date=scan_date, unit_id=unit_id)
    return any(marker in (comment.get("body") or "") for comment in comments)


def has_run_note(comments: list[dict], run_id: str) -> bool:
    marker = RUN_NOTE_MARKER.format(run_id=run_id)
    return any(marker in (comment.get("body") or "") for comment in comments)


def render_run_note(
    run_id: str, scan_date: str, headline: str, lines: list[str], notify: str
) -> str:
    """The one comment that actively pings a human, rather than waiting to be found."""
    parts = [
        RUN_NOTE_MARKER.format(run_id=run_id),
        f"**{headline}**",
        "",
        *lines,
        "",
        f"<sub>alignment scan `{scan_date}`, run `{run_id}`</sub>",
    ]
    if notify:
        parts += ["", notify]
    return "\n".join(parts) + "\n"


def find_draft(comments: list[dict], unit_id: str) -> dict:
    marker = UNIT_MARKER.format(unit_id=unit_id)
    matches = [comment for comment in comments if marker in (comment.get("body") or "")]
    if not matches:
        fail(f"No draft comment carries the marker for `{unit_id}`.")
    if len(matches) > 1:
        fail(f"{len(matches)} draft comments carry the marker for `{unit_id}`.")
    return matches[0]


def parse_draft(body: str, unit_id: str) -> tuple[str, str]:
    already = FILED_MARKER_RE.search(body)
    if already:
        fail(f"`{unit_id}` was already filed as #{already.group(1)}.")
    title_match = TITLE_LINE_RE.search(body)
    if not title_match:
        fail(f"The draft for `{unit_id}` has no `### <title>` line.")
    title = title_match.group(1).strip()
    if not title.startswith(ISSUE_TITLE_PREFIX):
        fail(f"The draft title for `{unit_id}` does not start with `{ISSUE_TITLE_PREFIX}`.")
    issue_body = body[title_match.end() :].strip()
    if not issue_body:
        fail(f"The draft for `{unit_id}` has an empty body.")
    return title, issue_body


def issue_number(issue_url: str) -> str:
    return issue_url.rstrip("/").rsplit("/", 1)[-1]


def filed_body(body: str, unit_id: str, issue_url: str) -> str:
    marker = UNIT_MARKER.format(unit_id=unit_id)
    return body.replace(
        marker,
        f"{marker}\n{FILED_MARKER.format(number=issue_number(issue_url))}\n\n"
        f"**Filed as {issue_url}**",
        1,
    )


def find_published_issue(repo: str, unit_id: str) -> str | None:
    marker = PUBLISHED_UNIT_MARKER.format(unit_id=unit_id)
    page = 1
    while True:
        issues = json.loads(
            gh(
                [
                    "api",
                    f"repos/{repo}/issues?state=all&labels=ai_generated&per_page=100&page={page}",
                ]
            )
        )
        for issue in issues:
            if marker in (issue.get("body") or ""):
                return str(issue["html_url"])
        if len(issues) < 100:
            return None
        page += 1


def create_issue(repo: str, title: str, body: str, unit_id: str) -> str:
    if not title.startswith(ISSUE_TITLE_PREFIX):
        fail(f"Refusing to file `{title}`: the title must start with `{ISSUE_TITLE_PREFIX}`.")
    if not UNIT_ID_RE.fullmatch(unit_id):
        fail(f"Refusing to file an invalid unit id: `{unit_id}`.")
    existing = find_published_issue(repo, unit_id)
    if existing:
        return existing
    published_body = f"{PUBLISHED_UNIT_MARKER.format(unit_id=unit_id)}\n{body.rstrip()}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(published_body)
        body_file = handle.name
    try:
        command = ["issue", "create", "--repo", repo, "--title", title, "--body-file", body_file]
        for label in ISSUE_LABELS:
            command += ["--label", label]
        return gh(command).strip().splitlines()[-1].strip()
    finally:
        os.unlink(body_file)


def update_comment(repo: str, comment_id: int, body: str) -> None:
    gh(
        ["api", "-X", "PATCH", f"repos/{repo}/issues/comments/{comment_id}", "--input", "-"],
        stdin=json.dumps({"body": body}),
    )
