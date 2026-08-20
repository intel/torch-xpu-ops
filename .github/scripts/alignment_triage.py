#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""The comment protocol of the standing XPU alignment triage issue.

Drafts live as bot comments carrying a unit marker. Publishing a draft copies
its bytes verbatim into a new issue, so a human reads exactly what gets filed
and no agent runs between approval and filing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

UNIT_MARKER = "<!-- alignment-unit: {unit_id} -->"
RUN_MARKER = "<!-- alignment-run: {run_id} {scan_date} -->"
FILED_MARKER = "<!-- alignment-unit-filed: #{number} -->"
FILED_MARKER_RE = re.compile(r"<!-- alignment-unit-filed: #(\d+) -->")
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


def render_draft(unit_id: str, title: str, body: str, run_id: str, scan_date: str) -> str:
    return (
        f"{UNIT_MARKER.format(unit_id=unit_id)}\n"
        f"{RUN_MARKER.format(run_id=run_id, scan_date=scan_date)}\n"
        f"### {title}\n\n{body}\n"
    )


def render_filed_note(unit_id: str, issue_url: str, run_id: str, scan_date: str) -> str:
    """The audit line for a unit the workflow filed without human approval."""
    return (
        f"{UNIT_MARKER.format(unit_id=unit_id)}\n"
        f"{FILED_MARKER.format(number=issue_number(issue_url))}\n"
        f"{RUN_MARKER.format(run_id=run_id, scan_date=scan_date)}\n"
        f"`{scan_date}` automatically filed `{unit_id}` as {issue_url}\n"
    )


def has_unit(comments: list[dict], unit_id: str) -> bool:
    marker = UNIT_MARKER.format(unit_id=unit_id)
    return any(marker in (comment.get("body") or "") for comment in comments)


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


def create_issue(repo: str, title: str, body: str) -> str:
    if not title.startswith(ISSUE_TITLE_PREFIX):
        fail(f"Refusing to file `{title}`: the title must start with `{ISSUE_TITLE_PREFIX}`.")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(body.rstrip() + "\n")
        body_file = handle.name
    try:
        command = ["issue", "create", "--repo", repo, "--title", title, "--body-file", body_file]
        for label in ISSUE_LABELS:
            command += ["--label", label]
        return gh(command).strip().splitlines()[-1].strip()
    finally:
        os.unlink(body_file)


def post_comment(repo: str, issue: int, body: str) -> None:
    gh(
        ["api", "-X", "POST", f"repos/{repo}/issues/{issue}/comments", "--input", "-"],
        stdin=json.dumps({"body": body}),
    )


def update_comment(repo: str, comment_id: int, body: str) -> None:
    gh(
        ["api", "-X", "PATCH", f"repos/{repo}/issues/comments/{comment_id}", "--input", "-"],
        stdin=json.dumps({"body": body}),
    )
