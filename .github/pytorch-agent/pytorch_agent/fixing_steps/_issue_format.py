"""Shared issue/PR formatting helpers."""
from __future__ import annotations

import re


def parse_issue_sections(body: str) -> dict[str, str]:
    """Parse an issue body into titled sections.

    Looks for markdown headings (### Heading) and collects text until the
    next heading.
    """
    sections: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    for line in (body or "").splitlines():
        heading = re.match(r"^#{1,4}\s+(.+)", line)
        if heading:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def build_pr_body(
    *,
    upstream_issue_repo: str,
    source_number: int,
    title: str,
    triage_reason: str | None,
    issue_body: str,
    include_diff_stat: bool = False,
    diff_stat: str = "",
    reviewer: str = "",
) -> str:
    """Build a PR description from issue details."""
    issue_url = f"https://github.com/{upstream_issue_repo}/issues/{source_number}"

    body = (
        f"## Summary\n\n"
        f"Fix for [{upstream_issue_repo}#{source_number}]({issue_url})\n\n"
        f"**Issue:** {title}\n\n"
    )
    if triage_reason:
        body += f"**Root Cause:** {triage_reason}\n\n"

    sections = parse_issue_sections(issue_body)
    if sections.get("Failed Tests"):
        body += f"**Failed Tests:**\n{sections['Failed Tests']}\n\n"
    if sections.get("Failure Type"):
        body += f"---\n\n**Failure Type:** {sections['Failure Type']}\n\n"

    if include_diff_stat and diff_stat:
        body += f"---\n\n**Diff stat:**\n```\n{diff_stat}\n```\n\n"

    body += "---\n\n"
    if reviewer:
        body += f"cc @{reviewer}\n"

    return body
