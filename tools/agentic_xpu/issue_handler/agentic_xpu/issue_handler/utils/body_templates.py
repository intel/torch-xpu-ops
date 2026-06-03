# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Issue body read/write helpers — all state lives in the issue body.

Pure string manipulation, no GitHub API calls — callers handle reading/writing.
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import yaml


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

def parse_sections(body: str) -> dict[str, str]:
    """Parse markdown body into {section_name: content} dict.

    Sections are delimited by ## or ### headings.
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


def update_section(body: str, section: str, content: str) -> str:
    """Replace content of a named section, preserving other sections.

    Matches heading levels ## through #### (H2–H4) as section delimiters.
    H1 headings (#) are intentionally excluded — they may appear inside
    section content (e.g. "# Observe: ..." comment lines) and must not be
    treated as section boundaries.

    If the section doesn't exist, append it before Action Items (or at the end).
    """
    # Regex: capture the heading line (group 1) and everything after it
    # (group 2: content) until the next H2–H4 heading or end of string.
    pattern = re.compile(
        r"(^#{2,4}\s+" + re.escape(section) + r"\s*\n)(.*?)(?=^#{2,4}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    if match:
        return body[:match.start(2)] + content + "\n\n" + body[match.end(2):]

    # Section doesn't exist — insert before Action Items or append
    action_match = re.search(r"^## Action Items", body, re.MULTILINE)
    insert_point = action_match.start() if action_match else len(body)
    new_section = f"\n## {section}\n{content}\n\n"
    return body[:insert_point] + new_section + body[insert_point:]


# ---------------------------------------------------------------------------
# Status block  <!-- agent:status:STAGE -->
# ---------------------------------------------------------------------------

STATUS_PATTERN = re.compile(r"<!-- agent:status:(\w+) -->")


def get_status(body: str) -> str | None:
    """Extract stage from <!-- agent:status:STAGE --> marker. Returns None if absent."""
    match = STATUS_PATTERN.search(body or "")
    return match.group(1) if match else None


def set_status(body: str, stage: str) -> str:
    """Update or insert the status marker with new stage."""
    new_marker = f"<!-- agent:status:{stage} -->"
    if STATUS_PATTERN.search(body):
        return STATUS_PATTERN.sub(new_marker, body)
    return new_marker + "\n\n" + body


# ---------------------------------------------------------------------------
# Action items checkboxes
# ---------------------------------------------------------------------------

def check_action_item(body: str, item_substring: str) -> str:
    """Mark a checkbox done: - [ ] ...item_substring... → - [x] ...item_substring..."""
    def _replace(m: re.Match) -> str:
        line = m.group(0)
        if item_substring.lower() in line.lower():
            return line.replace("- [ ]", "- [x]", 1)
        return line

    return re.sub(r"^- \[ \] .+$", _replace, body, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Folded logs  <!-- agent:MARKER-log -->
# ---------------------------------------------------------------------------

def append_log(body: str, marker: str, log_text: str) -> str:
    """Append text inside a <details> block identified by <!-- agent:MARKER-log -->."""
    marker_tag = f"<!-- agent:{marker}-log -->"
    if marker_tag not in body:
        body += (
            f"\n<details><summary>{marker} log</summary>\n"
            f"{log_text}\n"
            f"{marker_tag}\n"
            f"</details>\n"
        )
        return body

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marker_match = re.search(r'^([ \t]*)' + re.escape(marker_tag), body, re.MULTILINE)
    indent = marker_match.group(1) if marker_match else ""
    log_lines = f"**[{ts}]**\n{log_text}".splitlines()
    indented_log = "\n".join(indent + line if line.strip() else "" for line in log_lines)
    entry = f"\n{indented_log}\n"
    return body.replace(marker_tag, entry + indent + marker_tag)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

from .config import CONFIG_DIR
ISSUE_TEMPLATE_PATH = CONFIG_DIR / "agent-issue-body.yml"
NONBUG_TEMPLATE_PATH = CONFIG_DIR / "agent-issue-body-nonbug.yml"
PR_TEMPLATE_PATH = CONFIG_DIR / "pr-body-template.md"


def build_body(template_path: Path, **kwargs: str) -> str:
    """Load a YAML template file and fill placeholders in its 'body' field."""
    with open(template_path, encoding="utf-8") as f:
        template_data = yaml.safe_load(f)
    template = template_data.get("body", "")
    return template.format(**kwargs)


def render_initial_body(
    *,
    test_type: str = "",
    category: str = "",
    dependency: str = "",
    platform: str = "",
    root_cause: str = "_Pending triage_",
    fix_strategy: str = "_Pending triage_",
    target_repo: str = "_Pending triage_",
    context: str = "",
    original_issue: str = "",
) -> str:
    """Render the bug issue body suffix (agent pipeline sections only)."""
    return build_body(
        ISSUE_TEMPLATE_PATH,
        test_type=test_type,
        category=category,
        dependency=dependency,
        platform=platform,
        root_cause=root_cause,
        fix_strategy=fix_strategy,
        target_repo=target_repo,
        context=context,
        original_issue=original_issue,
    )


def set_metadata(body: str, key: str, value: str) -> str:
    """Set or update a metadata HTML comment. Adds if missing."""
    pattern = rf"(<!--\s*{re.escape(key)}:\s*)#?(.+?)(\s*-->)"
    if re.search(pattern, body):
        return re.sub(pattern, rf"\g<1>{value}\3", body)
    return body + f"\n<!-- {key}: {value} -->\n"


def get_metadata(body: str, key: str) -> str | None:
    """Extract a metadata value from an HTML comment like <!-- key: value -->."""
    m = re.search(rf"<!--\s*{re.escape(key)}:\s*#?(.+?)\s*-->", body)
    return m.group(1).strip() if m else None


def render_nonbug_body(
    *,
    category: str = "",
    platform: str = "",
    related_components: str = "",
    context: str = "",
    original_issue: str = "",
) -> str:
    """Render the non-bug issue body suffix (agent pipeline sections only)."""
    return build_body(
        NONBUG_TEMPLATE_PATH,
        category=category,
        platform=platform,
        related_components=related_components,
        context=context,
        original_issue=original_issue,
    )


def render_pr_body(
    *,
    upstream_issue_repo: str,
    source_number: int,
    title: str,
    triage_reason: str | None = None,
    issue_body: str = "",
    include_diff_stat: bool = False,
    diff_stat: str = "",
    reviewer: str = "",
) -> str:
    """Build a PR description from issue details using pr_body_template.md."""
    issue_url = f"https://github.com/{upstream_issue_repo}/issues/{source_number}"

    root_cause_section = f"**Root Cause:** {triage_reason}\n" if triage_reason else ""

    sections = parse_sections(issue_body)
    failed_tests_section = (
        f"**Failed Tests:**\n{sections['Failed Tests']}\n" if sections.get("Failed Tests") else ""
    )
    failure_type_section = (
        f"---\n\n**Failure Type:** {sections['Failure Type']}\n" if sections.get("Failure Type") else ""
    )
    diff_stat_section = (
        f"---\n\n**Diff stat:**\n```\n{diff_stat}\n```\n" if include_diff_stat and diff_stat else ""
    )
    reviewer_section = f"---\n\ncc @{reviewer}\n" if reviewer else ""

    return build_body(
        PR_TEMPLATE_PATH,
        upstream_issue_repo=upstream_issue_repo,
        source_number=source_number,
        issue_url=issue_url,
        title=title,
        root_cause_section=root_cause_section,
        failed_tests_section=failed_tests_section,
        failure_type_section=failure_type_section,
        diff_stat_section=diff_stat_section,
        reviewer_section=reviewer_section,
    )
