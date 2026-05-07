"""Issue body read/write helpers — all state lives in the issue body.

Replaces state.py comment-based tracking. Pure string manipulation,
no GitHub API calls — callers handle reading/writing the issue.
"""
from __future__ import annotations

import re
from datetime import datetime


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

    If the section doesn't exist, append it before Action Items
    (or at the end).
    """
    # Find the section heading and replace content until next heading
    pattern = re.compile(
        rf"(^#{{{1,4}}}\s+{re.escape(section)}\s*\n)(.*?)(?=^#{{{1,4}}}\s|\Z)",
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
    # Insert at the very top
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
    """Append text inside a <details> block identified by <!-- agent:MARKER-log -->.

    Looks for the marker comment and inserts log_text before it.
    """
    marker_tag = f"<!-- agent:{marker}-log -->"
    if marker_tag not in body:
        # Append a new details block at the end
        body += (
            f"\n<details><summary>{marker} log</summary>\n"
            f"{log_text}\n"
            f"{marker_tag}\n"
            f"</details>\n"
        )
        return body

    # Insert before the marker
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n**[{ts}]**\n{log_text}\n"
    return body.replace(marker_tag, entry + marker_tag)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

ISSUE_TEMPLATE = """\
<!-- agent:status:{stage} -->

## Summary
{summary}

## Test Info
- **Test Type:** {test_type}
- **Category:** {category}
- **Dependency:** {dependency}
- **Platform:** {platform}

## Failed Tests
{failed_tests}

## Error Log
```
{error_log}
```

## Reproducer
```bash
{reproducer}
```

## Commit Scope
{commit_scope}

## Root Cause Analysis
{root_cause}

## Proposed Fix Strategy
{fix_strategy}

## Action Items
- [ ] 🔍 Issue formatted (Discovery Agent)
  <details><summary>Discovery log</summary>
  <!-- agent:discovery-log -->
  </details>
- [ ] 🧠 Root cause identified (Triage Agent)
  <details><summary>Triage log</summary>
  <!-- agent:triage-log -->
  </details>
- [ ] 🔧 Fix implemented (Fix Agent)
  <details><summary>Fix log</summary>
  <!-- agent:fix-log -->
  </details>
- [ ] ✅ Fix verified locally
- [ ] 📋 PR proposed
- [ ] 👀 Human review
- [ ] 🎉 PR merged

## Context
{context}
"""


def render_initial_body(
    *,
    stage: str = "TRIAGING",
    summary: str = "_Pending discovery_",
    test_type: str = "",
    category: str = "",
    dependency: str = "",
    platform: str = "",
    failed_tests: str = "",
    error_log: str = "",
    reproducer: str = "",
    commit_scope: str = "",
    root_cause: str = "_Pending triage_",
    fix_strategy: str = "_Pending triage_",
    context: str = "",
) -> str:
    """Render the full issue body template from structured data."""
    return ISSUE_TEMPLATE.format(
        stage=stage,
        summary=summary,
        test_type=test_type,
        category=category,
        dependency=dependency,
        platform=platform,
        failed_tests=failed_tests,
        error_log=error_log,
        reproducer=reproducer,
        commit_scope=commit_scope,
        root_cause=root_cause,
        fix_strategy=fix_strategy,
        context=context,
    )


# ---------------------------------------------------------------------------
# Metadata extraction helpers (HTML comment markers)
# ---------------------------------------------------------------------------

def get_metadata(body: str, key: str) -> str | None:
    """Extract a metadata value from an HTML comment like <!-- key: value -->."""
    m = re.search(rf"<!--\s*{re.escape(key)}:\s*#?(.+?)\s*-->", body)
    return m.group(1).strip() if m else None


def sync_labels(repo: str, number: int, stage: str) -> None:
    """Ensure issue labels match the current stage."""
    from . import git as gh
    from .config import STAGE_TO_LABEL, ALL_AGENT_LABELS
    target_label = STAGE_TO_LABEL.get(stage)
    for label in ALL_AGENT_LABELS:
        if label == target_label:
            gh.add_label(repo, number, label)
        else:
            try:
                gh.remove_label(repo, number, label)
            except Exception:
                pass


def set_metadata(body: str, key: str, value: str) -> str:
    """Set or update a metadata HTML comment. Adds if missing."""
    pattern = rf"(<!--\s*{re.escape(key)}:\s*)#?(.+?)(\s*-->)"
    if re.search(pattern, body):
        return re.sub(pattern, rf"\g<1>{value}\3", body)
    return body + f"\n<!-- {key}: {value} -->\n"
