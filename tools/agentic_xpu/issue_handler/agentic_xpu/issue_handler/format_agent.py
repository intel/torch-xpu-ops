"""Discovery agent — extract content from raw issue, build formatted body from template.

Approach:
  1. Regex split: break the raw body into (heading, content) sections.
  2. Classify: rule-based matching assigns each section to a template slot.
     Ambiguous or mixed sections go to the LLM for classification/splitting.
  3. Build: fill the template and write back.

Entry point:
  python -m agentic_xpu.issue_handler.format_agent --issue 123
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

import yaml

from .utils import git as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS, AGENT_DIR
from .utils.body_templates import (
    get_status, build_body, check_action_item, append_log,
    ISSUE_TEMPLATE_PATH,
)
from .utils.agent_backend import get_backend
from .utils.json_utils import extract_json
from .utils.logger import log
from .utils.stages import Skill, Stage
from .utils.git import sync_labels


# ---------------------------------------------------------------------------
# Step 1: Regex split into raw sections
# ---------------------------------------------------------------------------

@dataclass
class RawSection:
    """A section parsed from the raw issue body."""
    heading: str          # e.g. "### 🐛 Describe the bug", "" for header
    content: str          # text content under the heading
    level: int = 0        # heading level (2 for ##, 3 for ###, 0 for no heading)


def _split_into_sections(body: str) -> list[RawSection]:
    """Split body at ## or ### headings into (heading, content) pairs.

    The first block before any heading becomes a section with heading="".
    Headings inside code fences are ignored.
    """
    fence_ranges = _code_fence_ranges(body)
    # Find all ## and ### headings outside fences
    heading_pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
    matches = []
    for m in heading_pattern.finditer(body):
        if not _is_inside_fence(m.start(), fence_ranges):
            matches.append(m)

    sections: list[RawSection] = []

    if not matches:
        # No headings — entire body is one section
        return [RawSection(heading="", content=body.strip(), level=0)]

    # Header: content before first heading
    header_text = body[:matches[0].start()].strip()
    if header_text:
        sections.append(RawSection(heading="", content=header_text, level=0))

    # Each heading + content until next heading
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        level = len(m.group(1))
        heading = m.group(0).strip()
        sections.append(RawSection(heading=heading, content=content, level=level))

    return sections


# ---------------------------------------------------------------------------
# Step 2: Classify each section into a template slot
# ---------------------------------------------------------------------------

# Slot names matching template placeholders
SLOT_DESCRIPTION = "description"
SLOT_REPRODUCER = "reproducer"
SLOT_ERROR_LOG = "error_log"
SLOT_ENVIRONMENT = "environment"
SLOT_ADDITIONAL = "additional_context"
SLOT_HEADER = "_header"  # metadata header, goes into additional_context

# Rule-based classification patterns
_DESCRIPTION_RE = re.compile(
    r'describe\s+the\s+bug|description|summary|overview|bug\s+report',
    re.IGNORECASE,
)
_REPRODUCER_RE = re.compile(
    r'reproducer|repro(?:duction)?(?:\s+steps)?|steps\s+to\s+reproduce|minimal\s+repro|how\s+to\s+reproduce',
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r'observed\s+output|error(?:\s+log|\s+output)?|traceback|'
    r'actual\s+(?:behavior|output|result)|failed\s+output|output\s+on\s+\w+',
    re.IGNORECASE,
)
_ENV_RE = re.compile(
    r'environment|versions|collect_env|system\s+info',
    re.IGNORECASE,
)

# Content-based heuristics
_ENV_CONTENT_MARKERS = [
    'PyTorch version:', 'Is debug build:', '[pip3]', '[conda]',
    'Collecting environment information', 'CUDA used to build',
    'Is XPU available:', 'ROCM used to build',
]
_REPRODUCER_CONTENT_MARKERS = ['import torch', 'torch.device', 'torch.randn']
# Markers that identify a code block as error/diagnostic evidence rather than
# a reproducer or environment block.  Checked only in _split_mixed_section to
# avoid misclassifying a short inline code snippet as the error log.
_ERROR_CONTENT_MARKERS = [
    'BUG CONFIRMED', 'Traceback (most recent call last)', 'AssertionError',
    'RuntimeError', 'FAILED', 'Error:', 'assert ', '>>> ', 'bug confirmed',
]


def _classify_by_heading(heading: str) -> str | None:
    """Classify a section by its heading text. Returns slot name or None."""
    if not heading:
        return SLOT_HEADER

    # Strip heading markers and emoji
    text = re.sub(r'^#{1,3}\s*', '', heading)
    text = re.sub(r'[🐛🔧✅📋💡🎯🧠🔍]', '', text).strip()

    if _REPRODUCER_RE.search(text):
        return SLOT_REPRODUCER
    if _ERROR_RE.search(text):
        return SLOT_ERROR_LOG
    if _ENV_RE.search(text):
        return SLOT_ENVIRONMENT
    if _DESCRIPTION_RE.search(text):
        return SLOT_DESCRIPTION
    return None


def _classify_by_content(content: str) -> str | None:
    """Classify a section by its content heuristics. Returns slot name or None."""
    if any(marker in content for marker in _ENV_CONTENT_MARKERS):
        return SLOT_ENVIRONMENT
    return None


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (lang, code) pairs from code fences in text.

    Accepts up to 3 spaces of indentation before opening/closing fences
    (CommonMark §4.5).  Without this allowance, a fence indented by even a
    single space inside a list item would not be recognised, silently
    dropping the entire reproducer.
    """
    blocks = []
    pattern = re.compile(
        r'^ {0,3}(`{3,})(\w*)\s*\n(.*?)^ {0,3}\1\s*$',
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        blocks.append((m.group(2), m.group(3).strip()))
    return blocks


def _strip_code_blocks(text: str) -> str:
    """Remove code fences from text, leaving only prose."""
    pattern = re.compile(
        r'^ {0,3}`{3,}\w*\s*\n.*?^ {0,3}`{3,}\s*$',
        re.MULTILINE | re.DOTALL,
    )
    return pattern.sub('', text).strip()


def _strip_details_blocks(text: str) -> str:
    """Remove <details>...</details> blocks from text."""
    return re.sub(r'<details\b[^>]*>.*?</details>', '', text, flags=re.DOTALL).strip()


def _split_mixed_section(section: RawSection) -> dict[str, str]:
    """Split a mixed section (description + code blocks) into separate slots.

    Used when a single heading like "### 🐛 Describe the bug" contains
    description text, reproducer code, and error output.

    Classification order for each code block:
      1. Reproducer markers (import torch, etc.) → SLOT_REPRODUCER
      2. Environment markers (PyTorch version, etc.) → SLOT_ENVIRONMENT
      3. Error/diagnostic markers (BUG CONFIRMED, Traceback, etc.) → SLOT_ERROR_LOG
      4. First remaining block → SLOT_ERROR_LOG (evidence of the bug)
      5. Remaining blocks → SLOT_ADDITIONAL
    """
    result: dict[str, str] = {}

    code_blocks = _extract_code_blocks(section.content)
    # Strip code fences AND <details> blocks — the latter often contain
    # comment lines like "# Observe: ..." that leak into prose after stripping.
    prose = _strip_details_blocks(_strip_code_blocks(section.content))

    if prose:
        result[SLOT_DESCRIPTION] = prose

    for lang, code in code_blocks:
        if any(marker in code for marker in _REPRODUCER_CONTENT_MARKERS):
            if SLOT_REPRODUCER not in result:
                result[SLOT_REPRODUCER] = f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"
            else:
                # Second reproducer variant → additional context
                existing = result.get(SLOT_ADDITIONAL, "")
                block_text = f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"
                result[SLOT_ADDITIONAL] = (existing + "\n\n" + block_text).strip()
        elif any(marker in code for marker in _ENV_CONTENT_MARKERS):
            result[SLOT_ENVIRONMENT] = f"```\n{code}\n```"
        elif any(marker in code for marker in _ERROR_CONTENT_MARKERS):
            if SLOT_ERROR_LOG not in result:
                result[SLOT_ERROR_LOG] = f"```\n{code}\n```"
            else:
                existing = result.get(SLOT_ADDITIONAL, "")
                result[SLOT_ADDITIONAL] = (existing + "\n\n" + f"```\n{code}\n```").strip()
        elif SLOT_ERROR_LOG not in result:
            # First unclassified non-reproducer, non-env code block → error log
            # (typically the short snippet showing the bug inline)
            result[SLOT_ERROR_LOG] = f"```\n{code}\n```"
        else:
            # Additional code blocks go to additional context
            existing = result.get(SLOT_ADDITIONAL, "")
            block_text = f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"
            result[SLOT_ADDITIONAL] = (existing + "\n\n" + block_text).strip()

    return result


def classify_sections(sections: list[RawSection]) -> dict[str, str]:
    """Classify all sections and return {slot_name: content}.

    Rule-based classification first. Unmatched sections go to additional_context.
    Mixed sections (description heading with code blocks) get split.
    """
    slots: dict[str, str] = {}

    def _append_slot(slot: str, content: str) -> None:
        if slot in slots:
            slots[slot] = slots[slot] + "\n\n" + content
        else:
            slots[slot] = content

    for section in sections:
        # Try heading-based classification
        slot = _classify_by_heading(section.heading)

        # If heading says description but content has code blocks, split it
        if slot == SLOT_DESCRIPTION:
            code_blocks = _extract_code_blocks(section.content)
            if code_blocks:
                # Mixed section — split into description + reproducer + error
                parts = _split_mixed_section(section)
                for part_slot, part_content in parts.items():
                    _append_slot(part_slot, part_content)
                continue
            else:
                _append_slot(slot, section.content)
                continue

        if slot == SLOT_HEADER:
            _append_slot(SLOT_ADDITIONAL, section.content)
            continue

        if slot is not None:
            _append_slot(slot, section.content)
            continue

        # Try content-based classification
        slot = _classify_by_content(section.content)
        if slot is not None:
            _append_slot(slot, section.content)
            continue

        # Unmatched — goes to additional context
        label = section.heading + "\n" if section.heading else ""
        _append_slot(SLOT_ADDITIONAL, label + section.content)

    return slots


# ---------------------------------------------------------------------------
# Utility: code fence detection
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r'^(`{3,}|~{3,})', re.MULTILINE)


def _code_fence_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of all code-fenced blocks in text."""
    ranges = []
    fences = list(_CODE_FENCE.finditer(text))
    i = 0
    while i < len(fences):
        open_fence = fences[i]
        open_char = open_fence.group(1)[0]
        open_len = len(open_fence.group(1))
        j = i + 1
        while j < len(fences):
            close_fence = fences[j]
            if (close_fence.group(1)[0] == open_char
                    and len(close_fence.group(1)) >= open_len):
                end = text.find('\n', close_fence.end())
                if end == -1:
                    end = len(text)
                ranges.append((open_fence.start(), end))
                i = j + 1
                break
            j += 1
        else:
            ranges.append((open_fence.start(), len(text)))
            break
    return ranges


def _is_inside_fence(pos: int, ranges: list[tuple[int, int]]) -> bool:
    """Check if a position falls inside any code fence range."""
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 3: Extract footer
# ---------------------------------------------------------------------------

def _extract_footer(body: str) -> tuple[str, str]:
    """Extract trailing footer (---\\n*...*) from body.

    Returns (footer_text, body_without_footer).
    Footer is the pattern: ---\\n*italic attribution line*
    """
    m = re.search(r'\n---\n(\*[^\n]+\*)\s*$', body)
    if m:
        footer = m.group(0).strip()
        return footer, body[:m.start()].rstrip()
    return "", body


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _load_label_mapping() -> dict[str, str]:
    """Load label prefix → field mapping from config/agent_config.yml."""
    path = AGENT_DIR / "config" / "agent_config.yml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("label_prefixes", {})
    return {"agent_test": "test_type", "agent_category": "category",
            "agent_dependency": "dependency"}


def _extract_label_info(labels: list[dict]) -> dict[str, str]:
    """Extract test_type, category, dependency from label names."""
    mapping = _load_label_mapping()
    info = dict.fromkeys(mapping.values(), "")
    for label in labels:
        name = label.get("name", "") if isinstance(label, dict) else label
        for prefix, field_name in mapping.items():
            if name.startswith(f"{prefix}:"):
                info[field_name] = name.split(":", 1)[1].strip()
    return info


# ---------------------------------------------------------------------------
# LLM: classify ambiguous sections + extract metadata
# ---------------------------------------------------------------------------

_LLM_METADATA_ONLY_PROMPT = """\
You are extracting metadata from a GitHub issue for the XPU (Intel GPU) pipeline.

## Issue
Title: {title}
Labels: {labels}

## Body Sections
{classified_summary}

Return ONLY valid JSON, no markdown fences:
```
{{
  "metadata": {{
    "test_type": "ut | e2e | ...",
    "category": "category if identifiable",
    "dependency": "upstream | ...",
    "platform": "xpu | BMG | ...",
    "context": "upstream links, version info, brief notes"
  }},
  "issue_type": "bug"
}}
```

Rules:
- issue_type: "bug" for test failures/errors/crashes, "nonbug" for features/tasks/enhancements
- test_type: "ut" for unit tests, "e2e" for end-to-end, "" if unclear
- dependency: "upstream" if fix exists/needed in pytorch/pytorch, "" otherwise
- platform: "xpu" unless specific GPU mentioned (BMG, etc.)
- context: brief one-line summary with upstream links if any
"""


# ---------------------------------------------------------------------------
# Reset and Run
# ---------------------------------------------------------------------------

def reset(issue_number: int) -> None:
    """Reset issue body to original raw content for re-run."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    m = re.search(
        r'## Original Issue\s*\n<details><summary>.*?</summary>\s*\n(.*?)\n\s*</details>',
        body, re.DOTALL,
    )
    if not m:
        log("WARN", f"Issue #{issue_number} has no Original Issue section, cannot reset",
            issue=issue_number)
        return
    raw = m.group(1).strip()
    gh.update_issue_body(ISSUE_REPO, issue_number, raw)
    log("INFO", f"Issue #{issue_number} reset to original body ({len(raw)} chars)",
        issue=issue_number)


def run(issue_number: int) -> None:
    """Format a raw issue: split sections, classify, build from template."""
    # Read issue
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    labels = detail.get("labels", [])

    # Skip if already formatted — must have BOTH a status marker AND the
    # template sections (## Reproducer is the canonical sentinel).  A body
    # that only has a status marker but no template was stamped by the scan
    # script without running format_agent; we must re-format it.
    # Also re-format if the reproducer is a code_fix artifact path (e.g.
    # "agent_space/repro_issueN.py") — these are generated by the fix agent
    # and must not replace the original reproducer from the issue.
    _TEMPLATE_SENTINEL = "## Reproducer"
    _ARTIFACT_REPRODUCER_RE = re.compile(
        r"## Reproducer\s*```[^\n]*\n\s*(python\s+agent_space/|bash\s+agent_space/)",
        re.IGNORECASE,
    )
    _reproducer_is_artifact = bool(_ARTIFACT_REPRODUCER_RE.search(body))
    if get_status(body) is not None and _TEMPLATE_SENTINEL in body and not _reproducer_is_artifact:
        log("INFO", f"Issue #{issue_number} already formatted, skipping discovery",
            issue=issue_number)
        return
    if _reproducer_is_artifact:
        log("WARN",
            f"Issue #{issue_number} has artifact reproducer (agent_space/ path) — re-running format",
            issue=issue_number)
    if get_status(body) is not None and _TEMPLATE_SENTINEL not in body:
        log("INFO",
            f"Issue #{issue_number} has status marker but no template sections "
            f"(missing '{_TEMPLATE_SENTINEL}') — re-running format",
            issue=issue_number)

    # --- Step 1: Extract footer, then split into sections ---
    footer, body_no_footer = _extract_footer(body)
    sections = _split_into_sections(body_no_footer)
    log("INFO",
        f"Split into {len(sections)} section(s), footer={'yes' if footer else 'no'}",
        issue=issue_number)

    # --- Step 2: Classify sections ---
    slots = classify_sections(sections)

    # Log classification results
    classified = [s for s in [SLOT_DESCRIPTION, SLOT_REPRODUCER, SLOT_ERROR_LOG,
                               SLOT_ENVIRONMENT, SLOT_ADDITIONAL] if s in slots]
    log("INFO",
        f"Classified slots: {', '.join(f'{s}={len(slots[s])}ch' for s in classified)}",
        issue=issue_number)

    # --- Step 3: LLM for metadata extraction ---
    label_info = _extract_label_info(labels)
    label_names = [l.get("name", "") if isinstance(l, dict) else l for l in labels]

    classified_summary = "\n".join(
        f"- {slot}: {len(content)} chars"
        + (f" — {content[:100]}..." if len(content) > 100 else f" — {content}")
        for slot, content in slots.items()
    )

    prompt = _LLM_METADATA_ONLY_PROMPT.format(
        title=detail.get("title", ""),
        labels=", ".join(label_names) if label_names else "none",
        classified_summary=classified_summary,
    )

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("DISCOVERED", 300)
    output, log_path, session_id, token_usage = backend.run(
        prompt, skill=Skill.FORMAT,
        issue=issue_number, stage="DISCOVERED", timeout=timeout,
    )
    log("INFO", f"Discovery agent log: {log_path} | {token_usage.summary()}",
        issue=issue_number)

    # Parse LLM response
    try:
        json_str = extract_json(output)
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        log("WARN", f"Failed to parse discovery output as JSON: {e}",
            issue=issue_number)
        data = {"metadata": {}, "issue_type": "bug"}

    issue_type = data.get("issue_type", "bug")
    metadata = data.get("metadata", {})

    # Labels are authoritative over LLM
    for key, value in label_info.items():
        if value:
            metadata[key] = value

    log("INFO",
        f"LLM result: type={issue_type} | "
        f"category={metadata.get('category', 'n/a')} | "
        f"platform={metadata.get('platform', 'n/a')} | "
        f"dependency={metadata.get('dependency', 'n/a')}",
        issue=issue_number)

    # --- Step 4: Build body from template ---
    new_body = build_body(
        ISSUE_TEMPLATE_PATH,
        description=slots.get(SLOT_DESCRIPTION, "_No description extracted_"),
        reproducer=slots.get(SLOT_REPRODUCER, "_No reproducer found_"),
        error_log=slots.get(SLOT_ERROR_LOG, "_No error log found_"),
        environment=slots.get(SLOT_ENVIRONMENT, "_No environment info found_"),
        footer=footer + "\n" if footer else "",
        test_type=metadata.get("test_type", ""),
        category=metadata.get("category", ""),
        dependency=metadata.get("dependency", ""),
        platform=metadata.get("platform", ""),
        root_cause="_Pending triage_",
        fix_strategy="_Pending triage_",
        target_repo="_Pending triage_",
        additional_context=slots.get(SLOT_ADDITIONAL, ""),
        original_issue=body,
    )

    # --- Step 5: Action item + discovery log ---
    new_body = check_action_item(new_body, "Issue formatted")
    log_summary = (
        f"**Type:** {issue_type}\n"
        f"**Sections found:** {', '.join(classified) if classified else 'none'}\n"
        f"**Tokens:** {token_usage.summary()}"
    )
    new_body = append_log(new_body, "discovery", log_summary)

    # --- Step 6: Write back and sync labels ---
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
    sync_labels(ISSUE_REPO, issue_number, Stage.DISCOVERED)
    body_delta = len(new_body) - len(body)
    log("INFO", f"Discovery complete for #{issue_number} "
        f"({len(new_body)} chars, {body_delta:+d} from original)",
        issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a raw issue")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--reset", action="store_true",
                        help="Reset issue to original body before re-running discovery")
    args = parser.parse_args()
    if args.reset:
        reset(args.issue)
    run(args.issue)


if __name__ == "__main__":
    main()
