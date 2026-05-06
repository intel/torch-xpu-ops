"""Discovery agent — read raw issue, format into structured template.

Entry point:
  python -m pytorch_agent.discovery_agent --issue 123
"""
from __future__ import annotations

import argparse
import json

from .utils import github_client as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS
from .utils.issue_body import (
    get_status, render_initial_body, check_action_item, append_log,
)
from .utils.agent_backend import get_backend
from .utils.logger import log


DISCOVERY_PROMPT_TEMPLATE = """Analyze this raw GitHub issue and extract structured information.

## Issue #{number}: {title}

### Labels
{labels}

### Body
{body}

## Instructions
Extract the following fields from the issue. Use the labels, body text,
error logs, and any other information available. If a field is not present,
use an empty string.

Respond with ONLY valid JSON (no markdown fences):
{{
  "summary": "one-line description of the failure",
  "test_type": "ut or e2e (from agent_test labels)",
  "category": "category from agent_category labels (e.g. Torch Ops, Inductor, Distributed)",
  "dependency": "dependency from agent_dependency labels (e.g. upstream-pytorch, oneDNN)",
  "platform": "platform if mentioned (e.g. PVC, ATS-M, DG2)",
  "failed_tests": "list of failing test cases, one per line, backtick-wrapped",
  "error_log": "relevant error output (last ~50 lines)",
  "reproducer": "commands to reproduce the failure",
  "commit_scope": "commit range if mentioned",
  "context": "any additional context useful for triage"
}}
"""


def _extract_label_info(labels: list[dict]) -> dict[str, str]:
    """Extract test_type, category, dependency from label names."""
    info = {"test_type": "", "category": "", "dependency": ""}
    for label in labels:
        name = label.get("name", "") if isinstance(label, dict) else label
        if name.startswith("agent_test:"):
            info["test_type"] = name.split(":", 1)[1].strip()
        elif name.startswith("agent_category:"):
            info["category"] = name.split(":", 1)[1].strip()
        elif name.startswith("agent_dependency:"):
            info["dependency"] = name.split(":", 1)[1].strip()
    return info


def run(issue_number: int) -> None:
    """Format a raw issue into the structured template."""
    # Read issue
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    labels = detail.get("labels", [])

    # Skip if already formatted
    if get_status(body) is not None:
        log("INFO", f"Issue #{issue_number} already formatted, skipping discovery",
            issue=issue_number)
        return

    # Extract label info for fallback
    label_info = _extract_label_info(labels)
    label_names = [l.get("name", "") if isinstance(l, dict) else l
                   for l in labels]

    # Build prompt
    prompt = DISCOVERY_PROMPT_TEMPLATE.format(
        number=issue_number,
        title=detail.get("title", ""),
        labels="\n".join(f"- {l}" for l in label_names),
        body=body[:5000],  # Truncate very long bodies
    )

    # Call LLM
    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("DISCOVERY", 300)
    output, log_path, session_id = backend.run(
        prompt, skill="pytorch-issue-discovery",
        issue=issue_number, stage="DISCOVERY",
    )
    log("INFO", f"Discovery agent log: {log_path}", issue=issue_number)

    # Parse JSON from LLM output
    try:
        # Find JSON in output (may have surrounding text)
        json_match = _extract_json(output)
        data = json.loads(json_match)
    except (json.JSONDecodeError, ValueError) as e:
        log("WARN", f"Failed to parse discovery output as JSON: {e}",
            issue=issue_number)
        # Fallback: use label info + raw body
        data = {
            "summary": detail.get("title", ""),
            "failed_tests": "",
            "error_log": "",
            "reproducer": "",
            "commit_scope": "",
            "context": body[:2000],
            **label_info,
        }

    # Merge label info (labels are authoritative over LLM extraction)
    for key in ("test_type", "category", "dependency"):
        if label_info.get(key):
            data[key] = label_info[key]

    # Render formatted body
    new_body = render_initial_body(
        stage="TRIAGING",
        summary=data.get("summary", detail.get("title", "")),
        test_type=data.get("test_type", ""),
        category=data.get("category", ""),
        dependency=data.get("dependency", ""),
        platform=data.get("platform", ""),
        failed_tests=data.get("failed_tests", ""),
        error_log=data.get("error_log", ""),
        reproducer=data.get("reproducer", ""),
        commit_scope=data.get("commit_scope", ""),
        context=data.get("context", ""),
    )

    # Check the action item and append log
    new_body = check_action_item(new_body, "Issue formatted")
    new_body = append_log(new_body, "discovery",
                          f"Extracted from raw issue by discovery agent.\n"
                          f"Log: `{log_path.name}`")

    # Write back to issue
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
    gh.add_label(ISSUE_REPO, issue_number, "agent:active")

    log("INFO", f"Discovery complete for #{issue_number}", issue=issue_number)


def _extract_json(text: str) -> str:
    """Extract the first JSON object from text."""
    # Try to find JSON between braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    raise ValueError("No JSON object found in text")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a raw issue")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
