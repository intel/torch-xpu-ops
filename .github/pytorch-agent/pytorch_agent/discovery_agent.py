"""Discovery agent — read raw issue, format into structured template.

Entry point:
  python -m pytorch_agent.discovery_agent --issue 123
"""
from __future__ import annotations

import argparse
import json
import re

import yaml

from .utils import git as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS, AGENT_DIR
from .utils.body_templates import (
    get_status, render_initial_body, check_action_item, append_log, sync_labels,
)
from .utils.agent_backend import get_backend
from .utils.json_utils import extract_json
from .utils.logger import log


def _load_label_mapping() -> dict[str, str]:
    """Load label prefix → field mapping from config/agent_config.yml."""
    path = AGENT_DIR / "config" / "agent_config.yml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("label_prefixes", {})
    # Fallback hardcoded
    return {"agent_test": "test_type", "agent_category": "category",
            "agent_dependency": "dependency"}


def _extract_label_info(labels: list[dict]) -> dict[str, str]:
    """Extract test_type, category, dependency from label names."""
    mapping = _load_label_mapping()
    info = {field: "" for field in mapping.values()}
    for label in labels:
        name = label.get("name", "") if isinstance(label, dict) else label
        for prefix, field in mapping.items():
            if name.startswith(f"{prefix}:"):
                info[field] = name.split(":", 1)[1].strip()
    return info


def reset(issue_number: int) -> None:
    """Reset issue body to original raw content for re-run."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    m = re.search(
        r'<details><summary>Original issue body</summary>\s*\n(.*?)\n\s*</details>',
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


def _extract_environment(body: str) -> str:
    """Extract collect_env / Versions section from raw issue body."""
    # Try <details> block containing collect_env
    m = re.search(
        r'<details>\s*\n\s*<summary>.*?collect_env.*?</summary>\s*\n(.*?)</details>',
        body, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Try ### Versions section
    m = re.search(r'### Versions\s*\n(.*?)(?=\n### |\n## |\Z)', body, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


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

    # Extract label info
    label_info = _extract_label_info(labels)
    label_names = [l.get("name", "") if isinstance(l, dict) else l for l in labels]

    # Call LLM with skill (no inline prompt)
    prompt = (
        f"Read the pytorch-issue-discovery skill and extract structured info "
        f"from issue #{issue_number}.\n\n"
        f"## Issue #{issue_number}: {detail.get('title', '')}\n\n"
        f"### Labels\n" + "\n".join(f"- {l}" for l in label_names) + "\n\n"
        f"### Body\n{body[:5000]}"
    )

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("DISCOVERY", 300)
    output, log_path, session_id = backend.run(
        prompt, skill="pytorch-issue-discovery",
        issue=issue_number, stage="DISCOVERY", timeout=timeout,
    )
    log("INFO", f"Discovery agent log: {log_path}", issue=issue_number)

    # Parse JSON from LLM output
    try:
        json_match = extract_json(output)
        data = json.loads(json_match)
    except (json.JSONDecodeError, ValueError) as e:
        log("WARN", f"Failed to parse discovery output as JSON: {e}",
            issue=issue_number)
        data = {
            "summary": detail.get("title", ""),
            "failed_tests": "",
            "error_log": "",
            "reproducer": "",
            "commit_scope": "",
            "context": body[:2000],
            **label_info,
        }

    # Labels are authoritative over LLM extraction
    for key, value in label_info.items():
        if value:
            data[key] = value

    # Coerce list fields to strings
    ft = data.get("failed_tests", "")
    if isinstance(ft, list):
        data["failed_tests"] = "\n".join(ft)

    # Render formatted body
    new_body = render_initial_body(
        stage="DISCOVERED",
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
        environment=_extract_environment(body),
        original_issue=body,
    )

    # Check action item and append log with extraction summary
    new_body = check_action_item(new_body, "Issue formatted")
    log_summary = (
        f"**Summary:** {data.get('summary', 'N/A')}\n"
        f"**Failed tests:** {data.get('failed_tests', 'N/A')}\n"
        f"**Dependency:** {data.get('dependency', 'N/A')}\n"
        f"**Commit scope:** {data.get('commit_scope', 'N/A')}"
    )
    new_body = append_log(new_body, "discovery", log_summary)

    # Write back and sync labels
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
    sync_labels(ISSUE_REPO, issue_number, "DISCOVERED")
    log("INFO", f"Discovery complete for #{issue_number}", issue=issue_number)





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
