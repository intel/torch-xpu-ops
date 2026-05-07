"""Discovery agent — read raw issue, format into structured template.

Entry point:
  python -m pytorch_agent.discovery_agent --issue 123
"""
from __future__ import annotations

import argparse
import json

import yaml

from .utils import git as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS, AGENT_DIR
from .utils.body_templates import (
    get_status, render_initial_body, check_action_item, append_log,
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
        original_issue=body,
    )

    # Check action item and append log
    new_body = check_action_item(new_body, "Issue formatted")
    new_body = append_log(new_body, "discovery",
                          f"Extracted from raw issue by discovery agent.\n"
                          f"Log: `{log_path.name}`")

    # Write back (no agent:active label — discovery just formats)
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
    log("INFO", f"Discovery complete for #{issue_number}", issue=issue_number)





def main() -> None:
    parser = argparse.ArgumentParser(description="Format a raw issue")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
