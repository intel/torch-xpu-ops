"""Triage agent — analyze formatted issue, determine root cause and fix strategy.

Entry point:
  python -m issue_handler.triage_agent --issue 123
"""
from __future__ import annotations

import argparse
import json

from .utils import git as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS
from .utils.body_templates import (
    get_status, update_section, set_status,
    check_action_item, append_log,
)
from .utils.agent_backend import get_backend
from .utils.json_utils import extract_json
from .utils.logger import log


TRIAGE_SKILL = "issue-triage"





def run(issue_number: int) -> tuple[str, str]:
    """Triage an issue. Returns (verdict, reason)."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    # Check status — triage accepts DISCOVERED or TRIAGING
    status = get_status(body)
    if status is not None and status not in ("DISCOVERED", "TRIAGING"):
        log("INFO", f"Issue #{issue_number} at stage {status}, skipping triage",
            issue=issue_number)
        return ("skip", f"already at {status}")

    # Select skill and call LLM (no inline prompt)
    prompt = (
        f"Read the {TRIAGE_SKILL} skill and triage issue #{issue_number}.\n\n"
        f"## Issue #{issue_number}: {detail.get('title', '')}\n\n"
        f"{body[:8000]}"
    )

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("TRIAGING", 300)
    output, log_path, _ = backend.run(
        prompt, skill=TRIAGE_SKILL,
        issue=issue_number, stage="TRIAGING",
        timeout=timeout,
    )
    log("INFO", f"Triage agent log: {log_path}", issue=issue_number)

    # Parse result
    try:
        json_str = extract_json(output)
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        log("WARN", f"Failed to parse triage output: {e}", issue=issue_number)
        data = {
            "root_cause": f"Could not parse triage output (length={len(output)})",
            "fix_strategy": "",
            "verdict": "NEEDS_HUMAN",
            "reason": "Triage output parsing failed",
        }

    verdict = data.get("verdict", "NEEDS_HUMAN").upper()
    reason = data.get("reason", "")
    root_cause = data.get("root_cause", "")
    fix_strategy = data.get("fix_strategy", "")

    # Update issue body
    new_body = body
    new_body = update_section(new_body, "Root Cause Analysis", root_cause)
    new_body = update_section(new_body, "Proposed Fix Strategy", fix_strategy)
    new_body = check_action_item(new_body, "Root cause identified")

    if verdict == "IMPLEMENTING":
        new_body = set_status(new_body, "IMPLEMENTING")
    else:
        new_body = set_status(new_body, "NEEDS_HUMAN")

    new_body = append_log(
        new_body, "triage",
        f"**Verdict:** {verdict}\n**Reason:** {reason}\n\n"
        f"**Root Cause:** {root_cause}\n\n"
        f"**Fix Strategy:** {fix_strategy}\n\n"
        f"Log: `{log_path.name}`",
    )

    # Write back
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"Triage result for #{issue_number}: {verdict} — {reason}",
        issue=issue_number)
    return verdict, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a formatted issue")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    verdict, reason = run(args.issue)
    log("INFO", f"Verdict: {verdict} — {reason}", issue=args.issue)


if __name__ == "__main__":
    main()
