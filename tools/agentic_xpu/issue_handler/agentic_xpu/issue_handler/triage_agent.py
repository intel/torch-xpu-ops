# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Triage agent — analyze formatted issue, determine root cause and fix strategy.

Entry point:
  python -m issue_handler.triage_agent --issue 123
"""
from __future__ import annotations

import argparse
import json
import re

from .utils import git as gh
from .utils.config import ISSUE_REPO, STAGE_TIMEOUTS
from .utils.body_templates import (
    get_status, update_section, set_status,
    check_action_item, append_log, set_metadata,
)
from .utils.agent_backend import get_backend
from .utils.json_utils import extract_json
from .utils.logger import log
from .utils.stages import Skill, Stage


def _parse_markdown_triage(text: str) -> dict | None:
    """Fallback parser for when triage agent returns markdown instead of JSON."""
    # Look for **Verdict:** `VALUE` or **Verdict:** VALUE
    verdict_m = re.search(r'\*\*Verdict:\*\*\s*`?(\w+)`?', text)
    if not verdict_m:
        return None
    verdict = verdict_m.group(1).upper()
    # Extract other fields
    root_m = re.search(r'\*\*Root Cause:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)', text, re.DOTALL)
    fix_m = re.search(r'\*\*Fix Strategy:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)', text, re.DOTALL)
    reason_m = re.search(r'\*\*Reason:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)', text, re.DOTALL)
    return {
        "verdict": verdict,
        "root_cause": root_m.group(1).strip() if root_m else "",
        "fix_strategy": fix_m.group(1).strip() if fix_m else "",
        "reason": reason_m.group(1).strip() if reason_m else "",
    }

TRIAGE_SKILL = Skill.TRIAGE





def run(issue_number: int) -> tuple[str, str]:
    """Triage an issue. Returns (verdict, reason)."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    # Check status — triage accepts DISCOVERED or TRIAGING
    status = get_status(body)
    if status is not None and status not in (Stage.DISCOVERED, Stage.TRIAGING):
        log("INFO", f"Issue #{issue_number} at stage {status}, skipping triage",
            issue=issue_number)
        return ("skip", f"already at {status}")

    # Strip the Environment section (collect_env) — it's huge and not useful for triage
    triage_body = re.sub(
        r'## Environment\s*\n<details>.*?</details>',
        '## Environment\n_(stripped for triage prompt)_',
        body, flags=re.DOTALL,
    )

    # Select skill and call LLM (no inline prompt)
    prompt = (
        f"Read the {TRIAGE_SKILL} skill and triage issue #{issue_number}.\n\n"
        f"## Issue #{issue_number}: {detail.get('title', '')}\n\n"
        f"{triage_body[:4000]}"
    )

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("TRIAGING", 300)
    output, log_path, session_id, token_usage = backend.run(
        prompt, skill=TRIAGE_SKILL,
        issue=issue_number, stage="TRIAGING",
        timeout=timeout,
    )
    log("INFO", f"Triage agent log: {log_path} (session: {session_id}) | {token_usage.summary()}",
        issue=issue_number)

    # Parse result
    try:
        json_str = extract_json(output)
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: try to parse structured markdown output
        data = _parse_markdown_triage(output)
        if not data:
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
    target_repo = data.get("target_repo", "").lower().strip()
    # Infer target_repo from fix_strategy if not explicitly provided
    if not target_repo:
        fs_lower = fix_strategy.lower()
        if "src/aten/native/xpu" in fs_lower or "torch-xpu-ops" in fs_lower:
            target_repo = "torch-xpu-ops"
        else:
            target_repo = "pytorch"
    # Update issue body
    new_body = body
    new_body = update_section(new_body, "Root Cause Analysis", root_cause)
    new_body = update_section(new_body, "Proposed Fix Strategy", fix_strategy)
    new_body = set_metadata(new_body, "target_repo", target_repo)
    new_body = check_action_item(new_body, "Root cause identified")

    if verdict == "IMPLEMENTING":
        # Triage complete — mark TRIAGED so a human or downstream agent can pick it up
        new_body = set_status(new_body, Stage.TRIAGED)
        new_body = check_action_item(new_body, "Triage complete")
    else:
        new_body = set_status(new_body, Stage.NEEDS_HUMAN)
        # Add visible reason why the agent can't handle this
        assessment = (
            f"**Verdict:** {verdict}\n\n"
            f"**Why this needs human intervention:**\n{reason}\n\n"
            f"**Root Cause:** {root_cause}\n\n"
            f"**Proposed Fix Strategy:** {fix_strategy}"
        )
        new_body = update_section(new_body, "Agent Assessment", assessment)

    new_body = update_section(new_body, "Target Repository", target_repo)
    new_body = append_log(
        new_body, "triage",
        f"**Verdict:** {verdict}\n**Reason:** {reason}\n\n"
        f"**Target Repository:** `{target_repo}`\n\n"
        f"**Root Cause:** {root_cause}\n\n"
        f"**Fix Strategy:** {fix_strategy}\n\n"
        f"**Tokens:** {token_usage.summary()}\n"
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
