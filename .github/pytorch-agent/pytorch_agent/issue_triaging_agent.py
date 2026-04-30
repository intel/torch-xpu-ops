"""Issue triaging agent — decide if fix belongs in pytorch or torch-xpu-ops.

Entry point:
  python -m pytorch_agent.issue_triaging_agent --issue 123

Decision 1: When triage verdict is "pytorch", writes stage directly to
IMPLEMENTING (skips TRIAGING as a persistent state).
"""
from __future__ import annotations

import argparse
import re

from .utils import github_client as gh
from .utils.config import UPSTREAM_ISSUE_REPO
from .utils.state import (
    find_tracked_by_issue, update_stage, save_state, TrackedIssue,
)
from .utils.agent_backend import get_backend
from .utils.logger import log
from .issue_discovery import process_issue


TRIAGE_PROMPT_TEMPLATE = """Analyze the following GitHub issue and decide where the fix belongs.

## Issue #{number}: {title}

{body}

## Instructions
Respond with EXACTLY one of these two lines:
VERDICT: pytorch — <reason>
VERDICT: skip — <reason>

Choose "pytorch" if the fix requires changes to pytorch/pytorch source code.
Choose "skip" if the fix belongs in intel/torch-xpu-ops or is not actionable.
"""


def triage_issue(issue_number: int) -> tuple[str, str]:
    """Triage an issue. Returns (verdict, reason).

    verdict is "pytorch" or "skip".
    """
    # Ensure tracked
    tracked = find_tracked_by_issue(issue_number)
    if tracked is None:
        tracked = process_issue(issue_number)

    # Idempotency: skip if already past triage
    if tracked.stage not in ("DISCOVERED", "TRIAGING"):
        log("INFO", f"Issue #{issue_number} already at stage {tracked.stage}, skipping triage",
            issue=issue_number)
        return ("pytorch" if tracked.stage != "SKIPPED" else "skip",
                tracked.triage_reason or "already triaged")

    # Get issue details for prompt
    detail = gh.get_issue_detail(UPSTREAM_ISSUE_REPO, issue_number)
    prompt = TRIAGE_PROMPT_TEMPLATE.format(
        number=issue_number,
        title=detail.get("title", ""),
        body=detail.get("body", ""),
    )

    # Dispatch agent — post session ID to issue for live monitoring
    def _post_session_id(sid: str):
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, issue_number,
            f"🔗 **Triage agent session started**\n\n"
            f"**Attach to watch live:**\n"
            f"```bash\ncd ~/pytorch && opencode -s {sid}\n```\n"
            f"Session ID: `{sid}`",
        )

    backend = get_backend()
    output, log_path, session_id = backend.run(
        prompt, skill="pytorch-triage",
        issue=issue_number, stage="TRIAGING",
        on_session_start=_post_session_id,
    )
    log("INFO", f"Triage agent log: {log_path}", issue=issue_number)

    # Parse verdict
    verdict, reason = _parse_verdict(output)
    tracked.triage_reason = reason

    if verdict == "pytorch":
        update_stage(tracked, "IMPLEMENTING",
                     f"Triage verdict: pytorch fix needed. Reason: {reason}")
    else:
        update_stage(tracked, "SKIPPED",
                     f"Triage verdict: skip. Reason: {reason}")

    # Post triage result to source issue
    emoji = "🔧" if verdict == "pytorch" else "⏭️"
    gh.add_issue_comment(
        UPSTREAM_ISSUE_REPO, issue_number,
        f"{emoji} **Triage result:** `{verdict}`\n\n"
        f"**Reason:** {reason}\n\n"
        f"_Agent log: `{log_path.name}`_",
    )

    log("INFO", f"Triage result for #{issue_number}: {verdict} — {reason}",
        issue=issue_number)
    return verdict, reason


def _parse_verdict(output: str) -> tuple[str, str]:
    """Parse VERDICT line from agent output."""
    for line in output.splitlines():
        match = re.match(r"VERDICT:\s*(pytorch|skip)\s*[—–-]\s*(.*)", line, re.IGNORECASE)
        if match:
            return match.group(1).lower(), match.group(2).strip()
    # Fallback: if no clear verdict, default to skip
    return "skip", f"Could not parse verdict from agent output (length={len(output)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage an ai_generated issue")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    verdict, reason = triage_issue(args.issue)
    print(f"Verdict: {verdict} — {reason}")


if __name__ == "__main__":
    main()
