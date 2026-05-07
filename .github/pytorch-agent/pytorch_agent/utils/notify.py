"""Shared notification helpers for agent sessions.

Updates issue body Action Items and logs instead of posting comments.
"""
from __future__ import annotations

from pathlib import Path

from . import git as gh
from .body_templates import append_log


def post_session_started(
    repo: str, issue: int, stage: str, sid: str,
) -> None:
    """Log session start into the issue body's agent log."""
    detail = gh.get_issue_detail(repo, issue)
    body = detail.get("body", "") or ""
    new_body = append_log(
        body, "log",
        f"🔗 **{stage} agent session started** — ID: `{sid}`",
    )
    gh.update_issue_body(repo, issue, new_body)


def post_agent_completed(
    repo: str, issue: int, header: str, log_path: Path,
    output: str, *, tail: int = 50,
) -> None:
    """Log agent completion into the issue body's agent log."""
    lines = output.strip().splitlines()[-tail:] if output.strip() else ["(empty)"]
    detail = gh.get_issue_detail(repo, issue)
    body = detail.get("body", "") or ""
    log_text = (
        f"🤖 **{header}** — log: `{log_path.name}`\n\n"
        f"<details><summary>Agent output (last {tail} lines)</summary>\n\n"
        f"```\n{chr(10).join(lines)}\n```\n"
        f"</details>"
    )
    new_body = append_log(body, "log", log_text)
    gh.update_issue_body(repo, issue, new_body)
