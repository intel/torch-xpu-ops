"""Shared notification helpers for agent sessions."""
from __future__ import annotations

from . import github_client as gh


def post_session_started(
    repo: str, issue: int, stage: str, sid: str, workdir: str = "~/pytorch",
) -> None:
    """Post a standardised session-started comment to the source issue."""
    gh.add_issue_comment(
        repo, issue,
        f"\U0001f517 **{stage} agent session started**\n\n"
        f"**Attach to watch live:**\n"
        f"```bash\ncd {workdir} && opencode -s {sid}\n```\n"
        f"Session ID: `{sid}`",
    )


def post_agent_completed(
    repo: str, issue: int, header: str, log_path: "Path",
    output: str, *, tail: int = 50,
) -> None:
    """Post a standardised agent-completed comment with log tail."""
    lines = output.strip().splitlines()[-tail:] if output.strip() else ["(empty)"]
    gh.add_issue_comment(
        repo, issue,
        f"\U0001f916 **{header}** — log: `{log_path.name}`\n\n"
        f"<details><summary>Agent output (last {tail} lines)</summary>\n\n"
        f"```\n{chr(10).join(lines)}\n```\n"
        f"</details>",
    )
