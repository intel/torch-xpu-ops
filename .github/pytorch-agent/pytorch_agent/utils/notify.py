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
