"""Parse GitHub PR review state and /agent slash commands.

Review states
-------------
- "approved"          — at least one approving review since last push
- "changes_requested" — reviewer requested changes (no approval yet)
- "pending"           — no reviews yet
- "paused"            — a /agent pause comment was found

Slash commands (in PR or issue comments, prefixed with /agent)
--------------------------------------------------------------
- /agent pause    →  tracked.paused = True
- /agent resume   →  tracked.paused = False
"""
from __future__ import annotations


def get_review_state(pr_number: int, last_push_sha: str | None) -> str:
    """Return the current review state for *pr_number*.

    Only considers reviews posted *after* the commit at *last_push_sha*
    so that stale approvals / rejections are ignored after a new push.
    """
    raise NotImplementedError


def get_pending_reviews(pr_number: int, last_push_sha: str | None) -> list[dict]:
    """Return review objects with change requests since *last_push_sha*."""
    raise NotImplementedError


def format_reviews_for_prompt(reviews: list[dict]) -> str:
    """Format *reviews* into a clean block suitable for an agent prompt."""
    raise NotImplementedError
