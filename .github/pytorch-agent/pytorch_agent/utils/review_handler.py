"""Review handler — parse and format PR reviews for LLM prompts."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from . import github_client as gh
from .config import PRIVATE_REVIEW_REPO


def _resolve_push_dt(last_push_sha: str | None) -> Optional[datetime]:
    """Get the timestamp of a commit SHA, or None if unavailable."""
    if not last_push_sha:
        return None
    try:
        commit = gh._gh_api(
            f"/repos/{PRIVATE_REVIEW_REPO}/commits/{last_push_sha}"
        )
        push_ts = commit.get("commit", {}).get("committer", {}).get("date", "")
        if push_ts:
            return datetime.fromisoformat(push_ts.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def _is_after(timestamp: str, push_dt: Optional[datetime]) -> bool:
    """Return True if timestamp is after push_dt (or if either is unknown)."""
    if not push_dt or not timestamp:
        return True
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")) > push_dt
    except Exception:
        return True


def get_pending_reviews(tracking_pr_number: int,
                        last_push_sha: str | None = None) -> list[dict]:
    """Get unaddressed review feedback on the tracking PR.

    Collects from three sources:
    1. PR reviews (CHANGES_REQUESTED, COMMENTED with body)
    2. PR review comments (inline code comments)
    3. Issue comments on the PR (general comments, owner workaround)

    If last_push_sha is provided, filters to feedback submitted after
    the push that produced that SHA (by comparing timestamps).
    """
    push_dt = _resolve_push_dt(last_push_sha)

    feedback: list[dict] = []

    # 1. PR reviews
    reviews = gh.get_pr_reviews(PRIVATE_REVIEW_REPO, tracking_pr_number)
    for r in reviews:
        state = r.get("state", "")
        body = (r.get("body") or "").strip()
        if state == "CHANGES_REQUESTED" or (state == "COMMENTED" and body):
            if _is_after(r.get("submitted_at", ""), push_dt):
                feedback.append(r)

    # 2. PR review comments (inline)
    try:
        review_comments = gh.get_pr_review_comments(PRIVATE_REVIEW_REPO, tracking_pr_number)
        for c in review_comments:
            if _is_after(c.get("created_at", ""), push_dt):
                feedback.append({
                    "user": c.get("user", {}),
                    "state": "INLINE_COMMENT",
                    "body": f"File: `{c.get('path', '?')}` line {c.get('line', '?')}\n{c.get('body', '')}",
                    "submitted_at": c.get("created_at", ""),
                })
    except Exception:
        pass

    # 3. Issue comments on the PR — only /agent prefixed commands
    try:
        pr_comments = gh._gh_api(
            f"/repos/{PRIVATE_REVIEW_REPO}/issues/{tracking_pr_number}/comments"
        )
        for c in pr_comments:
            login = c.get("user", {}).get("login", "")
            body = (c.get("body") or "").strip()
            if (login.endswith("[bot]") or "AGENT_STATE:" in body
                    or "🤖" in body or not body):
                continue
            # Only process comments starting with /agent
            if not body.lower().startswith("/agent"):
                continue
            # Skip approval keywords — these aren't feedback to address
            body_lower = body.lower()
            if any(kw in body_lower for kw in
                   ("lgtm", "approved", "looks good", "ship it")):
                continue
            # Strip the /agent prefix for the task body
            agent_body = body[len("/agent"):].strip()
            if not agent_body or agent_body.lower() == "pause":
                continue
            if _is_after(c.get("created_at", ""), push_dt):
                feedback.append({
                    "user": c.get("user", {}),
                    "state": "COMMENT",
                    "body": agent_body,
                    "submitted_at": c.get("created_at", ""),
                })
    except Exception:
        pass

    return feedback


def format_reviews_for_prompt(reviews: list[dict]) -> str:
    """Format review comments as structured text for LLM prompt."""
    if not reviews:
        return "No pending reviews."

    parts = []
    for i, review in enumerate(reviews, 1):
        user = review.get("user", {}).get("login", "unknown")
        state = review.get("state", "unknown")
        body = review.get("body", "").strip()
        parts.append(f"### Review {i} by @{user} ({state})\n{body}")

    return "\n\n".join(parts)


def get_review_state(tracking_pr_number: int,
                     last_push_sha: str | None = None) -> str:
    """Return 'approved', 'changes_requested', or 'pending'.

    Three feedback sources (since repo owner can't "request changes" on own PR):
    1. PR reviews with APPROVED/CHANGES_REQUESTED state
    2. PR review comments (inline code comments)
    3. Issue comments on the PR (general comments)

    Any unaddressed comment/review after the last agent push = 'changes_requested'.
    Explicit APPROVED review = 'approved'.
    No feedback = 'pending'.
    """
    push_dt = _resolve_push_dt(last_push_sha)

    reviews = gh.get_pr_reviews(PRIVATE_REVIEW_REPO, tracking_pr_number)

    # --- /agent commands take highest priority ---
    # Check issue comments for /agent prefixed commands FIRST.
    # These are explicit user signals that override all other review state.
    try:
        pr_comments = gh._gh_api(
            f"/repos/{PRIVATE_REVIEW_REPO}/issues/{tracking_pr_number}/comments"
        )
        human_comments = [
            c for c in pr_comments
            if not (c.get("user", {}).get("login", "").endswith("[bot]"))
            and "AGENT_STATE:" not in c.get("body", "")
            and "🤖" not in c.get("body", "")
        ]
        new_comments = [
            c for c in human_comments
            if _is_after(c.get("created_at", ""), push_dt)
        ]
        agent_commands = [
            c for c in new_comments
            if (c.get("body") or "").strip().lower().startswith("/agent")
        ]
        if agent_commands:
            latest = agent_commands[-1]
            body = (latest.get("body") or "").strip()
            cmd = body[len("/agent"):].strip().lower()
            if cmd == "pause":
                return "paused"
            if cmd in ("approve", "lgtm", "approved", "looks good", "ship it"):
                return "approved"
            if cmd:
                return "changes_requested"
    except Exception:
        pass

    # --- GitHub review UI (APPROVED / CHANGES_REQUESTED) ---
    # Get latest review per user
    latest: dict[str, str] = {}
    for review in reviews:
        user = review.get("user", {}).get("login", "")
        state = review.get("state", "")
        if state in ("APPROVED", "CHANGES_REQUESTED"):
            latest[user] = state

    # Check for explicit states first
    if latest:
        states = set(latest.values())
        if "CHANGES_REQUESTED" in states:
            # Only count if there's new feedback after last push
            if not push_dt:
                return "changes_requested"
            # Check if any CHANGES_REQUESTED review is after push
            for review in reviews:
                if (review.get("state") == "CHANGES_REQUESTED"
                        and _is_after(review.get("submitted_at", ""), push_dt)):
                    return "changes_requested"
        if all(s == "APPROVED" for s in states):
            return "approved"

    # Check for COMMENTED reviews with non-empty body (owner workaround)
    comment_reviews = [
        r for r in reviews
        if r.get("state") == "COMMENTED" and (r.get("body") or "").strip()
        and _is_after(r.get("submitted_at", ""), push_dt)
    ]
    if comment_reviews:
        return "changes_requested"

    # Check for inline PR review comments after last push
    try:
        review_comments = gh.get_pr_review_comments(
            PRIVATE_REVIEW_REPO, tracking_pr_number)
        for c in review_comments:
            if _is_after(c.get("created_at", ""), push_dt):
                return "changes_requested"
    except Exception:
        pass

    return "pending"
