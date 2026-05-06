"""Discover new issues labeled `agent:new` and register them for tracking.

Flow
----
1. Query intel/torch-xpu-ops for issues with label "agent:new".
2. Skip any issue already tracked (state comment present).
3. Call process_issue() to create the initial TrackedIssue and state comment.
"""
from __future__ import annotations

from .utils.state import TrackedIssue


def discover_new_issues() -> list[dict]:
    """Return open issues with label "agent:new" that are not yet tracked."""
    raise NotImplementedError


def process_issue(issue_number: int) -> TrackedIssue:
    """Register *issue_number*: create TrackedIssue + state comment if not present.

    Idempotent — safe to call multiple times for the same issue.
    """
    raise NotImplementedError
