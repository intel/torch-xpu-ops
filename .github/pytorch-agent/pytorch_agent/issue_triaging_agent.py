"""Triage stage: AI decides whether to accept or skip an issue.

Accepts
-------
- CI failures with a clear reproduction path
- Failing XPU op tests with stack trace

Skips
-----
- Feature requests, docs, questions
- Issues already fixed or duplicate
- Insufficient reproduction info

On accept  →  stage becomes IMPLEMENTING, triage_reason recorded
On skip    →  stage becomes SKIPPED, reason posted to issue
"""
from __future__ import annotations

from .utils.state import TrackedIssue


TRIAGE_PROMPT_TEMPLATE = """You are triaging a PyTorch XPU CI issue. Decide: ACCEPT or SKIP.

## Issue #{number}: {title}

{body}

## Instructions
Reply with exactly one of:
  ACCEPT: <one-line reason why this is actionable>
  SKIP: <one-line reason why this should not be auto-fixed>
"""


def triage_issue(issue_number: int) -> TrackedIssue:
    """Run triage for *issue_number*. Returns the updated TrackedIssue.

    Posts a session-started comment, dispatches the agent, parses
    ACCEPT/SKIP verdict, and transitions stage accordingly.
    """
    raise NotImplementedError
