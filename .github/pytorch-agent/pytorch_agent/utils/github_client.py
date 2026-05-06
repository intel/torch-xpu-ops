"""Thin wrappers around the `gh` CLI for all GitHub API operations.

Token routing
-------------
- REVIEW_GH_TOKEN  →  PRIVATE_REVIEW_REPO and PUBLIC_TARGET_REPO operations
- GH_TOKEN         →  everything else (upstream issues, labels, comments)
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _token_for_repo(repo: str) -> str | None:
    """Return the correct GH token for *repo*."""
    raise NotImplementedError


def _gh(args: list[str], input_text: str | None = None,
        token: str | None = None) -> str:
    """Run `gh <args>` and return stdout. Raises CalledProcessError on failure."""
    raise NotImplementedError


def _gh_api(endpoint: str, method: str = "GET",
            token: str | None = None, **fields: Any) -> dict | list:
    """Call `gh api` and return parsed JSON."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

def get_issues(repo: str, label: str) -> list[dict]:
    """List open issues carrying *label*."""
    raise NotImplementedError


def get_issue_detail(repo: str, number: int) -> dict:
    """Return full issue details including body, labels, comments."""
    raise NotImplementedError


def add_issue_comment(repo: str, number: int, body: str) -> None:
    """Post a comment on an issue."""
    raise NotImplementedError


def get_issue_comments(repo: str, number: int) -> list[dict]:
    """Return all comments on an issue."""
    raise NotImplementedError


def update_issue_comment(repo: str, comment_id: int, body: str) -> None:
    """Edit an existing issue comment."""
    raise NotImplementedError


def close_issue(repo: str, number: int) -> None:
    """Close an issue."""
    raise NotImplementedError


def add_label(repo: str, number: int, label: str) -> None:
    """Add a label to an issue, creating it if necessary."""
    raise NotImplementedError


def remove_label(repo: str, number: int, label: str) -> None:
    """Remove a label from an issue."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Pull requests
# ---------------------------------------------------------------------------

def create_pr(repo: str, title: str, body: str, head: str,
              base: str = "main") -> dict:
    """Create a PR inside *repo*."""
    raise NotImplementedError


def create_cross_fork_pr(head_repo: str, head_branch: str,
                         base_repo: str, title: str, body: str) -> dict:
    """Open a PR from *head_repo*:*head_branch* into *base_repo*:main."""
    raise NotImplementedError


def add_pr_comment(repo: str, number: int, body: str) -> dict:
    """Post a comment on a PR and return the created comment dict."""
    raise NotImplementedError


def update_pr_comment(repo: str, comment_id: int, body: str) -> None:
    """Edit an existing PR comment."""
    raise NotImplementedError


def get_pr_status(repo: str, pr_number: int) -> str:
    """Return PR state: "open" | "closed" | "merged"."""
    raise NotImplementedError


def get_ci_checks(repo: str, pr_number: int) -> list[dict]:
    """Return all check-runs for the latest commit of a PR."""
    raise NotImplementedError


def delete_branch(repo: str, branch: str) -> None:
    """Delete *branch* from *repo* (best-effort, ignores 404)."""
    raise NotImplementedError
