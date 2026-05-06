"""Shared helpers for building PR titles, bodies, and parsing issue sections."""
from __future__ import annotations


def parse_issue_sections(body: str) -> dict[str, str]:
    """Parse a torch-xpu-ops issue body into named sections.

    Sections are delimited by Markdown headings (## Section Name).
    Returns a dict mapping section name → section text.
    """
    raise NotImplementedError


def build_pr_body(
    *,
    upstream_issue_repo: str,
    source_number: int,
    title: str,
    triage_reason: str | None,
    issue_body: str,
    reviewer: str = "",
) -> str:
    """Build a descriptive PR body for submission to pytorch/pytorch.

    Includes:
    - Link back to the upstream issue
    - Triage summary
    - Relevant excerpt from the issue body
    - Optional cc @reviewer line
    """
    raise NotImplementedError
