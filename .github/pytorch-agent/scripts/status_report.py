#!/usr/bin/env python3
"""Print status of all tracked issues (reads from issue body status)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from issue_handler.utils import git as gh
from issue_handler.utils.config import ISSUE_REPO
from issue_handler.utils.body_templates import get_status


def main() -> None:
    # Get all open issues with any agent label
    issues = gh.get_issues(ISSUE_REPO, label="agent:active")
    # Also get needs-human
    issues += gh.get_issues(ISSUE_REPO, label="agent:needs-human")

    if not issues:
        print("No tracked issues found.")
        return

    print(f"{'#':>5}  {'Stage':<15}  Title")
    print("-" * 70)
    for issue in sorted(issues, key=lambda x: x.get("number", 0)):
        body = issue.get("body", "") or ""
        status = get_status(body) or "UNKNOWN"
        title = issue.get("title", "")[:50]
        print(f"{issue['number']:>5}  {status:<15}  {title}")


if __name__ == "__main__":
    main()
