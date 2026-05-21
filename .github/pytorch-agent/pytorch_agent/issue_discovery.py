"""Issue discovery — find new ai_generated issues and start tracking.

Entry points:
  python -m pytorch_agent.issue_discovery --poll       # discover all new
  python -m pytorch_agent.issue_discovery --issue 123  # process one
"""
from __future__ import annotations

import argparse

from .utils import github_client as gh
from .utils.config import UPSTREAM_ISSUE_REPO, ISSUE_LABEL, ALL_AGENT_LABELS
from .utils.state import TrackedIssue, save_state, find_tracked_by_issue
from .utils.logger import log


def discover_new_issues() -> list[dict]:
    """Poll UPSTREAM_ISSUE_REPO for ai_generated issues not yet tracked.

    Explicitly checks for agent:tracking label to skip already-tracked issues.
    """
    all_issues = gh.get_issues(UPSTREAM_ISSUE_REPO, ISSUE_LABEL)
    new_issues = []
    for issue in all_issues:
        labels = [l.get("name", "") if isinstance(l, dict) else l
                  for l in issue.get("labels", [])]
        if not any(l in ALL_AGENT_LABELS for l in labels):
            new_issues.append(issue)
    log("INFO", f"Discovered {len(new_issues)} new issues out of {len(all_issues)} total")
    return new_issues


def process_issue(issue_number: int) -> TrackedIssue:
    """Start tracking a single issue. Returns TrackedIssue."""
    # Check if already tracked
    existing = find_tracked_by_issue(issue_number)
    if existing:
        log("INFO", f"Issue #{issue_number} already tracked at stage {existing.stage}",
            issue=issue_number)
        return existing

    # Get issue details
    detail = gh.get_issue_detail(UPSTREAM_ISSUE_REPO, issue_number)
    tracked = TrackedIssue(
        source_repo=UPSTREAM_ISSUE_REPO,
        source_number=issue_number,
        title=detail.get("title", f"Issue #{issue_number}"),
        stage="DISCOVERED",
        branch=f"agent/issue-{issue_number}",
    )
    save_state(tracked)
    log("INFO", f"Now tracking issue #{issue_number}: {tracked.title}",
        issue=issue_number)
    return tracked


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover ai_generated issues")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--poll", action="store_true", help="Discover all new issues")
    group.add_argument("--issue", type=int, help="Process a specific issue")
    args = parser.parse_args()

    if args.poll:
        new_issues = discover_new_issues()
        for issue in new_issues:
            process_issue(issue["number"])
    else:
        process_issue(args.issue)


if __name__ == "__main__":
    main()
