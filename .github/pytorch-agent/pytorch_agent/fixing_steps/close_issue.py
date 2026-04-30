"""Close the source issue after public PR is merged.

Entry point:
  python -m pytorch_agent.fixing_steps.close_issue --issue 123
"""
from __future__ import annotations

import argparse

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PUBLIC_TARGET_REPO, PRIVATE_REVIEW_REPO,
)
from ..utils.state import TrackedIssue, update_stage, load_tracked
from ..utils.logger import log


def run(tracked: TrackedIssue) -> None:
    """Close source issue and clean up."""
    if tracked.stage != "DONE":
        return

    if not tracked.public_pr_number:
        log("WARN", f"No public PR for issue #{tracked.source_number}",
            issue=tracked.source_number)
        return

    # Verify public PR is merged
    status = gh.get_pr_status(PUBLIC_TARGET_REPO, tracked.public_pr_number)
    if status != "merged":
        log("INFO", f"Public PR not yet merged for #{tracked.source_number} (status={status})",
            issue=tracked.source_number)
        return

    # Comment on source issue
    comment = (
        f"✅ Fixed in {PUBLIC_TARGET_REPO}#{tracked.public_pr_number} "
        f"— {tracked.public_pr_url}"
    )
    gh.add_issue_comment(UPSTREAM_ISSUE_REPO, tracked.source_number, comment)

    # Close source issue
    gh.close_issue(UPSTREAM_ISSUE_REPO, tracked.source_number)

    # Delete agent branch from review fork
    branch = tracked.branch or f"agent/issue-{tracked.source_number}"
    gh.delete_branch(PRIVATE_REVIEW_REPO, branch)

    log("INFO", f"Issue #{tracked.source_number} closed. "
                 f"Public PR: {tracked.public_pr_url}",
        issue=tracked.source_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
