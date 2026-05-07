"""Close the source issue after public PR is merged.

Entry point:
  python -m pytorch_agent.fixing_steps.close_issue --issue 123
"""
from __future__ import annotations

import argparse

from ..utils import git as gh
from ..utils.config import (
    ISSUE_REPO, PUBLIC_TARGET_REPO, PRIVATE_REVIEW_REPO,
)
from ..utils.body_templates import get_status, set_status, append_log, get_metadata, check_action_item
from ..utils.logger import log


def run(issue_number: int) -> None:
    """Close source issue and clean up."""
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    if get_status(body) != "MERGED":
        return

    public_pr_str = get_metadata(body, "public_pr")
    if not public_pr_str:
        log("WARN", f"No public PR in issue #{issue_number}", issue=issue_number)
        return
    public_pr = int(public_pr_str)

    # Verify merged
    status = gh.get_pr_status(PUBLIC_TARGET_REPO, public_pr)
    if status != "merged":
        log("INFO", f"Public PR not yet merged for #{issue_number} (status={status})",
            issue=issue_number)
        return

    # Update issue body
    new_body = set_status(body, "DONE")
    new_body = check_action_item(new_body, "PR merged")
    public_pr_url = f"https://github.com/{PUBLIC_TARGET_REPO}/pull/{public_pr}"
    new_body = append_log(new_body, "close",
                          f"Fixed in {PUBLIC_TARGET_REPO}#{public_pr} — {public_pr_url}")
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    # Close issue
    gh.close_issue(ISSUE_REPO, issue_number)

    # Delete agent branch from review fork
    branch = f"agent/issue-{issue_number}"
    gh.delete_branch(PRIVATE_REVIEW_REPO, branch)

    log("INFO", f"Issue #{issue_number} closed. Public PR: {public_pr_url}",
        issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
