"""Submit fix as public PR to pytorch/pytorch.

Entry point:
  python -m pytorch_agent.fixing_steps.public_submit --issue 123
"""
from __future__ import annotations

import argparse
import os
from subprocess import CalledProcessError

from ..utils import git as gh
from ..utils.git import build_pr_body
from ..utils.config import (
    ISSUE_REPO, UPSTREAM_ISSUE_REPO, PRIVATE_REVIEW_REPO,
    PUBLIC_TARGET_REPO,
)
from ..utils.issue_body import (
    get_status, set_status, parse_sections, append_log,
)
from ..utils.logger import log


def run(issue_number: int) -> None:
    """Create cross-fork PR from private review fork to pytorch/pytorch."""
    # Read issue
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    if get_status(body) != "PUBLIC_PR":
        return

    branch = f"agent/issue-{issue_number}"
    title = detail.get("title", "")
    if title.startswith("[Agent]"):
        title = title[len("[Agent]"):].strip()

    # Build PR body from issue details
    pr_body = build_pr_body(
        upstream_issue_repo=UPSTREAM_ISSUE_REPO,
        source_number=issue_number,
        title=title,
        triage_reason=parse_sections(body).get("Root Cause Analysis", ""),
        issue_body=body,
        reviewer=os.environ.get("PUBLIC_PR_REVIEWER", ""),
    )

    # Idempotent: check if PR already exists
    head_label = f"{PRIVATE_REVIEW_REPO.split('/')[0]}:{branch}"
    try:
        existing = gh.list_pulls(PUBLIC_TARGET_REPO, head=head_label, state="open")
        if existing:
            pr = existing[0]
            log("INFO", f"Public PR already exists: #{pr.get('number')}",
                issue=issue_number)
        else:
            pr = gh.create_cross_fork_pr(
                head_repo=PRIVATE_REVIEW_REPO,
                head_branch=branch,
                base_repo=PUBLIC_TARGET_REPO,
                title=title,
                body=pr_body,
            )
    except CalledProcessError as exc:
        log("WARN", f"PR creation failed, checking for existing: {exc}",
            issue=issue_number)
        existing = gh.list_pulls(PUBLIC_TARGET_REPO, head=head_label, state="open")
        if existing:
            pr = existing[0]
        else:
            raise

    public_pr_number = pr.get("number")
    public_pr_url = pr.get("html_url", pr.get("url"))

    # Update issue body: set status, record PR ref
    new_body = set_status(body, "CI_WATCH")
    # Metadata as top-level HTML comments (not inside <details> log)
    new_body += f"\n<!-- public_pr: #{public_pr_number} -->\n"
    new_body += f"<!-- ci_iteration: 0 -->\n"
    new_body = append_log(new_body, "public-submit",
                          f"Public PR created: {PUBLIC_TARGET_REPO}#{public_pr_number}\n"
                          f"URL: {public_pr_url}")
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    log("INFO", f"Public PR created for #{issue_number}: "
                 f"pytorch/pytorch#{public_pr_number}", issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
