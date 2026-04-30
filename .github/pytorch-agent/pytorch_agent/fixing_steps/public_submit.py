"""Submit fix as public PR to pytorch/pytorch.

Entry point:
  python -m pytorch_agent.fixing_steps.public_submit --issue 123
"""
from __future__ import annotations

import argparse
import os
import subprocess

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PRIVATE_REVIEW_REPO, REVIEW_REMOTE, PYTORCH_DIR,
    PUBLIC_TARGET_REPO,
)
from ..utils.state import TrackedIssue, update_stage, load_tracked
from ..utils.logger import log


def run(tracked: TrackedIssue) -> None:
    """Create cross-fork PR from private review fork to pytorch/pytorch."""
    if tracked.stage != "PUBLIC_PR":
        return

    branch = tracked.branch or f"agent/issue-{tracked.source_number}"

    # Clean title (no [Agent] prefix)
    title = tracked.title
    if title.startswith("[Agent]"):
        title = title[len("[Agent]"):].strip()

    # Build descriptive PR body from issue details + diff
    detail = gh.get_issue_detail(UPSTREAM_ISSUE_REPO, tracked.source_number)
    issue_url = f"https://github.com/{UPSTREAM_ISSUE_REPO}/issues/{tracked.source_number}"

    body = (
        f"## Summary\n\n"
        f"Fix for [{UPSTREAM_ISSUE_REPO}#{tracked.source_number}]({issue_url})\n\n"
        f"**Issue:** {detail.get('title', 'N/A')}\n\n"
    )
    if tracked.triage_reason:
        body += f"**Root Cause:** {tracked.triage_reason}\n\n"

    # Parse issue sections for context
    from ._issue_format import parse_issue_sections as _parse_issue_sections
    sections = _parse_issue_sections(detail.get("body", ""))
    if sections.get("Failed Tests"):
        body += f"**Failed Tests:**\n{sections['Failed Tests']}\n\n"
    if sections.get("Failure Type"):
        body += f"**Failure Type:** {sections['Failure Type']}\n\n"

    body += (
                (f"cc @{os.environ.get('PUBLIC_PR_REVIEWER', '')}\\n"
         if os.environ.get("PUBLIC_PR_REVIEWER") else "")
    )

    # Idempotent: check if PR already exists for this branch
    head_label = f"{PRIVATE_REVIEW_REPO.split('/')[0]}:{branch}"
    try:
        existing = gh._gh_api(
            f"/repos/{PUBLIC_TARGET_REPO}/pulls",
            head=head_label, state="open",
        )
        if existing:
            pr = existing[0]
            log("INFO", f"Public PR already exists: #{pr.get('number')}",
                issue=tracked.source_number)
        else:
            pr = gh.create_cross_fork_pr(
                head_repo=PRIVATE_REVIEW_REPO,
                head_branch=branch,
                base_repo=PUBLIC_TARGET_REPO,
                title=title,
                body=body,
            )
    except (subprocess.CalledProcessError, Exception) as exc:
        # 422 "PR already exists" — find it
        existing = gh._gh_api(
            f"/repos/{PUBLIC_TARGET_REPO}/pulls",
            head=head_label, state="open",
        )
        if existing:
            pr = existing[0]
        else:
            raise

    tracked.public_pr_number = pr.get("number")
    tracked.public_pr_url = pr.get("html_url", pr.get("url"))

    # Post public PR link to source issue
    gh.add_issue_comment(
        UPSTREAM_ISSUE_REPO, tracked.source_number,
        f"🚀 **Public PR created:** [{PUBLIC_TARGET_REPO}#{tracked.public_pr_number}]"
        f"({tracked.public_pr_url})\n\n"
        f"CI is now running. Agent will monitor and address failures.",
    )

    tracked.ci_iteration = 0
    update_stage(tracked, "CI_WATCH",
                 f"Public PR #{tracked.public_pr_number} created: {tracked.public_pr_url}")
    log("INFO", f"Public PR created for #{tracked.source_number}: "
                 f"pytorch/pytorch#{tracked.public_pr_number}",
        issue=tracked.source_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
