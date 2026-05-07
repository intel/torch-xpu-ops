"""Submit fix as public PR to pytorch/pytorch.

Entry point:
  python -m pytorch_agent.fixing_steps.public_submit --issue 123
"""
from __future__ import annotations

import argparse
import os
from subprocess import CalledProcessError

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PRIVATE_REVIEW_REPO,
    PUBLIC_TARGET_REPO,
)
from ..utils.state import TrackedIssue, update_stage, load_tracked
from ..utils.logger import log
from ._issue_format import build_pr_body


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

    body = build_pr_body(
        upstream_issue_repo=UPSTREAM_ISSUE_REPO,
        source_number=tracked.source_number,
        title=detail.get("title", "N/A"),
        triage_reason=tracked.triage_reason,
        issue_body=detail.get("body", ""),
        reviewer=os.environ.get("PUBLIC_PR_REVIEWER", ""),
    )

    # Idempotent: check if PR already exists for this branch
    head_label = f"{PRIVATE_REVIEW_REPO.split('/')[0]}:{branch}"
    try:
        existing = gh._gh_api(
            f"/repos/{PUBLIC_TARGET_REPO}/pulls",
            token=gh._token_for_repo(PUBLIC_TARGET_REPO),
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
    except CalledProcessError as exc:
        # 422 "PR already exists" — find it
        log("WARN", f"PR creation failed, checking for existing: {exc}",
            issue=tracked.source_number)
        existing = gh._gh_api(
            f"/repos/{PUBLIC_TARGET_REPO}/pulls",
            token=gh._token_for_repo(PUBLIC_TARGET_REPO),
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
