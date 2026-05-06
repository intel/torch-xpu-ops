"""PUBLIC_PR stage: open a cross-fork PR to pytorch/pytorch.

Steps
-----
1. Build a clean PR title (strip [Agent] prefix if present).
2. Build PR body from issue details via build_pr_body().
3. Check if a PR from the same branch already exists (idempotent).
4. Create the cross-fork PR using REVIEW_GH_TOKEN.
5. Post the public PR link back to the source issue.
6. Transition → CI_WATCH.

Entry point
-----------
    python -m pytorch_agent.fixing_steps.public_submit --issue 123
"""
from __future__ import annotations

import argparse

from ..utils.state import TrackedIssue, load_tracked


def run(tracked: TrackedIssue) -> None:
    """Create (or find existing) public PR and transition to CI_WATCH."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(load_tracked(args.issue))


if __name__ == "__main__":
    main()
