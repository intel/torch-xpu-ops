"""MERGED stage: close the source issue and clean up.

Steps
-----
1. Verify public PR is actually merged (guard against stage race).
2. Post a ✅ "Fixed in pytorch/pytorch#N" comment on the source issue.
3. Close the source issue.
4. Delete the agent branch from PRIVATE_REVIEW_REPO.
5. Transition → DONE.

Entry point
-----------
    python -m pytorch_agent.fixing_steps.close_issue --issue 123
"""
from __future__ import annotations

import argparse

from ..utils.state import TrackedIssue, load_tracked


def run(tracked: TrackedIssue) -> None:
    """Close source issue and transition to DONE."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(load_tracked(args.issue))


if __name__ == "__main__":
    main()
