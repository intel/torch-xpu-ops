"""IMPLEMENTING stage: generate a fix using an AI coding agent.

Steps
-----
1. Fetch issue body and parse sections (error log, reproduce steps, …).
2. Build an implementation prompt with the issue context.
3. Run the agent inside ~/pytorch (private fork working directory).
4. Auto-commit any changes the agent left uncommitted.
5. Push the branch to PRIVATE_REVIEW_REPO.
6. Open a PR on PRIVATE_REVIEW_REPO for human review.
7. Transition → IN_REVIEW.

Entry point
-----------
    python -m pytorch_agent.fixing_steps.implement --issue 123
"""
from __future__ import annotations

import argparse

from ..utils.state import TrackedIssue, load_tracked


IMPLEMENT_PROMPT_TEMPLATE = """Fix the following PyTorch XPU CI failure.

## Issue #{number}: {title}

### Error log
{error_log}

### Reproduce steps
{reproduce}

## Hard rules
- NEVER use @skipIfXpu, @skip, or any skip decorator — fix the test.
- Do NOT commit submodule changes (third_party/*).
- Run the affected test locally and confirm it passes before finishing.
"""


def run(tracked: TrackedIssue) -> None:
    """Run the implement stage for *tracked*."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(load_tracked(args.issue))


if __name__ == "__main__":
    main()
