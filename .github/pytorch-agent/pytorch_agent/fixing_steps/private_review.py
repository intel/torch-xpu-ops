"""IN_REVIEW stage: respond to review feedback on the private fork PR.

Review loop
-----------
1. Check review state (approved / changes_requested / pending / paused).
2. If approved        →  transition to PUBLIC_PR.
3. If paused          →  set tracked.paused, wait for /agent resume.
4. If pending         →  nothing to do yet.
5. If changes_requested:
   a. Extract actionable tasks from review comments (LLM + regex fallback).
   b. Post a task-list comment on the PR.
   c. Dispatch agent to address feedback.
   d. Commit + push changes.
   e. Update task-list comment with completion status.
   f. If review_iteration > MAX_REVIEW_ITERATIONS → NEEDS_HUMAN.

Entry point
-----------
    python -m pytorch_agent.fixing_steps.private_review --issue 123
"""
from __future__ import annotations

import argparse

from ..utils.state import TrackedIssue, load_tracked


REVIEW_FIX_PROMPT_TEMPLATE = """Address the following code review feedback on your PyTorch fix.

## Original Issue #{number}: {title}

## Review Feedback
{reviews}

## Instructions
1. Address EACH comment — do not skip any.
2. Make the requested changes.
3. Ensure tests still pass.

## Hard rules
- NEVER use @skipIfXpu or any skip decorator.
- Do NOT commit submodule changes (third_party/*).
"""


def run(tracked: TrackedIssue) -> None:
    """Run one review-response cycle for *tracked*."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(load_tracked(args.issue))


if __name__ == "__main__":
    main()
