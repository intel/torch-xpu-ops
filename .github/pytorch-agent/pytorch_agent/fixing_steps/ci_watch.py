"""CI_WATCH stage: monitor CI and auto-fix failures.

Loop (each cron cycle)
----------------------
1. Check if public PR is already merged → MERGED.
2. Fetch CI check-runs for the PR head commit.
3. If any are still pending → wait (return, try next cycle).
4. If all passed → wait for maintainer to merge (return).
5. If failures:
   a. Increment ci_iteration (saved before work to prevent lost counts).
   b. If ci_iteration > MAX_CI_ITERATIONS → NEEDS_HUMAN.
   c. Dispatch agent with failure details.
   d. Commit + push fixes.

Entry point
-----------
    python -m pytorch_agent.fixing_steps.ci_watch --issue 123
"""
from __future__ import annotations

import argparse

from ..utils.state import TrackedIssue, load_tracked


MAX_CI_ITERATIONS = 3

CI_FIX_PROMPT_TEMPLATE = """CI is failing on the public PR for your PyTorch fix.

## Failing checks
{failures}

## Instructions
1. Analyse each failure — related to your change or pre-existing?
2. If related: fix the code.
3. If unrelated: note it but do not change code.
"""


def run(tracked: TrackedIssue) -> None:
    """Run one CI-watch cycle for *tracked*."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(load_tracked(args.issue))


if __name__ == "__main__":
    main()
