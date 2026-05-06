"""Stage router: advance an issue by one step in the pipeline.

advance() loads the current stage and dispatches to the correct handler:

    IMPLEMENTING  →  fixing_steps.implement.run()
    IN_REVIEW     →  fixing_steps.private_review.run()
    PUBLIC_PR     →  fixing_steps.public_submit.run()
    CI_WATCH      →  fixing_steps.ci_watch.run()
    MERGED        →  fixing_steps.close_issue.run()
    (terminal)    →  no-op
"""
from __future__ import annotations



def advance(issue_number: int) -> None:
    """Load *issue_number* state and run the handler for its current stage.

    Called by the polling loop on every cron cycle.  Each handler is
    idempotent — calling advance() on an already-completed stage is safe.
    """
    raise NotImplementedError
