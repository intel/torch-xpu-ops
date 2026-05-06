"""Issue state management via GitHub issue comments + labels.

State storage
-------------
State lives entirely on the source issue in intel/torch-xpu-ops:

- A hidden HTML comment  <!-- AGENT_STATE: {...} -->  holds the full
  TrackedIssue JSON.  The pipeline reads/writes this on every transition.
- GitHub labels  (agent:active, agent:blocked, …) mirror the current
  stage for human visibility and issue-list filtering.

Stage machine
-------------
DISCOVERED → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → MERGED → DONE
                                                                       ↘ SKIPPED
                                                                       ↘ NEEDS_HUMAN
"""
from __future__ import annotations

from dataclasses import dataclass, field


STAGES = [
    "DISCOVERED", "IMPLEMENTING", "IN_REVIEW",
    "PUBLIC_PR", "CI_WATCH", "MERGED", "DONE", "SKIPPED", "NEEDS_HUMAN",
]

STATE_COMMENT_MARKER = "<!-- AGENT_STATE:"
STATE_COMMENT_END = "-->"


@dataclass
class TrackedIssue:
    """All pipeline state for one tracked issue.

    Persisted as JSON inside a hidden HTML comment on the source issue.
    """
    source_repo: str
    source_number: int
    title: str
    stage: str = "DISCOVERED"
    tracking_pr_number: int | None = None   # PR on PRIVATE_REVIEW_REPO
    tracking_pr_url: str | None = None
    public_pr_number: int | None = None     # PR on PUBLIC_TARGET_REPO
    public_pr_url: str | None = None
    branch: str | None = None
    triage_reason: str | None = None
    review_iteration: int = 0               # how many review cycles run
    attempt_count: int = 0
    last_push_sha: str | None = None
    paused: bool = False                    # set by /agent pause comment
    ci_iteration: int = 0                   # how many CI-fix cycles run
    _state_comment_id: int | None = field(default=None, repr=False)


def render_state_comment(tracked: TrackedIssue) -> str:
    """Render *tracked* as a GitHub comment (human-visible + hidden JSON)."""
    raise NotImplementedError


def parse_state_comment(comment_body: str) -> TrackedIssue | None:
    """Parse state from a comment body. Returns None if no marker found."""
    raise NotImplementedError


def save_state(tracked: TrackedIssue) -> None:
    """Upsert the state comment and sync GitHub labels."""
    raise NotImplementedError


def update_stage(tracked: TrackedIssue, new_stage: str, message: str) -> None:
    """Transition to *new_stage*, post a human-readable comment, save state."""
    raise NotImplementedError


def get_all_tracked() -> list[TrackedIssue]:
    """Return all issues currently carrying any agent: label."""
    raise NotImplementedError


def find_tracked_by_issue(source_number: int) -> TrackedIssue | None:
    """Return existing TrackedIssue for *source_number*, or None."""
    raise NotImplementedError


def load_tracked(issue_number: int) -> TrackedIssue:
    """Load TrackedIssue or raise ValueError if none found."""
    raise NotImplementedError
