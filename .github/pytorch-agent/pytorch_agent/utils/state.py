"""State management via source issue comments + labels.

State is stored on intel/torch-xpu-ops issues:
- Labels track current stage (agent:tracking, agent:implementing, etc.)
- JSON state lives in a hidden HTML comment in a dedicated issue comment
- Stage transitions post human-readable comments; detailed logs stay in logs/
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime

from . import github_client as gh
from .config import (
    UPSTREAM_ISSUE_REPO, STAGE_TO_LABEL, ALL_AGENT_LABELS,
)
from .logger import log

STAGES = [
    "DISCOVERED", "TRIAGING", "IMPLEMENTING", "IN_REVIEW",
    "PUBLIC_PR", "CI_WATCH", "DONE", "SKIPPED", "NEEDS_HUMAN",
]

STATE_COMMENT_MARKER = "<!-- AGENT_STATE:"
STATE_COMMENT_END = "-->"


@dataclass
class TrackedIssue:
    source_repo: str
    source_number: int
    title: str
    stage: str = "DISCOVERED"
    tracking_pr_number: int | None = None
    tracking_pr_url: str | None = None
    public_pr_number: int | None = None
    public_pr_url: str | None = None
    branch: str | None = None
    triage_reason: str | None = None
    review_iteration: int = 0
    attempt_count: int = 0
    last_push_sha: str | None = None
    paused: bool = False
    _state_comment_id: int | None = field(default=None, repr=False)


def render_state_comment(tracked: TrackedIssue) -> str:
    """Render TrackedIssue as hidden HTML comment with JSON."""
    data = asdict(tracked)
    # Remove internal field
    data.pop("_state_comment_id", None)
    json_str = json.dumps(data, indent=2)
    return (
        f"{STATE_COMMENT_MARKER}\n{json_str}\n{STATE_COMMENT_END}\n\n"
        f"🤖 **Agent tracking state** — do not edit this comment manually.\n"
        f"Stage: `{tracked.stage}` | Branch: `{tracked.branch or 'N/A'}`"
    )


def parse_state_comment(comment_body: str) -> TrackedIssue | None:
    """Parse hidden HTML comment back into TrackedIssue. Returns None if no marker."""
    match = re.search(
        rf"{re.escape(STATE_COMMENT_MARKER)}\s*\n(.*?)\n\s*{re.escape(STATE_COMMENT_END)}",
        comment_body, re.DOTALL,
    )
    if not match:
        return None
    data = json.loads(match.group(1))
    comment_id = data.pop("_state_comment_id", None)
    tracked = TrackedIssue(**data)
    tracked._state_comment_id = comment_id
    return tracked


def _find_state_comment(repo: str, issue_number: int) -> tuple[int | None, TrackedIssue | None]:
    """Find the state comment on an issue. Returns (comment_id, TrackedIssue)."""
    comments = gh.get_issue_comments(repo, issue_number)
    for comment in comments:
        body = comment.get("body", "")
        if STATE_COMMENT_MARKER in body:
            tracked = parse_state_comment(body)
            if tracked:
                tracked._state_comment_id = comment["id"]
                return comment["id"], tracked
    return None, None


def save_state(tracked: TrackedIssue) -> None:
    """Update (or create) the state comment on the source issue. Sync labels."""
    repo = tracked.source_repo
    number = tracked.source_number
    body = render_state_comment(tracked)

    if tracked._state_comment_id:
        gh.update_issue_comment(repo, tracked._state_comment_id, body)
    else:
        gh.add_issue_comment(repo, number, body)
        # Re-fetch to get the comment ID
        comment_id, _ = _find_state_comment(repo, number)
        tracked._state_comment_id = comment_id

    # Sync labels: set the label matching current stage, remove others
    target_label = STAGE_TO_LABEL.get(tracked.stage)
    for label in ALL_AGENT_LABELS:
        if label == target_label:
            gh.add_label(repo, number, label)
        else:
            try:
                gh.remove_label(repo, number, label)
            except Exception:
                pass  # Label may not be on this issue


def update_stage(tracked: TrackedIssue, new_stage: str, message: str) -> None:
    """Update stage, post human-readable comment, sync label, save state.

    Also updates the Action Items checklist in the issue body (if present).
    """
    old_stage = tracked.stage
    tracked.stage = new_stage
    save_state(tracked)

    # Update Action Items checklist in issue body
    _update_action_items_checklist(tracked)

    # Post human-readable transition comment
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comment = (
        f"🤖 **Stage transition:** `{old_stage}` → `{new_stage}`\n"
        f"📅 {ts}\n\n{message}"
    )
    gh.add_issue_comment(tracked.source_repo, tracked.source_number, comment)
    log("INFO", f"Stage {old_stage} → {new_stage}: {message}",
        issue=tracked.source_number)


# Maps each pipeline stage to which Action Items should be checked off (cumulative).
# These match the checklist from the CI failure tracking issue template.
_STAGE_CHECKLIST: dict[str, list[str]] = {
    "DISCOVERED":    [],
    "TRIAGING":      [],
    "IMPLEMENTING":  ["Reproduce on dev machine", "Identify root cause", "Implement fix", "Verify fix locally"],
    "IN_REVIEW":     ["Reproduce on dev machine", "Identify root cause", "Implement fix", "Verify fix locally", "PR proposal"],
    "PUBLIC_PR":     ["Reproduce on dev machine", "Identify root cause", "Implement fix", "Verify fix locally", "PR proposal", "Human review"],
    "CI_WATCH":      ["Reproduce on dev machine", "Identify root cause", "Implement fix", "Verify fix locally", "PR proposal", "Human review", "PR creation"],
    "DONE":          ["Reproduce on dev machine", "Identify root cause", "Implement fix", "Verify fix locally", "PR proposal", "Human review", "PR creation"],
}


def _update_action_items_checklist(tracked: TrackedIssue) -> None:
    """Update the Action Items checklist in the issue body to reflect current stage.

    Checks off completed items and unchecks future items based on _STAGE_CHECKLIST.
    No-op if the issue body has no Action Items section.
    """
    repo = tracked.source_repo
    number = tracked.source_number

    try:
        detail = gh.get_issue_detail(repo, number)
    except Exception:
        return

    body = detail.get("body", "") or ""
    if "### Action Items" not in body:
        return

    checked_items = set(_STAGE_CHECKLIST.get(tracked.stage, []))

    # Replace checkbox lines in the Action Items section
    lines = body.split("\n")
    in_action_items = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("### Action Items"):
            in_action_items = True
            new_lines.append(line)
            continue

        if in_action_items and line.startswith("### "):
            # Entered a new section — stop modifying
            in_action_items = False

        if in_action_items and re.match(r"^- \[[ xX]\] ", line):
            # Extract the item text
            item_text = re.sub(r"^- \[[ xX]\] ", "", line).strip()
            if item_text in checked_items:
                new_lines.append(f"- [x] {item_text}")
            else:
                new_lines.append(f"- [ ] {item_text}")
            continue

        new_lines.append(line)

    new_body = "\n".join(new_lines)
    if new_body != body:
        try:
            gh._gh_api(f"/repos/{repo}/issues/{number}", method="PATCH", body=new_body)
        except Exception as e:
            log("WARN", f"Failed to update Action Items checklist for #{number}: {e}",
                issue=number)


def get_all_tracked() -> list[TrackedIssue]:
    """List all issues with any agent: label, parse state comments."""
    seen = set()
    tracked_list = []
    for label in ALL_AGENT_LABELS:
        issues = gh.get_issues(UPSTREAM_ISSUE_REPO, label)
        for issue in issues:
            if issue["number"] in seen:
                continue
            seen.add(issue["number"])
            _, tracked = _find_state_comment(UPSTREAM_ISSUE_REPO, issue["number"])
            if tracked:
                tracked_list.append(tracked)
    return tracked_list


def find_tracked_by_issue(source_number: int) -> TrackedIssue | None:
    """Check if issue already has agent:tracking label + state comment."""
    _, tracked = _find_state_comment(UPSTREAM_ISSUE_REPO, source_number)
    return tracked


def load_tracked(issue_number: int) -> TrackedIssue:
    """Load TrackedIssue from source issue state comment (for CLI entry points)."""
    _, tracked = _find_state_comment(UPSTREAM_ISSUE_REPO, issue_number)
    if tracked is None:
        raise ValueError(
            f"No agent state found on issue #{issue_number}. "
            "Run issue_discovery first."
        )
    return tracked
