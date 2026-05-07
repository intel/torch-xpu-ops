"""Handle private review on the private review fork.

Entry point:
  python -m pytorch_agent.fixing_steps.private_review --issue 123
"""
from __future__ import annotations

import argparse
import subprocess

from ..utils import git as gh
from ..utils.git import git, add_and_commit
from ..utils.config import (
    PRIVATE_REVIEW_REPO, PYTORCH_DIR, REVIEW_REMOTE,
    MAX_REVIEW_ITERATIONS, ISSUE_REPO, STAGE_TIMEOUTS,
)
from ..utils.issue_body import (
    get_status, set_status, append_log,
)
from ..utils.review_handler import (
    get_pending_reviews, format_reviews_for_prompt, get_review_state,
)
from ..utils.agent_backend import get_backend
from ..utils.logger import log
from ..utils.notify import post_agent_completed, post_session_started


def _build_task_list(reviews: list[dict]) -> list[str]:
    """Extract actionable tasks from review comments using an LLM."""
    import re

    combined = []
    for review in reviews:
        body = (review.get("body") or "").strip().replace("\r", "")
        if body:
            combined.append(body)
    if not combined:
        return []

    review_text = "\n\n---\n\n".join(combined)

    extraction_prompt = (
        "Extract concise, actionable tasks from the following code review. "
        "Return ONLY a numbered list (1. 2. 3. ...), one task per line. "
        "Each task should be a concrete action the developer must take. "
        "Do NOT include section headers, verdicts, analysis, or explanations — "
        "only actionable items.\n\n"
        "Review:\n" + review_text
    )

    try:
        from ..utils.config import OPENCODE_CMD
        from ..utils.agent_backend import parse_opencode_events
        result = subprocess.run(
            [OPENCODE_CMD, "run", "--format", "json", "--dir", str(PYTORCH_DIR),
             "--dangerously-skip-permissions", extraction_prompt],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, text=True, timeout=120,
        )
        full_text = parse_opencode_events(result.stdout)
        if full_text:
            items = re.findall(r'^\s*\d+\.\s*(.+)', full_text, re.MULTILINE)
            if items:
                return [item.strip()[:200] for item in items if item.strip()]
    except (subprocess.TimeoutExpired, Exception) as e:
        log("WARN", f"LLM task extraction failed: {e}, falling back to regex")

    return _build_task_list_regex(combined)


def _build_task_list_regex(bodies: list[str]) -> list[str]:
    """Fallback regex-based task extraction."""
    import re
    tasks = []
    for body in bodies:
        numbered = re.findall(r'^\s*\d+\.\s*(.+)', body, re.MULTILINE)
        if numbered:
            tasks.extend(item.strip()[:200] for item in numbered if item.strip())
        else:
            paragraphs = re.split(r'\n\s*\n', body)
            for para in paragraphs:
                para = para.strip()
                if para:
                    tasks.append(para.split("\n")[0][:200])
    return tasks


def _format_task_comment(tasks: list[str], done: set[int] | None = None,
                         reviewer: str | None = None,
                         all_done: bool = False) -> str:
    """Format a task list comment with checkboxes."""
    if done is None:
        done = set()
    lines = ["🤖 **Addressing review feedback:**\n"]
    for i, task in enumerate(tasks):
        check = "x" if i in done else " "
        lines.append(f"- [{check}] {task}")
    if all_done and reviewer:
        lines.append(f"\n✅ All tasks addressed. @{reviewer} please re-review.")
    return "\n".join(lines)


def run(issue_number: int) -> None:
    """Handle private review cycle."""
    import re

    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    if get_status(body) != "IN_REVIEW":
        return

    # Get tracking PR number from body
    pr_match = re.search(r"tracking_pr:\s*#?(\d+)", body)
    if not pr_match:
        log("WARN", f"No tracking PR for issue #{issue_number}", issue=issue_number)
        return
    tracking_pr = int(pr_match.group(1))

    # Get last push sha
    sha_match = re.search(r"last_push_sha:\s*([a-f0-9]+)", body)
    last_push_sha = sha_match.group(1) if sha_match else None

    state = get_review_state(tracking_pr, last_push_sha)

    if state == "approved":
        new_body = set_status(body, "PUBLIC_PR")
        new_body = append_log(new_body, "review", "Private review approved. Ready for public PR.")
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        return

    if state == "paused":
        log("INFO", f"Agent paused for #{issue_number}", issue=issue_number)
        return

    if state == "pending":
        log("INFO", f"Review still pending for #{issue_number}", issue=issue_number)
        return

    # changes_requested — get iteration count
    iter_match = re.search(r"review_iteration:\s*(\d+)", body)
    review_iteration = int(iter_match.group(1)) if iter_match else 0
    review_iteration += 1

    if review_iteration > MAX_REVIEW_ITERATIONS:
        new_body = set_status(body, "NEEDS_HUMAN")
        new_body = append_log(new_body, "review",
                              f"Exceeded {MAX_REVIEW_ITERATIONS} review iterations. Needs human.")
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        return

    # Get review comments
    reviews = get_pending_reviews(tracking_pr, last_push_sha)
    if not reviews:
        log("INFO", f"No new review feedback after last push for #{issue_number}",
            issue=issue_number)
        return
    review_text = format_reviews_for_prompt(reviews)

    # Identify reviewer
    reviewer = None
    for r in reviews:
        login = r.get("user", {}).get("login", "")
        if login and not login.endswith("[bot]"):
            reviewer = login
            break

    # Build task list and post
    tasks = _build_task_list(reviews)
    if tasks:
        initial_comment = _format_task_comment(tasks)
        resp = gh.add_pr_comment(PRIVATE_REVIEW_REPO, tracking_pr, initial_comment)
        task_comment_id = resp.get("id")
    else:
        task_comment_id = None

    # Call agent with skill (no inline prompt)
    prompt = (
        f"Read the pytorch-review-fix skill and address review feedback on PR #{tracking_pr} "
        f"for issue #{issue_number}.\n\n"
        f"## Review Feedback\n{review_text}"
    )

    pre_sha = git("rev-parse", "HEAD", issue=issue_number).stdout.strip()

    def _post_session_id(sid: str):
        post_session_started(ISSUE_REPO, issue_number, "Review fix", sid)

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("IN_REVIEW", 1800)
    output, log_path, _ = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                    skill="pytorch-review-fix", timeout=timeout,
                                    issue=issue_number, stage="IN_REVIEW",
                                    on_session_start=_post_session_id)

    # Check changes
    post_sha = git("rev-parse", "HEAD", issue=issue_number).stdout.strip()
    uncommitted = git("status", "--porcelain", issue=issue_number).stdout.strip()
    has_changes = (post_sha != pre_sha) or bool(uncommitted)

    if uncommitted:
        committed = add_and_commit(
            f"Address review feedback (iteration {review_iteration})\n\n"
            f"intel/torch-xpu-ops#{issue_number}",
            issue=issue_number,
        )
        if committed:
            has_changes = True

    # Determine addressed tasks
    done_tasks: set[int] = set()
    if has_changes and output:
        output_lower = output.lower()
        for i, task in enumerate(tasks):
            task_words = [w.lower() for w in task.split() if len(w) > 4]
            matches = sum(1 for w in task_words if w in output_lower)
            if task_words and matches >= len(task_words) * 0.3:
                done_tasks.add(i)
    elif has_changes:
        done_tasks = set(range(len(tasks)))

    # Push
    branch = f"agent/issue-{issue_number}"
    if has_changes:
        git("push", REVIEW_REMOTE, branch, issue=issue_number)

    # Update iteration and sha in body
    new_sha = git("rev-parse", "HEAD", issue=issue_number).stdout.strip()
    new_body = body
    if iter_match:
        new_body = new_body[:iter_match.start()] + f"review_iteration: {review_iteration}" + new_body[iter_match.end():]
    else:
        new_body += f"\n<!-- review_iteration: {review_iteration} -->\n"
    if sha_match:
        new_body = new_body.replace(sha_match.group(0), f"last_push_sha: {new_sha}")
    else:
        new_body += f"\n<!-- last_push_sha: {new_sha} -->\n"

    new_body = append_log(new_body, "review",
                          f"Review iteration {review_iteration} — "
                          f"tasks addressed: {len(done_tasks)}/{len(tasks)}\n"
                          f"Log: `{log_path.name}`")
    gh.update_issue_body(ISSUE_REPO, issue_number, new_body)

    # Update task comment
    if task_comment_id and tasks:
        all_done = done_tasks == set(range(len(tasks)))
        comment = _format_task_comment(tasks, done=done_tasks, reviewer=reviewer, all_done=all_done)
        if not has_changes:
            comment += "\n\n⚠️ Agent produced no code changes."
        gh.update_pr_comment(PRIVATE_REVIEW_REPO, task_comment_id, comment)

    log("INFO", f"Review iteration {review_iteration} for #{issue_number}, "
                 f"tasks: {len(done_tasks)}/{len(tasks)}", issue=issue_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    run(args.issue)


if __name__ == "__main__":
    main()
