"""Handle private review on the private review fork.

Entry point:
  python -m pytorch_agent.fixing_steps.private_review --issue 123
"""
from __future__ import annotations

import argparse
import subprocess

from ..utils import github_client as gh
from ..utils.config import (
    PRIVATE_REVIEW_REPO, PYTORCH_DIR, REVIEW_REMOTE,
    MAX_REVIEW_ITERATIONS, UPSTREAM_ISSUE_REPO, STAGE_TIMEOUTS,
)
from ..utils.state import TrackedIssue, update_stage, save_state, load_tracked
from ..utils.review_handler import (
    get_pending_reviews, format_reviews_for_prompt, get_review_state,
)
from ..utils.agent_backend import get_backend
from ..utils.logger import log


REVIEW_FIX_PROMPT_TEMPLATE = """Address the following code review feedback on your PyTorch fix.

## Original Issue #{number}: {title}

## Review Feedback
{reviews}

## Instructions
1. Address EACH review comment separately — do not skip any.
2. Make the requested changes.
3. Ensure tests still pass.

## HARD RULES (violations will be rejected)
- NEVER use @skipIfXpu, @skip, unittest.skip, or any skip decorator. You must FIX the test, not skip it.
- Do NOT commit submodule pointer changes (third_party/*). Use `git add` on specific files only.
"""


def _build_task_list(reviews: list[dict]) -> list[str]:
    """Extract actionable tasks from review comments using an LLM.

    Sends the review text to opencode with a focused extraction prompt,
    then parses the numbered list from the response.  Falls back to a
    simple regex splitter if the LLM call fails or times out.
    """
    import json as _json
    import re

    # Combine all review bodies
    combined = []
    for review in reviews:
        body = (review.get("body") or "").strip().replace("\r", "")
        if body:
            combined.append(body)
    if not combined:
        return []

    review_text = "\n\n---\n\n".join(combined)

    # --- LLM extraction ---
    extraction_prompt = (
        "Extract concise, actionable tasks from the following code review. "
        "Return ONLY a numbered list (1. 2. 3. ...), one task per line. "
        "Each task should be a concrete action the developer must take "
        "(e.g. 'Revert changes to file X', 'Get CI failure traceback'). "
        "Do NOT include section headers, verdicts, analysis, or explanations — "
        "only actionable items.\n\n"
        "Review:\n" + review_text
    )

    try:
        result = subprocess.run(
            ["opencode", "run", "--format", "json", "--dir", str(PYTORCH_DIR),
             "--dangerously-skip-permissions", extraction_prompt],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=120,
        )
        # Parse JSON events for text output
        llm_output = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = _json.loads(line)
                if evt.get("type") == "text":
                    # Text lives in part.text, not top-level content
                    text = (evt.get("part", {}).get("text", "")
                            or evt.get("content", ""))
                    llm_output.append(text)
            except _json.JSONDecodeError:
                continue

        full_text = "".join(llm_output).strip()
        if full_text:
            # Parse numbered items: "1. ...", "2. ...", etc.
            items = re.findall(r'^\s*\d+\.\s*(.+)', full_text, re.MULTILINE)
            if items:
                log("INFO", f"LLM extracted {len(items)} tasks from review")
                return [item.strip()[:200] for item in items if item.strip()]

        log("WARN", "LLM extraction returned no parseable tasks, falling back to regex")
    except subprocess.TimeoutExpired:
        log("WARN", "LLM task extraction timed out (60s), falling back to regex")
    except Exception as e:
        log("WARN", f"LLM task extraction failed: {e}, falling back to regex")

    # --- Fallback: simple regex ---
    return _build_task_list_regex(combined)


def _build_task_list_regex(bodies: list[str]) -> list[str]:
    """Fallback regex-based task extraction."""
    import re
    tasks = []
    for body in bodies:
        # Extract numbered items (1. ..., 2. ...) as primary signal
        numbered = re.findall(r'^\s*\d+\.\s*(.+)', body, re.MULTILINE)
        if numbered:
            tasks.extend(item.strip()[:200] for item in numbered if item.strip())
        else:
            # Fall back to paragraph splitting
            paragraphs = re.split(r'\n\s*\n', body)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                summary = para.split("\n")[0][:200]
                tasks.append(summary)
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


def run(tracked: TrackedIssue) -> None:
    """Handle private review cycle."""
    if tracked.stage != "IN_REVIEW":
        return

    if not tracked.tracking_pr_number:
        log("WARN", f"No tracking PR for issue #{tracked.source_number}",
            issue=tracked.source_number)
        return

    state = get_review_state(tracked.tracking_pr_number,
                             tracked.last_push_sha)

    if state == "approved":
        update_stage(tracked, "PUBLIC_PR",
                     "Private review approved. Ready for public PR.")
        return

    if state == "paused":
        if not tracked.paused:
            tracked.paused = True
            save_state(tracked)
            log("INFO", f"Agent paused by /agent pause for #{tracked.source_number}",
                issue=tracked.source_number)
        else:
            log("INFO", f"Agent still paused for #{tracked.source_number}",
                issue=tracked.source_number)
        return

    # If we get feedback after being paused, resume
    if tracked.paused:
        tracked.paused = False
        save_state(tracked)
        log("INFO", f"Agent resumed for #{tracked.source_number}",
            issue=tracked.source_number)

    if state == "pending":
        log("INFO", f"Review still pending for #{tracked.source_number}",
            issue=tracked.source_number)
        return

    # changes_requested
    tracked.review_iteration += 1
    if tracked.review_iteration > MAX_REVIEW_ITERATIONS:
        update_stage(tracked, "NEEDS_HUMAN",
                     f"Exceeded {MAX_REVIEW_ITERATIONS} review iterations. Needs human.")
        gh.add_label(UPSTREAM_ISSUE_REPO, tracked.source_number, "agent:needs-human")
        return

    save_state(tracked)

    # Get review comments
    reviews = get_pending_reviews(tracked.tracking_pr_number,
                                  tracked.last_push_sha)
    # If no new reviews after last push, nothing to address — skip
    if not reviews:
        logger.info("No new review feedback after last push for #%s, skipping",
                     tracked.source_number)
        return
    review_text = format_reviews_for_prompt(reviews)

    # Identify the reviewer (first non-bot commenter)
    reviewer = None
    for r in reviews:
        login = r.get("user", {}).get("login", "")
        if login and not login.endswith("[bot]"):
            reviewer = login
            break

    # Build task list and post initial comment on the PR
    tasks = _build_task_list(reviews)
    if tasks:
        initial_comment = _format_task_comment(tasks)
        resp = gh.add_pr_comment(PRIVATE_REVIEW_REPO,
                                 tracked.tracking_pr_number, initial_comment)
        task_comment_id = resp.get("id")
    else:
        task_comment_id = None

    # Dispatch agent to fix
    prompt = REVIEW_FIX_PROMPT_TEMPLATE.format(
        number=tracked.source_number,
        title=tracked.title,
        reviews=review_text,
    )

    # Record pre-agent HEAD so we can check if agent made changes
    pre_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(PYTORCH_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    def _post_session_id(sid: str):
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🔗 **Review fix agent session started**\n\n"
            f"**Attach to watch live:**\n"
            f"```bash\ncd ~/pytorch && opencode -s {sid}\n```\n"
            f"Session ID: `{sid}`",
        )

    backend = get_backend()
    timeout = STAGE_TIMEOUTS.get("IN_REVIEW", 1800)
    output, log_path, _ = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                    skill="xpu-ops-pr-review", timeout=timeout,
                                    issue=tracked.source_number, stage="IN_REVIEW",
                                    on_session_start=_post_session_id)
    log("INFO", f"Review fix agent log: {log_path}",
        issue=tracked.source_number)

    # Check if agent actually made changes (committed or uncommitted)
    post_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(PYTORCH_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    uncommitted = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(PYTORCH_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    has_changes = (post_sha != pre_sha) or bool(uncommitted)

    # Auto-commit any uncommitted changes the agent left
    if uncommitted:
        # Exclude submodule pointers
        changed_files = [
            line.split(maxsplit=1)[1].strip()
            for line in uncommitted.split("\n")
            if line.strip() and not line.split(maxsplit=1)[1].strip().startswith("third_party/")
        ]
        if changed_files:
            subprocess.run(["git", "add", "--"] + changed_files,
                           cwd=str(PYTORCH_DIR), check=True)
            subprocess.run(["git", "commit", "-m",
                           f"Address review feedback (iteration {tracked.review_iteration})\n\n"
                           f"intel/torch-xpu-ops#{tracked.source_number}"],
                          cwd=str(PYTORCH_DIR), check=True)
            has_changes = True

    # Determine which tasks were addressed by checking agent output
    done_tasks: set[int] = set()
    if has_changes and output:
        output_lower = output.lower()
        for i, task in enumerate(tasks):
            # Simple heuristic: check if key words from task appear in output
            task_words = [w.lower() for w in task.split() if len(w) > 4]
            matches = sum(1 for w in task_words if w in output_lower)
            if task_words and matches >= len(task_words) * 0.3:
                done_tasks.add(i)
        # If agent made changes but we can't match specific tasks,
        # be honest — don't mark any as done
    elif has_changes:
        # Agent made changes but output is empty — mark all tentatively
        done_tasks = set(range(len(tasks)))

    # Push updated code (NEVER force push — it destroys reviewed commits)
    branch = tracked.branch or f"agent/issue-{tracked.source_number}"
    if has_changes:
        subprocess.run(
            ["git", "push", REVIEW_REMOTE, branch],
            cwd=str(PYTORCH_DIR), check=True,
        )

    # Update last_push_sha
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(PYTORCH_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    tracked.last_push_sha = sha
    save_state(tracked)

    # Update task comment with actual status
    if task_comment_id and tasks:
        all_done = done_tasks == set(range(len(tasks)))
        if not has_changes:
            status_note = "\n\n⚠️ Agent produced no code changes. Tasks may not be addressed."
        elif not all_done:
            status_note = (f"\n\n⚠️ Only {len(done_tasks)}/{len(tasks)} tasks "
                          f"appear addressed. @{reviewer} please verify.")
        else:
            status_note = ""
        comment = _format_task_comment(
            tasks, done=done_tasks, reviewer=reviewer, all_done=all_done,
        ) + status_note
        gh.update_pr_comment(PRIVATE_REVIEW_REPO, task_comment_id, comment)

    # Post log to source issue
    gh.add_issue_comment(
        UPSTREAM_ISSUE_REPO, tracked.source_number,
        f"🤖 **Review iteration {tracked.review_iteration} log:** `{log_path.name}`\n"
        f"Tasks addressed: {len(done_tasks)}/{len(tasks)}\n\n"
        f"<details><summary>Agent output summary</summary>\n\n"
        f"```\n{''.join(output.strip().splitlines(True)[-30:]) if output.strip() else '(empty)'}```\n"
        f"</details>",
    )

    log("INFO", f"Review iteration {tracked.review_iteration} for #{tracked.source_number}, "
                 f"tasks: {len(done_tasks)}/{len(tasks)}, changes: {has_changes}",
        issue=tracked.source_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
