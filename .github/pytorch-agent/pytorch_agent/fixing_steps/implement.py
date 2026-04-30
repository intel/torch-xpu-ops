"""Implement a fix for a triaged issue.

Entry point:
  python -m pytorch_agent.fixing_steps.implement --issue 123
"""
from __future__ import annotations

import argparse
import subprocess

from ..utils import github_client as gh
from ..utils.config import (
    UPSTREAM_ISSUE_REPO, PRIVATE_REVIEW_REPO, PYTORCH_DIR,
    REVIEW_REMOTE, MAX_AGENT_ATTEMPTS, STAGE_TIMEOUTS,
)
from ..utils.state import TrackedIssue, update_stage, save_state, load_tracked
from ..utils.agent_backend import get_backend
from ..utils.logger import log


IMPLEMENT_PROMPT_TEMPLATE = """Fix the following PyTorch CI failure for Intel XPU.

## Issue #{number}: {title}

{body_section}

## Instructions
1. Start by reproducing the failure using the repro commands above (if provided).
2. If a commit scope is given, use `git log --oneline` on that range to identify the likely breaking commit.
3. Read the error log carefully and trace the root cause in the PyTorch codebase.
4. Make the minimal fix. Prefer XPU-specific paths (aten/src/ATen/xpu/, torch/xpu/) when possible.
5. Run the failing test(s) listed above to verify your fix.
6. Ensure no regressions in related tests.

## HARD RULES (violations will be rejected)
- NEVER use @skipIfXpu, @skip, unittest.skip, or any skip decorator. You must FIX the test, not skip it.
- Do NOT commit submodule pointer changes (third_party/*). Use `git add` on specific files only.

Work in the current directory (~/pytorch).
"""


def _parse_issue_sections(body: str) -> dict[str, str]:
    """Parse structured issue body from CI failure template into sections."""
    sections: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []

    for line in (body or "").split("\n"):
        # GitHub renders template fields as ### headings
        if line.startswith("### "):
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[4:].strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _build_body_section(detail: dict) -> str:
    """Build the prompt body from parsed issue sections.

    If the issue follows the CI failure template, produce a structured prompt.
    Otherwise fall back to the raw issue body.
    """
    body = detail.get("body", "")
    sections = _parse_issue_sections(body)

    # If we got structured sections, build a focused prompt
    if sections.get("Failed Tests"):
        parts = []

        if sections.get("Failure Type"):
            parts.append(f"**Failure Type:** {sections['Failure Type']}")

        if sections.get("Commit Scope"):
            parts.append(f"### Commit Scope\n{sections['Commit Scope']}")

        parts.append(f"### Failed Tests\n{sections['Failed Tests']}")

        if sections.get("Error Log"):
            # Truncate very long logs to keep prompt focused
            error_log = sections["Error Log"]
            lines = error_log.split("\n")
            if len(lines) > 80:
                error_log = "\n".join(lines[-80:])
                error_log = f"... (truncated, showing last 80 lines)\n{error_log}"
            parts.append(f"### Error Log\n```\n{error_log}\n```")

        if sections.get("CI Job URL"):
            parts.append(f"**CI Job URL:** {sections['CI Job URL']}")

        if sections.get("Repro Commands"):
            parts.append(f"### Repro Commands\n```bash\n{sections['Repro Commands']}\n```")

        return "\n\n".join(parts)

    # Fallback: raw body
    return body


def run(tracked: TrackedIssue) -> None:
    """Implement a fix: branch, dispatch agent, push."""
    # Idempotency
    if tracked.stage != "IMPLEMENTING":
        log("INFO", f"Issue #{tracked.source_number} not in IMPLEMENTING stage, skipping",
            issue=tracked.source_number)
        return

    # Escalation check
    tracked.attempt_count += 1
    if tracked.attempt_count > MAX_AGENT_ATTEMPTS:
        update_stage(tracked, "NEEDS_HUMAN",
                     f"Exceeded {MAX_AGENT_ATTEMPTS} implementation attempts. Needs human.")
        return
    save_state(tracked)

    branch = f"agent/issue-{tracked.source_number}"
    tracked.branch = branch
    # Sync review/main with the latest upstream main to avoid divergence
    _git(["fetch", "upstream"], cwd=str(PYTORCH_DIR))
    _git(["fetch", REVIEW_REMOTE], cwd=str(PYTORCH_DIR))
    try:
        # Fast-forward review/main to upstream/main so PRs only show our commits
        _git(["push", REVIEW_REMOTE, "upstream/main:main"], cwd=str(PYTORCH_DIR))
        _git(["fetch", REVIEW_REMOTE, "main"], cwd=str(PYTORCH_DIR))
    except subprocess.CalledProcessError:
        # review/main might be protected or have diverged; log and continue
        log("WARN", "Could not sync review/main with upstream/main — PR may show extra commits",
            issue=tracked.source_number)
    try:
        _git(["checkout", "-b", branch, f"{REVIEW_REMOTE}/main"], cwd=str(PYTORCH_DIR))
    except subprocess.CalledProcessError:
        # Branch already exists — just checkout (do NOT reset, preserves prior commits)
        _git(["checkout", branch], cwd=str(PYTORCH_DIR))

    # Get issue details for prompt
    detail = gh.get_issue_detail(UPSTREAM_ISSUE_REPO, tracked.source_number)

    # Check if a prior run already produced changes (e.g. pipeline crashed after
    # opencode finished but before push/PR).  Skip re-running the agent.
    existing_diff = _git(["diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD"],
                         cwd=str(PYTORCH_DIR)).strip()
    if existing_diff:
        log("INFO", f"Branch {branch} already has changes vs {REVIEW_REMOTE}/main, "
                     "skipping agent re-run",
            issue=tracked.source_number)
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **Resuming pipeline** — branch `{branch}` already has changes from a "
            f"prior run, skipping to push + PR creation.",
        )
    else:
        body_section = _build_body_section(detail)
        prompt = IMPLEMENT_PROMPT_TEMPLATE.format(
            number=tracked.source_number,
            title=detail.get("title", ""),
            body_section=body_section,
        )

        # Announce that the agent is starting work
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **Agent starting implementation** (attempt {tracked.attempt_count})\n\n"
            f"Branch: `{branch}`\n"
            f"Working on: `{detail.get('title', 'unknown')}`\n\n"
            f"_Session ID will be posted shortly — you can attach to watch live._",
        )

        # Dispatch agent — post session ID to issue as soon as it's available
        from ..utils.notify import post_session_started

        def _post_session_id(sid: str):
            post_session_started(UPSTREAM_ISSUE_REPO, tracked.source_number,
                                 "Implementation", sid, str(PYTORCH_DIR))

        backend = get_backend()
        timeout = STAGE_TIMEOUTS.get("IMPLEMENTING", 1800)
        output, log_path, session_id = backend.run(prompt, workdir=str(PYTORCH_DIR),
                                        skill="xpu-ops-pr-creation", timeout=timeout,
                                        issue=tracked.source_number, stage="IMPLEMENTING",
                                        on_session_start=_post_session_id)
        log("INFO", f"Implementation agent log: {log_path}",
            issue=tracked.source_number)

        # Post log + session info to source issue
        session_info = ""
        if session_id:
            session_info = (
                f"\n\n**Debug / attach to this session:**\n"
                f"```bash\ncd {PYTORCH_DIR} && opencode -s {session_id}\n```\n"
                f"Session ID: `{session_id}`"
            )
        gh.add_issue_comment(
            UPSTREAM_ISSUE_REPO, tracked.source_number,
            f"🤖 **Attempt {tracked.attempt_count} completed** — log: `{log_path.name}`"
            f"{session_info}\n\n"
            f"<details><summary>Agent output (last 50 lines)</summary>\n\n"
            f"```\n{chr(10).join(output.strip().splitlines()[-50:]) if output.strip() else '(empty)'}\n```\n"
            f"</details>",
        )

    # Verify agent made changes (committed or uncommitted)
    # First check for uncommitted changes and auto-commit them
    status = _git(["status", "--porcelain"], cwd=str(PYTORCH_DIR)).strip()
    if status:
        # Only add tracked changes, exclude submodule pointers
        changed_files = [
            line.split(maxsplit=1)[1].strip()
            for line in status.split("\n")
            if not line.split(maxsplit=1)[1].strip().startswith("third_party/")
        ]
        if changed_files:
            _git(["add", "--"] + changed_files, cwd=str(PYTORCH_DIR))
            _git(["commit", "-m",
                  f"Fix for intel/torch-xpu-ops#{tracked.source_number}\n\n"
                  f"{detail.get('title', 'Agent fix')}"],
                 cwd=str(PYTORCH_DIR))

    # Now check if we have commits above review/main
    diff = _git(["diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD"], cwd=str(PYTORCH_DIR)).strip()
    if not diff:
        log("WARN", f"Agent produced no changes for #{tracked.source_number}, "
                     f"attempt {tracked.attempt_count}", issue=tracked.source_number)
        update_stage(tracked, "IMPLEMENTING",
                     f"Attempt {tracked.attempt_count}: agent produced no changes, will retry.")
        return

    # Squash all agent commits into one clean commit for a tidy PR
    commit_count = _git(["rev-list", "--count", f"{REVIEW_REMOTE}/main..HEAD"],
                        cwd=str(PYTORCH_DIR)).strip()
    if int(commit_count) > 1:
        log("INFO", f"Squashing {commit_count} commits into one", issue=tracked.source_number)
        _git(["reset", "--soft", f"{REVIEW_REMOTE}/main"], cwd=str(PYTORCH_DIR))
        _git(["commit", "-m",
              f"{detail.get('title', f'Fix for issue #{tracked.source_number}')}\n\n"
              f"Fixes intel/torch-xpu-ops#{tracked.source_number}"],
             cwd=str(PYTORCH_DIR))

    # Push to review remote
    try:
        _git(["push", REVIEW_REMOTE, branch], cwd=str(PYTORCH_DIR))
    except subprocess.CalledProcessError:
        # Branch may not exist yet or diverged from squash — force push is safe
        # since we haven't created the PR yet (or are updating pre-review)
        _git(["push", "--force-with-lease", "--set-upstream", REVIEW_REMOTE, branch],
             cwd=str(PYTORCH_DIR))

    # Get the push SHA
    sha = _git(["rev-parse", "HEAD"], cwd=str(PYTORCH_DIR)).strip()
    tracked.last_push_sha = sha

    # Build descriptive PR body from issue details + diff summary
    issue_url = f"https://github.com/{UPSTREAM_ISSUE_REPO}/issues/{tracked.source_number}"
    diff_stat = _git(["diff", "--stat", f"{REVIEW_REMOTE}/main..HEAD"], cwd=str(PYTORCH_DIR)).strip()
    diff_content = _git(["diff", f"{REVIEW_REMOTE}/main..HEAD"], cwd=str(PYTORCH_DIR)).strip()
    # Truncate diff for PR body (keep it readable)
    diff_lines = diff_content.split("\n")
    if len(diff_lines) > 100:
        diff_content = "\n".join(diff_lines[:100]) + "\n... (truncated)"

    pr_title = detail.get("title", f"Fix for issue #{tracked.source_number}")
    pr_body = (
        f"## Summary\n\n"
        f"Fix for [{UPSTREAM_ISSUE_REPO}#{tracked.source_number}]({issue_url})\n\n"
        f"**Issue:** {detail.get('title', 'N/A')}\n\n"
    )
    # Add triage reason if available
    if tracked.triage_reason:
        pr_body += f"**Root Cause:** {tracked.triage_reason}\n\n"
    # Add error context from issue body
    sections = _parse_issue_sections(detail.get("body", ""))
    if sections.get("Failed Tests"):
        pr_body += f"**Failed Tests:**\n{sections['Failed Tests']}\n\n"
    if sections.get("Failure Type"):
        pr_body += f"**Failure Type:** {sections['Failure Type']}\n\n"
    # Add diff summary
    pr_body += (
        f"## Changes\n\n"
        f"```\n{diff_stat}\n```\n\n"
        f"<details><summary>Full diff</summary>\n\n"
        f"```diff\n{diff_content}\n```\n\n"
        f"</details>\n"
    )
    try:
        pr = gh.create_draft_pr(
            PRIVATE_REVIEW_REPO, title=pr_title, body=pr_body,
            head=branch,
        )
    except subprocess.CalledProcessError:
        # PR may already exist for this branch — find and reuse it
        existing = gh.list_prs(PRIVATE_REVIEW_REPO, state="open",
                               search=f"head:{branch}")
        if existing:
            pr = existing[0]
            # Update the body to reflect the new push
            gh.update_pr_body(PRIVATE_REVIEW_REPO, pr["number"], pr_body)
        else:
            raise
    tracked.tracking_pr_number = pr.get("number")
    tracked.tracking_pr_url = pr.get("html_url", pr.get("url"))

    # Mark ready for review
    gh.mark_pr_ready(PRIVATE_REVIEW_REPO, tracked.tracking_pr_number)

    update_stage(tracked, "IN_REVIEW",
                 f"Implementation pushed to `{branch}`, PR created: {tracked.tracking_pr_url}")
    log("INFO", f"Implementation complete for #{tracked.source_number}, "
                 f"attempt {tracked.attempt_count}", issue=tracked.source_number)


def _git(args: list[str], cwd: str | None = None) -> str:
    """Run git command via shared helper. Returns stdout."""
    from ..utils.git import git as _git_impl
    result = _git_impl(*args, workdir=Path(cwd) if cwd else None)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    tracked = load_tracked(args.issue)
    run(tracked)


if __name__ == "__main__":
    main()
