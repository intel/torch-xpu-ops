"""Git and GitHub operations — single file for all git + gh CLI wrappers."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import PYTORCH_DIR
from .logger import log


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def git(*args: str, workdir: Path | None = None, check: bool = True,
        issue: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command with logging."""
    cwd = str(workdir or PYTORCH_DIR)
    cmd_str = "git " + " ".join(args)
    log("INFO", f"$ {cmd_str}", issue=issue)
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check,
        capture_output=True, text=True,
    )


def git_out(*args: str, **kwargs) -> str:
    """Run git and return stdout. Convenience wrapper around git()."""
    return git(*args, **kwargs).stdout


def add_and_commit(message: str, *, issue: int | None = None,
                   workdir: Path | None = None) -> bool:
    """Stage tracked files (excluding third_party/*) and commit if dirty.

    Returns True if a commit was made, False if tree was clean.
    """
    cwd = workdir or PYTORCH_DIR
    status = git("status", "--porcelain", workdir=cwd, issue=issue).stdout
    if not status.strip():
        return False

    # Filter out submodule pointer changes (third_party/*)
    files = []
    for line in status.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        fname = parts[1].strip()
        if " -> " in fname:
            fname = fname.split(" -> ", 1)[1]
        if fname.startswith("third_party/"):
            log("INFO", f"Skipping submodule change: {fname}", issue=issue)
            continue
        files.append(fname)

    if not files:
        return False

    git("add", "--", *files, workdir=cwd, issue=issue)
    git("commit", "-m", message, workdir=cwd, issue=issue)
    return True


# ---------------------------------------------------------------------------
# GitHub CLI helpers
# ---------------------------------------------------------------------------

def _token_for_repo(repo: str) -> str | None:
    """Pick the right GH token based on which repo we're accessing.

    REVIEW_GH_TOKEN → for PRIVATE_REVIEW_REPO, PUBLIC_TARGET_REPO, and ISSUE_REPO
    GH_TOKEN → for everything else (upstream issues)
    """
    review_repo = os.environ.get("PRIVATE_REVIEW_REPO", "")
    public_target = os.environ.get("PUBLIC_TARGET_REPO", "")
    issue_repo = os.environ.get("ISSUE_REPO", "")
    review_token = os.environ.get("REVIEW_GH_TOKEN")
    if review_token and repo in (review_repo, public_target, issue_repo):
        return review_token
    return os.environ.get("GH_TOKEN")


def _gh(args: list[str], input_text: str | None = None,
        token: str | None = None) -> str:
    """Run gh command, return stdout."""
    env = None
    gh_token = token or os.environ.get("GH_TOKEN")
    if gh_token:
        env = {**os.environ, "GH_TOKEN": gh_token}
    result = subprocess.run(
        ["gh"] + args, capture_output=True, text=True,
        input=input_text, env=env,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["gh"] + args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def _gh_api(endpoint: str, method: str = "GET",
            token: str | None = None, **fields: Any) -> dict | list:
    """Call gh api, return parsed JSON."""
    cmd = ["api", endpoint, "--method", method]
    for k, v in fields.items():
        if isinstance(v, str):
            cmd += ["-f", f"{k}={v}"]
        else:
            cmd += ["-F", f"{k}={json.dumps(v)}"]
    raw = _gh(cmd, token=token)
    return json.loads(raw) if raw.strip() else {}


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

def get_issues(repo: str, label: str) -> list[dict]:
    """List open issues with a given label."""
    raw = _gh([
        "issue", "list", "--repo", repo, "--label", label,
        "--state", "open", "--json",
        "number,title,labels,body,createdAt,url", "--limit", "100",
    ], token=_token_for_repo(repo))
    return json.loads(raw) if raw.strip() else []


def get_issue_detail(repo: str, number: int) -> dict:
    """Get full issue details."""
    raw = _gh([
        "issue", "view", str(number), "--repo", repo,
        "--json", "number,title,body,labels,comments,url,state,createdAt",
    ], token=_token_for_repo(repo))
    return json.loads(raw)


def add_issue_comment(repo: str, number: int, body: str) -> None:
    """Add a comment to an issue."""
    _gh(["issue", "comment", str(number), "--repo", repo, "--body", body])


def close_issue(repo: str, number: int) -> None:
    """Close an issue."""
    _gh(["issue", "close", str(number), "--repo", repo])


def update_issue_body(repo: str, number: int, body: str) -> None:
    """Update an issue's body text."""
    _gh_api(f"/repos/{repo}/issues/{number}", method="PATCH",
            token=_token_for_repo(repo), body=body)


def add_label(repo: str, number: int, label: str) -> None:
    """Add a label to an issue. Creates the label if it doesn't exist."""
    try:
        _gh(["issue", "edit", str(number), "--repo", repo, "--add-label", label])
    except subprocess.CalledProcessError:
        try:
            _gh_api(f"/repos/{repo}/labels", method="POST",
                    name=label, color="c5def5")
        except subprocess.CalledProcessError:
            pass
        _gh(["issue", "edit", str(number), "--repo", repo, "--add-label", label])


def remove_label(repo: str, number: int, label: str) -> None:
    """Remove a label from an issue."""
    _gh(["issue", "edit", str(number), "--repo", repo, "--remove-label", label])


def get_issue_comments(repo: str, number: int) -> list[dict]:
    """Get all comments on an issue."""
    return _gh_api(f"/repos/{repo}/issues/{number}/comments")


def update_issue_comment(repo: str, comment_id: int, body: str) -> None:
    """Update an existing issue comment."""
    _gh_api(f"/repos/{repo}/issues/comments/{comment_id}",
            method="PATCH", body=body)


# ---------------------------------------------------------------------------
# Pull Requests
# ---------------------------------------------------------------------------

def add_pr_comment(repo: str, number: int, body: str) -> dict:
    """Add a comment to a PR."""
    return _gh_api(f"/repos/{repo}/issues/{number}/comments",
                   method="POST", token=_token_for_repo(repo), body=body)


def update_pr_comment(repo: str, comment_id: int, body: str) -> None:
    """Update an existing PR comment."""
    _gh_api(f"/repos/{repo}/issues/comments/{comment_id}",
            method="PATCH", token=_token_for_repo(repo), body=body)


def create_draft_pr(repo: str, title: str, body: str, head: str,
                    base: str = "main") -> dict:
    """Create a draft PR via gh api."""
    return _gh_api(
        f"/repos/{repo}/pulls", method="POST",
        token=_token_for_repo(repo), title=title, body=body,
        head=head, base=base, draft=True,
    )


def update_pr_body(repo: str, pr_number: int, body: str) -> None:
    """Update a PR's body."""
    _gh_api(f"/repos/{repo}/pulls/{pr_number}", method="PATCH",
            token=_token_for_repo(repo), body=body)


def mark_pr_ready(repo: str, pr_number: int) -> None:
    """Mark a draft PR as ready for review."""
    _gh(["pr", "ready", str(pr_number), "--repo", repo],
        token=_token_for_repo(repo))


def get_pr_reviews(repo: str, pr_number: int) -> list[dict]:
    """Get all reviews on a PR."""
    return _gh_api(f"/repos/{repo}/pulls/{pr_number}/reviews",
                   token=_token_for_repo(repo))


def get_pr_review_comments(repo: str, pr_number: int) -> list[dict]:
    """Get all review comments on a PR."""
    return _gh_api(f"/repos/{repo}/pulls/{pr_number}/comments",
                   token=_token_for_repo(repo))


def get_pr_status(repo: str, pr_number: int) -> str:
    """Get PR merge status: 'open', 'closed', or 'merged'."""
    pr = _gh_api(f"/repos/{repo}/pulls/{pr_number}",
                 token=_token_for_repo(repo))
    if pr.get("merged"):
        return "merged"
    return pr.get("state", "unknown")


def create_cross_fork_pr(head_repo: str, head_branch: str,
                         base_repo: str, title: str, body: str,
                         base_branch: str = "main") -> dict:
    """Create a cross-fork PR (e.g. reviewfork:branch → pytorch/pytorch:main)."""
    owner = head_repo.split("/")[0]
    return _gh_api(
        f"/repos/{base_repo}/pulls", method="POST",
        token=_token_for_repo(base_repo), title=title, body=body,
        head=f"{owner}:{head_branch}", base=base_branch,
    )


def list_prs(repo: str, state: str = "open",
             search: str | None = None) -> list[dict]:
    """List PRs with optional search filter."""
    cmd = [
        "pr", "list", "--repo", repo, "--state", state,
        "--json", "number,title,body,url,state,isDraft,headRefName",
        "--limit", "100",
    ]
    if search:
        cmd += ["--search", search]
    raw = _gh(cmd, token=_token_for_repo(repo))
    return json.loads(raw) if raw.strip() else []


# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------

def get_ci_checks(repo: str, pr_number: int) -> list[dict]:
    """Get CI check runs for a PR's head commit."""
    token = _token_for_repo(repo)
    pr = _gh_api(f"/repos/{repo}/pulls/{pr_number}", token=token)
    sha = pr.get("head", {}).get("sha")
    if not sha:
        return []
    result = _gh_api(f"/repos/{repo}/commits/{sha}/check-runs", token=token)
    return result.get("check_runs", [])


def delete_branch(repo: str, branch: str) -> None:
    """Delete a branch from a remote repo."""
    try:
        _gh_api(f"/repos/{repo}/git/refs/heads/{branch}", method="DELETE",
                token=_token_for_repo(repo))
    except subprocess.CalledProcessError:
        pass


# ---------------------------------------------------------------------------
# PR body builder (merged from _issue_format.py)
# ---------------------------------------------------------------------------

def build_pr_body(
    *,
    upstream_issue_repo: str,
    source_number: int,
    title: str,
    triage_reason: str | None,
    issue_body: str,
    include_diff_stat: bool = False,
    diff_stat: str = "",
    reviewer: str = "",
) -> str:
    """Build a PR description from issue details."""
    from .issue_body import parse_sections

    issue_url = f"https://github.com/{upstream_issue_repo}/issues/{source_number}"

    body = (
        f"## Summary\n\n"
        f"Fix for [{upstream_issue_repo}#{source_number}]({issue_url})\n\n"
        f"**Issue:** {title}\n\n"
    )
    if triage_reason:
        body += f"**Root Cause:** {triage_reason}\n\n"

    sections = parse_sections(issue_body)
    if sections.get("Failed Tests"):
        body += f"**Failed Tests:**\n{sections['Failed Tests']}\n\n"
    if sections.get("Failure Type"):
        body += f"---\n\n**Failure Type:** {sections['Failure Type']}\n\n"

    if include_diff_stat and diff_stat:
        body += f"---\n\n**Diff stat:**\n```\n{diff_stat}\n```\n\n"

    if reviewer:
        body += f"---\n\ncc @{reviewer}\n"

    return body
