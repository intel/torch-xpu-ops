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


# ---------------------------------------------------------------------------
# GitHub CLI helpers
# ---------------------------------------------------------------------------

def _token_for_repo(repo: str) -> str | None:
    """Pick the right GH token based on which repo we're accessing."""
    from .config import PRIVATE_REVIEW_REPO, PUBLIC_TARGET_REPO, ISSUE_REPO
    review_token = os.environ.get("REVIEW_GH_TOKEN")
    if review_token and repo in (PRIVATE_REVIEW_REPO, PUBLIC_TARGET_REPO, ISSUE_REPO):
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


def update_issue_body(repo: str, number: int, body: str) -> None:
    """Update an issue's body text."""
    _gh_api(f"/repos/{repo}/issues/{number}", method="PATCH",
            token=_token_for_repo(repo), body=body)


def add_issue_comment(repo: str, number: int, body: str) -> None:
    """Add a comment to an issue."""
    _gh_api(f"/repos/{repo}/issues/{number}/comments", method="POST",
            token=_token_for_repo(repo), body=body)


def add_label(repo: str, number: int, label: str) -> None:
    """Add a label to an issue. Creates the label if it doesn't exist."""
    try:
        _gh(
            ["issue", "edit", str(number), "--repo", repo, "--add-label", label],
            token=_token_for_repo(repo),
        )
    except subprocess.CalledProcessError:
        try:
            _gh_api(f"/repos/{repo}/labels", method="POST",
                    token=_token_for_repo(repo), name=label, color="c5def5")
        except subprocess.CalledProcessError:
            pass
        _gh(
            ["issue", "edit", str(number), "--repo", repo, "--add-label", label],
            token=_token_for_repo(repo),
        )


def remove_label(repo: str, number: int, label: str) -> None:
    """Remove a label from an issue."""
    _gh(
        ["issue", "edit", str(number), "--repo", repo, "--remove-label", label],
        token=_token_for_repo(repo),
    )
