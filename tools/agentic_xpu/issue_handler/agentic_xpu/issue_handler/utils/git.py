# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

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
    """Run a git command with logging.

    On non-zero exit (when ``check=True``), raises CalledProcessError with the
    full stderr embedded in the message so the orchestrator log shows *why*
    git failed instead of just the argv. Without this, errors like
    'checkout would overwrite local changes' are invisible.
    """
    cwd = str(workdir or PYTORCH_DIR)
    cmd_str = "git " + " ".join(args)
    log("INFO", f"$ {cmd_str}", issue=issue)
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=False,
        capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()
        stdout_tail = (result.stdout or "").strip()
        detail_parts = [f"rc={result.returncode}"]
        if stderr_tail:
            detail_parts.append(f"stderr: {stderr_tail}")
        if stdout_tail and not stderr_tail:
            detail_parts.append(f"stdout: {stdout_tail}")
        detail = " | ".join(detail_parts)
        log("ERROR", f"git failed: {cmd_str} ({detail})", issue=issue)
        raise subprocess.CalledProcessError(
            result.returncode, ["git", *args],
            output=result.stdout, stderr=result.stderr,
        )
    return result


def git_out(*args: str, **kwargs) -> str:
    """Run git and return stdout. Convenience wrapper around git()."""
    return git(*args, **kwargs).stdout


# ---------------------------------------------------------------------------
# GitHub CLI helpers
# ---------------------------------------------------------------------------

def _token_for_repo(repo: str) -> str | None:
    """Return the GH_TOKEN for API calls."""
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
        "--json", "number,title,body,labels,comments,url,state,createdAt,assignees",
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


def close_issue(repo: str, number: int) -> None:
    """Close an issue."""
    _gh_api(f"/repos/{repo}/issues/{number}", method="PATCH",
            token=_token_for_repo(repo), state="closed")


def assign_issue(repo: str, number: int, assignee: str) -> None:
    """Assign a user to an issue."""
    _gh_api(f"/repos/{repo}/issues/{number}/assignees", method="POST",
            token=_token_for_repo(repo), assignees=[assignee])


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


def add_and_commit(message: str, *, issue: int | None = None,
                   workdir: Path | None = None) -> bool:
    """Stage tracked files (excluding third_party/*) and commit if dirty.

    Returns True if a commit was made, False if tree was clean.
    """
    cwd = workdir or PYTORCH_DIR
    status = git("status", "--porcelain", workdir=cwd, issue=issue).stdout
    if not status.strip():
        return False

    # Filter out submodule pointer changes (third_party/*) and untracked files
    files = []
    for line in status.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        status_code = parts[0]
        # Skip untracked files to avoid committing generated artifacts
        if status_code == "??":
            continue
        fname = parts[1].strip()
        if " -> " in fname:
            fname = fname.split(" -> ", 1)[1]
        if fname.startswith("third_party/"):
            # Allow file edits inside third_party/torch-xpu-ops/ (actual
            # source changes).  Only skip bare submodule-pointer updates
            # (e.g. "third_party/torch-xpu-ops" with no deeper path).
            if "/" not in fname[len("third_party/"):]:
                log("INFO", f"Skipping submodule pointer change: {fname}",
                    issue=issue)
                continue
            log("INFO", f"Including third_party file change: {fname}",
                issue=issue)
        files.append(fname)

    if not files:
        return False

    git("add", "--", *files, workdir=cwd, issue=issue)
    git("commit", "-m", message, workdir=cwd, issue=issue)
    return True


# ---------------------------------------------------------------------------
# Label sync
# ---------------------------------------------------------------------------


def sync_labels(repo: str, number: int, stage: str) -> None:
    """Ensure issue labels match the current stage.

    Fetches current labels first so we only call remove_label on labels
    that are actually present — avoids swallowing real errors in the
    except-pass block masking a failed removal.
    """
    from .config import STAGE_TO_LABEL, ALL_AGENT_LABELS
    target_label = STAGE_TO_LABEL.get(stage)

    # Fetch current labels on the issue to avoid spurious remove failures
    try:
        detail = get_issue_detail(repo, number)
        current_labels = {lbl["name"] for lbl in (detail.get("labels") or [])}
    except Exception:
        current_labels = None  # If we can't fetch, fall back to try-remove

    for label in ALL_AGENT_LABELS:
        if label == target_label:
            add_label(repo, number, label)
        else:
            # Only remove if label is confirmed present (or unknown)
            if current_labels is not None and label not in current_labels:
                continue
            try:
                remove_label(repo, number, label)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GitHub CLI helpers
# ---------------------------------------------------------------------------


def create_draft_pr(repo: str, title: str, body: str, head: str,
                    base: str = "main") -> dict:
    """Create a draft PR via gh api."""
    return _gh_api(
        f"/repos/{repo}/pulls", method="POST",
        token=_token_for_repo(repo), title=title, body=body,
        head=head, base=base, draft=True,
    )


def mark_pr_ready(repo: str, pr_number: int) -> None:
    """Mark a draft PR as ready for review."""
    _gh(["pr", "ready", str(pr_number), "--repo", repo],
        token=_token_for_repo(repo))


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


def update_pr_body(repo: str, pr_number: int, body: str) -> None:
    """Update a PR's body."""
    _gh_api(f"/repos/{repo}/pulls/{pr_number}", method="PATCH",
            token=_token_for_repo(repo), body=body)
