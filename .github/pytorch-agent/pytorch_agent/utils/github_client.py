"""Thin wrappers around `gh` CLI for GitHub API operations."""
import json
import os
import subprocess
from typing import Any


def _token_for_repo(repo: str) -> str | None:
    """Pick the right GH token based on which repo we're accessing.

    REVIEW_GH_TOKEN → for PRIVATE_REVIEW_REPO operations
    GH_TOKEN → for everything else (upstream issues, public PRs)
    """
    review_repo = os.environ.get("PRIVATE_REVIEW_REPO", "")
    review_token = os.environ.get("REVIEW_GH_TOKEN")
    if review_token and repo == review_repo:
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
    """Call gh api, return parsed JSON.

    Type dispatch: strings use -f, others use -F with JSON encoding.
    """
    cmd = ["api", endpoint, "--method", method]
    for k, v in fields.items():
        if isinstance(v, str):
            cmd += ["-f", f"{k}={v}"]
        else:
            cmd += ["-F", f"{k}={json.dumps(v)}"]
    raw = _gh(cmd, token=token)
    return json.loads(raw) if raw.strip() else {}


# ---------------------------------------------------------------------------
# Issues (intel/torch-xpu-ops)
# ---------------------------------------------------------------------------

def get_issues(repo: str, label: str) -> list[dict]:
    """List open issues with a given label."""
    raw = _gh([
        "issue", "list", "--repo", repo, "--label", label,
        "--state", "open", "--json",
        "number,title,labels,body,createdAt,url", "--limit", "100",
    ])
    return json.loads(raw) if raw.strip() else []


def get_issue_detail(repo: str, number: int) -> dict:
    """Get full issue details."""
    raw = _gh([
        "issue", "view", str(number), "--repo", repo,
        "--json", "number,title,body,labels,comments,url,state,createdAt",
    ])
    return json.loads(raw)


def add_issue_comment(repo: str, number: int, body: str) -> None:
    """Add a comment to an issue."""
    _gh(["issue", "comment", str(number), "--repo", repo, "--body", body])


def add_pr_comment(repo: str, number: int, body: str) -> dict:
    """Add a comment to a PR (uses correct token for repo)."""
    return _gh_api(f"/repos/{repo}/issues/{number}/comments",
                   method="POST", token=_token_for_repo(repo), body=body)


def update_pr_comment(repo: str, comment_id: int, body: str) -> None:
    """Update an existing PR comment (uses correct token for repo)."""
    _gh_api(f"/repos/{repo}/issues/comments/{comment_id}",
            method="PATCH", token=_token_for_repo(repo), body=body)


def close_issue(repo: str, number: int) -> None:
    """Close an issue."""
    _gh(["issue", "close", str(number), "--repo", repo])


def add_label(repo: str, number: int, label: str) -> None:
    """Add a label to an issue. Creates the label if it doesn't exist."""
    try:
        _gh(["issue", "edit", str(number), "--repo", repo, "--add-label", label])
    except subprocess.CalledProcessError:
        # Label may not exist — create it and retry
        try:
            _gh_api(f"/repos/{repo}/labels", method="POST",
                    name=label, color="c5def5")
        except subprocess.CalledProcessError:
            pass  # Label may already exist on repo but issue edit failed for other reason
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
# Pull Requests (PRIVATE_REVIEW_REPO, PUBLIC_TARGET_REPO)
# ---------------------------------------------------------------------------

def create_draft_pr(repo: str, title: str, body: str, head: str,
                    base: str = "main") -> dict:
    """Create a draft PR via gh api."""
    token = _token_for_repo(repo)
    return _gh_api(
        f"/repos/{repo}/pulls", method="POST",
        token=token, title=title, body=body, head=head, base=base, draft=True,
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
        pass  # Branch may already be deleted
