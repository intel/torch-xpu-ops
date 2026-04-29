#!/usr/bin/env python3
"""Create a tracking issue in torch-xpu-ops from CI failure results.

Reads the JSON output from check_nightly_status.py and creates a structured
GitHub issue containing:
  - Commit scope (last PASS → first FAIL) with compare link
  - Suspect commits within the scope
  - NEW / EXISTING / FIXED failure breakdown
  - Push runs timeline for manual bisect reference
"""
import os
import sys
import json
import argparse
import requests
from datetime import datetime, timezone

# ============================================================================
# Global Configuration
# ============================================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# Repository where tracking issues are created
TRACKING_REPO = os.environ.get("TRACKING_REPO", "intel/torch-xpu-ops")

# PyTorch main repo (for Compare API to determine commit scope)
PYTORCH_REPO = "pytorch/pytorch"


# ============================================================================
# Commit Scope Functions
# ============================================================================

def get_commit_count(base_sha, head_sha):
    """Get the number of commits between two SHAs via GitHub Compare API.

    Args:
        base_sha: Starting commit (last PASS)
        head_sha: Ending commit (first FAIL)

    Returns:
        int or "?": Number of commits, "?" on API failure
    """
    url = f"https://api.github.com/repos/{PYTORCH_REPO}/compare/{base_sha}...{head_sha}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    if resp.status_code == 200:
        return resp.json().get("total_commits", "?")
    return "?"


def find_commit_scope(all_runs_data, current_commit):
    """Determine commit scope from push runs timeline.

    Walks through push runs in chronological order to find:
      - last_pass: most recent PASS commit
      - first_fail: first FAIL commit after that PASS

    The guilty commit must be between these two boundaries.

    Args:
        all_runs_data: Contents of all_runs.json (contains push_runs array)
        current_commit: Current failure commit SHA

    Returns:
        (last_pass_sha, first_fail_sha, commit_count)
    """
    push_runs = all_runs_data.get("push_runs", [])
    if not push_runs:
        return None, current_commit, "?"

    # Sort chronologically (API returns newest first)
    runs_chrono = sorted(push_runs, key=lambda r: r.get("created_at", ""))

    last_pass = None
    first_fail = None
    for run in runs_chrono:
        conclusion = run.get("conclusion", "")
        sha = run.get("head_sha", "")
        if conclusion == "success":
            last_pass = sha
            first_fail = None  # Reset: we want the first FAIL after this PASS
        elif conclusion != "success" and conclusion not in ("skipped", None, "") and first_fail is None:
            first_fail = sha

    if not first_fail:
        first_fail = current_commit
    if not last_pass:
        return None, first_fail, "?"

    commit_count = get_commit_count(last_pass, first_fail)
    return last_pass, first_fail, commit_count


def get_suspect_commits_in_scope(last_pass, first_fail, test_files):
    """Find commits in scope that may be related to the failures.

    Uses GitHub Compare API to list all commits in scope, then filters
    out obviously irrelevant ones (doc-only, typo, lint, etc.).

    Args:
        last_pass: Last PASS commit SHA
        first_fail: First FAIL commit SHA
        test_files: List of failing test file paths (for relevance hints)

    Returns:
        list: Up to 10 suspect commits, each with sha, message, author
    """
    url = f"https://api.github.com/repos/{PYTORCH_REPO}/compare/{last_pass}...{first_fail}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    if resp.status_code != 200:
        return []

    data = resp.json()
    commits = data.get("commits", [])

    suspects = []
    for commit in commits:
        sha = commit["sha"]
        msg = commit["commit"]["message"].split("\n")[0][:120]
        author = (commit.get("author", {}).get("login", "unknown")
                  if commit.get("author") else "unknown")

        # Skip obviously irrelevant commits
        msg_lower = msg.lower()
        if any(kw in msg_lower for kw in ["typo", "noqa", "lint", "flake8", "doc:", "docs:"]):
            continue

        suspects.append({
            "sha": sha,
            "message": msg,
            "author": author,
        })

    return suspects[:10]


# ============================================================================
# Issue Body Formatting
# ============================================================================

def format_issue_body(data, all_runs_data=None):
    """Format CI failure data into a GitHub issue body.

    Structure:
    1. Header: date, status
    2. Commit Scope: last PASS → first FAIL with compare link
    3. Suspect Commits: candidates within scope
    4. NEW Failures: tests that just started failing
    5. EXISTING Failures: tests that were already failing
    6. FIXED: tests that stopped failing
    7. Push Runs Timeline: for manual bisect reference

    Args:
        data: ci_results.json contents from check_nightly_status.py
        all_runs_data: all_runs.json contents (push runs for commit scope)
    """
    from collections import OrderedDict

    commit = data.get("commit_sha", "unknown")[:12]
    full_commit = data.get("commit_sha", "")
    failures = data.get("failures", [])
    unique_tests = data.get("unique_failed_tests", [])
    new_tests = data.get("new_failed_tests", [])
    existing_tests = data.get("existing_failed_tests", [])
    fixed_tests = data.get("fixed_tests", [])

    n_new = len(new_tests)
    n_existing = len(existing_tests)
    n_fixed = len(fixed_tests)

    # --- Compute commit scope ---
    last_pass_sha = None
    first_fail_sha = full_commit
    commit_count = "?"
    suspects = []

    if all_runs_data:
        last_pass_sha, first_fail_sha, commit_count = find_commit_scope(
            all_runs_data, full_commit)

        # Find suspect commits if we have a scope and new failures
        if last_pass_sha and new_tests:
            test_files = list(OrderedDict.fromkeys(
                t.split("::")[0] for t in new_tests
            ))
            suspects = get_suspect_commits_in_scope(
                last_pass_sha, first_fail_sha, test_files)

    last_pass_short = last_pass_sha[:12] if last_pass_sha else "unknown"
    first_fail_short = first_fail_sha[:12] if first_fail_sha else commit

    # --- Assemble issue body ---

    # 1. Header
    lines = [
        "## XPU CI Nightly Status Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Status:** {'ALL PASS' if data.get('status') == 'ALL_PASS' else 'HAS FAILURES'}",
        "",
    ]

    # 2. Commit Scope table
    lines.extend([
        "### Commit Scope",
        "",
        "| | Commit | Link |",
        "|---|--------|------|",
    ])
    if last_pass_sha:
        lines.append(
            f"| ✅ Last PASS | `{last_pass_short}` | "
            f"[view](https://github.com/{PYTORCH_REPO}/commit/{last_pass_sha}) |"
        )
    lines.append(
        f"| ❌ First FAIL | `{first_fail_short}` | "
        f"[view](https://github.com/{PYTORCH_REPO}/commit/{first_fail_sha}) |"
    )
    if last_pass_sha:
        compare_url = f"https://github.com/{PYTORCH_REPO}/compare/{last_pass_sha}...{first_fail_sha}"
        lines.extend([
            "",
            f"**Commits in scope:** {commit_count} "
            f"([compare]({compare_url}))",
        ])

    lines.extend([
        "",
        f"**Failed Jobs:** {len(failures)}",
        f"**Total Failed Tests:** {len(unique_tests)}",
    ])
    if n_new > 0 or n_existing > 0:
        lines.append(f"**New Failures:** {n_new} | **Existing:** {n_existing} | **Fixed:** {n_fixed}")
    lines.extend(["", "---", ""])

    # Early return if all tests passed
    if not unique_tests:
        lines.append("All XPU tests passed! No action needed.")
        return "\n".join(lines)

    # 3. Suspect Commits
    if suspects:
        lines.append(f"### Suspect Commits ({len(suspects)} candidates in scope)")
        lines.append("")
        lines.append("| # | Commit | Author | Message |")
        lines.append("|---|--------|--------|---------|")
        for i, sc in enumerate(suspects, 1):
            sha_short = sc["sha"][:12]
            lines.append(
                f"| {i} | [`{sha_short}`](https://github.com/{PYTORCH_REPO}/commit/{sc['sha']}) "
                f"| @{sc['author']} | {sc['message']} |"
            )
        lines.extend(["", "---", ""])

    # --- Helper functions ---

    def group_by_file(test_list):
        """Group test IDs by their test file path."""
        groups = OrderedDict()
        for t in test_list:
            parts = t.split("::")
            f = parts[0] if parts else t
            if f not in groups:
                groups[f] = []
            groups[f].append(t)
        return groups

    def find_job(test_id):
        """Find which job (shard) a test belongs to."""
        for f in failures:
            if test_id in f.get("failed_tests", []):
                jn = f.get("job_name", "unknown")
                shard = "?"
                if "test (default," in jn:
                    shard = jn.split("test (default,")[1].split(",")[0].strip()
                return f.get("job_url", ""), jn, shard
        return "", "", "?"

    # 4. NEW Failures (highest priority)
    if new_tests:
        new_groups = group_by_file(new_tests)
        lines.append(f"### NEW Failures ({n_new} tests in {len(new_groups)} file(s))")
        lines.append("")
        # Summary table
        lines.append("| # | Test File | Count | Shard | Job |")
        lines.append("|---|-----------|:-----:|-------|-----|")
        for i, (test_file, tests) in enumerate(new_groups.items(), 1):
            short = test_file.split("/")[-1]
            url, jn, shard = find_job(tests[0])
            job_link = f"[shard {shard}]({url})" if url else f"shard {shard}"
            lines.append(f"| {i} | `{short}` | {len(tests)} | {job_link} | `{jn}` |")
        lines.append("")

        # Detailed test cases per file (collapsible)
        for i, (test_file, tests) in enumerate(new_groups.items(), 1):
            short = test_file.split("/")[-1]
            lines.append("<details>")
            lines.append(f"<summary><b>{i}. {short}</b> ({len(tests)} failures)</summary>")
            lines.append("")
            for t in tests:
                name = t.split("::")[-1]
                lines.append(f"- `{name}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # 5. EXISTING Failures (lower priority)
    if existing_tests:
        lines.append(f"### Existing Failures ({n_existing} tests, also failed in previous run)")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Show existing failures</summary>")
        lines.append("")
        for t in existing_tests:
            name = t.split("::")[-1]
            lines.append(f"- `{name}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # 6. FIXED (good news)
    if fixed_tests:
        lines.append(f"### Fixed ({n_fixed} tests, was failing, now passing)")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Show fixed tests</summary>")
        lines.append("")
        for t in fixed_tests:
            name = t.split("::")[-1]
            lines.append(f"- `{name}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # 7. Fallback: no new/existing classification (single run, no comparison)
    if not new_tests and not existing_tests:
        all_groups = group_by_file(unique_tests)
        lines.append(f"### All Failures ({len(unique_tests)} tests in {len(all_groups)} file(s))")
        lines.append("")
        for i, (test_file, tests) in enumerate(all_groups.items(), 1):
            short = test_file.split("/")[-1]
            lines.append("<details>")
            lines.append(f"<summary><b>{i}. {short}</b> ({len(tests)} failures)</summary>")
            lines.append("")
            for t in tests:
                name = t.split("::")[-1]
                lines.append(f"- `{name}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # 8. Footer
    lines.extend([
        "---",
        f"*Auto-generated by [CI Monitor](https://github.com/{TRACKING_REPO})*",
    ])

    return "\n".join(lines)


def format_push_runs_timeline(all_runs_data):
    """Format push runs as a timeline table appended to the issue body.

    Shows each push run's time, commit, PASS/FAIL status for manual
    confirmation of commit scope accuracy.
    """
    push_runs = all_runs_data.get("push_runs", [])
    if not push_runs:
        return ""

    runs = sorted(push_runs, key=lambda r: r.get("created_at", ""))

    lines = [
        "### Push Runs Timeline (Bisect Reference)",
        "",
        "| Time (UTC) | Commit | Status | Link |",
        "|------------|--------|--------|------|",
    ]
    for run in runs:
        sha = run.get("head_sha", "unknown")[:12]
        conclusion = run.get("conclusion") or ""
        icon = "✅" if conclusion == "success" else "❌" if conclusion == "failure" else "❓"
        status = conclusion.upper() if conclusion else "?"
        created = run.get("created_at", "?")
        url = run.get("html_url", "")
        lines.append(f"| {created} | `{sha}` | {icon} {status} | [link]({url}) |")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Issue CRUD Operations
# ============================================================================

def check_existing_issue(commit_short):
    """Check if an issue already exists for this commit (avoid duplicates).

    Searches TRACKING_REPO for open issues with label 'pytorch-ci-failure'
    whose title contains the short SHA.

    Returns:
        int or None: Existing issue number, or None if not found
    """
    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues"
    params = {"state": "open", "labels": "pytorch-ci-failure", "per_page": 100}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    if resp.status_code == 200:
        for issue in resp.json():
            if commit_short in issue["title"]:
                return issue["number"]
    return None


def create_issue(title, body, labels=None):
    """Create a GitHub issue via API.

    Returns:
        dict or None: Created issue object, or None on failure
    """
    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues"
    payload = {"title": title, "body": body, "labels": labels or []}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code == 201:
        issue = resp.json()
        print(f"Issue created: {issue['html_url']}")
        return issue
    else:
        print(f"Failed to create issue: {resp.status_code} {resp.text[:200]}")
        return None


def update_issue(issue_number, body):
    """Update an existing issue's body (same commit → update, not duplicate)."""
    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues/{issue_number}"
    payload = {"body": body}
    resp = requests.patch(url, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code == 200:
        print(f"Issue #{issue_number} updated")
        return resp.json()
    else:
        print(f"Failed to update issue: {resp.status_code}")
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create tracking issue from CI results")

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON from check_nightly_status.py")
    parser.add_argument("--all-runs", type=str, default=None,
                        help="all_runs.json for push runs timeline and commit scope")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print issue body without creating")

    args = parser.parse_args()

    if not GITHUB_TOKEN and not args.dry_run:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    # Read CI results JSON
    with open(args.input) as f:
        data = json.load(f)

    # Read all_runs.json if provided
    all_runs_data = {}
    if args.all_runs:
        with open(args.all_runs) as f:
            all_runs_data = json.load(f)

    # Generate issue body
    body = format_issue_body(data, all_runs_data)

    # Append push runs timeline
    timeline = format_push_runs_timeline(all_runs_data)
    if timeline:
        body = body + "\n\n" + timeline

    # Generate issue title
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit_short = data.get("commit_sha", "unknown")[:12]
    n_failures = len(data.get("unique_failed_tests", []))
    n_new = len(data.get("new_failed_tests", []))
    n_existing = len(data.get("existing_failed_tests", []))

    if data.get("status") == "ALL_PASS":
        title = f"[PyTorch CI] {date_str} - ALL PASS ({commit_short})"
    else:
        if n_new > 0 or n_existing > 0:
            title = f"[PyTorch CI] {date_str} - {n_new} new, {n_existing} existing failure(s) ({commit_short})"
        else:
            title = f"[PyTorch CI] {date_str} - {n_failures} failure(s) ({commit_short})"

    # Dry-run: print without creating
    if args.dry_run:
        print(f"=== TITLE ===\n{title}\n")
        print(f"=== BODY ===\n{body}")
        return

    # Check for existing issue with same commit
    existing = check_existing_issue(commit_short)
    issue_number = None
    if existing:
        print(f"Updating existing issue #{existing} (same commit {commit_short})")
        update_issue(existing, body)
        issue_number = existing
    else:
        labels = ["pytorch-ci-failure", "agent:blocked", "ai_generated"]
        if data.get("status") != "ALL_PASS":
            labels.append("has-failures")
        if n_new > 0:
            labels.append("new-failures")
        issue = create_issue(title, body, labels)
        if issue:
            issue_number = issue["number"]

    # Set GitHub Actions output for downstream steps
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output and issue_number:
        with open(gh_output, "a") as f:
            f.write(f"issue_number={issue_number}\n")


if __name__ == "__main__":
    main()
