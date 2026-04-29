#!/usr/bin/env python3
"""Create sub-issues for CI failure groups.

Reads the CI results JSON, groups failures by test file,
and creates a sub-issue per group linked to the summary issue.

Classification is fact-based only at this stage:
  - NEW_FAILURE: first time failing (not in previous run)
  - EXISTING_FAILURE: continuing to fail (also failed in previous run)

Root cause analysis happens in later steps after reproduction.
"""
import os
import sys
import json
import re
import argparse
import requests
from collections import OrderedDict

# ============================================================================
# Global Configuration
# ============================================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
PYTORCH_REPO = "pytorch/pytorch"
TRACKING_REPO = os.environ.get("TRACKING_REPO", "intel/torch-xpu-ops")


# ============================================================================
# Sub-Issue Creation
# ============================================================================

def create_sub_issue(summary_issue_num, group_num, test_file, test_names,
                     is_new, commit_sha="unknown"):
    """Create a sub-issue for a failure group.

    Args:
        summary_issue_num: Parent summary issue number
        group_num: Sequential group number
        test_file: Test file path (e.g. test/inductor/test_flex_attention.py)
        test_names: List of full test IDs
        is_new: True if NEW_FAILURE, False if EXISTING_FAILURE
        commit_sha: PyTorch commit SHA
    """
    short_file = test_file.split("/")[-1] if "/" in test_file else test_file
    count = len(test_names)
    tag = "NEW" if is_new else "EXISTING"

    title = f"[PyTorch CI] [{tag}] {short_file} ({count} failures) - Ref #{summary_issue_num}"

    # Extract test method names
    test_method_names = []
    for t in test_names:
        parts = t.split("::")
        test_method_names.append(parts[-1] if len(parts) > 1 else t)

    body_lines = [
        f"## Failure Group #{group_num}",
        "",
        f"**Summary Issue:** #{summary_issue_num}",
        f"**Test File:** `{test_file}`",
        f"**Failed Tests:** {count}",
        f"**Type:** `{tag}_FAILURE`",
        f"**PyTorch Commit:** [`{commit_sha[:12]}`](https://github.com/{PYTORCH_REPO}/commit/{commit_sha})",
        "",
        "---",
        "",
        "### Failed Tests",
        "",
    ]

    if count > 20:
        body_lines.append("<details>")
        body_lines.append(f"<summary>Show all {count} tests</summary>")
        body_lines.append("")
    for t in test_names:
        name = t.split("::")[-1]
        body_lines.append(f"- `{name}`")
    if count > 20:
        body_lines.append("")
        body_lines.append("</details>")
    body_lines.append("")

    # Action items checklist
    body_lines.extend([
        "### Action Items",
        "",
        "- [ ] Reproduce on dev machine",
        "- [ ] Identify root cause",
        "- [ ] Implement fix",
        "- [ ] Verify fix locally",
        "- [ ] PR submitted to pytorch/pytorch",
        "",
    ])

    # Machine-readable reproduce instructions (for automation)
    repro_data = {
        "commit_sha": commit_sha,
        "test_file": test_file,
        "test_names": test_method_names[:20],
        "failure_type": f"{tag}_FAILURE",
        "repro_commands": [
            f"git fetch origin && git checkout {commit_sha}",
            "pip install -e . -v --no-build-isolation",
        ] + [
            f"python {test_file} -k {name} 2>&1 | tail -80"
            for name in test_method_names[:5]
        ],
    }
    body_lines.extend([
        "---",
        "",
        "<!-- REPRO_START -->",
        "```json",
        json.dumps(repro_data, indent=2),
        "```",
        "<!-- REPRO_END -->",
    ])

    body = "\n".join(body_lines)

    labels = ["pytorch-ci-failure", "needs-repro"]
    if is_new:
        labels.append("new-failure")

    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues"
    payload = {"title": title, "body": body, "labels": labels}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code == 201:
        issue = resp.json()
        print(f"  Sub-issue created: {issue['html_url']}")
        return issue
    else:
        print(f"  Failed to create sub-issue: {resp.status_code} {resp.text[:200]}")
        return None


def group_by_file(tests):
    """Group test IDs by test file path."""
    groups = OrderedDict()
    for t in tests:
        f = t.split("::")[0] if "::" in t else t
        if f not in groups:
            groups[f] = []
        groups[f].append(t)
    return groups


# ============================================================================
# Auto-Close Fixed Sub-Issues
# ============================================================================

def close_fixed_sub_issues(fixed_tests, commit_short, dry_run=False):
    """Close sub-issues whose failures are all now fixed.

    Logic:
      1. Search all open pytorch-ci-failure sub-issues
         (sub-issue title contains [NEW] or [EXISTING])
      2. Extract test list from REPRO_START JSON block in body
      3. If all tests in sub-issue are in fixed_tests -> auto-close
      4. If partially fixed -> add comment noting which tests resolved

    Args:
        fixed_tests: List of test IDs marked as fixed in current scan
        commit_short: Current commit short SHA
        dry_run: Print only, don't modify issues
    """
    if not fixed_tests:
        print("\nNo fixed tests, skipping sub-issue close check")
        return

    fixed_set = set(fixed_tests)
    print(f"\n=== Checking {len(fixed_set)} fixed tests against open sub-issues ===")

    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues"
    params = {"state": "open", "labels": "pytorch-ci-failure", "per_page": 100}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"Failed to fetch open issues: {resp.status_code}")
        return

    for issue in resp.json():
        title = issue.get("title", "")
        # Only process sub-issues (title contains [NEW] or [EXISTING])
        if "[NEW]" not in title and "[EXISTING]" not in title:
            continue

        body = issue.get("body", "") or ""
        issue_num = issue["number"]

        sub_tests = _extract_tests_from_sub_issue(body)
        if not sub_tests:
            continue

        sub_set = set(sub_tests)
        resolved = sub_set & fixed_set
        remaining = sub_set - fixed_set

        if not resolved:
            continue

        if not remaining:
            # All fixed -> auto-close
            comment = (
                f"All failures in this sub-issue resolved as of commit `{commit_short}`.\n\n"
                f"Resolved tests ({len(resolved)}):\n"
                + "\n".join(f"- `{t}`" for t in sorted(resolved))
                + "\n\nAuto-closing."
            )
            if dry_run:
                print(f"  [DRY-RUN] Would close #{issue_num}: all {len(resolved)} tests fixed")
            else:
                _add_comment(issue_num, comment)
                _close_issue(issue_num)
                print(f"  Closed #{issue_num}: all {len(resolved)} tests fixed")
        else:
            # Partially fixed -> add comment
            comment = (
                f"Partial update as of commit `{commit_short}`:\n\n"
                f"Now passing ({len(resolved)}):\n"
                + "\n".join(f"- `{t}`" for t in sorted(resolved))
                + f"\n\nStill failing ({len(remaining)}):\n"
                + "\n".join(f"- `{t}`" for t in sorted(remaining))
            )
            if dry_run:
                print(f"  [DRY-RUN] Would comment on #{issue_num}: "
                      f"{len(resolved)} fixed, {len(remaining)} remaining")
            else:
                _add_comment(issue_num, comment)
                print(f"  Commented on #{issue_num}: "
                      f"{len(resolved)} fixed, {len(remaining)} remaining")


def _extract_tests_from_sub_issue(body):
    """Extract test list from REPRO_START JSON block in sub-issue body."""
    start = "<!-- REPRO_START -->"
    end = "<!-- REPRO_END -->"
    if start in body and end in body:
        block = body.split(start)[1].split(end)[0].strip()
        # Remove markdown code fence
        block = re.sub(r'^```\w*\n?', '', block)
        block = re.sub(r'\n?```$', '', block)
        try:
            repro = json.loads(block)
            return repro.get("test_names", [])
        except json.JSONDecodeError:
            pass

    # Fallback: regex match test names from body
    matches = re.findall(r'`(test_\S+)`', body)
    return list(set(matches)) if matches else []


def _add_comment(issue_number, body):
    """Add a comment to an issue."""
    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues/{issue_number}/comments"
    resp = requests.post(url, headers=HEADERS, json={"body": body}, timeout=60)
    if resp.status_code != 201:
        print(f"  Failed to add comment to #{issue_number}: {resp.status_code}")


def _close_issue(issue_number):
    """Close an issue."""
    url = f"https://api.github.com/repos/{TRACKING_REPO}/issues/{issue_number}"
    resp = requests.patch(url, headers=HEADERS, json={"state": "closed"}, timeout=60)
    if resp.status_code != 200:
        print(f"  Failed to close #{issue_number}: {resp.status_code}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create sub-issues for CI failure groups")
    parser.add_argument("--input", type=str, required=True,
                        help="CI results JSON")
    parser.add_argument("--summary-issue", type=int, required=True,
                        help="Summary issue number to reference")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print info without creating issues")
    args = parser.parse_args()

    if not GITHUB_TOKEN and not args.dry_run:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    commit_sha = data.get("commit_sha", "unknown")
    commit_short = commit_sha[:12]
    new_tests = data.get("new_failed_tests", [])
    existing_tests = data.get("existing_failed_tests", [])
    fixed_tests = data.get("fixed_tests", [])

    # Phase 1: Close fixed sub-issues
    close_fixed_sub_issues(fixed_tests, commit_short, dry_run=args.dry_run)

    # Phase 2: Create sub-issues for failure groups
    new_groups = group_by_file(new_tests)
    existing_groups = group_by_file(existing_tests)

    print(f"\n=== Creating sub-issues: {len(new_groups)} NEW + "
          f"{len(existing_groups)} EXISTING groups ===")
    print(f"    Summary issue: #{args.summary_issue}")
    print()

    group_num = 0

    # NEW failures -> create sub-issues
    for test_file, tests in new_groups.items():
        group_num += 1
        short = test_file.split("/")[-1]
        print(f"[{group_num}] NEW: {short} ({len(tests)} tests)")
        if not args.dry_run:
            create_sub_issue(args.summary_issue, group_num, test_file, tests,
                             is_new=True, commit_sha=commit_sha)
        print()

    # EXISTING failures -> create sub-issues
    for test_file, tests in existing_groups.items():
        group_num += 1
        short = test_file.split("/")[-1]
        print(f"[{group_num}] EXISTING: {short} ({len(tests)} tests)")
        if not args.dry_run:
            create_sub_issue(args.summary_issue, group_num, test_file, tests,
                             is_new=False, commit_sha=commit_sha)
        print()

    # Summary
    print(f"=== Done: {group_num} sub-issues, {len(fixed_tests)} fixed tests checked ===")
    print(f"  NEW_FAILURE: {len(new_groups)} groups ({len(new_tests)} tests)")
    print(f"  EXISTING_FAILURE: {len(existing_groups)} groups ({len(existing_tests)} tests)")
    print(f"  FIXED: {len(fixed_tests)} tests")


if __name__ == "__main__":
    main()
