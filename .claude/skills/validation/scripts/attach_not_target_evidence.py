#!/usr/bin/env python3
"""
Deterministic tracking-issue lookup for `Not Applicable` verdicts.

A `Not Applicable` verdict produced by the not-target sheet/code path does not
always carry a GitHub issue link, even when a tracking issue exists (e.g. XNNPACK
skips are tracked by intel/torch-xpu-ops#4179, labeled not_target+skipped, which
lists the exact test names in its body). This script performs a cheap, bounded,
deterministic lookup that string-matches the test name / class name against the
bodies of closed `not_target` and open `skipped` issues in intel/torch-xpu-ops,
and returns the matched issue link as supporting evidence.

It is NOT a full known-issue search: it only dumps two bounded label-filtered
issue lists and matches locally, so a `Not Applicable` row can be backfilled with
a link without changing the verdict.

Usage:
    python3 attach_not_target_evidence.py --name-xpu <test_name_xpu> \
        --classname-xpu <ClassNameXPU> [--repo intel/torch-xpu-ops] [--limit 200]

Output (stdout JSON):
    {
        "matched": bool,
        "issue_number": int | null,
        "url": str | null,
        "title": str | null,
        "state": str | null,
        "labels": [str],
        "match_evidence": str,
        "searched_labels": ["not_target(closed)", "skipped(open)"],
        "errors": [str]
    }

Requires the `gh` CLI to be authenticated. Network/gh failures are reported in
`errors` and yield `matched: false` (the caller keeps the Not Applicable verdict
and notes the empty result).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

DEFAULT_REPO = "intel/torch-xpu-ops"

# Each entry: (label, state, human-readable tag for searched_labels).
LABEL_QUERIES = (
    ("not_target", "closed", "not_target(closed)"),
    ("skipped", "open", "skipped(open)"),
)


def _gh_issue_list(repo: str, label: str, state: str, limit: int) -> tuple[list, str | None]:
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--label", label,
        "--state", state,
        "--limit", str(limit),
        "--json", "number,title,state,labels,url,body",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return [], f"gh issue list {label}/{state} failed: {exc}"
    if proc.returncode != 0:
        return [], f"gh issue list {label}/{state} exit {proc.returncode}: {proc.stderr.strip()}"
    try:
        return json.loads(proc.stdout or "[]"), None
    except json.JSONDecodeError as exc:
        return [], f"gh issue list {label}/{state} bad JSON: {exc}"


def _body_matches(body: str, name_xpu: str, classname_xpu: str) -> str | None:
    hay = (body or "").lower()
    name = (name_xpu or "").lower()
    cls = (classname_xpu or "").lower()
    if name and name in hay:
        return f"issue body contains literal test name '{name_xpu}'"
    # op_ut,<dotted.path>.<ClassName>,<full_test_name> skip-list format.
    if name and cls and f",{cls}," in hay and name in hay:
        return f"issue body contains op_ut skip-list entry for '{classname_xpu},{name_xpu}'"
    if cls and name and cls in hay and name in hay:
        return f"issue body references class '{classname_xpu}' and test '{name_xpu}'"
    return None


def lookup(repo: str, name_xpu: str, classname_xpu: str, limit: int) -> dict:
    errors: list[str] = []
    searched: list[str] = []
    for label, state, tag in LABEL_QUERIES:
        searched.append(tag)
        issues, err = _gh_issue_list(repo, label, state, limit)
        if err:
            errors.append(err)
            continue
        for issue in issues:
            evidence = _body_matches(issue.get("body", ""), name_xpu, classname_xpu)
            if evidence:
                labels = [lab.get("name", "") for lab in issue.get("labels", [])]
                return {
                    "matched": True,
                    "issue_number": issue.get("number"),
                    "url": issue.get("url"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "labels": labels,
                    "match_evidence": f"Deterministic body match ({label}/{state}): {evidence}",
                    "searched_labels": searched,
                    "errors": errors,
                }
    return {
        "matched": False,
        "issue_number": None,
        "url": None,
        "title": None,
        "state": None,
        "labels": [],
        "match_evidence": "No closed not_target or open skipped issue body references this test.",
        "searched_labels": searched,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-xpu", required=True, help="Full XPU test name (e.g. test_conv2d_transpose).")
    parser.add_argument("--classname-xpu", required=True, help="XPU test class name (e.g. TestXNNPACKOps).")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repo to search (default: {DEFAULT_REPO}).")
    parser.add_argument("--limit", type=int, default=200, help="Max issues per label query (default: 200).")
    args = parser.parse_args()

    result = lookup(args.repo, args.name_xpu, args.classname_xpu, args.limit)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
