#!/usr/bin/env python3
# Copyright 2024-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""AI-powered PR review using Claude via Anthropic API.

Usage:
    ANTHROPIC_API_KEY=... python bot_review.py --pr-number 123 --repo owner/repo
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "pr-review"

SYSTEM_PROMPT = """You are a code reviewer for the intel/torch-xpu-ops repository.
This repository provides XPU (Intel GPU) operator implementations for PyTorch ATen.

Review philosophy:
1. Only report problems. Do NOT mention things done correctly.
2. Focus on what CI cannot check: correctness against CPU/CUDA semantics,
   XPU-specific risks (synchronization, indexing, precision), test adequacy.
3. Be specific and actionable. Reference file paths and line numbers.
4. Assume the reader has PyTorch familiarity.
5. Use SYCL programming model terms (subgroup size, work-group, work-item),
   not hardware terms, in code review.

Output format:
## PR Review: #<number>
### Summary
What the PR does (1 sentence), then overall verdict.
### Correctness
[Problems only]
### XPU-Specific Risks
[Problems only]
### Testing
[Problems only]
### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Omit sections with no problems."""


def run(cmd, check=True):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def call_claude(system_prompt, user_prompt, api_key):
    """Call the Anthropic API with the given prompts."""
    import urllib.request

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(
        {
            "model": "claude-opus-4-20250514",
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["content"][0]["text"]
    except Exception as e:
        print(f"::error::Anthropic API call failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AI-powered PR review")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", type=str, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("::error::ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # Fetch PR metadata
    pr_json = run(
        f"gh pr view {args.pr_number} --repo {args.repo} "
        f"--json title,body,author,baseRefName,headRefName,files,additions,deletions"
    )
    pr = json.loads(pr_json)

    # Fetch PR diff
    diff = run(f"gh pr diff {args.pr_number} --repo {args.repo}")

    # Build file list
    file_list = "\n".join(
        f"  {f['path']} (+{f.get('additions', '?')}/-{f.get('deletions', '?')})"
        for f in pr.get("files", [])
    )

    # Construct user prompt
    user_prompt = f"""Review this PR.

Title: {pr["title"]}
Author: {pr["author"]["login"]}
Base: {pr["baseRefName"]} <- Head: {pr["headRefName"]}
Total: +{pr["additions"]}/-{pr["deletions"]}

Files changed:
{file_list}

PR description:
{pr.get("body", "(no description)")}

Diff:
{diff}"""

    # Truncate if too large (Claude context limit)
    max_chars = 180000
    if len(user_prompt) > max_chars:
        user_prompt = user_prompt[:max_chars] + "\n\n[Diff truncated due to size]"

    review = call_claude(SYSTEM_PROMPT, user_prompt, api_key)

    # Post the review as a PR comment
    # Write to temp file to avoid shell escaping issues
    with open("/tmp/review_body.md", "w") as f:
        f.write(review)
    run(
        f"gh pr comment {args.pr_number} --repo {args.repo} "
        f"--body-file /tmp/review_body.md"
    )
    print(f"Review posted to PR #{args.pr_number}")


if __name__ == "__main__":
    main()
