#!/usr/bin/env python3
# Copyright 2024-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Create a revert PR for a merged pull request.

Usage:
    python bot_revert.py --pr-number 123 --reason "broke nightly" --repo owner/repo
"""

import argparse
import json
import subprocess
import sys
import tempfile


def run(cmd, check=True, capture=True):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip() if capture else None


def main():
    parser = argparse.ArgumentParser(description="Create a revert PR")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--reason", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    args = parser.parse_args()

    # Get PR info
    pr_json = run(
        f"gh pr view {args.pr_number} --repo {args.repo} "
        f"--json title,mergeCommit,merged,baseRefName,headRefName"
    )
    pr = json.loads(pr_json)

    if not pr["merged"]:
        print(f"::error::PR #{args.pr_number} is not merged. Cannot revert.")
        # Post comment via gh
        run(
            f"gh pr comment {args.pr_number} --repo {args.repo} "
            f'--body "Cannot revert: PR #{args.pr_number} is not merged."'
        )
        sys.exit(1)

    merge_sha = pr["mergeCommit"]["oid"]
    base_branch = pr["baseRefName"]
    original_title = pr["title"]
    revert_branch = f"revert-pr-{args.pr_number}"

    # Configure git identity
    run('git config user.name "torchxpubot"')
    run('git config user.email "torchxpubot@users.noreply.github.com"')

    # Fetch and create revert branch
    run(f"git fetch origin {base_branch}")
    run(f"git checkout -b {revert_branch} origin/{base_branch}")

    # Revert the merge commit
    # -m 1 is needed for merge commits to specify the mainline parent
    revert_result = subprocess.run(
        f"git revert --no-edit -m 1 {merge_sha}",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if revert_result.returncode != 0:
        # Try without -m 1 (for squash merges which are not merge commits)
        revert_result = subprocess.run(
            f"git revert --no-edit {merge_sha}",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if revert_result.returncode != 0:
            print(f"::error::git revert failed: {revert_result.stderr}")
            run(
                f"gh pr comment {args.pr_number} --repo {args.repo} "
                f'--body "Failed to revert merge commit `{merge_sha[:10]}`. '
                f'Manual revert may be needed."'
            )
            sys.exit(1)

    # Push the revert branch
    run(f"git push origin {revert_branch}")

    # Create the revert PR (use temp files to avoid shell injection)
    revert_title = f'Revert "{original_title}"'
    revert_body = (
        f"Reverts #{args.pr_number}\n\n"
        f"**Reason:** {args.reason}\n\n"
        f"Original PR: #{args.pr_number}"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write(revert_title)
        title_file = tf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as bf:
        bf.write(revert_body)
        body_file = bf.name

    pr_url = run(
        f"gh pr create --repo {args.repo} "
        f"--base {base_branch} --head {revert_branch} "
        f"--title-file {title_file} "
        f"--body-file {body_file}"
    )

    # Comment on the original PR (use --body-file to avoid injection)
    comment_body = f"Revert PR created: {pr_url}\nReason: {args.reason}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as cf:
        cf.write(comment_body)
        comment_file = cf.name
    run(f"gh pr comment {args.pr_number} --repo {args.repo} --body-file {comment_file}")
    print(f"Revert PR created: {pr_url}")


if __name__ == "__main__":
    main()
