#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Cherry-pick a landed pull request onto a release branch.

Usage:
    python bot_cherry_pick.py --pr-number 4907 --onto release/2.14 \
        --classification critical --actor someone --repo intel/torch-xpu-ops

The calling workflow must check out with `fetch-depth: 0`. Under the default
shallow checkout the picked commit's parent may be absent, and git reports that
as a conflict, which is reported to the user as one.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

RELEASE_BRANCH_RE = re.compile(r"^release/(\d+\.\d+)$")

# Deliberately stricter than GitHub: no leading `-`, so a repo name can never
# be mistaken for a flag by `gh`, and ASCII only, unlike `\w`.
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

CLASSIFICATIONS = ["regression", "critical", "fixnewfeature", "docs", "release"]


def validate_onto(onto):
    """Return onto if it names a release branch, else raise ValueError."""
    if not RELEASE_BRANCH_RE.match(onto or ""):
        raise ValueError(
            f"`{onto}` is not a release branch. "
            "cherry-pick only targets branches named `release/x.xx`."
        )
    return onto


def release_version(onto):
    """`release/2.14` -> `2.14`."""
    return RELEASE_BRANCH_RE.match(onto).group(1)


def find_tracker_issue(issues, version):
    """Return the tracker issue number for a release version, or None.

    Anchors on the `[v.<version>.` prefix rather than searching for the
    version anywhere in the title: `2.1` is a substring of `[v.2.14.0]`.
    """
    prefix = f"[v.{version}."
    for issue in issues:
        title = issue["title"]
        if title.startswith(prefix) and "Release Tracker" in title:
            return issue["number"]
    return None


def branch_name(pr_number, actor):
    """Match upstream's `cherry-pick-<pr>-by-<actor>` naming.

    The actor is sanitized because it reaches a shell command line when the
    branch is pushed; do not drop the substitution to match upstream.
    """
    safe_actor = re.sub(r"[^0-9a-zA-Z]+", "_", actor)
    return f"cherry-pick-{pr_number}-by-{safe_actor}"


def cherry_pick_title(onto, original_title):
    return f"[Cherry-pick {onto}] {original_title}"


def cherry_pick_body(pr_number, onto, commit_sha, original_body):
    provenance = (
        f"Cherry-pick of #{pr_number} to `{onto}`.\n"
        f"Cherry-picked from commit {commit_sha}."
    )
    original = (original_body or "").strip()
    return f"{original}\n\n{provenance}" if original else provenance


def criteria_line(classification, fixes):
    parts = [classification.capitalize()]
    if fixes:
        parts.append(fixes)
    return "* " + " - ".join(parts)


def tracker_comment(repo, pr_number, cherry_pick_pr_url, *, classification, fixes):
    return "\n".join(
        (
            "Link to landed trunk PR (if applicable):",
            f"* https://github.com/{repo}/pull/{pr_number}",
            "",
            "Link to release branch PR:",
            f"* {cherry_pick_pr_url}",
            "",
            "Criteria Category:",
            criteria_line(classification, fixes),
        )
    )


def notification_comment(pr_number, onto, *, cherry_pick_pr_url, tracker_url):
    """The comment posted back on the original PR once the pick has landed."""
    message = (
        f"### Cherry picking #{pr_number}\n"
        f"The cherry pick PR is at {cherry_pick_pr_url}"
    )
    if tracker_url:
        return message + f"\n\nThe following tracker issue is updated:\n* {tracker_url}"
    return message + (
        f"\n\nNo open release tracker issue was found for `{onto}`, so no "
        "tracker comment was posted."
    )


class CommandError(Exception):
    """A shell command exited non-zero."""

    def __init__(self, cmd, stderr):
        super().__init__(f"Command failed: {cmd}")
        self.cmd = cmd
        self.stderr = stderr.strip()


class CherryPickConflict(Exception):
    """`git cherry-pick` could not apply the commit."""

    def __init__(self, stderr):
        super().__init__("cherry-pick failed")
        self.stderr = stderr.strip()


def run(cmd, check=True, env=None):
    """Run a shell command and return stdout. Raise CommandError on failure."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=False, env=env
    )
    if check and result.returncode != 0:
        raise CommandError(cmd, result.stderr)
    return result.stdout.strip()


def comment_on_pr(repo, pr_number, body, *, dry_run=False):
    """Post a comment on the original pull request."""
    if dry_run:
        print(f"[dry-run] would comment on #{pr_number}:\n{body}")
        return
    run(f"gh pr comment {pr_number} --repo {repo} --body {shlex.quote(body)}")


def fail(repo, pr_number, message, *, dry_run=False):
    """Report a failure on the original PR and exit non-zero."""
    print(f"::error::{message}", file=sys.stderr)
    try:
        comment_on_pr(repo, pr_number, message, dry_run=dry_run)
    except CommandError as error:
        # Already failing; a broken comment must not mask the real reason.
        print(f"Could not comment on #{pr_number}: {error}", file=sys.stderr)
    sys.exit(1)


def apply_commit(commit_sha):
    """Cherry-pick commit_sha onto the current branch, or raise a conflict.

    LC_ALL=C because the merge-commit retry below matches on git's message.
    """
    env = dict(os.environ, LC_ALL="C")
    try:
        run(f"git cherry-pick -x {commit_sha}", env=env)
        return
    except CommandError as error:
        stderr = error.stderr

    if "is a merge" in stderr:
        # A maintainer merged through the UI with "Create a merge commit";
        # -m 1 picks the trunk-side parent. Mirrors bot_revert.py.
        try:
            run(f"git cherry-pick -x -m 1 {commit_sha}", env=env)
            return
        except CommandError as error:
            stderr = error.stderr

    run("git cherry-pick --abort", check=False)
    raise CherryPickConflict(stderr)


def create_cherry_pick_branch(onto, *, commit_sha, branch):
    """Point branch at origin/onto and apply commit_sha to it."""
    run(f"git fetch origin {onto}")
    run(f"git checkout -B {branch} origin/{onto}")
    apply_commit(commit_sha)


def submit_pr(repo, *, branch, pr_number, onto, commit_sha, title, body):
    """Push the branch and open the cherry-pick PR. Return its URL."""
    run(f"git push origin {branch} --force")
    return run(
        f"gh pr create --repo {repo} --base {onto} --head {branch} "
        f"--title {shlex.quote(cherry_pick_title(onto, title))} "
        f"--body {shlex.quote(cherry_pick_body(pr_number, onto, commit_sha, body))}"
    )


def update_tracker(repo, onto, *, pr_number, cherry_pick_pr_url, classification, fixes):
    """Comment on the release tracker issue. Return the comment URL or None."""
    issues = json.loads(
        run(
            f"gh issue list --repo {repo} --state open --limit 100 "
            f'--search "Release Tracker in:title" --json number,title'
        )
        or "[]"
    )
    issue_number = find_tracker_issue(issues, release_version(onto))
    if not issue_number:
        return None
    body = tracker_comment(
        repo,
        pr_number,
        cherry_pick_pr_url,
        classification=classification,
        fixes=fixes,
    )
    return run(
        f"gh issue comment {issue_number} --repo {repo} --body {shlex.quote(body)}"
    )


def cherry_pick(args):
    """Do the work. Raises CommandError or CherryPickConflict on failure."""
    repo, pr_number, dry_run = args.repo, args.pr_number, args.dry_run

    try:
        onto = validate_onto(args.onto)
    except ValueError as error:
        fail(repo, pr_number, str(error), dry_run=dry_run)

    # `merged` is not a gh pr view field; state is OPEN, CLOSED, or MERGED.
    pr = json.loads(
        run(f"gh pr view {pr_number} --repo {repo} --json title,body,state,mergeCommit")
    )
    if pr["state"] != "MERGED":
        fail(
            repo,
            pr_number,
            f"Cannot cherry-pick: PR #{pr_number} is not merged.",
            dry_run=dry_run,
        )
    # GitHub reports a merged PR with mergeCommit: null in some states.
    merge_commit = pr.get("mergeCommit")
    if not merge_commit:
        fail(
            repo,
            pr_number,
            f"Cannot cherry-pick: PR #{pr_number} has no merge commit on record.",
            dry_run=dry_run,
        )
    commit_sha = merge_commit["oid"]

    if not run(f"git ls-remote --heads origin {onto}"):
        fail(
            repo,
            pr_number,
            f"Branch `{onto}` does not exist on origin.",
            dry_run=dry_run,
        )

    # The branch name is deterministic and the push below is a force-push, so
    # an already-open cherry-pick PR would silently lose review commits. This
    # runs before any branch work so a re-run does not clobber local state.
    branch = branch_name(pr_number, args.actor)
    existing = run(
        f"gh pr list --repo {repo} --head {branch} --state open "
        f"--json url --jq '.[0].url // empty'"
    )
    if existing:
        fail(
            repo,
            pr_number,
            f"A cherry-pick PR is already open at {existing}. "
            f"Close it or update `{branch}` by hand.",
            dry_run=dry_run,
        )

    run('git config user.name "torchxpubot"')
    run('git config user.email "torchxpubot@users.noreply.github.com"')
    create_cherry_pick_branch(onto, commit_sha=commit_sha, branch=branch)

    if dry_run:
        print(f"[dry-run] would push {branch} and open a PR onto {onto}")
        return

    cherry_pick_pr_url = submit_pr(
        repo,
        branch=branch,
        pr_number=pr_number,
        onto=onto,
        commit_sha=commit_sha,
        title=pr["title"],
        body=pr["body"],
    )
    # The PR exists by now, so a tracker problem is a warning, not a failure.
    try:
        tracker_url = update_tracker(
            repo,
            onto,
            pr_number=pr_number,
            cherry_pick_pr_url=cherry_pick_pr_url,
            classification=args.classification,
            fixes=args.fixes,
        )
    except CommandError as error:
        print(f"::warning::could not update the release tracker: {error}")
        tracker_url = None

    # Likewise: failing to announce success is not a failure to succeed.
    try:
        comment_on_pr(
            repo,
            pr_number,
            notification_comment(
                pr_number,
                onto,
                cherry_pick_pr_url=cherry_pick_pr_url,
                tracker_url=tracker_url,
            ),
        )
    except CommandError as error:
        print(f"::warning::could not comment on #{pr_number}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Cherry-pick a PR onto a release branch"
    )
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--onto", type=str, required=True)
    parser.add_argument("--classification", choices=CLASSIFICATIONS, required=True)
    parser.add_argument("--fixes", type=str, default="")
    parser.add_argument("--actor", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not REPO_RE.match(args.repo):
        raise SystemExit(f"invalid --repo: {args.repo}")

    # Single translation point: every terminal path reports on the original PR.
    try:
        cherry_pick(args)
    except CherryPickConflict as error:
        fail(
            args.repo,
            args.pr_number,
            f"Cherry-picking onto `{args.onto}` hit a conflict, so nothing was "
            "pushed. Please cherry-pick this PR by hand and open the pull "
            f"request manually.\n\n```\n{error.stderr}\n```",
            dry_run=args.dry_run,
        )
    except CommandError as error:
        fail(
            args.repo,
            args.pr_number,
            f"Cherry-pick failed: `{error.cmd}` exited non-zero.\n\n"
            f"```\n{error.stderr}\n```",
            dry_run=args.dry_run,
        )
    except Exception as error:  # noqa: BLE001
        # SystemExit is a BaseException, so fail() paths still propagate.
        fail(
            args.repo,
            args.pr_number,
            f"Cherry-pick failed unexpectedly: {type(error).__name__}: {error}",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
