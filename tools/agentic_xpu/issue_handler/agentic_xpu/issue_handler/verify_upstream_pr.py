# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Verify an upstream pytorch/pytorch PR before spending XPU triage effort.

Flow (inserted between ``verify_existence`` FAILS and ``triage``):

  1. Read the issue body. If we've already classified DEVICE_SPECIFIC,
     short-circuit straight to triage (record a log entry).
  2. Resolve a linked upstream PR — direct ref in body first, else
     GraphQL ``closedByPullRequestsReferences`` on any linked upstream
     issue.  If none → no work to do, fall through to triage.
  3. Fetch PR metadata + diff (truncate >20KB).  Hand to the
     ``upstream-pr-analysis`` skill.
  4. DEVICE_SPECIFIC: persist verdict + log to body, return False so the
     orchestrator runs triage in the next turn.
  5. DEVICE_AGNOSTIC: under ``pytorch_lock``, checkout the PR into a
     namespaced ref, rebuild if stale, run the refined reproducer.
       - Repro PASSES + PR merged → mark issue DONE, comment, close.
       - Repro PASSES + PR open  → mark WAITING_UPSTREAM, comment.
       - Repro FAILS → log "verdict not borne out", fall through to triage.

Stage transitions:
  DISCOVERED ──► UPSTREAM_VERIFYING ──► DONE | WAITING_UPSTREAM | DISCOVERED
  WAITING_UPSTREAM (re-entry, 12h throttle) ──► DONE | WAITING_UPSTREAM
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone

from .utils import git as gh
from .utils.body_templates import (
    append_log, get_metadata, get_status, set_metadata, set_status,
)
from .utils.build import incremental_build
from .utils.config import (
    ISSUE_REPO, PUBLIC_TARGET_REPO, PYTORCH_DIR, UPSTREAM_REMOTE,
)
from .utils.logger import log
from .utils.locks import pytorch_lock
from .utils.reproducer import (
    LLM_REPRO_MODEL, ReproResult, extract_reproducer_command,
    extract_reproducer_via_llm, persist_refined_command,
    run_reproducer_command,
)
from .utils.stages import Skill, Stage
from .utils.xpu_env import ensure_xpu_ready, sync_pytorch
from .utils.agent_backend import get_backend

# Avoid circular import with verify_fix at module load
from .verify_fix import _capture_head_ref, _restore_head_ref


DIFF_TRUNCATE_BYTES = 20_000
THROTTLE_HOURS = 12
VERIFY_BRANCH_MAX_AGE_DAYS = 14
VERIFY_BRANCH_PREFIX = "agentic/verify-upstream-"
VERIFY_EXISTENCE_BRANCH_PREFIX = "agentic/verify-existence-"
UPSTREAM_PR_REF_PREFIX = "refs/agentic/upstream-pr-"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UpstreamPRConflict(RuntimeError):
    """Cherry-picking the upstream PR onto current main produced conflicts.

    Treated as a DEVICE_SPECIFIC signal: the fix can no longer be cleanly
    replayed onto trunk, so we fall through to normal triage.
    """


class UpstreamPRBuildFailed(RuntimeError):
    """Incremental + clean build both failed after cherry-picking the PR."""


# ---------------------------------------------------------------------------
# Stale-branch GC
# ---------------------------------------------------------------------------

def _is_public_pytorch_remote(remote: str, issue: int | None = None) -> bool:
    """Return True iff ``remote``'s URL points at ``pytorch/pytorch``.

    Used to skip pruning ``refs/agentic/upstream-pr-*`` for the public
    upstream — those refs are cheap and useful for forensic re-fetch.
    """
    try:
        url = gh.git_out("remote", "get-url", remote,
                         workdir=PYTORCH_DIR, issue=issue).strip()
    except subprocess.CalledProcessError:
        return False
    return bool(re.search(r"pytorch/pytorch(?:\.git)?$", url))


def _gc_stale_verify_branches(max_age_days: int = VERIFY_BRANCH_MAX_AGE_DAYS,
                              issue: int | None = None) -> None:
    """Delete stale ``agentic/*`` refs older than ``max_age_days``.

    Prunes:
      - ``refs/heads/agentic/verify-upstream-*`` (local branches)
      - ``refs/heads/agentic/verify-existence-*`` (local branches)
      - ``refs/agentic/upstream-pr-<N>`` **only** when ``UPSTREAM_REMOTE``
        does not resolve to ``pytorch/pytorch`` (per Q-B).

    Skips the currently checked-out branch defensively. Logs each
    deletion (and each skipped public-pytorch ref) for forensics.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_ts = cutoff.timestamp()

    try:
        current_branch = gh.git_out(
            "rev-parse", "--abbrev-ref", "HEAD",
            workdir=PYTORCH_DIR, issue=issue,
        ).strip()
    except subprocess.CalledProcessError:
        current_branch = ""

    # Local branches: always prune at 14 d.
    branch_globs = [
        f"refs/heads/{VERIFY_BRANCH_PREFIX}*",
        f"refs/heads/{VERIFY_EXISTENCE_BRANCH_PREFIX}*",
    ]
    try:
        raw = gh.git_out(
            "for-each-ref",
            "--format=%(refname:short) %(committerdate:unix)",
            *branch_globs,
            workdir=PYTORCH_DIR, issue=issue,
        )
    except subprocess.CalledProcessError as e:
        log("WARN", f"GC: for-each-ref failed for local branches: {e}",
            issue=issue)
        raw = ""

    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        name, ts_str = parts
        try:
            ts = float(ts_str)
        except ValueError:
            continue
        if ts >= cutoff_ts:
            continue
        if name == current_branch:
            log("INFO", f"GC: skipping currently checked-out branch {name}",
                issue=issue)
            continue
        try:
            gh.git("branch", "-D", name,
                   workdir=PYTORCH_DIR, issue=issue)
            log("INFO", f"GC: deleted stale branch {name} "
                f"(age > {max_age_days}d)", issue=issue)
        except subprocess.CalledProcessError as e:
            log("WARN", f"GC: failed to delete {name}: {e}", issue=issue)

    # Upstream PR fetch refs: only prune when remote is NOT pytorch/pytorch.
    if _is_public_pytorch_remote(UPSTREAM_REMOTE, issue=issue):
        log("INFO",
            f"GC: skipping fetch-ref pruning under {UPSTREAM_REMOTE} "
            "(public pytorch/pytorch — keep for forensic re-fetch)",
            issue=issue)
        return

    try:
        raw = gh.git_out(
            "for-each-ref",
            "--format=%(refname) %(committerdate:unix)",
            f"{UPSTREAM_PR_REF_PREFIX}*",
            workdir=PYTORCH_DIR, issue=issue,
        )
    except subprocess.CalledProcessError as e:
        log("WARN", f"GC: for-each-ref failed for PR refs: {e}", issue=issue)
        raw = ""

    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        refname, ts_str = parts
        try:
            ts = float(ts_str)
        except ValueError:
            continue
        if ts >= cutoff_ts:
            continue
        try:
            gh.git("update-ref", "-d", refname,
                   workdir=PYTORCH_DIR, issue=issue)
            log("INFO", f"GC: deleted stale fetch ref {refname} "
                f"(age > {max_age_days}d)", issue=issue)
        except subprocess.CalledProcessError as e:
            log("WARN", f"GC: failed to delete {refname}: {e}", issue=issue)


# ---------------------------------------------------------------------------
# Upstream PR resolution
# ---------------------------------------------------------------------------

_DIRECT_PR_PATTERNS = [
    # https://github.com/pytorch/pytorch/pull/12345
    re.compile(rf"https?://github\.com/{re.escape(PUBLIC_TARGET_REPO)}/pull/(\d+)"),
    # pytorch/pytorch#12345
    re.compile(rf"{re.escape(PUBLIC_TARGET_REPO)}#(\d+)"),
]

_LINKED_ISSUE_PATTERNS = [
    re.compile(rf"https?://github\.com/{re.escape(PUBLIC_TARGET_REPO)}/issues/(\d+)"),
    re.compile(rf"{re.escape(PUBLIC_TARGET_REPO)}#(\d+)"),  # also matches issues
]


def _extract_direct_pr_ref(body: str) -> int | None:
    """Return the first pytorch/pytorch#NNNN PR reference in the body."""
    for pat in _DIRECT_PR_PATTERNS:
        m = pat.search(body or "")
        if m:
            return int(m.group(1))
    return None


def _extract_linked_upstream_issues(body: str) -> list[int]:
    """Return upstream issue numbers referenced anywhere in the body."""
    seen: set[int] = set()
    out: list[int] = []
    for pat in _LINKED_ISSUE_PATTERNS:
        for m in pat.finditer(body or ""):
            n = int(m.group(1))
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _query_closing_prs(issue_number: int, issue: int) -> dict | None:
    """Use GraphQL ``closedByPullRequestsReferences`` to find PRs that
    close a given upstream issue.  Returns the most relevant PR
    (first MERGED, else first OPEN), or None.
    """
    owner, name = PUBLIC_TARGET_REPO.split("/", 1)
    query = (
        "query($owner:String!, $name:String!, $number:Int!) {"
        "  repository(owner:$owner, name:$name) {"
        "    issue(number:$number) {"
        "      closedByPullRequestsReferences(first:10, includeClosedPrs:true) {"
        "        nodes { number state merged }"
        "      }"
        "    }"
        "  }"
        "}"
    )
    try:
        # gh CLI binds extra -f/-F flags to graphql variables. We bypass
        # _gh_api's automatic JSON-serialization of dict args by calling
        # _gh directly with the right argv.
        raw = gh._gh([
            "api", "graphql", "--method", "POST",
            "-f", f"query={query}",
            "-f", f"owner={owner}",
            "-f", f"name={name}",
            "-F", f"number={issue_number}",
        ])
        resp = json.loads(raw) if raw.strip() else {}
    except subprocess.CalledProcessError as e:
        log("WARN", f"GraphQL closedBy lookup failed for upstream#{issue_number}: {e}",
            issue=issue)
        return None

    try:
        nodes = (resp["data"]["repository"]["issue"]
                 ["closedByPullRequestsReferences"]["nodes"]) or []
    except (KeyError, TypeError):
        return None

    if not nodes:
        return None

    merged = next((n for n in nodes if n.get("merged")), None)
    if merged:
        return merged
    opn = next((n for n in nodes if n.get("state") == "OPEN"), None)
    return opn or nodes[0]


def _resolve_upstream_pr(body: str, issue: int) -> int | None:
    """Resolve an upstream PR for this issue, direct refs first."""
    pr = _extract_direct_pr_ref(body)
    if pr:
        log("INFO", f"Found direct upstream PR ref: #{pr}", issue=issue)
        return pr

    for linked in _extract_linked_upstream_issues(body):
        log("INFO", f"Querying closing PRs for upstream issue #{linked}",
            issue=issue)
        node = _query_closing_prs(linked, issue=issue)
        if node and node.get("number"):
            pr_num = int(node["number"])
            log("INFO",
                f"Resolved upstream PR #{pr_num} via issue #{linked} "
                f"(state={node.get('state')}, merged={node.get('merged')})",
                issue=issue)
            return pr_num

    log("INFO", "No upstream PR resolved for this issue", issue=issue)
    return None


# ---------------------------------------------------------------------------
# PR metadata + diff
# ---------------------------------------------------------------------------

def _fetch_pr_metadata(pr_num: int, issue: int) -> dict | None:
    """Fetch PR title/body/state/merged/head_ref/files via REST."""
    try:
        meta = gh._gh_api(f"repos/{PUBLIC_TARGET_REPO}/pulls/{pr_num}")
    except subprocess.CalledProcessError as e:
        log("WARN", f"Could not fetch upstream PR #{pr_num}: {e}", issue=issue)
        return None

    # Fetch first page of files separately — pulls endpoint doesn't include them
    try:
        files = gh._gh_api(
            f"repos/{PUBLIC_TARGET_REPO}/pulls/{pr_num}/files?per_page=100",
        )
    except subprocess.CalledProcessError:
        files = []

    return {
        "number": pr_num,
        "title": meta.get("title", ""),
        "body": meta.get("body", "") or "",
        "state": (meta.get("state") or "").upper(),  # "OPEN" / "CLOSED"
        "merged": bool(meta.get("merged")),
        "head_ref": (meta.get("head") or {}).get("ref", ""),
        "additions": meta.get("additions", 0),
        "deletions": meta.get("deletions", 0),
        "changed_files": meta.get("changed_files", 0),
        "files": [
            {"path": f.get("filename", ""),
             "additions": f.get("additions", 0),
             "deletions": f.get("deletions", 0)}
            for f in files
        ],
        "html_url": meta.get("html_url", ""),
    }


def _fetch_pr_diff(pr_num: int, issue: int) -> tuple[str, bool]:
    """Fetch the unified diff of an upstream PR. Returns (diff, truncated)."""
    try:
        # Use gh CLI directly for non-JSON response
        r = subprocess.run(
            ["gh", "api",
             f"repos/{PUBLIC_TARGET_REPO}/pulls/{pr_num}",
             "-H", "Accept: application/vnd.github.v3.diff"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        diff = r.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log("WARN", f"Could not fetch PR #{pr_num} diff: {e}", issue=issue)
        return "", False

    if len(diff) > DIFF_TRUNCATE_BYTES:
        note = (f"\n\n[NOTE: diff truncated at {DIFF_TRUNCATE_BYTES} bytes "
                f"— full diff is {len(diff)} bytes; "
                f"see https://github.com/{PUBLIC_TARGET_REPO}/pull/{pr_num}/files]\n")
        log("WARN",
            f"PR #{pr_num} diff truncated: {len(diff)} → {DIFF_TRUNCATE_BYTES}",
            issue=issue)
        return diff[:DIFF_TRUNCATE_BYTES] + note, True
    return diff, False


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

def _build_classifier_prompt(pr_meta: dict, diff: str, truncated: bool) -> str:
    files_block = "\n".join(
        f"  - {f['path']} (+{f['additions']} / -{f['deletions']})"
        for f in pr_meta["files"][:50]
    ) or "  (no file list available)"

    return (
        f"# Upstream PR to analyze\n\n"
        f"**PR:** {pr_meta['html_url']}\n"
        f"**Title:** {pr_meta['title']}\n"
        f"**State:** {pr_meta['state']} (merged={pr_meta['merged']})\n"
        f"**Changed files ({pr_meta['changed_files']}, "
        f"+{pr_meta['additions']} / -{pr_meta['deletions']}):**\n{files_block}\n\n"
        f"## PR description\n{pr_meta['body'][:4000]}\n\n"
        f"## Unified diff{' (truncated)' if truncated else ''}\n"
        f"```diff\n{diff}\n```\n\n"
        f"Apply the rules in your skill and emit the JSON verdict.\n"
    )


def _parse_verdict(text: str) -> dict:
    """Parse the agent's JSON verdict.  Bad output → low-confidence
    DEVICE_SPECIFIC so we don't accidentally short-circuit triage."""
    fallback = {
        "verdict": "DEVICE_SPECIFIC",
        "reason": "Could not parse classifier output — defaulting to device-specific",
        "confidence": "low",
    }
    if not text:
        return fallback
    # Try fenced JSON first, then bare JSON object
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{[^{}]*\"verdict\"[^{}]*\})", text, re.DOTALL)
    if not m:
        return fallback
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return fallback
    verdict = obj.get("verdict", "").upper()
    if verdict not in ("DEVICE_AGNOSTIC", "DEVICE_SPECIFIC"):
        return fallback
    return {
        "verdict": verdict,
        "reason": obj.get("reason", "").strip() or "(no reason given)",
        "confidence": (obj.get("confidence") or "medium").lower(),
    }


def _classify_with_llm(pr_meta: dict, diff: str, truncated: bool,
                       issue: int) -> dict:
    prompt = _build_classifier_prompt(pr_meta, diff, truncated)
    backend = get_backend()
    try:
        out, log_path, _sid, _tok = backend.run(
            prompt, skill=Skill.UPSTREAM_PR_ANALYSIS, timeout=180,
            issue=issue, stage="upstream-pr-analysis",
        )
    except Exception as e:
        log("WARN", f"Classifier agent failed: {e}", issue=issue)
        return {
            "verdict": "DEVICE_SPECIFIC",
            "reason": f"Classifier agent crashed: {e}",
            "confidence": "low",
        }
    log("INFO", f"Classifier log: {log_path}", issue=issue)
    return _parse_verdict(out)


# ---------------------------------------------------------------------------
# Upstream PR checkout
# ---------------------------------------------------------------------------

def _checkout_upstream_pr(pr_num: int, issue: int) -> str:
    """DEPRECATED: superseded by ``_prepare_upstream_pr_branch``.

    Kept for one release in case rollback is needed. New callers should
    use ``_prepare_upstream_pr_branch`` which cherry-picks the PR onto
    current upstream/main and builds incrementally, instead of rebuilding
    the whole PR HEAD (which is typically weeks behind main).

    Fetch upstream PR into a namespaced ref and detach-checkout it.

    Returns the local ref name (so callers can pass it to ``sync_pytorch``
    for an accurate staleness check).

    NOTE: ``git fetch`` has been observed to exit 0 without actually creating
    the target ref under rare race / transient conditions. When that happens
    the subsequent ``git checkout --detach <ref>`` fails with the misleading
    error "fatal: git checkout: --detach does not take a path argument …".
    We defend against this by verifying ref existence after fetch and retrying
    once before giving up with a clear error message.
    """
    local_ref = f"refs/agentic/upstream-pr-{pr_num}"
    last_err = ""
    for attempt in (1, 2):
        gh.git(
            "fetch", UPSTREAM_REMOTE, f"pull/{pr_num}/head:{local_ref}",
            workdir=PYTORCH_DIR, issue=issue,
        )
        rv = gh.git(
            "rev-parse", "--verify", "--quiet", local_ref,
            workdir=PYTORCH_DIR, issue=issue, check=False,
        )
        if rv.returncode == 0 and rv.stdout.strip():
            break
        last_err = (
            f"fetch rc=0 but ref {local_ref} not created "
            f"(attempt {attempt}/2)"
        )
        log("WARN", last_err, issue=issue)
    else:
        raise RuntimeError(
            f"git fetch did not create {local_ref} after 2 attempts; "
            f"last: {last_err}"
        )
    gh.git("checkout", "--detach", local_ref,
           workdir=PYTORCH_DIR, issue=issue)
    return local_ref


def _prepare_upstream_pr_branch(pr_num: int, issue: int) -> str:
    """Cherry-pick upstream PR onto current ``{UPSTREAM_REMOTE}/main`` and
    incrementally build only the PR's diff.

    Returns the local branch name (``agentic/verify-upstream-<N>``) the
    caller should keep checked out for the reproducer run.

    Sequence (all inside the caller's ``pytorch_lock``):

    1. Fetch ``{UPSTREAM_REMOTE}/main`` and the PR head into
       ``refs/agentic/upstream-pr-<N>``.
    2. Create/reset ``agentic/verify-upstream-<N>`` off current
       ``{UPSTREAM_REMOTE}/main``.
    3. Cherry-pick ``$(merge-base)..refs/agentic/upstream-pr-<N>`` onto
       the new branch. Single-commit PRs collapse to one pick;
       multi-commit PRs replay each commit in order.
    4. Sync submodules.
    5. Incremental build (via ``utils.build.incremental_build``) against
       ``{UPSTREAM_REMOTE}/main`` as base ref — only the PR's actual
       C++/SYCL deltas recompile.

    Raises
    ------
    UpstreamPRConflict
        Cherry-pick produced conflicts. ``cherry-pick --abort`` is run
        before raising. Caller should treat as DEVICE_SPECIFIC and fall
        through to triage.
    UpstreamPRBuildFailed
        Both incremental and clean builds failed after a clean
        cherry-pick. Caller routes through the existing build-failure
        persistence path.
    """
    local_ref = f"{UPSTREAM_PR_REF_PREFIX}{pr_num}"
    branch_name = f"{VERIFY_BRANCH_PREFIX}{pr_num}"
    main_ref = f"{UPSTREAM_REMOTE}/main"

    # 1. Fetch main + PR head. Retry the PR fetch once on the
    #    "rc=0 but ref missing" race seen in _checkout_upstream_pr.
    gh.git("fetch", UPSTREAM_REMOTE, "main",
           workdir=PYTORCH_DIR, issue=issue)
    last_err = ""
    for attempt in (1, 2):
        gh.git(
            "fetch", UPSTREAM_REMOTE, f"pull/{pr_num}/head:{local_ref}",
            workdir=PYTORCH_DIR, issue=issue,
        )
        rv = gh.git(
            "rev-parse", "--verify", "--quiet", local_ref,
            workdir=PYTORCH_DIR, issue=issue, check=False,
        )
        if rv.returncode == 0 and rv.stdout.strip():
            break
        last_err = (f"fetch rc=0 but ref {local_ref} not created "
                    f"(attempt {attempt}/2)")
        log("WARN", last_err, issue=issue)
    else:
        raise RuntimeError(
            f"git fetch did not create {local_ref} after 2 attempts; "
            f"last: {last_err}")

    # 1b. If the local clone is shallow, ``merge-base upstream/main local_ref``
    #     will return rc=1 (no common ancestor in the truncated history) even
    #     though both refs exist. The cherry-pick range we want to compute
    #     requires real history, so unshallow upstream now. Idempotent — a
    #     non-shallow repo silently no-ops on the next attempt.
    is_shallow = gh.git_out(
        "rev-parse", "--is-shallow-repository",
        workdir=PYTORCH_DIR, issue=issue,
    ).strip() == "true"
    if is_shallow:
        log("INFO",
            "Repo is shallow; running 'git fetch --unshallow upstream' so "
            "merge-base can find a common ancestor for cherry-pick "
            "(this can take a while on first invocation).",
            issue=issue)
        gh.git("fetch", "--unshallow", UPSTREAM_REMOTE,
               workdir=PYTORCH_DIR, issue=issue)

    # 2. Branch off current upstream/main (force-reset if it already exists
    #    from a prior run).
    gh.git("checkout", "-B", branch_name, main_ref,
           workdir=PYTORCH_DIR, issue=issue)

    # 3. Cherry-pick the PR's commit range onto the new branch.
    # NOTE: ``git merge-base`` returns rc=1 (not an error) when the two refs
    # have no common ancestor — e.g. ``refs/agentic/upstream-pr-N`` was created
    # from a stale fork or rebased onto an unrelated base. Treating that as a
    # CalledProcessError silently bubbles up to ``orchestrator.advance`` and
    # gets swallowed into a triage fall-through. Handle it explicitly.
    mb_rv = gh.git(
        "merge-base", main_ref, local_ref,
        workdir=PYTORCH_DIR, issue=issue, check=False,
    )
    if mb_rv.returncode == 1:
        raise RuntimeError(
            f"git merge-base {main_ref} {local_ref} found no common ancestor "
            f"(unrelated histories). PR #{pr_num}'s branch may have been "
            f"rebased onto a different base, or the local clone is missing "
            f"history. This needs human inspection — refusing to fabricate a "
            f"cherry-pick range."
        )
    if mb_rv.returncode != 0:
        raise RuntimeError(
            f"git merge-base {main_ref} {local_ref} failed rc={mb_rv.returncode}: "
            f"{(mb_rv.stderr or '').strip()[:300]}"
        )
    merge_base = (mb_rv.stdout or "").strip()
    if not merge_base:
        raise RuntimeError(
            f"git merge-base {main_ref} {local_ref} returned empty")

    pick_range = f"{merge_base}..{local_ref}"
    log("INFO",
        f"Cherry-picking {pick_range} onto {branch_name} "
        f"(merge-base={merge_base[:12]})",
        issue=issue)
    rv = gh.git(
        "cherry-pick", pick_range,
        workdir=PYTORCH_DIR, issue=issue, check=False,
    )
    if rv.returncode != 0:
        stderr = (rv.stderr or "").strip()
        log("WARN",
            f"Cherry-pick of upstream PR #{pr_num} onto {main_ref} failed: "
            f"{stderr[:500]}",
            issue=issue)
        # Best-effort abort. If --abort itself fails (e.g. no pick in
        # progress) just log and continue raising.
        try:
            gh.git("cherry-pick", "--abort",
                   workdir=PYTORCH_DIR, issue=issue)
        except subprocess.CalledProcessError as abort_exc:
            log("WARN", f"cherry-pick --abort failed: {abort_exc}",
                issue=issue)
        raise UpstreamPRConflict(
            f"upstream PR #{pr_num} conflicts with current {main_ref} — "
            "needs manual port")

    # 4. Submodule sync — cheap, picks up any third_party / xpu.txt deltas
    #    the PR introduced.
    try:
        gh.git("submodule", "sync", "--recursive",
               workdir=PYTORCH_DIR, issue=issue)
        gh.git("submodule", "update", "--init", "--recursive",
               workdir=PYTORCH_DIR, issue=issue)
    except subprocess.CalledProcessError as e:
        log("WARN", f"submodule sync/update failed (continuing): {e}",
            issue=issue)

    # 5. Incremental build against the freshly-fetched upstream/main.
    ok, output = incremental_build(
        workdir=PYTORCH_DIR,
        base_ref=main_ref,
        issue=issue,
    )
    if not ok:
        log("ERROR",
            f"Incremental build for upstream PR #{pr_num} failed:\n"
            f"{output[-2000:] if output else '(no output)'}",
            issue=issue)
        raise UpstreamPRBuildFailed(
            f"build failed after cherry-picking upstream PR #{pr_num} "
            f"onto {main_ref}")

    return branch_name


# ---------------------------------------------------------------------------
# Body persistence — append to <!-- agent:upstream-log -->
# ---------------------------------------------------------------------------

def _log_to_body(body: str, lines: list[str]) -> str:
    """Append a markdown block to the upstream-log details section."""
    text = "\n".join(lines)
    return append_log(body, "upstream", text)


def _format_pr_summary(pr_meta: dict, verdict: dict, truncated: bool) -> list[str]:
    state = "merged" if pr_meta["merged"] else pr_meta["state"].lower()
    out = [
        f"**Upstream PR:** [{PUBLIC_TARGET_REPO}#{pr_meta['number']}]"
        f"({pr_meta['html_url']}) — _{state}_",
        f"**Title:** {pr_meta['title']}",
        f"**Verdict:** `{verdict['verdict']}` "
        f"(confidence: {verdict['confidence']})",
        f"**Reason:** {verdict['reason']}",
        f"**Files changed:** {pr_meta['changed_files']} "
        f"(+{pr_meta['additions']} / -{pr_meta['deletions']})",
    ]
    if truncated:
        out.append("_(diff was truncated for classification)_")
    return out


# ---------------------------------------------------------------------------
# Throttle for WAITING_UPSTREAM re-entries
# ---------------------------------------------------------------------------

def _diagnose_reproducer_missing(body: str) -> str:
    """Produce a multi-line diagnostic explaining why
    ``extract_reproducer_command`` returned None — surfaced in the
    issue-body log and the agent log so the user can quickly see
    whether the Reproducer section is missing, empty, or just shaped
    differently than the extractor expects.
    """
    from .utils.body_templates import parse_sections
    lines = []
    sections = parse_sections(body or "")
    section_names = list(sections.keys())
    lines.append(f"body length: {len(body or '')} chars")
    lines.append(f"sections parsed: {section_names}")
    repro = sections.get("Reproducer")
    if repro is None:
        lines.append("Reproducer section: MISSING (verify_existence may "
                     "not have run, or the template was stripped).")
        return "\n".join(lines)
    lines.append(f"Reproducer section: present, {len(repro)} chars")
    lines.append(f"contains '**Refined command:**': "
                 f"{'yes' if '**Refined command:**' in repro else 'no'}")
    lines.append(f"contains fenced bash block (```bash...```): "
                 f"{'yes' if '```bash' in repro or '```sh' in repro else 'no'}")
    preview = (repro[:300] + "…") if len(repro) > 300 else repro
    lines.append("--- Reproducer section preview (first 300 chars) ---")
    lines.append(preview)
    return "\n".join(lines)


def _should_throttle(body: str) -> bool:
    last = get_metadata(body, "upstream-last-check")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - last_dt) < timedelta(hours=THROTTLE_HOURS)


# ---------------------------------------------------------------------------
# Outcome handlers
# ---------------------------------------------------------------------------

def _persist_metadata(body: str, *, pr_num: int, verdict: str,
                      pr_state: str) -> str:
    body = set_metadata(body, "upstream-pr-number", str(pr_num))
    body = set_metadata(body, "upstream-pr-verdict", verdict)
    body = set_metadata(body, "upstream-pr-state", pr_state)
    body = set_metadata(body, "upstream-last-check",
                        datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return body


def _commit_body(issue_number: int, body: str, stage: Stage) -> None:
    """Push body update + sync labels for a stage transition."""
    body = set_status(body, stage)
    gh.update_issue_body(ISSUE_REPO, issue_number, body)
    gh.sync_labels(ISSUE_REPO, issue_number, stage)


def _handle_device_specific(issue_number: int, body: str, pr_meta: dict,
                             verdict: dict, truncated: bool) -> bool:
    """Persist verdict, append log, restore DISCOVERED, fall through to triage."""
    summary = _format_pr_summary(pr_meta, verdict, truncated)
    summary.append("**Decision:** XPU-specific fix still required — handing to triage.")
    body = _log_to_body(body, summary)
    body = _persist_metadata(body, pr_num=pr_meta["number"],
                             verdict="DEVICE_SPECIFIC",
                             pr_state=pr_meta["state"])
    _commit_body(issue_number, body, Stage.DISCOVERED)
    log("INFO",
        f"Issue #{issue_number}: upstream PR #{pr_meta['number']} is "
        f"DEVICE_SPECIFIC — handing to triage",
        issue=issue_number)
    return False  # orchestrator continues to triage


def _handle_repro_inconclusive(issue_number: int, body: str, pr_meta: dict,
                                verdict: dict, repro: ReproResult) -> bool:
    """Agnostic verdict — repro still fails after cherry-pick.

    If the upstream PR is still open (not yet merged), the cherry-pick may
    not cleanly represent the final fix. Trust the classifier and set
    WAITING_UPSTREAM rather than downgrading to DEVICE_SPECIFIC.

    Only downgrade to DEVICE_SPECIFIC (→ triage) if the upstream PR is
    already merged and the repro still fails — that means the upstream fix
    genuinely does not cover XPU.
    """
    pr_merged = pr_meta.get("merged", False)
    block = _format_pr_summary(pr_meta, verdict, False)
    block.append("")
    block.append(
        f"**Local verification:** ❌ Reproducer still fails "
        f"({repro.reason}) — agnostic verdict not borne out by local repro."
    )

    if not pr_merged:
        # PR still open — cherry-pick may not apply cleanly yet.
        # Trust the DEVICE_AGNOSTIC classifier and wait for upstream merge.
        block.append(
            f"**Decision:** Upstream PR #{pr_meta['number']} is still open. "
            "Cannot confirm fix locally, but classifier verdict is DEVICE_AGNOSTIC. "
            "Setting WAITING_UPSTREAM — no XPU-specific fix needed."
        )
        body = _log_to_body(body, block)
        body = _persist_metadata(body, pr_num=pr_meta["number"],
                                 verdict="DEVICE_AGNOSTIC",
                                 pr_state=pr_meta["state"])
        _commit_body(issue_number, body, Stage.WAITING_UPSTREAM)
        gh.sync_labels(ISSUE_REPO, issue_number, Stage.WAITING_UPSTREAM)
        log("WARN",
            f"Issue #{issue_number}: agnostic verdict for PR "
            f"#{pr_meta['number']} not borne out by repro, but PR still open "
            "— setting WAITING_UPSTREAM",
            issue=issue_number)
        return True  # halt cycle, do not fall to triage

    # PR is merged and repro still fails — genuinely device-specific.
    block.append("**Decision:** Upstream PR is merged yet bug persists — XPU-specific fix required, handing to triage.")
    body = _log_to_body(body, block)
    body = _persist_metadata(body, pr_num=pr_meta["number"],
                             verdict="DEVICE_SPECIFIC",  # downgrade
                             pr_state=pr_meta["state"])
    _commit_body(issue_number, body, Stage.DISCOVERED)
    log("WARN",
        f"Issue #{issue_number}: agnostic verdict for PR "
        f"#{pr_meta['number']} not borne out by repro — falling to triage",
        issue=issue_number)
    return False


def _prune_stale_upstream_log(body: str) -> str:
    """Collapse stale ``upstream-log`` entries when a positive verdict
    supersedes them.

    Earlier runs may have appended NEEDS_HUMAN diagnostics (e.g. "could
    not extract reproducer") before a later run successfully verified
    the PR. Those entries are now misleading — they suggest the agent is
    stuck when in fact it has resolved the issue. We don't *delete* them
    (audit trail matters), but we wrap the existing ``<details>`` block
    in a collapsed "stale earlier attempts" wrapper so the fresh
    success entry rendered immediately after is the visible signal.

    No-op if no upstream-log block exists yet, or if the block was
    already pruned (idempotent across re-entries).
    """
    marker = "<!-- agent:upstream-log -->"
    if marker not in body:
        return body
    stale_open = "<details><summary>upstream log</summary>"
    pruned_open = (
        "<details><summary>upstream log "
        "(earlier attempts — superseded)</summary>"
    )
    # Only rewrite the first occurrence (the active log block).
    if stale_open in body and pruned_open not in body:
        body = body.replace(stale_open, pruned_open, 1)
    return body


def _handle_agnostic_verified(issue_number: int, body: str, pr_meta: dict,
                               verdict: dict, repro: ReproResult) -> bool:
    """Agnostic verdict + repro passes. Close (merged) or wait (open)."""
    # Collapse any prior NEEDS_HUMAN / diagnostic entries — the success
    # entry we are about to append supersedes them.
    body = _prune_stale_upstream_log(body)

    block = _format_pr_summary(pr_meta, verdict, False)
    block.append("")
    block.append("**Local verification:** ✅ Reproducer passes with PR checked out.")

    if pr_meta["merged"]:
        block.append(
            "**Conclusion:** Upstream PR is merged and resolves the issue on "
            "XPU. Closing as fixed."
        )
        body = _log_to_body(body, block)
        body = _persist_metadata(body, pr_num=pr_meta["number"],
                                 verdict="DEVICE_AGNOSTIC",
                                 pr_state="MERGED")
        _commit_body(issue_number, body, Stage.DONE)
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"Verified locally that upstream PR "
            f"[{PUBLIC_TARGET_REPO}#{pr_meta['number']}]({pr_meta['html_url']}) "
            f"resolves this issue on XPU. Closing as fixed.",
        )
        gh.close_issue(ISSUE_REPO, issue_number)
    else:
        block.append(
            f"**Conclusion:** Upstream PR is still _{pr_meta['state'].lower()}_. "
            f"Marking as `agent:waiting-upstream`; we will re-check after merge."
        )
        body = _log_to_body(body, block)
        body = _persist_metadata(body, pr_num=pr_meta["number"],
                                 verdict="DEVICE_AGNOSTIC",
                                 pr_state=pr_meta["state"])
        _commit_body(issue_number, body, Stage.WAITING_UPSTREAM)
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"Verified locally that upstream PR "
            f"[{PUBLIC_TARGET_REPO}#{pr_meta['number']}]({pr_meta['html_url']}) "
            f"will resolve this issue on XPU. Waiting on upstream merge.",
        )
    return True  # caller skips triage


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(issue_number: int) -> bool:
    """Run upstream PR verification.

    Returns True iff we handled the issue this turn (DONE or
    WAITING_UPSTREAM). Returns False to let the orchestrator continue
    with the normal triage flow.
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""
    status = get_status(body)

    if status not in (Stage.DISCOVERED, Stage.WAITING_UPSTREAM):
        return False

    # Throttle WAITING_UPSTREAM re-checks
    if status == Stage.WAITING_UPSTREAM and _should_throttle(body):
        log("INFO", f"Issue #{issue_number} WAITING_UPSTREAM — within "
            f"{THROTTLE_HOURS}h throttle, skipping re-check",
            issue=issue_number)
        return True

    # Short-circuit on prior DEVICE_SPECIFIC verdict
    prior_verdict = get_metadata(body, "upstream-pr-verdict")
    if status == Stage.DISCOVERED and prior_verdict == "DEVICE_SPECIFIC":
        body2 = _log_to_body(body, [
            "_Re-entered: already classified DEVICE_SPECIFIC; skipping re-analysis._",
        ])
        if body2 != body:
            gh.update_issue_body(ISSUE_REPO, issue_number, body2)
        return False  # continue to triage

    # Resolve upstream PR
    pr_num = _resolve_upstream_pr(body, issue=issue_number)
    if not pr_num:
        return False  # no PR → triage as normal

    # Transition into UPSTREAM_VERIFYING (only from DISCOVERED;
    # WAITING_UPSTREAM stays on its own label until we resolve)
    if status == Stage.DISCOVERED:
        body = set_status(body, Stage.UPSTREAM_VERIFYING)
        gh.update_issue_body(ISSUE_REPO, issue_number, body)
        gh.sync_labels(ISSUE_REPO, issue_number, Stage.UPSTREAM_VERIFYING)

    # Re-fetch body in case anything else mutated it
    body = gh.get_issue_detail(ISSUE_REPO, issue_number).get("body", "") or ""

    pr_meta = _fetch_pr_metadata(pr_num, issue=issue_number)
    if not pr_meta:
        # Fall through to triage on transient API error
        if status == Stage.DISCOVERED:
            _commit_body(issue_number, body, Stage.DISCOVERED)
        return False

    # Skip LLM if we already classified this PR as agnostic
    already_agnostic = (prior_verdict == "DEVICE_AGNOSTIC")
    if already_agnostic:
        log("INFO", f"PR #{pr_num} previously classified DEVICE_AGNOSTIC — "
            "skipping classifier, re-running verification",
            issue=issue_number)
        verdict = {"verdict": "DEVICE_AGNOSTIC",
                   "reason": "(cached from previous run)",
                   "confidence": "high"}
        truncated = False
    else:
        diff, truncated = _fetch_pr_diff(pr_num, issue=issue_number)
        verdict = _classify_with_llm(pr_meta, diff, truncated,
                                      issue=issue_number)
        log("INFO",
            f"Classifier verdict: {verdict['verdict']} "
            f"(confidence: {verdict['confidence']})",
            issue=issue_number)

    if verdict["verdict"] == "DEVICE_SPECIFIC":
        # Low-confidence DEVICE_SPECIFIC (e.g. classifier crashed or timed
        # out) must NOT be persisted to metadata.  If we write
        # upstream-pr-verdict=DEVICE_SPECIFIC at low confidence the
        # short-circuit above (prior_verdict == "DEVICE_SPECIFIC") will
        # permanently skip re-classification on every future run — even
        # though we never actually ran the LLM.
        # Instead: log a warning and fall through to triage WITHOUT writing
        # anything to metadata, so the next pipeline cycle retries the
        # classifier cleanly.
        if verdict.get("confidence") == "low":
            log("WARN",
                f"Issue #{issue_number}: upstream PR #{pr_meta['number']} "
                "classifier returned DEVICE_SPECIFIC with low confidence "
                f"(reason: {verdict['reason']}) — NOT persisting verdict; "
                "will retry classifier on next run. Falling through to triage.",
                issue=issue_number)
            # Reset stage to DISCOVERED (we transitioned to UPSTREAM_VERIFYING
            # earlier in this function) and return False so orchestrator
            # continues to triage as a fallback for this cycle.
            _commit_body(issue_number, body, Stage.DISCOVERED)
            return False
        return _handle_device_specific(issue_number, body, pr_meta,
                                        verdict, truncated)

    # AGNOSTIC path — need the refined reproducer
    cmd = extract_reproducer_command(body, issue=issue_number)
    cmd_source = "tier1-3" if cmd else None
    if not cmd:
        # Tier 4: LLM fallback via opencode + gpt-5-mini.
        log("INFO",
            "Tiers 1-3 failed to extract reproducer; trying LLM fallback",
            issue=issue_number)
        llm_result = extract_reproducer_via_llm(body, issue_number)
        if llm_result:
            cmd, llm_meta = llm_result
            cmd_source = "tier4-llm"
            body = _log_to_body(body, [
                f"_Refined reproducer extracted via LLM fallback "
                f"(`{LLM_REPRO_MODEL}`):_",
                f"```",
                cmd,
                f"```",
                f"**Reason:** {llm_meta['reason']}",
            ])

    # Persist refined command back to the Reproducer section so future
    # runs hit Tier 1 directly (no /tmp dependency, no repeated LLM cost,
    # full audit trail of what we actually executed). No-op if the body
    # already has a `**Refined command:**` line.
    if cmd:
        new_body = persist_refined_command(body, cmd)
        if new_body != body:
            log("INFO",
                f"Persisted refined command to body ({cmd_source}): {cmd[:120]}",
                issue=issue_number)
            body = new_body
        gh.update_issue_body(ISSUE_REPO, issue_number, body)

    if not cmd:
        # Q1 (2026-05-24): NEVER downgrade an AGNOSTIC verdict here.
        # The classifier already inspected the diff; missing repro is a
        # *workflow gap* (verify_existence didn't populate it), not
        # evidence that the fix is device-specific. Diagnose, persist
        # AGNOSTIC, escalate to NEEDS_HUMAN so the user can investigate.
        diag = _diagnose_reproducer_missing(body)
        log("WARN",
            "Cannot run local verification: no refined reproducer in "
            f"body (Tiers 1-4 all failed). Escalating to NEEDS_HUMAN. "
            f"Diagnostic: {diag}",
            issue=issue_number)
        body = _log_to_body(body, [
            "_Classifier returned **DEVICE_AGNOSTIC** but the refined "
            "reproducer command is missing from the issue body and the "
            "LLM fallback could not extract one — cannot verify locally. "
            "Escalating to NEEDS_HUMAN for investigation._",
            "",
            "**Reproducer extraction diagnostic:**",
            "```",
            diag,
            "```",
        ])
        body = _persist_metadata(body, pr_num=pr_num,
                                 verdict="DEVICE_AGNOSTIC",
                                 pr_state=pr_meta["state"])
        body = set_status(body, Stage.NEEDS_HUMAN)
        gh.update_issue_body(ISSUE_REPO, issue_number, body)
        gh.sync_labels(ISSUE_REPO, issue_number, Stage.NEEDS_HUMAN)
        return True  # terminal — orchestrator must not run triage

    # Heavy work under the global pytorch lock. On ANY exception below
    # (e.g. transient git checkout failure), reset the stage back to
    # DISCOVERED so the next pipeline cycle retries cleanly — otherwise
    # the issue is stranded at UPSTREAM_VERIFYING and every later stage
    # (triage/fix/verify_fix) silently skips it.
    try:
        with pytorch_lock(issue=issue_number):
            # Opportunistic GC of stale agentic/* refs (cheap; once per
            # verify_upstream_pr cycle, inside the lock for safety).
            try:
                _gc_stale_verify_branches(issue=issue_number)
            except Exception as gc_exc:
                log("WARN", f"_gc_stale_verify_branches failed: {gc_exc!r}",
                    issue=issue_number)

            if not ensure_xpu_ready(issue=issue_number):
                log("ERROR", "XPU env not ready — falling to triage",
                    issue=issue_number)
                _commit_body(issue_number, body, Stage.DISCOVERED)
                return False

            saved_head = _capture_head_ref(PYTORCH_DIR, issue_number)
            try:
                try:
                    _prepare_upstream_pr_branch(pr_num, issue=issue_number)
                except UpstreamPRConflict as conflict:
                    log("WARN",
                        f"Upstream PR #{pr_num} cannot be cherry-picked onto "
                        f"current {UPSTREAM_REMOTE}/main: {conflict}",
                        issue=issue_number)
                    # If the LLM already classified this PR as DEVICE_AGNOSTIC,
                    # a cherry-pick failure (e.g. ghstack "poisoned" commits,
                    # already-partial merge, or trivial empty-diff) does NOT
                    # mean the fix is XPU-specific.  The upstream PR will fix
                    # this issue once it lands — we just cannot verify locally.
                    # Correct action: WAITING_UPSTREAM + comment, no triage/fix.
                    # Only downgrade to DEVICE_SPECIFIC (→ triage) when the
                    # classifier itself said DEVICE_SPECIFIC.
                    if verdict["verdict"] == "DEVICE_AGNOSTIC":
                        log("INFO",
                            f"Issue #{issue_number}: PR #{pr_num} classified "
                            "DEVICE_AGNOSTIC but cherry-pick failed — cannot "
                            "verify locally. Setting WAITING_UPSTREAM.",
                            issue=issue_number)
                        block = _format_pr_summary(pr_meta, verdict, truncated)
                        block.append("")
                        block.append(
                            f"**Local verification:** ⚠️ Cherry-pick of PR #{pr_num} "
                            f"onto `{UPSTREAM_REMOTE}/main` failed ({conflict}). "
                            "This is expected for ghstack-formatted PRs or when "
                            "the PR base has since moved. The fix is classified "
                            "as device-agnostic — no XPU-specific port needed."
                        )
                        block.append(
                            f"**Conclusion:** Waiting for upstream PR #{pr_num} "
                            "to merge; will re-verify after merge."
                        )
                        body = _log_to_body(body, block)
                        body = _persist_metadata(body, pr_num=pr_num,
                                                 verdict="DEVICE_AGNOSTIC",
                                                 pr_state=pr_meta["state"])
                        _commit_body(issue_number, body, Stage.WAITING_UPSTREAM)
                        gh.add_issue_comment(
                            ISSUE_REPO, issue_number,
                            f"Upstream PR [{PUBLIC_TARGET_REPO}#{pr_num}]"
                            f"({pr_meta['html_url']}) is classified as "
                            "device-agnostic. Cherry-pick onto current "
                            f"`{UPSTREAM_REMOTE}/main` failed (ghstack format "
                            "or base drift) so local XPU verification was "
                            "skipped. No XPU-specific fix needed — marking as "
                            "`agent:waiting-upstream` until the PR merges.",
                        )
                        return True  # handled — skip triage
                    # Classifier said DEVICE_SPECIFIC: cherry-pick conflict
                    # means a manual port is genuinely needed → fall to triage.
                    body = _log_to_body(body, [
                        f"_Upstream PR #{pr_num} conflicts with current "
                        f"`{UPSTREAM_REMOTE}/main` — needs manual port. "
                        "Treating as DEVICE_SPECIFIC and falling through "
                        "to triage._",
                    ])
                    body = _persist_metadata(body, pr_num=pr_num,
                                             verdict="DEVICE_SPECIFIC",
                                             pr_state=pr_meta["state"])
                    _commit_body(issue_number, body, Stage.DISCOVERED)
                    return False
                except UpstreamPRBuildFailed as build_exc:
                    log("ERROR",
                        f"Rebuild for upstream PR #{pr_num} failed "
                        f"({build_exc}) — falling to triage",
                        issue=issue_number)
                    # Same logic as UpstreamPRConflict: if the LLM said
                    # DEVICE_AGNOSTIC, a local build failure doesn't mean
                    # a XPU-specific fix is needed — we just can't verify.
                    # Set WAITING_UPSTREAM rather than silently downgrading.
                    if verdict["verdict"] == "DEVICE_AGNOSTIC":
                        log("INFO",
                            f"Issue #{issue_number}: PR #{pr_num} is "
                            "DEVICE_AGNOSTIC but local build failed — "
                            "cannot verify. Setting WAITING_UPSTREAM.",
                            issue=issue_number)
                        block = _format_pr_summary(pr_meta, verdict, truncated)
                        block.append("")
                        block.append(
                            f"**Local verification:** ⚠️ Build failed after "
                            f"cherry-picking PR #{pr_num} ({build_exc}). "
                            "Fix is classified as device-agnostic — no "
                            "XPU-specific port needed."
                        )
                        block.append(
                            f"**Conclusion:** Waiting for upstream PR #{pr_num} "
                            "to merge; will re-verify after merge."
                        )
                        body = _log_to_body(body, block)
                        body = _persist_metadata(body, pr_num=pr_num,
                                                 verdict="DEVICE_AGNOSTIC",
                                                 pr_state=pr_meta["state"])
                        _commit_body(issue_number, body, Stage.WAITING_UPSTREAM)
                        gh.add_issue_comment(
                            ISSUE_REPO, issue_number,
                            f"Upstream PR [{PUBLIC_TARGET_REPO}#{pr_num}]"
                            f"({pr_meta['html_url']}) is classified as "
                            "device-agnostic but local build failed after "
                            "cherry-pick. No XPU-specific fix needed — "
                            "marking as `agent:waiting-upstream`.",
                        )
                        return True
                    body = _log_to_body(body, [
                        f"_Build failed for upstream PR #{pr_num} after "
                        f"cherry-pick onto `{UPSTREAM_REMOTE}/main` — "
                        "cannot verify locally. Falling through to triage._",
                    ])
                    body = _persist_metadata(body, pr_num=pr_num,
                                             verdict="DEVICE_SPECIFIC",
                                             pr_state=pr_meta["state"])
                    _commit_body(issue_number, body, Stage.DISCOVERED)
                    return False

                repro = run_reproducer_command(cmd, issue=issue_number)
            finally:
                _restore_head_ref(PYTORCH_DIR, saved_head, issue_number)
    except Exception as exc:
        log("ERROR",
            f"verify_upstream_pr crashed: {exc!r} — resetting stage to "
            "DISCOVERED so next cycle retries",
            issue=issue_number, exc=exc)
        try:
            fresh_body = gh.get_issue_detail(ISSUE_REPO, issue_number).get(
                "body", "") or ""
            fresh_body = _log_to_body(fresh_body, [
                f"_verify_upstream_pr crashed: `{type(exc).__name__}: {exc}` — "
                "stage reset to DISCOVERED for retry on next pipeline tick._",
            ])
            _commit_body(issue_number, fresh_body, Stage.DISCOVERED)
        except Exception as recovery_exc:
            log("ERROR",
                f"Failed to reset stage after crash: {recovery_exc!r}",
                issue=issue_number, exc=recovery_exc)
        raise

    if not repro.passed:
        return _handle_repro_inconclusive(issue_number, body, pr_meta,
                                           verdict, repro)
    return _handle_agnostic_verified(issue_number, body, pr_meta,
                                      verdict, repro)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("issue", type=int)
    args = p.parse_args()
    handled = run(args.issue)
    print(f"handled={handled}")


if __name__ == "__main__":
    main()
