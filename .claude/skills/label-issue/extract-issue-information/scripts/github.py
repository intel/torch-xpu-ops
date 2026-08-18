# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import json
import re
import subprocess
import sys

from patterns import PRIORITY_MAP, PYTORCHXPU_FIELD_MAP


class PullRequestReference(Exception):
    """The reference is a pull request, not an issue.

    Deliberately NOT a ValueError subclass: callers map ValueError to the
    "malformed reference" exit code, and a well-formed PR URL is a different
    failure that must not be reported as malformed.
    """


def parse_issue_ref(ref, default_owner="intel", default_repo="torch-xpu-ops"):
    """Parse an issue reference into (owner, repo, number).

    A full GitHub issue URL supplies its own owner/repo. A bare issue number
    uses the provided defaults. A /pull/ URL raises PullRequestReference so it
    is reported as a rejected PR rather than as a malformed reference.
    """
    if isinstance(ref, int):
        return default_owner, default_repo, ref
    if not isinstance(ref, str):
        raise ValueError(f"Invalid issue reference: {ref!r}")

    ref = ref.strip()
    if ref.isdigit():
        return default_owner, default_repo, int(ref)

    # Full GitHub issue URL: https://github.com/OWNER/REPO/issues/N
    m = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)(?:[/?#].*)?$", ref)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)(?:[/?#].*)?$", ref)
    if m:
        raise PullRequestReference(
            f"{m.group(1)}/{m.group(2)}#{m.group(3)} is a pull request, not an issue"
        )

    raise ValueError(f"Invalid issue reference: {ref!r}")


def fetch_issue(owner, repo, number):
    cmd = ["gh", "api", f"repos/{owner}/{repo}/issues/{number}"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI not found on PATH; install and authenticate it") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out after 120s fetching issue {number} from {owner}/{repo}") from exc

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if "Not Found" in message:
            raise RuntimeError(f"Issue {number} not found in {owner}/{repo}")
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: {message or 'gh api returned a non-zero exit code'}")

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: empty response from gh api")

    try:
        issue = json.loads(stdout)  # pyright: ignore[reportAny]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: non-JSON response from gh api") from exc

    if not isinstance(issue, dict):
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: unexpected JSON response")

    if issue.get("pull_request") is not None:
        raise SystemExit(f"{owner}/{repo}#{number} is a pull request, not an issue")

    return issue


def resolve_ref_kind(owner, repo, number):
    """Return "pr", "issue", or None when the kind cannot be determined.

    The issues endpoint serves both and includes a pull_request key only for a
    PR. None means unresolved (no gh, timeout, 404, private repo) - never guess.
    """
    cmd = ["gh", "api", f"repos/{owner}/{repo}/issues/{int(number)}"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, check=False, text=True, timeout=120
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"PR-ref resolution failed for {owner}/{repo}#{number}: {exc}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        print(
            f"PR-ref resolution failed for {owner}/{repo}#{number}: "
            f"{(proc.stderr or '').strip() or 'non-zero exit'}",
            file=sys.stderr,
        )
        return None

    try:
        payload = json.loads(proc.stdout or "")
    except json.JSONDecodeError:
        print(f"PR-ref resolution returned non-JSON for {owner}/{repo}#{number}", file=sys.stderr)
        return None

    if not isinstance(payload, dict):
        return None

    return "pr" if payload.get("pull_request") is not None else "issue"


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def rest_to_core(issue):
    assignee = _as_dict(issue.get("assignee")).get("login", "")
    if not assignee:
        assignees = issue.get("assignees") or []
        if assignees:
            assignee = _as_dict(assignees[0]).get("login", "")

    return {
        "issue_id": issue.get("number") or 0,
        "title": issue.get("title") or "",
        "status": issue.get("state") or "",
        "assignee": assignee,
        "reporter": _as_dict(issue.get("user")).get("login", ""),
        "labels": [
            _as_dict(label).get("name", "")
            for label in (issue.get("labels") or [])
        ],
        "created_time": issue.get("created_at") or "",
        "updated_time": issue.get("updated_at") or "",
        "milestone": _as_dict(issue.get("milestone")).get("title", ""),
    }


def normalize_priority(raw):
    token = re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()
    if token in PRIORITY_MAP:
        return PRIORITY_MAP[token]
    m = re.search(r"\bP[0-3]\b", str(raw or "").upper())
    return PRIORITY_MAP.get(m.group(0), "") if m else ""


# Bare field names in PyTorchXPU project -> output key.
def fetch_project_and_type(owner, repo, number):
    """Fetch native issueType and PyTorchXPU project field values for one issue.

    Uses `gh api graphql` (not requests) because gh CLI's auth handles the
    read:project scope correctly. On any failure the function degrades
    gracefully: it prints a warning to stderr and returns an all-empty dict.
    It never raises.
    """
    result = {
        "github_type": "",
        "priority": "",
        "project_status": "",
        "project_estimate": "",
        "project_depending": "",
        "project_short_comments": "",
    }

    query = """
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          issueType { name }
          projectItems(first: 20) {
            nodes {
              project { title number }
              fieldValues(first: 50) {
                nodes {
                  ... on ProjectV2ItemFieldTextValue        { text   field { ... on ProjectV2FieldCommon { name } } }
                  ... on ProjectV2ItemFieldSingleSelectValue { name   field { ... on ProjectV2FieldCommon { name } } }
                  ... on ProjectV2ItemFieldNumberValue      { number field { ... on ProjectV2FieldCommon { name } } }
                }
              }
            }
          }
        }
      }
    }
    """

    args = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={repo}",
        "-F",
        f"number={int(number)}",
    ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"PyTorchXPU project fetch failed for issue {number}: {exc}", file=sys.stderr)
        return result

    if proc.returncode != 0:
        print(
            f"PyTorchXPU project fetch failed for issue {number}: {proc.stderr.strip()}",
            file=sys.stderr,
        )
        return result

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"PyTorchXPU project fetch returned non-JSON for issue {number}: {exc}",
            file=sys.stderr,
        )
        return result

    if not isinstance(data, dict):
        print(
            f"PyTorchXPU project fetch returned unexpected JSON for issue {number}",
            file=sys.stderr,
        )
        return result

    if data.get("errors"):
        print(
            f"PyTorchXPU project GraphQL errors for issue {number}: {data['errors']}",
            file=sys.stderr,
        )
        return result

    issue = _as_dict(_as_dict(data.get("data")).get("repository")).get("issue")
    issue = _as_dict(issue)

    result["github_type"] = _as_dict(issue.get("issueType")).get("name") or ""

    nodes = [
        n for n in (_as_dict(issue.get("projectItems")).get("nodes") or [])
        if isinstance(n, dict)
    ]

    # Prefer the PyTorchXPU project item if present, but the field-name map is
    # the real filter, so process all items if no titled match exists.
    preferred = [n for n in nodes if _as_dict(n.get("project")).get("title") == "PyTorchXPU"]
    items = preferred if preferred else nodes

    for item in items:
        for fv in (_as_dict(item.get("fieldValues")).get("nodes") or []):
            if not isinstance(fv, dict):
                continue
            fname = str(_as_dict(fv.get("field")).get("name") or "").strip()
            key = PYTORCHXPU_FIELD_MAP.get(fname)
            if key is None:
                continue
            raw = ""
            for candidate in (
                str(fv.get("name") or "").strip(),
                str(fv.get("text") or "").strip(),
                str(fv.get("number") or "").strip(),
            ):
                if candidate:
                    raw = candidate
                    break
            if key == "priority":
                result["priority"] = normalize_priority(raw)
            else:
                result[key] = raw

    return result
