# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import json
import re
import subprocess
import sys

from patterns import PYTORCHXPU_FIELD_MAP


def parse_issue_ref(ref, default_owner="intel", default_repo="torch-xpu-ops"):
    """Parse an issue reference into (owner, repo, number).

    A full GitHub issue URL supplies its own owner/repo. A bare issue number
    uses the provided defaults.
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

    raise ValueError(f"Invalid issue reference: {ref!r}")


def fetch_issue(owner, repo, number):
    cmd = ["gh", "api", f"repos/{owner}/{repo}/issues/{number}"]
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)

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


def rest_to_core(issue):
    assignee = ""
    if issue.get("assignee"):
        assignee = issue["assignee"].get("login", "")
    elif issue.get("assignees"):
        first_assignee = issue["assignees"][0] or {}
        assignee = first_assignee.get("login", "")

    milestone = ""
    if issue.get("milestone"):
        milestone = issue["milestone"].get("title", "")

    return {
        "issue_id": issue["number"],
        "title": issue.get("title") or "",
        "status": issue.get("state") or "",
        "assignee": assignee,
        "reporter": issue.get("user", {}).get("login", ""),
        "labels": [label.get("name", "") for label in issue.get("labels", [])],
        "created_time": issue.get("created_at") or "",
        "updated_time": issue.get("updated_at") or "",
        "milestone": milestone,
    }


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

    if data.get("errors"):
        print(
            f"PyTorchXPU project GraphQL errors for issue {number}: {data['errors']}",
            file=sys.stderr,
        )
        return result

    issue = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}

    result["github_type"] = ((issue.get("issueType") or {}).get("name")) or ""

    nodes = (issue.get("projectItems") or {}).get("nodes") or []

    # Prefer the PyTorchXPU project item if present, but the field-name map is
    # the real filter, so process all items if no titled match exists.
    preferred = [n for n in nodes if ((n.get("project") or {}).get("title") == "PyTorchXPU")]
    items = preferred if preferred else nodes

    for item in items:
        for fv in (item.get("fieldValues") or {}).get("nodes") or []:
            field = fv.get("field") or {}
            fname = str(field.get("name") or "").strip()
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
                m = re.search(r"\bP[0-3]\b", raw.upper())
                result["priority"] = m.group(0) if m else ""
            else:
                result[key] = raw

    return result
