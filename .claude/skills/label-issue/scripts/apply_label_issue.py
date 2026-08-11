#!/usr/bin/env python3
# Copyright 2020-2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""
Apply GitHub Issue labels/type/priority/comment derived from a label-issue
`labels.md` report.

Reads the markdown table produced by the label-issue skill, maps each row to
a concrete GitHub mutation, and applies them via `gh`:

  - `issue_type: <Bug|Task|Feature|Epic>` row
      -> native GitHub Issue Type (repo-level `issueTypes`, GraphQL
         `updateIssueIssueType`) if the repo supports it, else falls back to
         a `type: <Value>` label (created if missing).
  - `test_module: <value>` row
      -> `module: <value>` label (created if missing). test_module=ut maps
         to the existing `module: ut` label convention used in this repo.
  - `module: <bucket>` row
      -> `module: <bucket>` label (created if missing).
  - `P0`/`P1`/`P2`/`P3` row
      -> native repo-level Issue Field "Priority" (GraphQL
         `setIssueFieldValue`) with P0->Urgent, P1->High, P2->Medium,
         P3->Low. Skipped with a warning if the repo has no such field.
  - `dependency component: <component>` row
      -> `dependency component: <component>` label, skipped entirely if the
         row is absent or its value is `null`/empty.
  - `duplicated` row
      -> `duplicate` label. Skipped if the row is absent (no duplicate
         found).
  - `not_target` row
      -> `not_target` label. Skipped if the row is absent (issue is in
         scope for this repo).
  - A comment starting with `[agent_triage_result]` containing the full
    labels.md content is always posted last, after every label/field
    mutation has been attempted. If the currently authenticated `gh` user
    already posted an `[agent_triage_result]` comment on this issue, that
    comment is edited in place instead of appending a duplicate.

Analysis in `labels.md` is produced separately by the label-issue skill.
This script is the only piece that mutates GitHub state.

Usage:
    python3 apply_label_issue.py <issue_ref> --labels-md PATH [--repo owner/name] [--dry-run]

Exit codes:
    0  - all reachable actions attempted (some may have been skipped with a
         warning if the repo lacks a native feature); comment posted.
    1  - hard error (bad ref, gh not authenticated, labels.md unreadable,
         issue fetch failed).
"""

import argparse
import json
import re
import subprocess
import sys


def run_gh(args: list[str], check: bool = True, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def run_gh_graphql(query: str, fields: dict) -> dict:
    args = ["api", "graphql", "-f", f"query={query}"]
    for key, value in fields.items():
        args.extend(["-F", f"{key}={value}"])
    out = run_gh(args)
    data = json.loads(out)
    if data.get("errors"):
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def parse_issue_ref(ref: str, default_repo: str | None) -> tuple[str, int]:
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)", ref)
    if m:
        return m.group(1), int(m.group(2))
    if ref.isdigit():
        if not default_repo:
            raise ValueError(f"Bare issue number {ref} requires --repo")
        return default_repo, int(ref)
    raise ValueError(f"Cannot parse issue reference: {ref}")


# ---------------------------------------------------------------------------
# labels.md parsing
# ---------------------------------------------------------------------------

ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*$")


def parse_labels_md(path: str) -> dict:
    """Parse the label-issue markdown table into a structured dict.

    Returns:
        {
          "raw_text": <full file contents>,
          "rows": [ (label_cell:str, reason:str), ... ],
        }
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        label_cell, reason = m.group(1).strip(), m.group(2).strip()
        rows.append((label_cell, reason))
    return {"raw_text": text, "rows": rows}


def extract_row_value(rows: list[tuple[str, str]], prefix: str) -> str | None:
    """Find a row like `prefix: value` (or a bare `prefix` row) and return
    `value` (or "" for a bare match). Returns None if no such row exists.
    """
    for label_cell, _reason in rows:
        if label_cell == prefix:
            return ""
        if label_cell.startswith(prefix + ":"):
            return label_cell.split(":", 1)[1].strip()
    return None


def extract_bare_row(rows: list[tuple[str, str]], name: str) -> bool:
    return any(label_cell == name for label_cell, _ in rows)


# ---------------------------------------------------------------------------
# GitHub capability probing
# ---------------------------------------------------------------------------

TYPES_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    issueTypes(first: 20) { nodes { id name } }
  }
}
"""

FIELDS_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    viewerCanSeeIssueFields
    issueFields(first: 20) {
      nodes {
        ... on IssueFieldSingleSelect { id name options { id name } }
      }
    }
  }
}
"""


def fetch_repo_capabilities(owner: str, name: str) -> dict:
    """Probe Issue Types and Issue Fields independently. Either query can be
    rejected on its own (unsupported field, insufficient token scope, schema
    mismatch) without the other; each degrades to an empty/absent capability
    rather than aborting the whole run, per the documented "skipped, not a
    hard error" behavior for repos lacking one or the other native feature.
    """
    issue_types: dict = {}
    try:
        data = run_gh_graphql(TYPES_QUERY, {"owner": owner, "name": name})
        repo = data["repository"]
        issue_types = {n["name"]: n["id"] for n in (repo.get("issueTypes") or {}).get("nodes") or []}
    except (RuntimeError, KeyError):
        pass

    issue_fields: dict = {}
    viewer_can_see_issue_fields = False
    try:
        data = run_gh_graphql(FIELDS_QUERY, {"owner": owner, "name": name})
        repo = data["repository"]
        viewer_can_see_issue_fields = bool(repo.get("viewerCanSeeIssueFields"))
        for node in (repo.get("issueFields") or {}).get("nodes") or []:
            if not node or "name" not in node:
                continue
            options = {opt["name"]: opt["id"] for opt in node.get("options") or []}
            issue_fields[node["name"]] = {"id": node["id"], "options": options}
    except (RuntimeError, KeyError):
        pass

    return {
        "issue_types": issue_types,
        "issue_fields": issue_fields,
        "viewer_can_see_issue_fields": viewer_can_see_issue_fields,
    }


def fetch_issue_node(owner: str, name: str, number: int) -> dict:
    data = run_gh_graphql(
        """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $number) { id }
          }
        }
        """,
        {"owner": owner, "name": name, "number": number},
    )
    return data["repository"]["issue"]


def fetch_current_labels(repo: str, number: int) -> set[str]:
    out = run_gh(["issue", "view", str(number), "--repo", repo, "--json", "labels"])
    data = json.loads(out)
    return {l["name"] for l in data.get("labels", [])}


def fetch_all_label_names(repo: str) -> set[str]:
    out = run_gh(["label", "list", "--repo", repo, "--limit", "300", "--json", "name"])
    data = json.loads(out)
    return {l["name"] for l in data}


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

class Actions:
    def __init__(self, repo: str, number: int, dry_run: bool, repo_labels: set[str]):
        self.repo = repo
        self.number = number
        self.dry_run = dry_run
        self.repo_labels = repo_labels
        self.applied: list[str] = []
        self.skipped: list[str] = []
        self.errors: list[str] = []

    def ensure_label_exists(self, name: str):
        if name in self.repo_labels:
            return
        if self.dry_run:
            self.applied.append(f"[dry-run] would create label '{name}'")
            self.repo_labels.add(name)
            return
        try:
            run_gh(["label", "create", name, "--repo", self.repo, "--force"])
            self.repo_labels.add(name)
        except RuntimeError as exc:
            self.errors.append(f"failed to create label '{name}': {exc}")

    def add_label(self, name: str, issue_labels: set[str]):
        if name in issue_labels:
            self.skipped.append(f"label '{name}' already present on issue")
            return
        self.ensure_label_exists(name)
        if self.dry_run:
            self.applied.append(f"[dry-run] would add label '{name}'")
            return
        try:
            run_gh(["issue", "edit", str(self.number), "--repo", self.repo, "--add-label", name])
            self.applied.append(f"added label '{name}'")
        except RuntimeError as exc:
            self.errors.append(f"failed to add label '{name}': {exc}")

    def set_issue_type(self, type_value: str, caps: dict, issue_node_id: str):
        type_id = caps["issue_types"].get(type_value)
        if not type_id:
            self.skipped.append(
                f"native Issue Type unsupported or missing option '{type_value}' on {self.repo}"
            )
            return
        if self.dry_run:
            self.applied.append(f"[dry-run] would set Issue Type to '{type_value}'")
            return
        try:
            run_gh_graphql(
                """
                mutation($issueId: ID!, $issueTypeId: ID) {
                  updateIssueIssueType(input: {issueId: $issueId, issueTypeId: $issueTypeId}) {
                    issue { id }
                  }
                }
                """,
                {"issueId": issue_node_id, "issueTypeId": type_id},
            )
            self.applied.append(f"set Issue Type to '{type_value}'")
        except RuntimeError as exc:
            self.errors.append(f"failed to set Issue Type: {exc}")

    def set_priority_field(self, priority_value: str, caps: dict, issue_node_id: str):
        field = caps["issue_fields"].get("Priority")
        mapping = {"P0": "Urgent", "P1": "High", "P2": "Medium", "P3": "Low"}
        option_name = mapping.get(priority_value)
        if not option_name:
            self.skipped.append(f"unrecognized priority value '{priority_value}'")
            return
        if not field:
            if not caps.get("viewer_can_see_issue_fields"):
                reason = (
                    f"viewer cannot see native Issue Fields on {self.repo} "
                    "(insufficient token scope or feature disabled)"
                )
            else:
                reason = f"native Priority Issue Field unsupported on {self.repo}"
            self.skipped.append(f"{reason}; skipping {priority_value}->{option_name}")
            return
        option_id = field["options"].get(option_name)
        if not option_id:
            self.skipped.append(
                f"Priority field has no option '{option_name}' on {self.repo}"
            )
            return
        if self.dry_run:
            self.applied.append(
                f"[dry-run] would set Priority field to '{option_name}' ({priority_value})"
            )
            return
        try:
            run_gh_graphql(
                """
                mutation($issueId: ID!, $fieldId: ID!, $optionId: ID!) {
                  setIssueFieldValue(input: {
                    issueId: $issueId,
                    issueFields: [{fieldId: $fieldId, singleSelectOptionId: $optionId}]
                  }) {
                    issue { id }
                  }
                }
                """,
                {"issueId": issue_node_id, "fieldId": field["id"], "optionId": option_id},
            )
            self.applied.append(f"set Priority field to '{option_name}' ({priority_value})")
        except RuntimeError as exc:
            self.errors.append(f"failed to set Priority field: {exc}")

    def find_last_agent_comment_id(self) -> int | None:
        """Return the id of the most recent `[agent_triage_result]` comment
        authored by the currently authenticated `gh` user, or None if the
        viewer has never posted one on this issue.
        """
        try:
            viewer = run_gh(["api", "user", "-q", ".login"])
            out = run_gh(
                [
                    "api",
                    f"repos/{self.repo}/issues/{self.number}/comments",
                    "--paginate",
                    "--slurp",
                ]
            )
            pages = json.loads(out)
            comments = [comment for page in pages for comment in page]
        except (RuntimeError, ValueError):
            return None
        for comment in reversed(comments):
            if (
                comment.get("user", {}).get("login") == viewer
                and comment.get("body", "").startswith("[agent_triage_result]")
            ):
                return comment["id"]
        return None

    def post_comment(self, body: str):
        existing_id = self.find_last_agent_comment_id()
        if self.dry_run:
            self.applied.append(
                "[dry-run] would edit previous [agent_triage_result] comment"
                if existing_id
                else "[dry-run] would post [agent_triage_result] comment"
            )
            return
        try:
            if existing_id:
                run_gh(
                    [
                        "api",
                        "-X",
                        "PATCH",
                        f"repos/{self.repo}/issues/comments/{existing_id}",
                        "-F",
                        "body=@-",
                    ],
                    input_text=body,
                )
                self.applied.append(
                    f"edited previous [agent_triage_result] comment (id={existing_id})"
                )
            else:
                run_gh(
                    ["issue", "comment", str(self.number), "--repo", self.repo, "--body-file", "-"],
                    input_text=body,
                )
                self.applied.append("posted [agent_triage_result] comment")
        except RuntimeError as exc:
            self.errors.append(f"failed to post/edit comment: {exc}")


PRIORITY_VALUES = {"P0", "P1", "P2", "P3"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply labels/type/priority/comment from a label-issue labels.md report"
    )
    parser.add_argument("issue_ref", help="Bare issue number or full issue URL")
    parser.add_argument("--labels-md", required=True, help="Path to the label-issue labels.md file")
    parser.add_argument("--repo", default=None, help="owner/name, required for a bare issue number")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without mutating GitHub")
    parser.add_argument("--output", "-o", help="Write JSON result summary to this file")
    args = parser.parse_args()

    try:
        repo, number = parse_issue_ref(args.issue_ref, args.repo)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    owner, name = repo.split("/", 1)

    try:
        parsed = parse_labels_md(args.labels_md)
    except OSError as exc:
        print(f"ERROR: cannot read labels.md at {args.labels_md}: {exc}", file=sys.stderr)
        return 1
    rows = parsed["rows"]
    if not rows:
        print(f"ERROR: no label rows found in {args.labels_md}", file=sys.stderr)
        return 1

    try:
        run_gh(["auth", "status"])
    except RuntimeError as exc:
        print(f"ERROR: gh not authenticated: {exc}", file=sys.stderr)
        return 1

    try:
        caps = fetch_repo_capabilities(owner, name)
        issue_node = fetch_issue_node(owner, name, number)
        existing_labels = fetch_current_labels(repo, number)
        repo_labels = fetch_all_label_names(repo)
    except (RuntimeError, KeyError) as exc:
        print(f"ERROR: failed to fetch issue/repo state: {exc}", file=sys.stderr)
        return 1

    issue_node_id = issue_node["id"]
    actions = Actions(repo, number, args.dry_run, repo_labels)

    # --- issue_type -> native Type, fallback to `type: <Value>` label ---
    issue_type_value = extract_row_value(rows, "issue_type")
    if issue_type_value:
        if issue_type_value in caps["issue_types"]:
            actions.set_issue_type(issue_type_value, caps, issue_node_id)
        else:
            actions.add_label(f"type: {issue_type_value}", existing_labels)

    # --- test_module -> `module: <value>` label ---
    test_module_value = extract_row_value(rows, "test_module")
    if test_module_value:
        actions.add_label(f"module: {test_module_value}", existing_labels)

    # --- module -> `module: <bucket>` label ---
    module_value = extract_row_value(rows, "module")
    if module_value:
        actions.add_label(f"module: {module_value}", existing_labels)

    # --- priority (bare `P0`..`P3` row) -> native Priority Issue Field ---
    priority_value = None
    for label_cell, _reason in rows:
        if label_cell in PRIORITY_VALUES:
            priority_value = label_cell
            break
    if priority_value:
        actions.set_priority_field(priority_value, caps, issue_node_id)

    # --- dependency component -> label, skip if null/absent ---
    dependency_value = extract_row_value(rows, "dependency component")
    if dependency_value and dependency_value.lower() not in ("null", "none", ""):
        actions.add_label(f"dependency component: {dependency_value}", existing_labels)

    # --- duplicated -> `duplicate` label, skip if row absent ---
    if extract_bare_row(rows, "duplicated"):
        actions.add_label("duplicate", existing_labels)

    # --- not_target -> `not_target` label, skip if row absent ---
    if extract_bare_row(rows, "not_target"):
        actions.add_label("not_target", existing_labels)

    # --- comment: edit the viewer's previous [agent_triage_result] comment
    #     on this issue if one exists, else post a new one, last ---
    comment_body = "[agent_triage_result]\n\n" + parsed["raw_text"]
    actions.post_comment(comment_body)

    result = {
        "issue_id": number,
        "repo": repo,
        "url": f"https://github.com/{repo}/issues/{number}",
        "dry_run": args.dry_run,
        "applied": actions.applied,
        "skipped": actions.skipped,
        "errors": actions.errors,
    }
    output_text = json.dumps(result, indent=2)
    print(output_text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)

    return 1 if actions.errors else 0


if __name__ == "__main__":
    sys.exit(main())
