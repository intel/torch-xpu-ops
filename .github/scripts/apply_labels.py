#!/usr/bin/env python3
"""Apply the axes decided in a label-issue labels.md to a GitHub issue.

Reads a labels.md artifact (the collapsible <details> table produced by the
label-issue skill), then writes to GitHub:

  * label rows (test:, module:, dtype:, symptom, dependency component:, and the
    triage duplicate/wontfix/need_split rows)  -> gh issue edit --add-label
  * the `type` row                             -> gh issue edit --type  (native Type field)
  * the `priority` row                         -> native org issue field "Priority" (GraphQL setIssueFieldValue)
  * the full labels.md content                 -> posted as an issue comment

The Priority field is GitHub's new native issue field (org Settings -> Planning
-> Issue fields), NOT a project field. Its options are the tier names
(Urgent/High/Medium/Low), so the labels.md tier maps to the option name directly.
Writing needs only the issue's `viewerCanSetFields` permission (ordinary `repo`
scope), not the `project` scope.

Dry-run by default: prints exactly what it would do. Pass --apply to write.

For a `need_split` issue, the per-group axes (module, dtype, dependency
component, symptom, duplicate, and the priority) are NOT applied to the umbrella
issue -- they belong to the individual sub-issues created after the split. Only
issue-wide axes (need_split, type, test, os, hw) are applied.

Usage:
  apply_labels.py path/to/labels.md            # dry run
  apply_labels.py path/to/labels.md --apply    # write to GitHub
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROPOSED = os.path.join(HERE, "..", "reference", "proposed_labels.json")

# Name of the native org-level issue field that holds the priority tier.
PRIORITY_FIELD_NAME = "Priority"

# Axes that are NOT applied as labels.
NATIVE_TYPE_AXIS = "type"
PRIORITY_AXIS = "priority"


def run(cmd, check=True, capture=True):
    """Run a command, returning stdout. Raises on non-zero when check=True."""
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and res.returncode != 0:
        if "INSUFFICIENT_SCOPES" in (res.stderr or ""):
            sys.stderr.write(
                "error: gh token is missing a scope required for this write.\n"
                "  The native Priority issue field needs ordinary write access "
                "to the issue (the 'repo' scope); re-authenticate with\n"
                "  gh auth login   (or add the missing scope to your PAT at "
                "https://github.com/settings/tokens)\n"
            )
            sys.exit(1)
        sys.stderr.write(f"command failed: {' '.join(cmd)}\n{res.stderr}\n")
        sys.exit(1)
    return (res.stdout or "").strip()


def load_priority_order():
    """tier name (e.g. 'Medium') -> urgency rank (0 = most urgent).

    The native issue field's option names are the tier names themselves, so no
    tier->option mapping is needed; this only gives an ordering so the most
    urgent tier can be chosen across a multi-group issue.
    """
    with open(PROPOSED) as f:
        d = json.load(f)
    return {v["tier"]: i for i, v in enumerate(d["priority_field"]["values"])}


def parse_labels_md(path):
    """Return (repo, issue_id, rows) where rows is list of (axis, value)."""
    with open(path) as f:
        text = f.read()

    m = re.search(r"label-issue:\s*([^\s#]+)#(\d+)", text)
    if not m:
        sys.stderr.write(
            "could not find 'label-issue: <repo>#<id>' in the file summary\n"
        )
        sys.exit(1)
    repo, issue_id = m.group(1), m.group(2)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        axis = cells[0].strip().strip("`").lower()
        value = cells[1].strip().strip("`").strip()
        # skip header and separator rows
        if axis in ("axis", "") or set(value) <= {"-"} or value == "value":
            continue
        rows.append((axis, value))
    return repo, issue_id, rows, text


def classify(rows, priority_order):
    """Split parsed rows into labels, native type, priority tier.

    A multi-group issue carries one type/priority row per group, but GitHub's
    Type and the native Priority issue field are per-issue. Collapse type to the
    single distinct value (warn on conflict) and priority to the most urgent
    tier across groups (rank from priority_order, 0 = most urgent).

    When the issue is `need_split`, the per-group axes (module, dtype,
    dependency component, symptom, duplicate, and the priority) describe the
    individual sub-issues, not the umbrella issue, so they are dropped here and
    left for the post-split sub-issues.
    """
    need_split = any(
        axis == "triage" and value == "need_split" for axis, value in rows
    )
    # axis names (parsed, lowercased) suppressed on a need_split issue
    split_drop_axes = {"module", "dtype", "dependency component", "symptom"}

    labels = []
    types = []
    priorities = []
    seen = set()

    for axis, value in rows:
        if axis == NATIVE_TYPE_AXIS:
            if value not in types:
                types.append(value)
        elif axis == PRIORITY_AXIS:
            if need_split:
                continue
            if value not in priorities:
                priorities.append(value)
        else:
            if need_split and axis in split_drop_axes:
                continue
            if need_split and axis == "triage" and value == "duplicate":
                continue
            # everything else is a label token, applied verbatim
            if value not in seen:
                seen.add(value)
                labels.append(value)

    native_type = types[0] if types else None
    if len(types) > 1:
        sys.stderr.write(
            f"warning: multiple type values {types}; using {native_type}\n"
        )

    priority_tier = None
    if priorities:
        # most urgent = smallest rank; unknown tiers sort last
        priority_tier = min(
            priorities, key=lambda t: priority_order.get(t, 99)
        )
        if len(priorities) > 1:
            sys.stderr.write(
                f"warning: multiple priority values {priorities}; "
                f"using most urgent {priority_tier}\n"
            )
    return labels, native_type, priority_tier


def issue_node_id(repo, issue_id):
    owner, name = repo.split("/", 1)
    q = """
    query($owner:String!,$name:String!,$num:Int!){
      repository(owner:$owner,name:$name){
        issue(number:$num){ id }
      }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"owner={owner}",
        "-F", f"name={name}",
        "-F", f"num={issue_id}",
    ]
    data = json.loads(run(cmd))
    return data["data"]["repository"]["issue"]["id"]


def resolve_issue_priority_field(org):
    """Return (field_id, {option_name: option_id}) for the org's Priority field.

    'Priority' is a native org-level issue field (Settings -> Planning -> Issue
    fields), a single-select whose option names are the tier names.
    """
    q = """
    query($org:String!){
      organization(login:$org){
        issueFields(first:50){
          nodes{
            ... on IssueFieldSingleSelect { id name options{ id name } }
          }
        }
      }
    }"""
    cmd = ["gh", "api", "graphql", "-f", f"query={q}", "-F", f"org={org}"]
    nodes = json.loads(run(cmd))["data"]["organization"]["issueFields"]["nodes"]
    for n in nodes:
        if n and n.get("name") == PRIORITY_FIELD_NAME:
            return n["id"], {o["name"]: o["id"] for o in n["options"]}
    sys.stderr.write(
        f"native issue field {PRIORITY_FIELD_NAME!r} not found in org {org}\n"
    )
    sys.exit(1)


def set_issue_priority(issue_id, field_id, option_id):
    q = """
    mutation($issue:ID!,$field:ID!,$opt:ID!){
      setIssueFieldValue(input:{
        issueId:$issue,
        issueFields:[{fieldId:$field, singleSelectOptionId:$opt}]
      }){ issue{ id } }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"issue={issue_id}",
        "-F", f"field={field_id}",
        "-F", f"opt={option_id}",
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_md", help="path to a labels.md artifact")
    ap.add_argument("--apply", action="store_true",
                    help="actually write to GitHub (default: dry run)")
    ap.add_argument("--no-comment", action="store_true",
                    help="do not post the labels.md content as an issue comment")
    args = ap.parse_args()

    priority_order = load_priority_order()
    repo, issue_id, rows, full_text = parse_labels_md(args.labels_md)
    labels, native_type, priority_tier = classify(rows, priority_order)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] {repo}#{issue_id}")
    print(f"  labels     : {labels}")
    print(f"  type       : {native_type}")
    print(f"  priority   : {priority_tier}")
    print(f"  comment    : {'no' if args.no_comment else 'yes'} "
          f"({len(full_text)} chars)")

    if not args.apply:
        print("\n-- dry run; re-run with --apply to write to GitHub --")
        return

    # 1. labels + native type
    edit_cmd = ["gh", "issue", "edit", issue_id, "--repo", repo]
    for lb in labels:
        edit_cmd += ["--add-label", lb]
    if native_type:
        edit_cmd += ["--type", native_type]
    if labels or native_type:
        run(edit_cmd, capture=False)
        print("  applied labels/type")

    # 2. comment
    if not args.no_comment:
        run(["gh", "issue", "comment", issue_id, "--repo", repo,
             "--body", full_text], capture=False)
        print("  posted comment")

    # 3. native org Priority issue field
    if priority_tier:
        org = repo.split("/", 1)[0]
        node_id = issue_node_id(repo, issue_id)
        field_id, options = resolve_issue_priority_field(org)
        if priority_tier not in options:
            sys.stderr.write(
                f"priority tier {priority_tier!r} not in issue field "
                f"options {list(options)}; skipping priority\n"
            )
        else:
            set_issue_priority(node_id, field_id, options[priority_tier])
            print(f"  set Priority = {priority_tier}")


if __name__ == "__main__":
    main()
