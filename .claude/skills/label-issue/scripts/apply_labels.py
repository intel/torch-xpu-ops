#!/usr/bin/env python3
"""Apply the axes decided in a label-issue labels.md to a GitHub issue.

Reads a labels.md artifact (the collapsible <details> table produced by the
label-issue skill), then writes to GitHub:

  * label rows (test:, module:, dtype:, symptom, dependency component:, and the
    triage duplicate/wontfix/need_split rows)  -> gh issue edit --add-label
  * the `type` row                             -> gh issue edit --type  (native Type field)
  * the `priority` row                         -> PyTorchXPU project Priority field (GraphQL)
  * the full labels.md content                 -> posted as an issue comment

Dry-run by default: prints exactly what it would do. Pass --apply to write.

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

PROJECT_TITLE = "PyTorchXPU"
PROJECT_ORG = "intel"

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
        sys.stderr.write(f"command failed: {' '.join(cmd)}\n{res.stderr}\n")
        sys.exit(1)
    return (res.stdout or "").strip()


def gql(query, **variables):
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        cmd += ["-f", f"{k}={v}"]
    return json.loads(run(cmd))


def load_priority_map():
    """tier (e.g. 'Medium') -> project option name (e.g. 'P2')."""
    with open(PROPOSED) as f:
        d = json.load(f)
    return {v["tier"]: v["option"] for v in d["priority_field"]["values"]}


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


def classify(rows, priority_map):
    """Split parsed rows into labels, native type, priority option.

    A multi-group issue carries one type/priority row per group, but GitHub's
    Type and the project's Priority are per-issue. Collapse type to the single
    distinct value (warn on conflict) and priority to the most urgent across
    groups (P0 < P1 < P2 < P3).
    """
    labels = []
    types = []
    priorities = []
    seen = set()

    for axis, value in rows:
        if axis == NATIVE_TYPE_AXIS:
            if value not in types:
                types.append(value)
        elif axis == PRIORITY_AXIS:
            opt = priority_map.get(value, value)
            if opt not in priorities:
                priorities.append(opt)
        else:
            # everything else is a label token, applied verbatim
            if value not in seen:
                seen.add(value)
                labels.append(value)

    native_type = types[0] if types else None
    if len(types) > 1:
        sys.stderr.write(
            f"warning: multiple type values {types}; using {native_type}\n"
        )

    priority_option = None
    if priorities:
        # most urgent = smallest P-number; unknown options sort last
        def rank(opt):
            m = re.match(r"P(\d+)", opt)
            return int(m.group(1)) if m else 99
        priority_option = min(priorities, key=rank)
        if len(priorities) > 1:
            sys.stderr.write(
                f"warning: multiple priority values {priorities}; "
                f"using most urgent {priority_option}\n"
            )
    return labels, native_type, priority_option


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


def resolve_project():
    q = """
    query($org:String!,$query:String!){
      organization(login:$org){
        projectsV2(first:10, query:$query){ nodes{ id title } }
      }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"org={PROJECT_ORG}",
        "-F", f"query={PROJECT_TITLE}",
    ]
    nodes = json.loads(run(cmd))["data"]["organization"]["projectsV2"]["nodes"]
    for n in nodes:
        if n["title"] == PROJECT_TITLE:
            return n["id"]
    sys.stderr.write(f"project {PROJECT_TITLE!r} not found in org {PROJECT_ORG}\n")
    sys.exit(1)


def resolve_priority_field(project_id):
    q = """
    query($id:ID!){
      node(id:$id){
        ... on ProjectV2 {
          field(name:"Priority"){
            ... on ProjectV2SingleSelectField { id options{ id name } }
          }
        }
      }
    }"""
    cmd = ["gh", "api", "graphql", "-f", f"query={q}", "-F", f"id={project_id}"]
    field = json.loads(run(cmd))["data"]["node"]["field"]
    return field["id"], {o["name"]: o["id"] for o in field["options"]}


def add_item_to_project(project_id, content_id):
    q = """
    mutation($proj:ID!,$content:ID!){
      addProjectV2ItemById(input:{projectId:$proj, contentId:$content}){
        item{ id }
      }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"proj={project_id}",
        "-F", f"content={content_id}",
    ]
    return json.loads(run(cmd))["data"]["addProjectV2ItemById"]["item"]["id"]


def set_priority(project_id, item_id, field_id, option_id):
    q = """
    mutation($proj:ID!,$item:ID!,$field:ID!,$opt:String!){
      updateProjectV2ItemFieldValue(input:{
        projectId:$proj, itemId:$item, fieldId:$field,
        value:{ singleSelectOptionId:$opt }
      }){ projectV2Item{ id } }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"proj={project_id}",
        "-F", f"item={item_id}",
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

    priority_map = load_priority_map()
    repo, issue_id, rows, full_text = parse_labels_md(args.labels_md)
    labels, native_type, priority_option = classify(rows, priority_map)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] {repo}#{issue_id}")
    print(f"  labels     : {labels}")
    print(f"  type       : {native_type}")
    print(f"  priority   : {priority_option}")
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

    # 3. project priority
    if priority_option:
        content_id = issue_node_id(repo, issue_id)
        project_id = resolve_project()
        field_id, options = resolve_priority_field(project_id)
        if priority_option not in options:
            sys.stderr.write(
                f"priority option {priority_option!r} not in project "
                f"options {list(options)}; skipping priority\n"
            )
        else:
            item_id = add_item_to_project(project_id, content_id)
            set_priority(project_id, item_id, field_id, options[priority_option])
            print(f"  set project Priority = {priority_option}")


if __name__ == "__main__":
    main()
