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

Every write is best-effort and independent: labels, native type, and the Priority
field each warn-and-skip on failure instead of aborting, so one unavailable
surface (e.g. the native issue-fields API not enabled for the token) cannot undo
the writes that already succeeded. The Priority step runs last for this reason.
The script exits 0 on such partial success; it exits non-zero only on an
unexpected hard error (e.g. a required read failing with `check=True`) or on a
detected destructive replace (see below), which exits 2.

setIssueFieldValue's issueFields:[...] merge-vs-replace semantics are not verified
live. As a safeguard the Priority step snapshots the issue's other native field
values before the write and re-reads after: if the write cleared any of them
(i.e. it replaced rather than merged) it prints a loud ERROR and exits 2 -- this
is genuine silent data loss, not a partial success, so a wrapping workflow must
fail rather than go green with the message buried in the log. This detects, but
cannot prevent, a destructive replace -- a live merge check is still recommended
before merge.

The native Type and Priority fields are only written when the issue has no value
set for them yet; an existing value (e.g. set manually) is left unchanged, so
re-runs are idempotent and never clobber a human's triage. When an existing value
disagrees with what labels.md proposes, the field is still left untouched and the
mismatch is highlighted in the posted comment (a `> [!WARNING]` block) for a human
to reconcile. If a pre-write read degrades (None), the value can't be checked and
the write proceeds.

Dry-run by default: prints exactly what it would do. Pass --apply to write.

labels.md is derived from attacker-controlled issue text, so this script only
ever ADDS: every proposed label must appear in labels.json (unknown ones are
dropped), at most MAX_LABELS are applied, no existing label is removed, and the
native Type/Priority fields are never overwritten.

For a `need_split` issue, the per-group axes (module, dtype, dependency
component, symptom, duplicate, wontfix, and the priority) are NOT applied to the
umbrella issue -- they belong to the individual sub-issues created after the
split. Only issue-wide axes (need_split, type, test, os, hw) are applied.

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
# labels.json lives with the label-issue skill. Resolve it robustly:
# the workflow may colocate it next to this script (HERE/../reference), otherwise
# fall back to the in-repo skill copy. First existing path wins.
_LABELS_JSON_CANDIDATES = [
    os.path.join(HERE, "..", "reference", "labels.json"),
    os.path.join(
        HERE, "..", "..", ".claude", "skills", "label-issue",
        "reference", "labels.json",
    ),
]
LABELS_JSON = next(
    (p for p in _LABELS_JSON_CANDIDATES if os.path.isfile(p)),
    _LABELS_JSON_CANDIDATES[0],
)

# Upper bound on how many labels a single run may add. labels.md is derived from
# attacker-controlled issue text, so cap the blast radius of a bad artifact.
MAX_LABELS = 15

# Name of the native org-level issue field that holds the priority tier.
PRIORITY_FIELD_NAME = "Priority"

# Axes that are NOT applied as labels.
NATIVE_TYPE_AXIS = "type"
PRIORITY_AXIS = "priority"

# Issue-wide axes that carry exactly one label per issue. A multi-group artifact
# holds one row per group for each, so they are collapsed rather than unioned.
SINGLE_LABEL_AXES = ("test", "os", "hw")


def run(cmd, check=True, capture=True):
    """Run a command, returning stdout (str). When check=False, returns None on
    a non-zero exit instead of raising, so callers can handle partial failures."""
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        if not check:
            sys.stderr.write(f"command failed: {' '.join(cmd)}\n{res.stderr}\n")
            return None
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
    with open(LABELS_JSON) as f:
        d = json.load(f)
    return {v["tier"]: i for i, v in enumerate(d["priority_field"]["values"])}


def load_known_labels():
    """Every label name defined in labels.json, as a set.

    labels.md is generated from issue text, which is attacker-controlled, so the
    proposed labels are an allowlist match against this set before anything is
    written: an injected instruction can then at worst pick a wrong label from a
    fixed vocabulary, never invent one.
    """
    with open(LABELS_JSON) as f:
        d = json.load(f)
    return {
        lb["name"]
        for cat in d["categories"].values()
        for lb in cat.get("labels", [])
    }


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

    A multi-group issue carries one row per group for axes that are per-issue on
    GitHub, so each is collapsed here: type to a single value (Bug wins over any
    other tier, since a reported defect dominates a feature request on an
    umbrella issue), priority to the most urgent tier across groups (rank from
    priority_order, 0 = most urgent), and the single-label issue-wide axes
    (test, os, hw) to their shared value -- dropped entirely when the groups
    disagree, because an umbrella issue must never carry two of them at once.

    When the issue is `need_split`, the per-group axes (module, dtype,
    dependency component, symptom, duplicate, wontfix, and the priority) describe
    the individual sub-issues, not the umbrella issue, so they are dropped here
    and left for the post-split sub-issues.
    """
    need_split = any(
        axis == "triage" and value == "need_split" for axis, value in rows
    )
    # axis names (parsed, lowercased) suppressed on a need_split issue
    split_drop_axes = {"module", "dtype", "dependency component", "symptom"}

    labels = []
    types = []
    priorities = []
    single = {}  # single-label issue-wide axis -> list of distinct values
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
        elif axis in SINGLE_LABEL_AXES:
            vals = single.setdefault(axis, [])
            if value not in vals:
                vals.append(value)
        else:
            if need_split and axis in split_drop_axes:
                continue
            if need_split and axis == "triage" and value in ("duplicate", "wontfix"):
                continue
            # everything else is a label token, applied verbatim
            if value not in seen:
                seen.add(value)
                labels.append(value)

    for axis, vals in single.items():
        if len(vals) > 1:
            sys.stderr.write(
                f"warning: groups disagree on {axis} {vals}; "
                f"applying no {axis} label\n"
            )
            continue
        if vals[0] not in seen:
            seen.add(vals[0])
            labels.append(vals[0])

    native_type = None
    if types:
        native_type = "Bug" if "Bug" in types else types[0]
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
    out = run(cmd, check=False)
    if out is None:
        return None
    data = json.loads(out).get("data") or {}
    repo_node = (data.get("repository") or {}).get("issue") or {}
    return repo_node.get("id")


def resolve_issue_priority_field(org):
    """Return (field_id, {option_name: option_id}) for the org's Priority field,
    or None if the native issue-fields surface is unavailable to the token or the
    field is not defined. The caller treats None as "skip Priority" rather than a
    hard failure, so a mostly-successful run still exits 0.

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
    out = run(cmd, check=False)
    if out is None:
        return None
    data = json.loads(out).get("data") or {}
    nodes = ((data.get("organization") or {}).get("issueFields") or {}).get(
        "nodes"
    )
    if not nodes:
        return None
    for n in nodes:
        if n and n.get("name") == PRIORITY_FIELD_NAME:
            return n["id"], {o["name"]: o["id"] for o in n["options"]}
    sys.stderr.write(
        f"native issue field {PRIORITY_FIELD_NAME!r} not found in org {org}\n"
    )
    return None


def read_issue_field_values(node_id):
    """Return {field_name: rendered_value} for the issue's currently-populated
    native fields, or None if the read fails/degrades.

    Covers the single-select, text, number, and date value types. Any other
    value type (e.g. a future iteration/user field) matches no fragment and is
    silently skipped, so it would NOT be seen as cleared by the replace detector.
    Empty ({}) means "read OK, no fields of a covered type set".

    Used to snapshot other native fields before writing Priority so a
    replace-style setIssueFieldValue (which would silently clear them) can be
    detected after the fact.
    """
    q = """
    query($id:ID!){
      node(id:$id){
        ... on Issue {
          issueFieldValues(first:50){
            nodes{
              ... on IssueFieldSingleSelectValue {
                name
                field{ ... on IssueFieldCommon { name } }
              }
              ... on IssueFieldTextValue {
                text
                field{ ... on IssueFieldCommon { name } }
              }
              ... on IssueFieldNumberValue {
                number
                field{ ... on IssueFieldCommon { name } }
              }
              ... on IssueFieldDateValue {
                date
                field{ ... on IssueFieldCommon { name } }
              }
            }
          }
        }
      }
    }"""
    cmd = ["gh", "api", "graphql", "-f", f"query={q}", "-f", f"id={node_id}"]
    out = run(cmd, check=False)
    if out is None:
        return None
    data = json.loads(out).get("data") or {}
    node = data.get("node") or {}
    fv = node.get("issueFieldValues")
    if fv is None:
        return None
    result = {}
    for n in fv.get("nodes") or []:
        if not n:
            continue
        name = ((n.get("field") or {}).get("name"))
        # exactly one value key is present per node, depending on the field type
        val = next(
            (n[k] for k in ("name", "text", "number", "date") if n.get(k) is not None),
            None,
        )
        if name and val is not None:
            result[name] = val
    return result


def read_issue_type(repo, issue_id):
    """Return the issue's current native Type name, "" if it has none, or None
    if the read failed/degraded. Used to decide whether to apply the proposed
    Type and to flag a mismatch, without ever overwriting an existing Type.
    """
    owner, name = repo.split("/", 1)
    q = """
    query($owner:String!,$name:String!,$num:Int!){
      repository(owner:$owner,name:$name){
        issue(number:$num){ issueType{ name } }
      }
    }"""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={q}",
        "-F", f"owner={owner}",
        "-F", f"name={name}",
        "-F", f"num={issue_id}",
    ]
    out = run(cmd, check=False)
    if out is None:
        return None
    data = json.loads(out).get("data") or {}
    issue = (data.get("repository") or {}).get("issue")
    if issue is None:
        return None
    it = issue.get("issueType")
    return (it or {}).get("name") or ""


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
        "-f", f"issue={issue_id}",
        "-f", f"field={field_id}",
        "-f", f"opt={option_id}",
    ]
    return run(cmd, check=False)


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

    # Allowlist: only labels defined in labels.json may be written.
    known = load_known_labels()
    unknown = [lb for lb in labels if lb not in known]
    if unknown:
        sys.stderr.write(
            f"warning: dropping label(s) not defined in labels.json: {unknown}\n"
        )
        labels = [lb for lb in labels if lb in known]
    if len(labels) > MAX_LABELS:
        sys.stderr.write(
            f"warning: {len(labels)} labels exceeds the cap of {MAX_LABELS}; "
            f"dropping {labels[MAX_LABELS:]}\n"
        )
        labels = labels[:MAX_LABELS]

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

    # Read existing native Type and Priority up front. Both fields are applied
    # ONLY when the issue has no value yet; an existing value is never
    # overwritten, and a value that disagrees with labels.md is highlighted in
    # the posted comment for a human to reconcile. Reads are best-effort: a
    # degraded read (None) is treated as "unknown" and the field is applied.
    node_id = issue_node_id(repo, issue_id)
    existing_type = read_issue_type(repo, issue_id)
    field_values = read_issue_field_values(node_id) if node_id else None
    existing_priority = (field_values or {}).get(PRIORITY_FIELD_NAME)

    mismatch_notes = []  # highlighted lines appended to the comment

    apply_type = bool(native_type)
    if native_type and existing_type:
        apply_type = False  # already set; never overwrite
        if existing_type != native_type:
            mismatch_notes.append(
                f"- **Type**: proposed `{native_type}`, but the issue already "
                f"has `{existing_type}` -- left unchanged."
            )

    apply_priority = bool(priority_tier)
    if priority_tier and existing_priority:
        apply_priority = False  # already set; never overwrite
        if existing_priority != priority_tier:
            mismatch_notes.append(
                f"- **Priority**: proposed `{priority_tier}`, but the issue "
                f"already has `{existing_priority}` -- left unchanged."
            )

    # 1. labels + native type; apply each separately so one unknown label
    # (which makes gh reject the whole command) cannot drop the others or --type.
    for lb in labels:
        rc = run(["gh", "issue", "edit", issue_id, "--repo", repo,
                  "--add-label", lb], check=False, capture=False)
        if rc is None:
            print(f"  warning: failed to add label {lb!r}; skipped")
        else:
            print(f"  added label {lb}")
    if native_type and not apply_type:
        print(f"  Type already set on {repo}#{issue_id}; leaving it unchanged")
    elif apply_type:
        rc = run(["gh", "issue", "edit", issue_id, "--repo", repo,
                  "--type", native_type], check=False, capture=False)
        if rc is None:
            print(f"  warning: failed to set type {native_type!r}; skipped")
        else:
            print(f"  set type {native_type}")

    # 2. comment (labels.md body, plus any native-field mismatch highlight)
    if not args.no_comment:
        body = full_text
        if mismatch_notes:
            body += (
                "\n\n> [!WARNING]\n"
                "> Native field(s) already set and NOT changed; the values below "
                "disagree with this analysis -- please reconcile manually:\n"
                + "\n".join(f"> {n}" for n in mismatch_notes)
            )
        run(["gh", "issue", "comment", issue_id, "--repo", repo,
             "--body", body], capture=False)
        print("  posted comment")
    elif mismatch_notes:
        # The comment is their only GitHub-side channel, so with --no-comment
        # print them instead of losing the disagreement entirely.
        sys.stderr.write(
            "warning: native field(s) already set and NOT changed; the values "
            "below disagree with this analysis:\n"
            + "\n".join(mismatch_notes) + "\n"
        )

    # 3. native org Priority issue field. Best-effort: the native issue-fields
    # surface may be unavailable to the token, so any failure here warns and
    # skips rather than failing the whole run (labels/type/comment already wrote).
    if priority_tier and not apply_priority:
        print(f"  Priority already set on {repo}#{issue_id}; leaving it unchanged")
    elif apply_priority:
        org = repo.split("/", 1)[0]
        resolved = resolve_issue_priority_field(org) if node_id else None
        if node_id is None or resolved is None:
            print("  warning: could not resolve native Priority field; "
                  "Priority skipped")
        else:
            field_id, options = resolved
            if priority_tier not in options:
                sys.stderr.write(
                    f"priority tier {priority_tier!r} not in issue field "
                    f"options {list(options)}; skipping priority\n"
                )
            else:
                # field_values (read before any native-field write) doubles as
                # the pre-mutation snapshot for replace detection below
                # (None = read degraded, verification skipped).
                before = field_values
                if set_issue_priority(node_id, field_id, options[priority_tier]) is None:
                    print(f"  warning: failed to set Priority = {priority_tier}; "
                          "skipped")
                else:
                    print(f"  set Priority = {priority_tier}")
                    # setIssueFieldValue's issueFields:[...] merge-vs-replace
                    # semantics are not verified live. If it REPLACES, writing only
                    # Priority silently clears the issue's other native fields, and
                    # the mutation still returns success -- so verify by re-reading:
                    # warn loudly if any field populated before is now gone.
                    if before is not None:
                        after = read_issue_field_values(node_id)
                        if after is not None:
                            cleared = [
                                f for f in before
                                if f != PRIORITY_FIELD_NAME and f not in after
                            ]
                            if cleared:
                                lost = {f: before[f] for f in cleared}
                                sys.stderr.write(
                                    "ERROR: setting Priority appears to have CLEARED "
                                    f"other native fields {lost} on {repo}#{issue_id}. "
                                    "setIssueFieldValue replaced rather than merged; "
                                    "restore those fields manually and do not reuse "
                                    "this write path until fixed.\n"
                                )
                                # Unlike the other best-effort failures (which are
                                # legitimate partial successes), a destructive
                                # replace is silent data loss: exit non-zero so a
                                # wrapping workflow fails loudly instead of going
                                # green with the ERROR buried in the log.
                                sys.exit(2)


if __name__ == "__main__":
    main()
