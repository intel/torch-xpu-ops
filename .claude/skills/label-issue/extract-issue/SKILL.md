---
name: extract-issue
description: Extract metadata from a single intel/torch-xpu-ops GitHub issue and output JSON, using only gh and your own reading of the issue. Use when you need issue_id, title, status, labels, issue_type, test cases, traceback, reproduce steps, platform, and PyTorchXPU project fields for ONE issue given its number or URL. Emits the extraction JSON consumed by the parent label-issue skill without running any script.
---

# Extract Issue Info

Fetch ONE GitHub issue with `gh` and emit its metadata plus classification as
JSON. You do the parsing and classification yourself by reading the issue; no
Python, no scripts.

This is the only extraction path for the parent `label-issue` skill.

This skill is **analysis-only**. It fetches issue data read-only and never
writes to GitHub — no label edits, comments, or issue creation. Its single
artifact is the extraction JSON.

Do NOT use this for batch/multi-issue runs or Excel output; it handles exactly
one issue per invocation.

## Prerequisites

Authenticated `gh` CLI on `PATH`, with `read:project` scope for the GraphQL
project fields. `gh`, plus `ls`/`cat` for the benchmark model lists, are the only
commands used.

## Inputs

| Argument | Purpose |
|---|---|
| `<issue>` | Bare issue number, or a full GitHub issue URL. |
| `repo` | Repo for a bare number. Default `intel/torch-xpu-ops`. A full URL's own owner/name wins. |
| `pytorch_folder` | Local checkout, read only to load benchmark model lists. |
| `output` | Path to also write the JSON. It is always printed to the conversation. |

## Workflow

### Step 1 - Resolve the reference

| Input | Result |
|---|---|
| Bare digits | Use `repo` (default `intel/torch-xpu-ops`). |
| `https://github.com/<owner>/<repo>/issues/<n>` | That owner/repo/number. |
| `https://github.com/<owner>/<repo>/pull/<n>` | **hard-stop**: `<owner>/<repo>#<n> is a pull request, not an issue`. |
| Anything else, or a `repo` without a `/` | **hard-stop**: `Invalid issue reference: <ref>`. |

The PR case is its own rejection, not a malformed input.

### Step 2 - Fetch

```bash
gh api repos/<owner>/<repo>/issues/<number>
```

Non-zero exit, or empty/non-JSON output, is a **hard-stop**; `Not Found` in the
error means the issue does not exist. A `pull_request` key in the response is the
PR rejection from Step 1.

Then fetch the native issue type, the native `Priority` issue field, and the
PyTorchXPU project fields:

```bash
gh api graphql -f query='
query($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    issue(number:$number) {
      issueType { name }
      issueFieldValues(first: 30) {
        nodes {
          ... on IssueFieldSingleSelectValue { name field { ... on IssueFieldCommon { name } } }
        }
      }
      projectItems(first: 10) {
        nodes {
          project { title }
          fieldValues(first: 30) {
            nodes {
              ... on ProjectV2ItemFieldSingleSelectValue { name field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldTextValue      { text field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldNumberValue    { number field { ... on ProjectV2FieldCommon { name } } }
            }
          }
        }
      }
    }
  }
}' -F owner=<owner> -F name=<repo> -F number=<number>
```

`priority` comes from the issue's native org-level `Priority` issue field
(Settings -> Planning -> Issue fields) read from `issueFieldValues`. It is NOT a
project field: do NOT read `Priority` from the PyTorchXPU project. Take the
single-select value whose field `name` is `Priority` (the tier name, e.g.
`High`), or `""`.

From the project titled `PyTorchXPU`, read only the `pytorchxpu_*` fields,
mapping by field name: `Status`, `Estimate`, `Depending`, and `Short Comments`
to the matching `pytorchxpu_*` fields.

This fetch is **best-effort**. On any failure, or for a repo/org without the
field or project, set `issue_type`, `priority`, and every `pytorchxpu_*` field to
`""` and continue. That is not a hard-stop.

### Step 3 - Classify

Based on the issue title, labels, fields, body,
fill in every field per the **Output schema** table below, reading the rule pack
named in a field's Rule cell before deciding that field. Do not classify from
memory.

Each axis is independent, with one exception the pack itself states: decide
`test` before `test_cases`, because an `e2e` issue uses the E2E case shape
and skips unit-test parsing entirely.

### Step 4 - Emit

Emit one JSON object with exactly the fields in **Output schema**, in that order.
Print it, and write it to `output` when that was given.

## Output schema

One JSON object. Emit exactly these fields, in this order. A blank source
always yields `""`.

| Field | Source | Rule |
|-------|--------|------|
| issue_id | gh REST | Issue number (integer). |
| repo | input | Resolved `owner/name`. |
| title | gh REST | Issue title. |
| body | gh REST | Raw body, verbatim - never summarized, reflowed, or truncated. |
| status | gh REST | `open` or `closed`. |
| assignee | gh REST | First assignee login, or "". |
| reporter | gh REST | Issue author login. |
| labels | gh REST | Array of label name strings. |
| created_time / updated_time | gh REST | ISO 8601 timestamps. |
| milestone | gh REST | Milestone title, or "". |
| summary | gh REST | The `title`, verbatim. |
| issue_type | gh GraphQL | The GitHub **Type** field (`issueType.name`) verbatim: `Bug` \| `Task` \| `Feature` \| `Epic`. |
| pytorchxpu_status / _estimate / _depending / _short_comments | gh GraphQL | Project fields, or "". |
| priority | gh GraphQL / labels | The native org-level `Priority` issue field (from `issueFieldValues`) verbatim, else a human-set priority label from `categories.priority` of `../reference/proposed_labels.json`. NOT the PyTorchXPU project field. `""` when neither is set — the parent decides `priority` itself when blank. Never inferred from severity text. |
| os | you | An `os` code from `categories.os` of `../reference/proposed_labels.json`, or "", derived per [../reference/platform_specific.md](../reference/platform_specific.md). |
| hw | you | A `hw` code from `categories.hw` of `../reference/proposed_labels.json`, or "", derived per [../reference/platform_specific.md](../reference/platform_specific.md). |
| platform_specific | you | `true`/`false`, derived per [../reference/platform_specific.md](../reference/platform_specific.md). Judged from the text; never probe local hardware. |
| test | you | `ut` \| `e2e` \| `oob`, per [../reference/testcase_rules.md](../reference/testcase_rules.md) (keywords in `categories.test` of `../reference/proposed_labels.json`). |
| dependency | labels | A `dependency` code from `categories.dependency` of `../reference/proposed_labels.json` when a human-set dependency label is present on the issue, else `""`. Copied from the label only; never inferred from the traceback — the parent re-derives it from evidence when blank. |
| module | labels | A `module` code from `categories.module` of `../reference/proposed_labels.json` when a human-set module label is present on the issue, else `""`. Copied from the label only; the parent re-derives it from the traced root cause when blank. |
| traceback | you | Full Python traceback, chained segments included, per [../reference/text_rules.md](../reference/text_rules.md). |
| error_message | you | Issue-level normalized error/exception header (the sole failure signature), or "". Per-case `error_message` lives on each `test_cases[]` entry per [../reference/testcase_rules.md](../reference/testcase_rules.md). |
| reproduce_steps | you | Shell command lines, newline-joined, prose excluded, per [../reference/text_rules.md](../reference/text_rules.md). |
| test_file / test_class / test_case | you | Mirror of the first unit-test-shaped `test_cases` entry, per the **Top-level mirror fields** section of [../reference/testcase_rules.md](../reference/testcase_rules.md). All "" on an E2E issue. |
| test_cases | you | Every parsed case, in the scan order fixed by the **Ordering** contract of [../reference/testcase_rules.md](../reference/testcase_rules.md) - which the parent relies on for a stable `test_cases[0]`. |
| pr_link | you | PR URL the issue is tied to, per [../reference/text_rules.md](../reference/text_rules.md). |

Fields sourced from `gh REST` or `gh GraphQL` are copied from that one response.
Do not re-derive `issue_type` from the title text, the body, or the labels: a
blank Type field means `""`. `priority`, `dependency`, and `module` are copied
from the native `Priority` issue field (priority only) or a human-set label; when no
such field/label exists they are `""` and the parent decides them itself (e.g.
`priority` in Step 8 of the label-issue skill), so an invented value would
suppress that.

## Authoritative-source fields

Five fields read ONE authoritative source rather than guessing from keywords. A
blank source yields `""`; the parent re-derives these axes anyway, so an empty
field beats a guess.

| Field | Source |
|---|---|
| `summary` | The full title, verbatim. |
| `issue_type` | The GitHub Type field only, so `""` when unset. |
| `platform_specific` | Judged from the issue text, never from local hardware. |
| `pr_link` | A `/pull/` URL, or a resolved `owner/repo#N` or bare `#N`. |

## Constraints

1. Read-only with respect to GitHub. Never `gh issue edit`, `gh issue create`,
   `gh issue close`, `gh issue comment`, or any mutation. Fetch only.
2. Do all parsing and classification by reading the issue. Never run a Python
   script for extraction, and never shell out to one; `python3` being available
   is not a reason to use it.
3. Do not modify `pytorch_folder`. Read the model lists only.
4. Never invent a `file:line`, a test case, or a model name. Absent evidence
   yields `""`.

## Hard stops

- Missing issue reference, a malformed reference, or a `repo` without a `/`.
- The reference is a pull request, whether a bare number or a `/pull/<n>` URL.
- `gh` missing or unauthenticated.
- Fetch failure: 404, network error, empty or non-JSON response.

Normal degraded outcomes, NOT hard stops: a failed project/GraphQL fetch, a
missing `pytorch_folder`, unavailable benchmark model lists, and any `""` axis.
