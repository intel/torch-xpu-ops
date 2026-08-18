---
name: extract-issue
description: Extract metadata from a single intel/torch-xpu-ops GitHub issue and output JSON, using only gh and your own reading of the issue. Use when you need issue_id, title, status, labels, type/issue_type, test_module, test cases, traceback, reproduce steps, platform, and PyTorchXPU project fields for ONE issue given its number or URL, and python3 or the extraction scripts are unavailable. Reads module and dependency off existing labels only. Emits the same JSON schema as extract-issue-information without running any script.
---

# Extract Issue Info (script-free)

Fetch ONE GitHub issue with `gh` and emit its metadata plus classification as
JSON. You do the parsing and classification yourself by reading the issue; no
Python, no `patterns.json`.

Output is **schema-identical** to
`label-issue/extract-issue-information`, so the parent `label-issue` skill
consumes either interchangeably.

Do NOT use this for batch/multi-issue runs or Excel output; it handles exactly
one issue per invocation.

## Prerequisites

- Authenticated `gh` CLI on `PATH`, with `read:project` scope for the GraphQL
  project fields.
- No Python. No script execution. `gh`, plus `ls`/`cat` to read the benchmark
  model lists, are the only commands used.

## Inputs

| Argument | Purpose |
|---|---|
| `<issue>` | Bare issue number, or a full GitHub issue URL. |
| `repo` | Repo for a bare number. Default `intel/torch-xpu-ops`. A full URL's own owner/name wins. |
| `pytorch_folder` | Local checkout used only to load benchmark model lists. Never modified. |
| `output` | Path to also write the JSON. It is always printed to the conversation. |

## Reference

Read these before classifying the axis they govern. Do not classify from memory.

| Axis | File |
|---|---|
| `test_module`, `test_cases` | [reference/testcase_rules.md](reference/testcase_rules.md) |
| `os`, `platform`, `platform_specific` | [reference/platform_rules.md](reference/platform_rules.md) |
| `traceback`, `reproduce_steps`, `pr_link` | [reference/text_rules.md](reference/text_rules.md) |

`summary`, `issue_type`, `priority`, `module`, and `dependency` have no rule
pack: each is copied from a single source. See **Direct fields** and **Module
and dependency** below.

## Direct fields

Copy each from its one source. No keyword matching, no inference. `""` when the
source is blank.

| Field | Source |
|---|---|
| `summary` | The issue `title`, verbatim. |
| `issue_type` | The GitHub **Type** field (`issueType.name` in the GraphQL response). |
| `priority` | The PyTorchXPU project **Priority** field. |
| `github_type` | Same as `issue_type`: the raw GitHub Type field. |
| `type` | Lowercase of `issue_type` (`Bug` -> `bug`), or `""`. |

Normalize `priority` only: `P0` -> `Urgent`, `P1` -> `High`, `P2` -> `Medium`,
`P3` -> `Low`. The names `Urgent`/`High`/`Medium`/`Low` pass through as-is.
Anything else -> `""`.

Do not derive any of these from the title text, the body, or the labels. A blank
source field means `""`, not a guess - the parent skill decides `priority`
itself from `reference/priority.md` when this field is empty, and an invented
value would suppress that.

## Module and dependency

Read these two axes off the issue's EXISTING labels only. Do not infer them from
the title, body, or traceback.

| Field | Rule |
|---|---|
| `module_label` | The issue's `module: <x>` label, verbatim. `""` when it carries none. |
| `module` | The bucket for that label, per [../reference/module.md](../reference/module.md). `""` when there is no label. |
| `dependency_label` | The issue's `dependency component: <x>` or `dependency: third_party packages` label, verbatim. `""` when it carries none. |
| `dependency` | The taxonomy value for that label, per [../reference/dependency.md](../reference/dependency.md). `""` when there is no label. |

Two label names do not follow the common prefix pattern, so map them through the
reference tables rather than string-stripping the prefix:

- `module: ao` -> bucket `torchAO`, and `module: core` -> bucket `torch-runtime`
- `dependency: third_party packages` -> value `third_party_packages`

`module: ut` is NOT a module value. It is a `test_module` signal; ignore it here.

If the issue has several `module:` labels, take the first one GitHub returned and
emit only that. Same for `dependency`.

**Why label-only.** The parent `label-issue` skill re-derives both axes from
`reference/module.md` and `reference/dependency.md` against the traced root cause
(its Steps 3 and 6), and it overrides whatever appears here when the trace
disagrees. Keyword-guessing these fields would add a second, weaker opinion that
the parent then has to discard. An empty value is the honest answer when the
issue is unlabeled, and it is strictly more useful to the parent than a guess.

Consequence: unlike the script, `module` here can be `""`. The script always
emits a bucket because it keyword-classifies as a fallback. The parent tolerates
this - it re-derives the axis regardless - but do not report `others` as if a
bucket had been determined.

## Workflow

### Step 1 - Resolve the reference

Accept a bare number or a full URL.

- Bare digits -> use `repo` (default `intel/torch-xpu-ops`).
- `https://github.com/<owner>/<repo>/issues/<n>` -> that owner/repo/number.
- `https://github.com/<owner>/<repo>/pull/<n>` -> **hard-stop**, reason
  `<owner>/<repo>#<n> is a pull request, not an issue`. This is the PR rejection,
  not a malformed input.
- Anything else -> **hard-stop**, reason `Invalid issue reference: <ref>`.

A `repo` without a `/` is also a malformed-input hard-stop.

### Step 2 - Fetch

```bash
gh api repos/<owner>/<repo>/issues/<number>
```

Non-zero exit or empty/non-JSON output -> **hard-stop**. `Not Found` in the
error means the issue does not exist. If the response carries a `pull_request`
key, treat it as the PR rejection from Step 1.

Then fetch the PyTorchXPU project fields and native issue type:

```bash
gh api graphql -f query='
query($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    issue(number:$number) {
      issueType { name }
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

Take field values from the project titled `PyTorchXPU`. Map by field name:
`Status` -> `pytorchxpu_status`, `Estimate` -> `pytorchxpu_estimate`,
`Depending` -> `pytorchxpu_depending`, `Short Comments` ->
`pytorchxpu_short_comments`, `Priority` -> `priority`.

This fetch is **best-effort**. On any failure, or for a repo outside the
project, set `github_type`, `priority`, and every `pytorchxpu_*` field to `""`
and continue. That is not a hard-stop.

### Step 3 - Classify

Work through the reference files. Order matters in one place: decide
`test_module` before `test_cases`, because an `e2e` issue uses the E2E case
shape and skips unit-test parsing entirely.

Take `module` and `dependency` straight from the labels per **Module and
dependency** above. Every other axis is independent of the rest.

### Step 4 - Emit

Emit one JSON object with exactly the fields in **Output schema** below, in that
order. Print it, and write it to `output` when that was given.

Then resolve `low_confidence` inline: for each field named there, re-read
`title`/`body`, fill the field in, and remove its name from the list. Do not
leave a populated `low_confidence` in the final answer unless the evidence
genuinely is not in the issue.

## Output schema

A single JSON object. Same field names, order, and types as
`extract-issue-information`.

| Field | Source | Notes |
|-------|--------|-------|
| issue_id | gh REST | Issue number (integer). |
| repo | input | Resolved `owner/name`. |
| title | gh REST | Issue title. |
| body | gh REST | Raw body, verbatim. |
| status | gh REST | `open` or `closed`. |
| assignee | gh REST | First assignee login, or "". |
| reporter | gh REST | Issue author login. |
| labels | gh REST | Array of label name strings. |
| created_time / updated_time | gh REST | ISO 8601 timestamps. |
| milestone | gh REST | Milestone title, or "". |
| summary | you | The `title`, verbatim. "" when blank. |
| type | you | Lowercase of `issue_type` (e.g. `bug`), or "". |
| issue_type | gh GraphQL | The GitHub Type field verbatim (`Bug` \| `Task` \| `Feature` \| `Epic`), or "". |
| github_type | gh GraphQL | Native GitHub issue type name, or "". Same source as `issue_type`. |
| module | you | Bucket for the issue's `module: <x>` label, or "" when it has none. Label-only; never keyword-inferred. |
| module_label | you | The `module: <x>` label verbatim, or "". Emit this, not the bucket. |
| test_module | you | `ut` \| `e2e` \| `build` \| `infrastructure`. |
| dependency | you | Taxonomy value for the issue's dependency label, or "". Label-only; never keyword-inferred. `AO` is never a value. |
| dependency_label | you | The dependency label verbatim, or "". |
| priority | gh GraphQL | PyTorchXPU project Priority field. `P0`->`Urgent` .. `P3`->`Low`; named tiers pass through. "" when the field is blank. |
| pytorchxpu_status / _estimate / _depending / _short_comments | gh GraphQL | Project fields, or "". |
| os | you | `Linux` \| `Windows` \| "". |
| platform | you | PVC, BMG, ARC, ARL, LNL, MTL, CRI, or "". |
| platform_specific | you | `true` when the issue is reported as hardware-specific: a `[PLATFORM]` title tag, a `hw:` label, an "only on X" claim, or a "passed on X, failed on Y" contrast. Else `false`. |
| traceback | you | Full Python traceback, chained segments included, else "". |
| reproduce_steps | you | Shell command lines, newline-joined; "" if none. Prose excluded. |
| test_file / test_class / test_case | you | Mirror of the first **unit-test-shaped** entry in `test_cases` (the first entry with no `benchmark` key), NOT necessarily `test_cases[0]`. "" on an E2E issue. |
| test_cases | you | All parsed cases, ordered per testcase_rules.md. |
| pr_link | you | PR URL the issue is tied to; "" when none. A `/pull/` URL is trusted; an `owner/repo#N` or bare `#N` ref is resolved with one `gh api` call. |
| low_confidence | you | Field names needing a second pass. |

Field names and types match `extract-issue-information` exactly. These are
derived differently, by design - each reads one authoritative source instead of
keyword-classifying:

- `summary` - the full title, where the script truncates to 150 characters.
- `issue_type` / `type` - the GitHub Type field only. The script falls back to
  labels and then to a keyword heuristic, so it always produces a value; here a
  blank Type field yields `""`.
- `module` / `dependency` - existing labels only, so they can be `""` where the
  script would keyword-classify a bucket.
- `platform_specific` - judged from the issue text, where the script compares
  the issue platform against the local GPU. This skill never probes hardware.
- `pr_link` - a bare `#N` is resolved against the current repo. The script does
  not match that form at all, so it can return `""` where this skill finds a PR.

In every case a blank source yields `""`. The parent skill re-derives these axes
from its own reference packs, so an empty field is strictly better than a guess.

## Determinism

The parent skill takes `test_cases[0]` as the analyzed case and needs that
choice stable across runs. Emit `test_cases` in the scan order defined in
`reference/testcase_rules.md` and never re-rank by severity, dtype, or
alphabet. Two runs on one issue must agree on index 0.

## Inline LLM fallback

`low_confidence` lists ONLY these fields, only under these conditions:

| Field | Listed when |
|---|---|
| `reproduce_steps` | No shell command found AND the issue is not a unit test (a unit test's id is its own reproducer). |
| `test_cases` | No case parsed but `test_module` is `ut` or `e2e`. |
| `pr_link` | No PR found, but the body signals a non-main context ("this PR", "my branch", "cherry-pick", "backport", a CI run URL), or an `owner/repo#N` / bare `#N` ref could not be resolved. |

`dependency`, `traceback`, `os`, and `platform` are NEVER flagged.

## Constraints

1. Read-only with respect to GitHub. Never `gh issue edit`, `gh issue create`,
   `gh issue close`, `gh issue comment`, or any mutation. Fetch only.
2. Never run a Python script or any part of
   `extract-issue-information/scripts/`. If `python3` is available, that is
   still not a reason to use it: use the sibling skill instead.
3. Do not modify `pytorch_folder`. Read the model lists only.
4. Quote `body` verbatim in the output. Do not summarize, reflow, or truncate
   it - the parent resolves `low_confidence` from it.
5. Never invent a `file:line`, a test case, or a model name. Absent evidence
   yields "" or an entry in `low_confidence`.
6. Read the reference file for an axis before deciding it.

## Hard stops

- Missing issue reference.
- `gh` missing or unauthenticated.
- Fetch failure: 404, network error, empty or non-JSON response.
- The reference is a pull request, whether given as a bare number or a
  `/pull/<n>` URL.
- Malformed reference, or a `repo` without a `/`.

Not hard stops (normal degraded outcomes): a failed project/GraphQL fetch, a
missing `pytorch_folder`, unavailable benchmark model lists, and any "" axis.
