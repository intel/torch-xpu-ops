---
name: auto-label-issues
description: "Label proposal for a single GitHub issue (intel/torch-xpu-ops by default). Takes an issue id or URL and an optional pytorch_folder, extracts issue metadata, root-causes the failure against the local checkout when one is given (or from issue evidence alone when it is not), then applies the reference rule packs to derive target_component/need_action, dependency, duplicate, module, and priority. Emits a markdown label+reason table under agent_space/auto_label_issues/, then automatically runs apply_auto_label_issues.py to apply the derived Type/labels/Priority field and post an [agent_triage_result] comment to GitHub. Use when you want labels for an issue without running the full issue-triage pipeline (no local reproduce, no per-axis subagent fan-out)."
---

# Auto Label Issues

Derive the label set for ONE GitHub issue and write a `label | reason` table to
disk. This is the fast path: no local test reproduce, no per-axis subagent
fan-out. Steps 1-7 (analysis) never add labels, close issues, or comment
themselves. Step 8 then **automatically** runs `scripts/apply_auto_label_issues.py`
to apply the derived table to GitHub — labels.md analysis and its application
are one continuous run by default; pass `skip_apply: true` (or `--dry-run`) to
stop short of mutating GitHub when analysis-only output is wanted instead.


## Inputs

| Input | Required | Default | Notes |
|---|---|---|---|
| `issue_ref` | yes | — | Bare issue number or full issue URL. |
| `pytorch_folder` | no | — | Local PyTorch checkout for root-cause tracing. If omitted or nonexistent, Step 2 runs in **evidence-only mode** (see Step 2). |
| `repo` | no | `intel/torch-xpu-ops` | Used only for a bare number; a URL's own owner/name wins. |
| `skip_apply` | no | `false` | When `true`, stop after writing `labels.md` (Step 7) and skip Step 8 entirely — analysis-only, no GitHub mutation. |

Missing `issue_ref` -> **hard-stop**. A missing `pytorch_folder` is NOT a hard
stop; it degrades the trace, not the run.

## Prerequisites

- Authenticated `gh` CLI on PATH (`read:project` scope for project fields).
- `python3` available. If missing, activate a `.venv` in the repo root or a
  parent directory and retry. Do NOT install tooling.
- `pytorch_folder`, when given, exists and is a git checkout. When it is absent
  or not a checkout, continue in evidence-only mode instead of stopping.

## Reference rule packs

All rules live in `.claude/skills/auto-label-issues/reference/`. Read the
file for an axis before deciding it; do not decide from memory.

| Axis | File |
|---|---|
| `target_component`, `need_action` | `target_component.md` |
| `dependency` | `dependency.md` (plus `xpu_supported_operators_complete_list.md` for oneMKL/oneDNN) |
| `duplicates` | `duplicates.md` |
| `module` (11-bucket category enum) | `module.md` |
| `priority` | `priority.md` |
| `labels.md` output format | `output_format.md` |

## Workflow

### Step 1 — Extract issue information

Run the `auto-label-issues/extract-issue-information` skill's script:

```bash
mkdir -p agent_space/auto_label_issues/<repo_underscored>_issue_<id>
python3 .claude/skills/auto-label-issues/extract-issue-information/scripts/extract_basic_info.py \
  <issue_ref> [--repo <repo>] [--pytorch-folder <pytorch_folder>] \
  --output agent_space/auto_label_issues/<repo_underscored>_issue_<id>/extract.json
```

`<repo_underscored>` replaces `/` with `_` (e.g. `intel_torch-xpu-ops`).
`--pytorch-folder` is optional; it only enriches benchmark model lists, so the
script still exits 0 and returns full `title`/`body`/`traceback` without it.

Non-zero exit -> **hard-stop**. If `low_confidence` is non-empty, resolve those
fields inline from the returned `body`/`title` per that skill's Inline LLM
fallback, then overwrite them in `extract.json`.


### Step 2 — Root cause

From `extract.json` (`traceback`, `test_cases`, `reproduce_steps`, `title`,
`body`), establish the defect and its owner. Mode depends on `pytorch_folder`:

| | Mode A — traced | Mode B — evidence-only |
|---|---|---|
| When | `pytorch_folder` given and exists | absent or nonexistent |
| Sources | the checkout, plus Step 1 evidence | Step 1 evidence and `gh` only |
| Never | propose a fix | clone, fetch, or search a checkout |

**Mode A.** Delegate the trace to a read-only deep analysis subagent. Have it establish:
the call path to the failure with `file:line`; whether the owner is the test file,
`pytorch/{aten,torch,c10}`, `third_party/torch-xpu-ops/`, or a third party. 
If it runs in the background, await it rather than repeating the search yourself.

**Mode B.** Conclude a cause ONLY when the evidence is self-sufficient (e.g. the
traceback names the owning file and the error states the defect); cite what you
used. Otherwise set `root_cause` to exactly:

```
insufficient information for root causing: no pytorch_folder provided and issue evidence is not self-sufficient
```

Never guess an owner or infer a `file:line` you did not read.

**Both.** Record `trace_mode` (`traced` / `evidence-only`) for the Output note,
and `root_cause` in **at most 2 lines** — the defect plus its `file:line` (drop
`file:line` in Mode B when nothing was read). Keep the call path, ruled-out
alternatives, and mechanism narrative out of the report. An inconclusive trace
is allowed; it drives `NEED_HUMAN` in Step 3.

### Step 3 — target_component and need_action

Read `reference/target_component.md`. Map the traced fix location to
`target_component` (`test-case` | `pytorch` | `torch-xpu-ops` | `third-party` |
`N/A`) and to `need_action` (`NEED_FIX` | `NEED_FIX_CASE` |
`NEED_FIX_3RDPARTY` | `NEED_HUMAN`).

A skip or xfail decorator is never a fix. An inconclusive trace is
`N/A` + `NEED_FIX` — including every evidence-only run that could not
establish a root cause.

### Step 4 — dependency

Read `reference/dependency.md`. Return exactly one taxonomy value, `none`, or
`null`. For oneMKL/oneDNN, confirm the operator mapping against
`reference/xpu_supported_operators_complete_list.md`. Ambiguous or missing
evidence -> `null`; do not guess from issue prose.

### Step 5 — duplicates

Read `reference/duplicates.md`. Run the enriched `gh search issues` set
concurrently across BOTH `intel/torch-xpu-ops` and `pytorch/pytorch`, one set
per `test_cases[]` entry, always requesting `state,labels,body` and appending
`is:issue`. Apply self-exclusion, the two-of-three signal rule, and the
`relevance` / `recommended_action` tables. Also record inherited `not_target` /
`wontfix` from a HIGH or MEDIUM duplicate.

This step needs no checkout and runs identically in both trace modes.

### Step 6 — module

Read `reference/module.md`. Pick ONE value from the 11-bucket enum using the
Decision Priority Order (first match wins). Base it on the traced root cause,
not on keyword matching in the title. In evidence-only mode, base it on the
traceback's owning frames; if even the bucket is unclear, use `others`.

### Step 7 — priority

Read `reference/priority.md`. Apply the P0-P3 decision tree against the failure
mode from Steps 2-3. If `extract.json` already carries a non-empty `priority`
from the PyTorchXPU project field, preserve it verbatim and note that in the
reason. Priority derives from the observable failure mode (crash vs assertion,
case count, cited regression percentage), so it stays decidable in evidence-only
mode.

## Output

Write `agent_space/auto_label_issues/<repo_underscored>_issue_<id>/labels.md`
following the exact table format, field rules, and brevity/evidence-only
examples in `reference/output_format.md`. Read that file before producing
`labels.md`. Also print the same table to stdout.

## Apply (automatic, mutates GitHub)

Steps 1-7 only write `labels.md`; they do not themselves call `gh issue edit`,
`gh issue close`, or post comments. Step 8 runs immediately after Step 7 by
default: it invokes `scripts/apply_auto_label_issues.py` for real (no `--dry-run`)
to apply the derived table and post the `[agent_triage_result]` comment. Skip
Step 8 only when `skip_apply: true` was given, or when the user explicitly
asked for analysis-only output.

```bash
python3 .claude/skills/auto-label-issues/scripts/apply_auto_label_issues.py \
  <issue_ref> --repo <owner/name> \
  --labels-md agent_space/auto_label_issues/<repo_underscored>_issue_<id>/labels.md \
  --output agent_space/auto_label_issues/<repo_underscored>_issue_<id>/apply_result.json
```

Run it for real directly — do not gate behind `--dry-run` or ask for
confirmation first, since running this skill already implies the intent to
apply labels. Report the printed action list (`applied`/`skipped`/`errors`)
back to the user after it completes. If the user wants a preview before
mutating, they should ask for `--dry-run` explicitly or set `skip_apply: true`
and run Step 8 by hand afterward.

The script maps each `labels.md` row to a mutation:

| `labels.md` row | GitHub mutation |
|---|---|
| `issue_type: <Bug\|Task\|Feature\|Epic>` | Native repo Issue Type via GraphQL `updateIssueIssueType`, if the repo defines that type name; else falls back to a `type: <Value>` label (created if the repo lacks it). |
| `test_module: <value>` | `module: <value>` label (created if missing). |
| `module: <bucket>` | `module: <bucket>` label (created if missing). |
| `P0`/`P1`/`P2`/`P3` (bare row) | Native repo Issue Field "Priority" via GraphQL `setIssueFieldValue`, mapped P0->Urgent, P1->High, P2->Medium, P3->Low. Skipped with a warning if the repo has no native "Priority" Issue Field. |
| `dependency component: <component>` | `dependency component: <component>` label (created if missing). Row omitted or value `null`/`none` -> skipped entirely. |
| `duplicated` | `duplicate` label (created if missing). Row absent -> skipped entirely. |
| `not_target` | `not_target` label (created if missing). Row absent -> skipped entirely. |
| (always, last) | A comment starting with `[agent_triage_result]` containing the full `labels.md` text, posted via `gh issue comment`. If the authenticated `gh` user already left an `[agent_triage_result]` comment on this issue, that comment is edited in place (`gh api -X PATCH`) instead of appending a new one, so re-running the skill does not accumulate duplicate comments. |

Notes:

- A repo without native GitHub Issue Types (`issueTypes` returns null via
  GraphQL) or native Issue Fields (`viewerCanSeeIssueFields: false`) cannot
  receive the Type or Priority-field mutations; the script records these as
  `skipped` with a reason and still applies every label it can and posts the
  comment. Issue Types and Issue Fields are probed with two independent
  GraphQL queries, each degrading to an empty capability set on its own
  error, so a schema mismatch or missing scope on one probe never blocks the
  other or aborts the run.
- The native repo-level "Priority" Issue Field (Urgent/High/Medium/Low) is
  distinct from the org-level PyTorchXPU Project's "Priority" single-select
  field (P0/P1/P2/P3) used by `validation/issue-triage/update-label`. This
  script only touches the native repo Issue Field.
- `gh issue edit --add-label` fails if the label does not exist on the repo;
  the script creates any missing label via `gh label create --force` before
  adding it.
- The script exits 0 when every reachable action succeeded (skips are not
  errors); it exits 1 only on a hard error (bad ref, `gh` unauthenticated,
  unreadable `labels.md`, issue fetch failure) or an action-level `gh`
  failure recorded in `errors`.

## Constraints

1. Steps 1-7 (analysis) never call `gh issue edit`, `gh issue close`, or post
   comments directly — only `apply_auto_label_issues.py` (Step 8) mutates GitHub,
   and it does so using the exact `labels.md` table Steps 1-7 produced.
2. Step 8 runs automatically after Step 7 unless `skip_apply: true` was given
   or the user explicitly asked for analysis-only output.
3. No local test reproduce. That is `validation/issue-triage`'s job.
4. Read the reference file for an axis before deciding it. This applies even
   when the issue already carries a prior `[agent_triage_result]` comment or
   any other existing label/analysis: re-derive every axis (Steps 3-7) from
   the reference files and current evidence this run. An existing comment may
   be read as one input to Step 2's root-cause trace, but it can never
   substitute for reading `reference/target_component.md`, `dependency.md`,
   `module.md`, or `priority.md`, and it can never be copied into `labels.md`
   without independently re-running the corresponding step. Re-running the
   skill on the same issue therefore replaces the prior `[agent_triage_result]`
   comment in place (see Apply step) rather than appending a duplicate.
5. Never report the source issue as its own duplicate.
6. Do not edit `pytorch_folder` or any product code.
7. Never clone or fetch a checkout to substitute for a missing `pytorch_folder`.
   Absent means evidence-only, not "go find one".

## Hard Stops

- Missing `issue_ref`.
- `gh` CLI missing or unauthenticated.
- `extract_basic_info.py` exits non-zero (404, network failure, PR reference,
  malformed input).
- `apply_auto_label_issues.py` hard error (bad ref, unreadable `labels.md`, issue
  fetch failure) — report the stderr and the partial `labels.md` already
  written; do not retry blindly.

Not hard stops (normal degraded outcomes): a missing or nonexistent
`pytorch_folder`, an inconclusive trace, `insufficient information for root
causing`, any `null` axis, and any individual `apply_auto_label_issues.py` action
recorded under `skipped` (e.g. a repo lacking a native Priority field).

