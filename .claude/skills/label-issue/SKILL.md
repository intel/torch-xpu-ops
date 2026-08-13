---
name: label-issue
description: "Label proposal for a single GitHub issue (intel/torch-xpu-ops by default). Takes an issue id or URL and an optional pytorch_folder, extracts issue metadata, root-causes the failure against the local checkout when one is given (or from issue evidence alone when it is not), then applies the reference rule packs to derive target_component/need_action, dependency, duplicate, module, and priority. Emits a markdown label+reason table under agent_space/label_issue/ for a human to review and apply. Use when you want labels for an issue without running the full issue-triage pipeline (no local reproduce, no per-axis subagent fan-out)."
---

# Label Issue

Derive the label set for ONE GitHub issue and write a `label | reason` table to
disk. This is the fast path: no local test reproduce, no per-axis subagent
fan-out.

This skill is **analysis-only**. It never adds labels, closes issues, or posts
comments. Its single artifact is `labels.md`, for a human to review and apply.


## Inputs

| Input | Required | Default | Notes |
|---|---|---|---|
| `issue_ref` | yes | — | Bare issue number or full issue URL. |
| `pytorch_folder` | no | — | Local PyTorch checkout for root-cause tracing. If omitted or nonexistent, Step 2 runs in **evidence-only mode** (see Step 2). |
| `repo` | no | `intel/torch-xpu-ops` | Used only for a bare number; a URL's own owner/name wins. |

Missing `issue_ref` -> **hard-stop**. A missing `pytorch_folder` is NOT a hard
stop; it degrades the trace, not the run.

## Prerequisites

- Authenticated `gh` CLI on PATH (`read:project` scope for project fields).
- `python3` available. If missing, activate a `.venv` in the repo root or a
  parent directory and retry. Do NOT install tooling.
- `pytorch_folder`, when given, exists and is a git checkout. When it is absent
  or not a checkout, continue in evidence-only mode instead of stopping.

## Reference rule packs

All rules live in `.claude/skills/label-issue/reference/`. Read the
file for an axis before deciding it; do not decide from memory.

| Axis | File |
|---|---|
| `target_component`, `need_action` | `target_component.md` |
| `dependency` | `dependency.md` (plus `xpu_operator_dependency_list.md` for oneMKL/oneDNN) |
| `duplicates` | `duplicates.md` |
| `module` (11-bucket category enum) | `module.md` |
| `priority` | `priority.md` |
| `labels.md` output format | `output_format.md` |

## Workflow

### Step 1 — Extract issue information

Run the `label-issue/extract-issue-information` skill's script:

```bash
mkdir -p agent_space/label_issue/<repo_underscored>_issue_<id>
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py \
  <issue_ref> [--repo <repo>] [--pytorch-folder <pytorch_folder>] \
  --output agent_space/label_issue/<repo_underscored>_issue_<id>/extract.json
```

`<repo_underscored>` replaces `/` with `_` (e.g. `intel_torch-xpu-ops`).
`--pytorch-folder` is optional; it only enriches benchmark model lists, so the
script still exits 0 and returns full `title`/`body`/`traceback` without it.

Non-zero exit -> **hard-stop**. If `low_confidence` is non-empty, resolve those
fields inline from the returned `body`/`title` per that skill's Inline LLM
fallback, then overwrite them in `extract.json`.

### Step 1.5 submit-subissues — Split multiple failure kinds

Inspect `extract.json`'s `traceback`/`test_cases`/`body` for evidence of more
than one distinct kind of failure in this single issue (e.g. different error
messages/exception types, or unrelated test cases failing for different
reasons). Group the evidence by distinct error message + associated test
case(s).

- **One group (or all failures share the same cause)**: no sub-issue is
  created. Continue to Step 2 and run Steps 2-8 once, as usual, against the
  original issue.
- **Multiple groups**: for each group, run `gh issue create` on the same repo
  with a title/body scoped to that group's error message and test case(s),
  then set the original issue as parent via `gh issue edit <new_issue> --add-parent-issue <original_issue>`.
  Then run Steps 2-8 independently for each newly created sub-issue, writing a
  separate `labels.md` per sub-issue. Report the created sub-issues (number,
  URL, title) to the user alongside the per-sub-issue `labels.md` paths.

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
`reference/xpu_operator_dependency_list.md`. Ambiguous or missing
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

### Step 8 — Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md`
following the exact table format, field rules, and brevity/evidence-only
examples in `reference/output_format.md`. Read that file before producing
`labels.md`. Also print the same table to stdout.

This is the final step. Report the `labels.md` path to the user; do not apply
the table to GitHub.

## Constraints

1. This skill is analysis-only: it never applies the derived labels. Never call
   `gh issue edit --add-label`/`--remove-label`, `gh issue close`,
   `gh issue comment`, `gh label create`, or any GraphQL Type/Priority
   mutation. The only artifact is `labels.md` on disk. Step 1.5 is the sole
   exception, and only for splitting a multi-failure issue: it may run
   `gh issue create` and `gh issue edit --add-parent-issue`, neither of which
   applies a label.
2. No local test reproduce.
3. Read the reference file for an axis before deciding it. This applies even
   when the issue already carries a prior triage comment or any other existing
   label/analysis: re-derive every axis (Steps 3-7) from the reference files
   and current evidence this run. An existing comment may be read as one input
   to Step 2's root-cause trace, but it can never substitute for reading
   `reference/target_component.md`, `dependency.md`, `module.md`, or
   `priority.md`, and it can never be copied into `labels.md` without
   independently re-running the corresponding step.
4. Never report the source issue as its own duplicate.
5. Do not edit `pytorch_folder` or any product code.
6. Never clone or fetch a checkout to substitute for a missing `pytorch_folder`.
   Absent means evidence-only, not "go find one".

## Hard Stops

- Missing `issue_ref`.
- `gh` CLI missing or unauthenticated.
- `extract_basic_info.py` exits non-zero (404, network failure, PR reference,
  malformed input).

Not hard stops (normal degraded outcomes): a missing or nonexistent
`pytorch_folder`, an inconclusive trace, `insufficient information for root
causing`, and any `null` axis.
