---
name: label-issue
description: "Label proposal for a single GitHub issue (intel/torch-xpu-ops by default). Takes an issue id or URL and an optional pytorch_folder, extracts issue metadata, root-causes the failure against the local checkout when one is given (or from issue evidence alone when it is not), then applies the reference rule packs to derive dependency, duplicate, module, dtype, symptom, and priority. All label names and their keyword lists come from reference/proposed_labels.json; the skill never hard-codes a keyword the JSON already carries. When the issue reports several failing cases, only the first is analyzed and labels.md names it; when the failures form more than one group it also emits need_split. Emits a markdown labels table under agent_space/label_issue/; it never writes to GitHub. Use when you want labels for an issue without a local reproduce or per-axis subagent fan-out."
---

# Label Issue

Derive the label set for ONE GitHub issue and write a `label | reason` table to
disk. This is the fast path: no local test reproduce, no per-axis subagent
fan-out.

When the issue reports several failing cases, exactly one — the first — is
analyzed, and `labels.md` names it. When the failures fall into more than one
group (distinct normalized error signature), `labels.md` also carries
`need_split` as a recommendation. The issue is never split into sub-issues.

This skill is **analysis-only**. It never adds labels, closes issues, posts
comments, or creates issues. Its single artifact is `labels.md`, for a workflow
or human to apply.


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
- `python3` is NOT required: Step 1 uses the script-free `extract-issue` skill.
- `pytorch_folder`, when given, exists and is a git checkout. When it is absent
  or not a checkout, continue in evidence-only mode instead of stopping.

## Reference rule packs

All rules live in `.claude/skills/label-issue/reference/`. Read the
file for an axis before deciding it; do not decide from memory.

Every label name and its keyword list is defined once in
`reference/proposed_labels.json` (each label carries a `keywords` array where an
axis is keyword-driven). Read keywords from that JSON — never hard-code a keyword
in a rule pack or in this skill when the JSON already carries it. The axis rule
packs supplement the JSON keywords with the reasoning (decision-priority order,
traceback-origin evidence); they do not restate the keyword lists.

| Axis | File |
|---|---|
| `dependency` | `dependency.md` (plus `xpu_operator_dependency_list.md` for oneMKL/oneDNN) |
| `duplicates` | `duplicates.md` |
| `module` (13-bucket category enum) | `module.md` |
| `dtype` | `dtype.md` |
| `symptom` | `symptom.md` |
| `priority` | `priority.md` |
| `labels.md` output format | `output_format.md` |
| all label names + keyword lists | `proposed_labels.json` |

## Workflow

### Step 1 — Extract issue information

Run the `label-issue/extract-issue` skill. It uses `gh` plus its own reading of
the issue — no script, no `python3`.

```bash
mkdir -p agent_space/label_issue/<repo_underscored>_issue_<id>
```

Invoke `extract-issue` with `<issue_ref>`, the optional `repo`, the optional
`pytorch_folder`, and
`output = agent_space/label_issue/<repo_underscored>_issue_<id>/extract.json`.

`<repo_underscored>` replaces `/` with `_` (e.g. `intel_torch-xpu-ops`).
`pytorch_folder` is optional; it only enriches benchmark model lists, so the
skill still returns full `title`/`body`/`traceback` without it.

A hard stop inside that skill (missing/unauthenticated `gh`, 404, network
failure, PR reference, malformed input) is a **hard-stop** here too.

`extract-issue` is the only extraction path for this skill. Do not substitute a
Python extraction script for it, and do not switch away from it because `python3`
happens to be available.

### Step 1.5 — Group the failures, then select the case to analyze

An issue may report several failing cases. This skill analyzes exactly ONE, so
the label set has a single unambiguous subject. It still reports whether the
issue *should* have been filed as more than one issue.

**Group the failures.** From `extract.json`'s `traceback`, `test_cases`, and
`body`, group the reported failures by cause, using the **normalized error
message** as the primary key and the **test function** only as a tie-breaker.

Normalize an error signature by dropping run-specific noise: addresses, tensor
shapes, numeric tolerances and deltas, device ids, file paths, dtype and
parametrization suffixes, and model names. Two failures are the SAME group when
their normalized signatures match — even when they fail in different test
functions, different files, or many parametrizations of one operator.

Use the test function only to SPLIT failures that share a generic signature but
demonstrably different causes, e.g. a bare `AssertionError` raised from two
unrelated code paths with different tracebacks. Never use it to split failures
that already share a specific signature.

Distinct groups mean distinct causes, e.g. a `RuntimeError` from a missing oneDNN
primitive plus an `AssertionError` on a numeric tolerance.

- **One group** -> nothing further; do not emit `need_split`.
- **Two or more groups** -> record `group_count` and a one-line signature per
  group. Step 9 emits `need_split`. This skill never acts on it: it does not
  split the issue, file sub-issues, or edit anything.

Case count alone is NOT the signal, and neither is the number of distinct test
functions. One `NotImplementedError` for an unimplemented dtype reported across
29 cases in 11 test functions is ONE group and gets no `need_split`. Judge the
normalized error signature, never the number of `test_cases[]` entries.

**Select the analyzed case.** Take `extract.json`'s `test_cases[0]` — the first
entry, in the order `extract-issue` emitted it — as the **analyzed case**. Do not
reorder, re-rank, or pick by severity; index 0 is the rule, so two runs on one
issue always agree.

- `test_cases` has 0 or 1 entries: nothing to select. Run Steps 2-9 against the
  issue as a whole and emit no case note in Step 9.
- `test_cases` has 2 or more entries: record `analyzed_case` (its
  `test_file`/`test_class`/`test_case`, or for an E2E entry its
  `benchmark`/`model`/`phase`/`dtype`, since E2E entries carry no test-file
  fields) and `case_count`. Scope Steps 2-9 to that case ONLY: root-cause it,
  and ignore the other entries' tracebacks and error messages when deciding
  every axis **except priority** — priority's case-count rows (Step 8) are a
  property of the whole issue, not the analyzed case, so they count every
  `test_cases[]` entry regardless of scoping. Step 9 then names the analyzed
  case and the number left unanalyzed.

Never split the issue, never file a sub-issue, and never edit the issue. The
unanalyzed cases are simply out of scope for this run and are reported as such.

### Step 2 — Root cause

From `extract.json` (`traceback`, `test_cases`, `reproduce_steps`, `title`,
`body`), establish the defect and its owner **for the Step 1.5 analyzed case
only**. Mode depends on `pytorch_folder`:

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
is allowed.

### Step 3 — dependency

Read `reference/dependency.md`. Return exactly one taxonomy value, `none`, or
`null`. For oneMKL/oneDNN, confirm the operator mapping against
`reference/xpu_operator_dependency_list.md`. Ambiguous or missing
evidence -> `null`; do not guess from issue prose.

`extract.json` carries the value in `dependency`, taken from the issue's existing
dependency label when it has one. When it is non-blank, preserve it and return
that value directly — do not re-decide. It is `""` when the issue has no
dependency label — that is not evidence of `none`, so decide this axis from
`reference/dependency.md`. Emit the label, not the value: the prefix is not
uniform (`third_party_packages` maps to `dependency: third_party`). Take
the label from the mapping table in `reference/dependency.md`.

### Step 4 — duplicates

Read `reference/duplicates.md`. Run the enriched `gh search issues` set
concurrently across BOTH `intel/torch-xpu-ops` and `pytorch/pytorch` for the
Step 1.5 analyzed case, always requesting `state,labels,body` and appending
`is:issue`. Apply self-exclusion, the two-of-three signal rule (and its
single-signal body-match exception), and the
`relevance` / `recommended_action` tables. Also record inherited `wontfix`
from a HIGH or MEDIUM duplicate (a legacy `not_target` label counts as
`wontfix`). When `wontfix` is `true` (own labels or inherited), emit the
`wontfix` label per `reference/duplicates.md`.

This step needs no checkout and runs identically in both trace modes.

### Step 5 — module

Read `reference/module.md`. Pick ONE value from the 13-bucket enum using the
Decision Priority Order (first match wins). Base it on the traced root cause,
not on keyword matching in the title. In evidence-only mode, base it on the
traceback's owning frames; if even the bucket is unclear, use `others`.

`extract.json` may already carry a bucket in `module`, taken from the issue's
existing `module:` label. When it is non-blank, preserve it and return that
bucket directly — do not re-decide. When it is `""` — the issue carries no
`module:` label — derive the bucket yourself from the traced root cause; never
emit `""` for this axis. Take the label from the mapping table in
`reference/module.md`. Emit the label (`module: ao`), never the bucket
(`torchAO`).

### Step 6 — dtype

Read `reference/dtype.md`. This is a multi-label axis: emit one `dtype: <value>`
row per dtype the analyzed case implicates, or none when the failure is
dtype-agnostic. Match on the dtype parametrization suffix, error message,
traceback, and reproduce command for the Step 1.5 analyzed case, using the
`keywords` in `categories.dtype` of `reference/proposed_labels.json` — do not
hard-code dtype spellings here. For an E2E entry, prefer `extract.json`'s E2E
`dtype`. Emit the label column (e.g. `dtype: bfloat16`), never the bare `code`.
An empty dtype axis is a valid, common outcome.

### Step 7 — symptom

Read `reference/symptom.md`. This is a multi-label axis: emit one row per matched
symptom (`Accuracy`, `performance`, `regression`, `random`, `inference`,
`training`), or none. Match against the analyzed case's title/body/traceback
using the `keywords` in `categories.symptom` of `reference/proposed_labels.json`
— do not hard-code symptom keywords here. Require direct evidence, not an
incidental keyword in the environment dump. Emit each name verbatim
(case-sensitive) from the JSON. An empty symptom axis is a valid, common outcome.

### Step 8 — priority

Read `reference/priority.md`. Apply the Urgent/High/Medium/Low decision tree
against the failure mode from Step 2. Count failed UT cases (the
`>6` / `1-6` rows) across the **whole issue** — every `test_cases[]` entry,
not just the Step 1.5 analyzed case — since severity is a property of the
issue, not of any single case. If `extract.json` already carries a
non-empty `priority` from the PyTorchXPU project field, preserve it verbatim and
note that in the reason. Priority derives from the observable failure mode (crash
vs assertion, case count, cited regression percentage), so it stays decidable in
evidence-only mode.

### Step 9 — Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md`
following the exact table format, field rules, and brevity/evidence-only
examples in `reference/output_format.md`. Read that file before producing
`labels.md`. Also print the table to stdout.

When Step 1.5 found 2 or more cases, emit the `Analyzed case:` note defined in
`reference/output_format.md` so the reader knows which case the labels describe.
Omit that note entirely for a single-case issue.

When Step 1.5 found 2 or more failure groups, emit the `need_split` label row.
Omit it for a single group. `need_split` recommends splitting; this skill never
splits the issue itself.

This is the final step. Report the `labels.md` path to the user; do not apply
the table to GitHub.

## Constraints

1. This skill is analysis-only: it never applies the derived labels and never
   writes to GitHub at all. Never call `gh issue edit`, `gh issue create`,
   `gh issue close`, `gh issue comment`, `gh label create`, or any GraphQL
   Type/Priority mutation. The only artifact is `labels.md` on disk.
2. No local test reproduce.
3. Analyze exactly one case per run, chosen by Step 1.5. Never split a
   multi-case issue into sub-issues.
4. Read the reference file for an axis before deciding it. This applies even
   when the issue already carries a prior triage comment or any other existing
   label/analysis: re-derive every axis (Steps 3-8) from the reference files
   and current evidence this run. An existing comment may be read as one input
   to Step 2's root-cause trace, but it can never substitute for reading
   `reference/dependency.md`, `module.md`, `dtype.md`, `symptom.md`, or
   `priority.md`, and it can never be copied into `labels.md` without
   independently re-running the corresponding step.
5. Never report the source issue as its own duplicate.
6. Do not edit `pytorch_folder` or any product code.
7. Never clone or fetch a checkout to substitute for a missing `pytorch_folder`.
   Absent means evidence-only, not "go find one".

## Hard Stops

- Missing `issue_ref`.
- `gh` CLI missing or unauthenticated.
- Step 1 extraction hard-stops (404, network failure, PR reference, malformed
  input) from `extract-issue`.

Not hard stops (normal degraded outcomes): a missing or nonexistent
`pytorch_folder`, an inconclusive trace, `insufficient information for root
causing`, and any `null` axis.
