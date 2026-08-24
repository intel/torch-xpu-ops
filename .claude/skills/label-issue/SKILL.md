---
name: label-issue
description: "Propose GitHub issue labels for a single intel/torch-xpu-ops issue (or any repo) without a local test reproduce or per-axis subagent fan-out. Use when you want a label set for an issue given its number or URL, with an optional pytorch_folder for root-cause tracing. Extracts issue metadata via the extract-issue subskill, groups the reported failures, then root-causes and labels the first group's first case. Every axis (type, test, module, priority, dtype, symptom, dependency, os, hw, and the triage duplicate/wontfix/need_split labels) is derived from the label definitions, evidence, and keywords in reference/proposed_labels.json; it never hard-codes label data the JSON already carries. Emits a markdown labels table under agent_space/label_issue/ and never writes to GitHub."
---

# Label Issue

Derive the label set for ONE GitHub issue and write a `label | reason` table to
disk. This is the fast path: no local test reproduce, no per-axis subagent
fan-out.

The skill is **analysis-only**. It never adds labels, closes issues, posts
comments, or creates issues. Its single artifact is `labels.md`, for a workflow
or human to apply.

## Label definitions are data, not code

Every label — its exact name, its `keywords`, and its `evidence` criterion —
is defined once in `reference/proposed_labels.json`:

- `categories.<axis>` holds the per-axis label list (`type`, `test`, `module`,
  `os`, `hw`, `dependency`, `dtype`, `triage`, `symptom`, ...).
- `priority_field` holds the project Priority tiers.

Read label names, keywords, and evidence from that JSON at decision time. NEVER
hard-code a label spelling, a keyword, or an evidence rule in this skill or in a
reference pack when the JSON already carries it. The reference packs supply only
the reasoning the JSON cannot (decision-priority order, traceback-origin
evidence, operator-to-dependency mapping).

## Inputs

| Input | Required | Default | Notes |
|---|---|---|---|
| `issue_ref` | yes | — | Bare issue number or full issue URL. |
| `pytorch_folder` | no | — | Local PyTorch checkout for root-cause tracing. If omitted or nonexistent, Step 3.1 runs in **evidence-only mode**. |
| `repo` | no | `intel/torch-xpu-ops` | Used only for a bare number; a URL's own owner/name wins. |

Missing `issue_ref` -> **hard-stop**. A missing `pytorch_folder` is NOT a hard
stop; it degrades the trace, not the run.

## Prerequisites

- Authenticated `gh` CLI on PATH (`read:project` scope for project fields).
- `python3` is NOT required: Step 1 uses the script-free `extract-issue` skill.
- `pytorch_folder`, when given, exists and is a git checkout. When absent or not
  a checkout, continue in evidence-only mode instead of stopping.

## Reference packs

Under `.claude/skills/label-issue/reference/`. Before deciding an axis, read its
source; do not decide from memory. Two kinds of source:

- **Reasoning packs** (`.md`) — add the judgment the JSON cannot carry, and defer
  label names/keywords/evidence to `proposed_labels.json`.
- **JSON-only axes** — decided straight from a `proposed_labels.json` section; no
  `.md` pack.

| Axis | Reasoning pack | JSON-only source |
|---|---|---|
| grouping | `group_issue.md` | `categories.triage` (split label) |
| `dependency` | `dependency.md` (+ `dependency_info.md` for oneMKL/oneDNN) | — |
| duplicate / wontfix | `duplicates.md` | `categories.triage` |
| `os` / `hw` | `platform_specific.md` | `categories.os` / `categories.hw` |
| output format | `output_format.md` | — |
| `module` | — | `categories.module` (priority-ordered; each label has `evidence`/`keywords`) |
| `dtype` | — | `categories.dtype` |
| `symptom` | — | `categories.symptom` |
| priority | — | `priority_field` |

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

`extract.json` carries `title`, `body`, `traceback`, `test_cases`,
`reproduce_steps`, and the pre-read project fields (`os`, `hw`,
`platform_specific`, `test`, `dependency`, `module`, `priority`, `issue_type`).

A hard stop inside that skill (missing/unauthenticated `gh`, 404, network
failure, PR reference, malformed input) is a **hard-stop** here too. Do not
substitute a Python extraction script for `extract-issue`.

### Step 2 — Group the failures

An issue may report several failing cases. Follow `reference/group_issue.md` to
group `extract.json`'s failures by cause: it defines the ordered ladder of
grouping keys (error message -> dtype -> op/kernel -> parameters -> tensor shape,
first key that resolves the split wins), the signal-vs-noise rule, and the
one-group / two-or-more-group outcome.

- **One group** -> no split signal.
- **Two or more groups** -> the issue mixes distinct causes. Emit the
  split-recommendation label from `categories.triage` (match its `evidence`;
  read the name from the JSON). Record a one-line signature per group for the
  output. This skill NEVER splits, files sub-issues, or edits anything.

### Step 3 — Analyze the representative case

Take group 0 and, within it, `extract.json`'s `test_cases[0]` — the first entry
in emission order — as the **representative case** for the whole issue. Do not
reorder or re-rank; index 0 is the rule so two runs agree. If `test_cases` has
0 or 1 entries, the representative case is the issue as a whole.

Everything below is decided for the representative case ONLY, except where a
step explicitly counts the whole issue. Decide each axis from the JSON `evidence`
and `keywords` for that axis — never from a hard-coded label list here.

#### 3.1 — Root cause

From `extract.json` (`traceback`, `test_cases`, `reproduce_steps`, `title`,
`body`), establish the defect and its owner for the representative case. Mode
depends on `pytorch_folder`:

| | Mode A — traced | Mode B — evidence-only |
|---|---|---|
| When | `pytorch_folder` given and exists | absent or nonexistent |
| Sources | the checkout, plus Step 1 evidence | Step 1 evidence and `gh` only |
| Never | propose a fix | clone, fetch, or search a checkout |

**Mode A.** Delegate the trace to a read-only deep-analysis subagent: the call
path to the failure with `file:line`, and whether the owner is the test file,
`pytorch/{aten,torch,c10}`, `third_party/torch-xpu-ops/`, or a third party.
Await it rather than repeating the search.

**Mode B.** Conclude a cause ONLY when the evidence is self-sufficient (the
traceback names the owning file and the error states the defect); cite what you
used. Otherwise set `root_cause` to exactly:

```
insufficient information for root causing: no pytorch_folder provided and issue evidence is not self-sufficient
```

Never guess an owner or infer a `file:line` you did not read. Record
`trace_mode` and `root_cause` in at most 2 lines.

#### 3.2 — Duplicate

Follow `reference/duplicates.md` for the representative case: it defines the
full duplicate-search and relevance procedure. Emit the duplicate label from
`categories.triage` when its `evidence` is met (read the name from the JSON).
Needs no checkout; identical in both trace modes.

#### 3.3 — Wontfix short-circuit

If the issue is a duplicate of another issue and that other issue is out of
scope / by design — i.e. it carries the wontfix triage label per
`categories.triage` (its `evidence` absorbs the legacy `not_target`) — then this
issue inherits wontfix from that HIGH/MEDIUM-relevance duplicate. The issue's own
labels can also set wontfix directly.

When wontfix holds (own or inherited): emit the wontfix label, SKIP the remaining
checks (3.4 and 3.5), and go straight to Step 4 output.

#### 3.4 — Type, priority, module, symptom, dtype

Decide each from its JSON section. For every axis, if `extract.json` already
carries a human-set value for that axis, preserve it verbatim and note the human
origin; otherwise derive it:

- **type** — `issue_type_field`: the native GitHub Type (`Bug` \| `Feature` \|
  `Task` \| `Epic`). Preserve a non-empty `extract.json` `issue_type` verbatim.
  Otherwise evaluate the `values` and take the FIRST whose `evidence`/`keywords`
  match `lowercase(title + " " + body + " " + traceback)` (a reported failure ->
  `Bug`, new functionality -> `Feature`); the `keywords` are hints only and never
  override an explicitly set Type. An empty axis is valid.
- **priority** — `priority_field`: the tiers, per-tier `evidence`, and fallback
  `keywords`. Preserve a non-empty `extract.json` `priority`. Otherwise evaluate
  tiers in severity order and stop at the first matching `evidence`, defaulting
  to the tier the JSON marks as default. The whole-issue case-count conditions
  count every `test_cases[]` entry, not just the representative case.
- **module** — `categories.module`: pick exactly ONE label. The `labels` array
  is ordered by decision priority, so walk it top-to-bottom and take the FIRST
  whose `evidence` is met, driven by the traced root cause (keywords are hints
  only). The axis `description` carries the tie-break rules. Preserve a non-empty
  `extract.json` `module`.
- **symptom** — `categories.symptom`: multi-label; independently evaluate EVERY
  label and emit one row for EACH whose `evidence`/`keywords` are met by the
  representative case — a single issue routinely carries several symptom labels
  (e.g. `regression` + `inference`, or `Accuracy` + `training`). Do not stop at
  the first match. An empty axis is valid.
- **dtype** — `categories.dtype`: multi-label; follow the axis `description`
  (structured-field-first, part-of-failure-signature, AMP disambiguation) and
  per-label `evidence`. An empty axis is valid.

When falling back to keyword matching for symptom/dtype, match against
`lowercase(title + " " + body + " " + traceback)`, excluding the
`## Versions` / `Collecting environment` dump.

The `os` / `hw` axes are already decided by `extract-issue` per
`reference/platform_specific.md` (emitted only when the issue is
platform-specific); carry `extract.json`'s `os` / `platform` straight through.

#### 3.5 — Dependency

Follow `reference/dependency.md` (which itself points to `dependency_info.md`
for the oneMKL/oneDNN operator mapping) to decide the dependency for the
representative case. It returns exactly one value from `categories.dependency`,
`none`, or `null`, and defines how to emit the matching label. Preserve a
non-empty `extract.json` `dependency`.

### Step 4 — Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md`
following the exact table format and field rules in `reference/output_format.md`
(read it first). Emit each label name verbatim from `proposed_labels.json`, with
a one-line evidence reason. Also print the table to stdout.

- When Step 2 found 2 or more cases, emit the `Analyzed case:` note naming the
  representative case and the count left unanalyzed. Omit for a single case.
- When Step 2 found 2 or more groups, emit the split-recommendation triage row.
- When Step 3.3 short-circuited, emit the wontfix row and the axes decided so
  far only.

This is the final step. Report the `labels.md` path; do not apply to GitHub.

## Constraints

1. Analysis-only: never `gh issue edit`, `gh issue create`, `gh issue close`,
   `gh issue comment`, `gh label create`, or any GraphQL Type/Priority mutation.
   The only artifact is `labels.md`.
2. No local test reproduce.
3. Analyze exactly one representative case (group 0, `test_cases[0]`). Never
   split a multi-group issue into sub-issues.
4. Decide every axis from `proposed_labels.json` (and the axis's reference pack
   for reasoning) this run. An existing triage comment or label is at most one
   input to Step 3.1; it never substitutes for reading the JSON evidence and
   re-deriving the axis.
5. Never report the source issue as its own duplicate.
6. Do not edit `pytorch_folder` or any product code.
7. Never clone or fetch a checkout to substitute for a missing `pytorch_folder`.

## Hard Stops

- Missing `issue_ref`.
- `gh` CLI missing or unauthenticated.
- Step 1 extraction hard-stops (404, network failure, PR reference, malformed
  input) from `extract-issue`.

Not hard stops (normal degraded outcomes): a missing/nonexistent
`pytorch_folder`, an inconclusive trace, `insufficient information for root
causing`, and any `null` axis.
