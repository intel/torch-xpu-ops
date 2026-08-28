---
name: label-issue
description: "Propose GitHub issue labels for a single intel/torch-xpu-ops issue (or any repo) without a local test reproduce or per-axis subagent fan-out. Use when you want a label set for an issue given its number or URL, or when an issue reports multiple failure groups and you want one labels table per group, with an optional pytorch_folder for root-cause tracing. Extracts issue metadata via the extract-issue subskill, groups the reported failures, then root-causes and labels one representative case per group, emitting a labels table for each group one by one. Every axis (type, test, module, priority, dtype, symptom, dependency, os, hw, and the triage duplicate/wontfix/need_split labels) is derived from the label definitions, evidence, and keywords in reference/proposed_labels.json; it never hard-codes label data the JSON already carries. Emits a markdown labels table under agent_space/label_issue/ and never writes to GitHub."
---

# Label Issue

Derive the label set for ONE GitHub issue and write a `label | reason` table to
disk. This is the fast path: no local test reproduce, no per-axis subagent
fan-out.

When the issue splits into multiple failure groups, analyze one representative
case per group and emit a separate labels table for each group, one by one.

## Quick start

Given an issue reference (and optionally a local `pytorch_folder`):

```
label-issue issue_ref=4752 pytorch_folder=~/pytorch
```

Extracts the issue, groups its failures, root-causes one representative case per
group, and writes
`agent_space/label_issue/intel_torch-xpu-ops_issue_4752/labels.md` — a
`label | reason` table per group. Analysis-only: nothing is applied to GitHub.

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
| root cause | `triage_issue.md` | — |
| `dependency` | `dependency.md` (+ `dependency_info.md` for oneMKL/oneDNN) | — |
| duplicate / wontfix | `duplicates.md` | `categories.triage` |
| `os` / `hw` | `platform_specific.md` | `categories.os` / `categories.hw` |
| output format | `output_format.md` | — |
| `module` | — | `categories.module` (priority-ordered; each label has `evidence`/`keywords`) |
| `dtype` | — | `categories.dtype` |
| `symptom` | — | `categories.symptom` |
| priority | — | `priority_field` |
| type | — | `issue_type_field` (native Type; preserve `extract.json` `issue_type`) |

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
group `extract.json`'s failures by cause and decide the one-group /
two-or-more-group outcome.

### Step 3 — Analyze one representative case per group

Step 2 yields one or more groups. Process **every** group in group order. For
each group, pick its representative case and run the full per-case analysis
(3.1-3.5) below, independently of the other groups. Two runs must agree, so the
selection rule is fixed:

- The representative case of a group is its FIRST member in `extract.json`
  `test_cases` emission order — do not reorder or re-rank.
- If a group has exactly one case, that case is the representative.
- If `test_cases` has 0 or 1 entries total, there is a single group and the
  representative case is the issue as a whole.

Everything in 3.1-3.5 is decided for the group's representative case ONLY, and
any case-count condition (e.g. priority) counts only the cases in THIS group,
not the whole issue. Decide each axis from the JSON `evidence`
and `keywords` for that axis — never from a hard-coded label list here.

Repeat 3.1-3.5 for each group before moving to Step 4, keeping each group's
axis results separate.

#### 3.1 — Root cause

Follow `reference/triage_issue.md` to establish the defect and its owner for the
representative case. It defines the `error_message`/`traceback` reading order, the
Mode A (traced) vs Mode B (evidence-only) split on `pytorch_folder`, and how to
record `trace_mode` and `root_cause`.

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

Decide each axis from its JSON section, applying these shared rules:

- **Human value wins.** If `extract.json` already carries a human-set value for
  the axis, preserve it verbatim, note the human origin, and skip deriving it.
- **Match on `evidence`, not `keywords`.** `keywords` are only hints; they never
  override an explicit value or an `evidence` match. When matching text, use
  `lowercase(title + " " + body + " " + traceback)`, excluding the
  `## Versions` / `Collecting environment` dump.
- **Single vs multi.** A *single*-label axis takes the FIRST matching entry in
  JSON order; a *multi*-label axis emits one row per matching entry. An empty axis
  is always valid.

Per-axis specifics:

| Axis | JSON section | Kind | How to decide |
|---|---|---|---|
| type | `issue_type_field` | single | Match `values` by `evidence`/`keywords`. |
| priority | `priority_field` | single | Evaluate tiers in severity order by `evidence`; default to the tier the JSON marks as default. Case-count conditions count only the cases in THIS group, not the whole issue. |
| module | `categories.module` | single | `labels` is ordered by decision priority; take the FIRST whose `evidence` is met, driven by the traced root cause. The axis `description` carries the tie-break rules. |
| symptom | `categories.symptom` | multi | Evaluate EVERY label against the representative case. One issue routinely carries several (e.g. `regression` + `inference`, or `Accuracy` + `training`). |
| dtype | `categories.dtype` | multi | Follow the axis `description` (structured-field-first, part-of-failure-signature, AMP disambiguation) and per-label `evidence`. |

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
(read it first). Wrap the entire artifact in a collapsible `<details>` block whose
`<summary>` is the `label-issue: <repo>#<id>` title, so the content stays hidden
until clicked. Emit each label name verbatim from `proposed_labels.json`, with
a one-line evidence reason. Also print to stdout.

Emit one labels section per analyzed group, in group order, one by one:

- When Step 2 found 2 or more groups, first emit a top-level output table
  containing the `need_split` triage row (once, from `categories.triage`)
  recommending the issue be split.
- Head each group section with `## Group <n> — <summary of the group of tests>`
  (a short phrase for what the group's tests share, not just the representative
  case id).
- Under each head, emit that group's own `label | reason` table decided in
  Step 3 for its representative case, then end the block with a
  `Test cases (<M>):` list enumerating every test case in the group.
- For any group whose Step 3.3 short-circuited, emit that group's wontfix row
  and only the axes decided so far.
- For a single-group issue, emit one section without the top-level table and
  without the `## Group` head, and omit the unanalyzed-count note.

This is the final step. Report the `labels.md` path; do not apply to GitHub.

### Optional — Apply to GitHub

The skill itself never writes to GitHub. To act on a finished `labels.md`, the
user can run `scripts/apply_labels.py <labels.md>` (dry run by default; add
`--apply` to write). It parses the `<summary>` for `<repo>#<id>` and every table
row, then applies label rows via `gh issue edit --add-label`, the `type` row via
`gh issue edit --type` (native Type field), the `priority` row to the PyTorchXPU
project Priority field via GraphQL (tier mapped to `P0`-`P3` through
`proposed_labels.json`), and posts the full `labels.md` as an issue comment
(suppress with `--no-comment`). For multi-group issues it dedupes labels,
collapses `type` to one value, and picks the most urgent priority across groups.

## Constraints

1. Analysis-only: never `gh issue edit`, `gh issue create`, `gh issue close`,
   `gh issue comment`, `gh label create`, or any GraphQL Type/Priority mutation.
   The only artifact is `labels.md`.
2. No local test reproduce.
3. Analyze exactly one representative case per group (each group's first
   `test_cases` entry). Never split a multi-group issue into sub-issues — only
   recommend the split via the `need_split` row.
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
