# Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md` in the
layout below, and also print it to stdout. This file defines the artifact's
*shape*; the label names and their evidence come from `proposed_labels.json`, and
each axis is decided in the label-issue skill's Step 3.

## Skeleton

Every `<value>` below is one of the enum values defined in
`proposed_labels.json` (see **Axis sources**); this file never enumerates them.

```markdown
label-issue: <repo>#<id>

Root cause: <=2 lines, specific, with file:line when a trace read one>

<optional header notes, one line each - see below>

| label | reason |
|---|---|
| `type: <value>` | ... |
| `test: <value>` | ... |
| `module: <value>` | ... |
| `priority: <value>` | ... |
| `os: <value>` | ... |
| `hw: <value>` | ... |
| `dtype: <value>` | ... |
| `dependency component: <value>` | ... |
| `<symptom label>` | ... |
| `duplicate` | Duplicate of <url> (<relevance>, <recommended_action>) |
| `wontfix` | <own_labels or duplicate:<repo>#<n>> |
| `need_split` | <N> groups: <one-line signature each> |
```

The table is a **contract**: a workflow applies every row verbatim as a label
(or, for `type` and `priority`, as the issue's native Type / project Priority
field). Emit each name exactly as it appears in `proposed_labels.json`.

## Axis sources

Read every enum value, spelling, and casing from these `proposed_labels.json`
locations — never hard-code them:

| Row | JSON source | Emitted as |
|---|---|---|
| `type` | `issue_type_field.values[].name` | native Type (not a label) |
| `priority` | `priority_field.values[].tier` | project Priority (not a label) |
| `test` | `categories.test` | label |
| `module` | `categories.module` | label |
| `os` / `hw` | `categories.os` / `categories.hw` | label |
| `dtype` | `categories.dtype` | label(s) |
| symptom | `categories.symptom` | label(s), case-sensitive |
| `dependency component` | `categories.dependency` | label (match `code` -> emit `name`) |
| `duplicate` / `wontfix` / `need_split` | `categories.triage` | label |

## Header notes

Each appears as a single line above the table, only when its condition holds:

- `Analyzed case: <id> (case 1 of <N>; the other <N-1> not analyzed)` — when
  `extract.json`'s `test_cases` has 2+ entries. The labels describe this case
  only. Identify an E2E case by `benchmark`/`model`/`phase`/`dtype`. Omit for a
  single case; never write "case 1 of 1".
- `Trace mode: evidence-only (no pytorch_folder provided)` — in evidence-only
  mode. Cite `no local checkout provided` in any `null` reason a trace would have
  resolved.
- `Duplicate search: failed (<reason>)` — when every duplicate query failed or
  returned nothing parseable; also omit the `duplicate` row (a failed search is
  not `has_duplicate: false`).

## Row rules

| Row(s) | Emit when | Notes |
|---|---|---|
| `type`, `test` | always | `type` mirrors GitHub's native Type; when `issue_type` is `""`, infer the tier from evidence using `issue_type_field` keywords (reported failure -> Bug, new capability -> Feature). `test` distinguishes a build/infra failure as a `module`, not a `test`. |
| `module`, `priority` | always | Emit the chosen value verbatim from its Axis source. |
| `os`, `hw` | value present | Straight from `extract.json`; omit when blank (not OS/HW-specific). |
| `dtype`, symptom | value present | Multi-label: one row per value, omit the axis when none. |
| `dependency component` | value present | Match the decided value to a label by its `code`, emit its `name`; when `exists_in_repo` is false, note in the reason it must be created. Omit on `none`/`null`. |
| `duplicate`, `wontfix` | true | Omit otherwise. |
| `need_split` | 2+ groups | Reason = group count + one-line signature per group. Never emit for one group; never write "1 group". |

Omitting a row means the axis produced no value — not that its step was skipped
(Step 3.5 must still have run to conclude `null`/`none`). Do not add any prose
explaining which rows were omitted.

## Reasons

- **One line each, <=~140 chars.** State the deciding signal and its concrete
  evidence — a `file:line`, issue URL, traceback frame, or measured percentage —
  and nothing else. No multi-sentence justifications, no restating the rule, no
  listing ruled-out alternatives.

## Examples

Multi-group (analyzed-case note + `need_split`; all 4 cases share one signature
for group 0, so the labels describe only case 1):

```markdown
label-issue: intel/torch-xpu-ops#4200

Root cause: oneDNN has no bf16 `addmm` matmul primitive on this platform
(`aten/src/ATen/native/mkldnn/xpu/Blas.cpp:214`).

Analyzed case: test/xpu/test_matmul_xpu.py::TestMatmulXPU::test_addmm_bfloat16 (case 1 of 4; the other 3 not analyzed).

| label | reason |
|---|---|
| `type: Bug` | `issue_type` is `Bug`. |
| `test: ut` | Reproduce steps run `pytest test/xpu/test_matmul_xpu.py`. |
| `module: gemm` | Fails in the oneDNN matmul path, the addmm/gemm family. |
| `priority: Medium` | 4 UT cases, RuntimeError without crash. |
| `dtype: bfloat16` | Analyzed case is `test_addmm_bfloat16`; error names the bf16 matmul primitive. |
| `dependency component: oneDNN` | `RuntimeError` names the oneDNN matmul primitive descriptor. |
| `need_split` | 2 groups: RuntimeError missing addmm primitive in test_addmm_bfloat16; AssertionError tolerance in test_div_float64. |
```

Evidence-only (no `pytorch_folder`; owner not pinned, so `type` is inferred and
`module` falls to the catch-all):

```markdown
label-issue: intel/torch-xpu-ops#4302

Root cause: insufficient information for root causing: no pytorch_folder provided
and issue evidence is not self-sufficient

Analyzed case: test/xpu/test_ops_xpu.py::TestOpsXPU::test_foo_xpu (case 1 of 3; the other 2 not analyzed).

Trace mode: evidence-only (no pytorch_folder provided).

| label | reason |
|---|---|
| `type: Bug` | `issue_type` empty; heuristic matches `AssertionError` in body. |
| `test: ut` | Traceback shows a `pytest` frame with no e2e/oob markers. |
| `module: infra` | Owning component not identifiable from traceback alone; catch-all. |
| `priority: Medium` | 3 UT cases, AssertionError without crash. |
```
