# Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md` in the
layout below, and also print it to stdout. This file defines the artifact's
*shape*; the label names and their evidence come from `label_def.json`, and
each axis is decided in the label-issue skill's Step 3.

## Skeleton

Every `<value>` below is one of the enum values defined in
`label_def.json` (see **Axis sources**); this file never enumerates them.

Wrap the ENTIRE artifact in a collapsible `<details>` block whose `<summary>` is
the `label-issue: <repo>#<id>` title, so the content is hidden until clicked. Keep
a blank line after the `<summary>` line and before the closing `</details>` so
GitHub renders the Markdown tables inside. This wrapping applies to both the
single-group and multi-group layouts.

When the issue has 2+ groups, first emit a **top-level output table** holding the
`need_split` row (and any other issue-wide triage rows), then repeat the per-group
block once per group in group order. Each group head is
`## Group <n> — <summary of the group of tests>` (a short phrase describing what
that group of tests has in common, not just the representative case id). End each
group's block with a `Test cases (<M>):` list enumerating every test case in that
group. For a single group, drop the top-level table and the `## Group` head and
emit one block (no `Test cases` list needed unless the group has 2+ cases).

```markdown
<details>
<summary>label-issue: <repo>#<id></summary>

| axis | value | reason |
|---|---|---|
| triage | `need_split` | <N> groups: <one-line signature each> |

## Group <n> — <summary of the group of tests>

Root cause: <=2 lines, specific, with file:line when a trace read one>

<optional header notes, one line each - see below>

| axis | value | reason |
|---|---|---|
| `type` | `<value>` | ... |
| `test` | `test: <value>` | ... |
| `module` | `module: <value>` | ... |
| `priority` | `<value>` | ... |
| `os` | `os: <value>` | ... |
| `hw` | `hw: <value>` | ... |
| `dtype` | `dtype: <value>` | ... |
| `dependency component` | `dependency component: <value>` | ... |
| symptom | `<symptom label>` | ... |
| triage | `duplicate` | Duplicate of <url> (<relevance>, <recommended_action>) |
| triage | `wontfix` | <own_labels or duplicate:<repo>#<n>> |

Test cases (<M>):
- <case 1>
- <case 2>

</details>
```

The `need_split` row lives ONLY in the top-level table; never repeat it inside a
group block. Per-group triage rows (`duplicate`, `wontfix`) stay in that group's
table. When a header note names the representative case, the `## Group` summary
still describes the whole group, not that one case.

The `value` column is the token a workflow applies verbatim: a label name for
label axes (with its `type:` / `module:` / etc. prefix), or the bare enum value
for the native `type` (GitHub Type) and `priority` (native org Priority issue
field) fields.
Emit each value exactly as it appears in `label_def.json`. (Exception: on a
`need_split` issue a consumer applies only the issue-wide axes to the umbrella
issue — see the need_split suppression policy below.)

## Axis sources

Read every enum value, spelling, and casing from these `label_def.json`
locations — never hard-code them:

| Row | JSON source | Emitted as |
|---|---|---|
| `type` | `issue_type_field.values[].name` | native Type (not a label) |
| `priority` | `priority_field.values[].tier` | native org Priority issue field (not a label) |
| `test` | `categories.test` | label |
| `module` | `categories.module` | label |
| `os` / `hw` | `categories.os` / `categories.hw` | label |
| `dtype` | `categories.dtype` | label(s) |
| symptom | `categories.symptom` | label(s), case-sensitive |
| `dependency component` | `categories.dependency` | label (match `code` -> emit `name`) |
| `duplicate` / `wontfix` / `need_split` | `categories.triage` | label |

## Header notes

Each appears as a single line above the table, only when its condition holds:

- `Analyzed case: <id> (case 1 of <M>; the other <M-1> not analyzed)` — when this
  group's `test_cases` has 2+ entries. The labels describe this representative case
  only, while the `## Group` summary and the trailing `Test cases` list describe the
  whole group. Identify an E2E case by `benchmark`/`model`/`phase`/`dtype`. Omit for
  a single case; never write "case 1 of 1".
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
| `need_split` | 2+ groups | Emitted once in the top-level output table (never inside a group block). Reason = group count + one-line signature per group. Never emit for one group; never write "1 group". |
| `Test cases (<M>)` list | per group when 2+ groups, or a single group with 2+ cases | Enumerate every test case in the group at the END of that group's block, after the table. |

Omitting a row means the axis produced no value — not that its step was skipped
(Step 3.5 must still have run to conclude `null`/`none`). Do not add any prose
explaining which rows were omitted.

### need_split suppression policy

When the `need_split` row is present, the per-group axes describe the individual
sub-issues that will be created by the split, not the umbrella issue. A consumer
applying labels.md to the umbrella issue (e.g. `apply_labels.py`) therefore
applies ONLY the issue-wide axes — `need_split`, `type`, `test`, `os`, `hw` — and
suppresses the per-group axes `module`, `dtype`, `dependency component`,
`symptom`, `duplicate`, `wontfix`, and `priority`. Those carry over to the
sub-issues once the split happens. The artifact still emits every row for the
human reader and for the sub-issues; the suppression is a write-time policy, not
a reason to drop rows from labels.md.

## Reasons

- **One line each, <=~140 chars.** State the deciding signal and its concrete
  evidence — a `file:line`, issue URL, traceback frame, or measured percentage —
  and nothing else. No multi-sentence justifications, no restating the rule, no
  listing ruled-out alternatives.

## Examples

Multi-group (top-level `need_split` table, then one block per group, each ending
with its `Test cases` list):

```markdown
<details>
<summary>label-issue: intel/torch-xpu-ops#4200</summary>

| axis | value | reason |
|---|---|---|
| triage | `need_split` | 2 groups: RuntimeError missing addmm primitive in test_addmm_bfloat16; AssertionError tolerance in test_div_float64. |

## Group 1 — bf16 addmm matmul missing oneDNN primitive

Root cause: oneDNN has no bf16 `addmm` matmul primitive on this platform
(`aten/src/ATen/native/mkldnn/xpu/Blas.cpp:214`).

| axis | value | reason |
|---|---|---|
| `type` | `Bug` | `issue_type` is `Bug`. |
| `test` | `test: ut` | Reproduce steps run `pytest test/xpu/test_matmul_xpu.py`. |
| `module` | `module: gemm` | Fails in the oneDNN matmul path, the addmm/gemm family. |
| `priority` | `Medium` | 3 UT cases in this group, RuntimeError without crash. |
| `dtype` | `dtype: bfloat16` | Case is `test_addmm_bfloat16`; error names the bf16 matmul primitive. |
| `dependency component` | `dependency component: oneDNN` | `RuntimeError` names the oneDNN matmul primitive descriptor. |

Test cases (3):
- test/xpu/test_matmul_xpu.py::TestMatmulXPU::test_addmm_bfloat16
- test/xpu/test_matmul_xpu.py::TestMatmulXPU::test_addmm_bfloat16_batched
- test/xpu/test_matmul_xpu.py::TestMatmulXPU::test_baddbmm_bfloat16

## Group 2 — fp64 div elementwise tolerance mismatch

Root cause: fp64 `div` result exceeds tolerance vs CPU reference
(`aten/src/ATen/native/xpu/sycl/BinaryDivKernels.cpp:88`).

| axis | value | reason |
|---|---|---|
| `type` | `Bug` | `issue_type` is `Bug`. |
| `test` | `test: ut` | Reproduce steps run `pytest test/xpu/test_binary_ufuncs_xpu.py`. |
| `module` | `module: eltwise` | Elementwise div kernel mismatch. |
| `priority` | `Medium` | 1 UT case in this group, AssertionError without crash. |
| `dtype` | `dtype: float64` | Case is `test_div_float64`; tolerance failure on fp64. |
| symptom | `Accuracy` | Numeric mismatch vs CPU reference, not a functional error. |

Test cases (1):
- test/xpu/test_binary_ufuncs_xpu.py::TestBinaryUfuncsXPU::test_div_float64

</details>
```

Evidence-only (no `pytorch_folder`; owner not pinned, so `type` is inferred and
`module` falls to the catch-all):

```markdown
<details>
<summary>label-issue: intel/torch-xpu-ops#4302</summary>

Root cause: insufficient information for root causing: no pytorch_folder provided
and issue evidence is not self-sufficient

Analyzed case: test/xpu/test_ops_xpu.py::TestOpsXPU::test_foo_xpu (case 1 of 3; the other 2 not analyzed)

Trace mode: evidence-only (no pytorch_folder provided).

| axis | value | reason |
|---|---|---|
| `type` | `Bug` | `issue_type` empty; heuristic matches `AssertionError` in body. |
| `test` | `test: ut` | Traceback shows a `pytest` frame with no e2e/oob markers. |
| `module` | `module: infra` | Owning component not identifiable from traceback alone; catch-all. |
| `priority` | `Medium` | 3 UT cases, AssertionError without crash. |

Test cases (3):
- test/xpu/test_ops_xpu.py::TestOpsXPU::test_foo_xpu
- test/xpu/test_ops_xpu.py::TestOpsXPU::test_bar_xpu
- test/xpu/test_ops_xpu.py::TestOpsXPU::test_baz_xpu

</details>
```
