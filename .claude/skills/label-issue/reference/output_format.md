# Output

Write `agent_space/label_issue/<repo_underscored>_issue_<id>/labels.md`:

```markdown
label-issue: <repo>#<id>

Root cause: <=2 lines, specific, with file:line when a trace read one>

<When the issue reports 2+ cases, add exactly one line naming the analyzed case:
Analyzed case: <test_file>::<test_class>::<test_case> (case 1 of <N>; the other
<N-1> not analyzed). Omit this line entirely for a single-case issue.>

<In evidence-only mode add exactly one line:
Trace mode: evidence-only (no pytorch_folder provided).>

<When every duplicate search query failed or returned nothing parseable, add
exactly one line, and omit the `duplicated` row below:
Duplicate search: failed (<one-line reason, e.g. all queries returned non-JSON>).>

| label | reason |
|---|---|
| `type: <Bug\|Task\|Feature\|Epic>` | <`issue_type` from extract.json, or the failure evidence when it is blank, 1 line> |
| `test_module: <ut\|e2e\|build\|infrastructure>` | <deciding signal from extract.json, 1 line> |
| `module: <value>` | <bucket-deciding signal, 1 line> |
| `priority: <Urgent\|High\|Medium\|Low>` | <matched rule + evidence, 1 line> |
| `dependency component: <value>` | <direct evidence, 1 line; omit row when none/null> |
| `duplicated` | Duplicate of <url> (<relevance>, <recommended_action>); omit row when no duplicate |
| `not_target` | <own_labels or duplicate:<repo>#<n>>; omit row when false |

Report only — not GitHub labels, read by the calling workflow:

| field | reason |
|---|---|
| `need_split` | <N> distinct failure groups: <one-line signature each>; omit row when only one group |
| `target_component: <value>` | <traced fix location file:line, or the confirmed dependency, 1 line> |
| `need_action: <verdict>` | <deriving condition, 1 line> |
| `pr_link` | <PR URL from extract.json; omit row when pr_link is blank> |
```

Rules for the tables:

- **The two tables are a contract, not a formatting choice.** A GitHub workflow
  applies every row of the first table as a label (or, for `type`, as the
  issue's native Type) verbatim. The second table is report-only: `need_split`,
  `target_component`, `need_action`, and `pr_link` are never applied as labels —
  no `target_component: oneDNN` or `need_action: NEED_FIX` label exists in the
  repo. Never merge the two tables, and never move a row across them.

- The `need_split` row appears **only** when Step 1.5 found 2 or more distinct
  failure groups. Groups are keyed on the normalized error message, with the test
  function as a tie-breaker only. It is a recommendation for a human — the skill
  never splits the issue. Its reason names the group count and gives a one-line
  signature per group, e.g.
  `2 groups: RuntimeError missing addmm primitive in test_addmm_bfloat16; AssertionError tolerance in test_div_float64`.
  Judge groups by error signature, never by the number of cases or test
  functions: 29 cases across 11 functions sharing one error are ONE group and get
  no row. Do not write "1 group".
- The `Analyzed case:` line appears **only** when `extract.json`'s `test_cases`
  holds 2 or more entries. It names `test_cases[0]` — the case every row in
  both tables describes — and states how many cases went unanalyzed, so a
  reader is never misled into thinking the labels cover the whole issue. For
  an E2E entry,
  identify the case by `benchmark`/`model`/`phase`/`dtype` instead of
  `file::class::case`. A single-case issue omits the line; do not write
  "case 1 of 1" and do not add a paragraph explaining the omission.
- The dependency row's label comes from the mapping table in
  [dependency.md](dependency.md), keyed on Step 3's `dependency` value. Emit
  the label column verbatim, never the raw enum value — e.g. `oneDNN` maps to
  the label `dependency component: oneDNN`, so the row is
  `` `dependency component: oneDNN` ``, not `` `dependency: oneDNN` ``. Most
  values carry the `dependency component: ` prefix, but `third_party_packages`
  maps to the label `dependency: third_party packages` — a different prefix,
  and a space. Three labels (`oneCCL`, `IGC`, `Level_Zero`) do not exist in the
  repo yet; emit them anyway and note in the reason that the label must be
  created.
- The `target_component` row names the owner. When the dependency axis (Step 3)
  returned a taxonomy value, that value IS the `target_component` — emit
  `target_component: oneDNN`, never `target_component: third-party` — and the
  reason cites the dependency evidence rather than a product `file:line`. The
  dependency row and the `target_component` row therefore agree on the component
  in that case. See [target_component.md](target_component.md).
- The module row's label comes from the mapping table in
  [module.md](module.md), keyed on Step 6's `module` bucket. Emit the label (e.g.
  `module: ao`, `module: core`), never the raw bucket name. Two buckets differ
  from their label — `torchAO` -> `module: ao` and `torch-runtime` -> `module:
  core` — so emitting the bucket would produce a label that does not exist in the
  repo.
- `type` and `test_module` come straight from `extract.json` (Step 1); no
  `label-issue/reference/` pack governs them, and no step re-derives them here.
  (`test_module` was itself decided in Step 1 per
  `extract-issue/reference/testcase_rules.md`.) Always emit both rows — they are
  never omitted.
  The `type` row carries `extract.json`'s **`issue_type`** field (the canonical
  `Bug`/`Task`/`Feature`/`Epic` value). It mirrors GitHub's native issue
  **Type**, so it is applied as the issue type, not as a label. When
  `issue_type` is `""` — the issue has no
  GitHub Type set — infer the tier from the failure evidence (a reported failure
  is `Bug`; a request for new functionality is `Feature`) and say so in the
  reason. The row is still required.
- `priority` is the PyTorchXPU project's `Priority` field, not a label. Emit the
  canonical tier name (`Urgent`/`High`/`Medium`/`Low`) as produced by
  `extract.json`'s `priority`; see [priority.md](priority.md) for the mapping to
  the field's current `P0`-`P3` options.
- Emit a row only when the axis produced a value. Omit the dependency,
  `duplicated`, `not_target`, `need_split`, and `pr_link` rows entirely when they
  do not apply, but Step 3 must still have been run against
  `reference/dependency.md` to reach that `null`/`none` conclusion — an omitted
  row is not a skipped step.
- The `pr_link` row carries the bare URL from `extract.json` as its reason, with
  no prose. Omit the row when `pr_link` is `""`.
- **Every reason is ONE line: one sentence or clause, no more than ~140
  characters.** State the deciding signal and its evidence, nothing else. No
  multi-sentence justifications, no restating the rule text, no listing the
  alternatives that were ruled out, no explaining what was not chosen.
- Every reason must cite concrete evidence: a `file:line`, an issue URL, a
  traceback frame, or a measured percentage. No bare restatements of the rule.
- `null` (insufficient evidence) is a valid outcome. For an always-emitted axis —
  `target_component` and `need_action` — emit the row with reason
  `insufficient evidence: <what was missing>` rather than guessing. For the
  omittable axes listed above, `null` means the row is omitted instead; the two
  rules do not conflict.
- Omitted rows need no explanation. Do not append a paragraph describing which
  rows were omitted or why.
- In evidence-only mode, emit the one-line `Trace mode:` note and cite
  `no local checkout provided` as the missing evidence in any `null` or `N/A`
  reason that a trace would have resolved.
- When every duplicate search query failed or returned nothing parseable (per
  [duplicates.md](duplicates.md)), emit the one-line `Duplicate search: failed`
  note and omit the `duplicated` row — a failed search is not evidence of
  `has_duplicate: false`, so it must not be silently reported the same way.

Also print both tables to stdout.

### Brevity example

Good — one sentence per reason, no argument for the verdict. See the
Multi-case example below for a full worked table in this style.

Bad — a paragraph of root cause, and reasons that argue the case:

```markdown
Root cause: Three cases assert that the bisector isolates a subsystem, and on
Windows it instead returns None. The mechanism is upstream and device-neutral:
BACKENDS registers pre_grad_passes, joint_graph_passes, ... [15 more lines]

| `priority: Medium` | No project Priority field is set, so the decision tree
applies. Not Urgent (no SIGSEGV on a Core API, no measured >7% regression). Not
High (3 distinct cases, not >6; no cited pass-then-fail pair). Matches Medium
twice: ... |
```

### Multi-case example

`test_cases` holds 3 entries, so the analyzed case is named and the labels
describe only it. All 3 share one error signature, so this is ONE group and
`need_split` is **not** emitted:

```markdown
label-issue: intel/torch-xpu-ops#4171

Root cause: `do_bisect()` returns a result with `.subsystem` unset when no subsystem
isolates the failure (`torch/_inductor/compiler_bisector.py:646-653`).

Analyzed case: test/inductor/test_compiler_bisector.py::TestCompilerBisector::test_bad_backend (case 1 of 3; the other 2 not analyzed).

| label | reason |
|---|---|
| `type: Bug` | `issue_type` field is empty; `AssertionError` in title/body maps to Bug. |
| `test_module: ut` | Reproduce steps run `pytest test/inductor/test_compiler_bisector.py`. |
| `module: inductor` | Fails in `torch/_inductor/compiler_bisector.py:60-72` via the torch.compile path. |
| `priority: Medium` | 3 UT cases, AssertionError without crash. |

Report only — not GitHub labels, read by the calling workflow:

| field | reason |
|---|---|
| `target_component: pytorch` | Traced to `torch/_inductor/compiler_bisector.py:646-653`, upstream of torch-xpu-ops. |
| `need_action: NEED_FIX` | `pytorch` target_component. |
```

### Multi-group example

Two unrelated error signatures in one issue, so `need_split` is emitted while the
labels still describe only the analyzed case:

```markdown
label-issue: intel/torch-xpu-ops#4200

Root cause: oneDNN has no bf16 `addmm` matmul primitive on this platform
(`aten/src/ATen/native/mkldnn/xpu/Blas.cpp:214`).

Analyzed case: test/xpu/test_matmul_xpu.py::TestMatmulXPU::test_addmm_bfloat16 (case 1 of 4; the other 3 not analyzed).

| label | reason |
|---|---|
| `type: Bug` | `issue_type` is `Bug`. |
| `test_module: ut` | Reproduce steps run `pytest test/xpu/test_matmul_xpu.py`. |
| `module: torch-ops-gemm` | Fails in the oneDNN matmul path, the addmm/gemm family. |
| `priority: Medium` | 4 UT cases, RuntimeError without crash. |
| `dependency component: oneDNN` | `RuntimeError` names the oneDNN matmul primitive descriptor. |

Report only — not GitHub labels, read by the calling workflow:

| field | reason |
|---|---|
| `need_split` | 2 groups: RuntimeError missing addmm primitive in test_addmm_bfloat16; AssertionError tolerance in test_div_float64. |
| `target_component: oneDNN` | Confirmed oneDNN dependency owns the fix; `Blas.cpp:214` is the caller. |
| `need_action: NEED_FIX_3RDPARTY` | `oneDNN` target_component. |
```

The dependency row and the `target_component` row name the same component here
(see the bullet above) — `Blas.cpp:214` is the caller, so it belongs in the
`target_component` reason, never as the `target_component` value.

### Evidence-only example

No `pytorch_folder`, and the issue evidence does not pin an owner. This example
assumes a non-Windows issue; on Windows the last row would be `NEED_HUMAN` per
the verdict table in [target_component.md](target_component.md):

```markdown
label-issue: intel/torch-xpu-ops#4302

Root cause: insufficient information for root causing: no pytorch_folder provided
and issue evidence is not self-sufficient

Analyzed case: test/xpu/test_ops_xpu.py::TestOpsXPU::test_foo_xpu (case 1 of 3; the other 2 not analyzed).

Trace mode: evidence-only (no pytorch_folder provided).

| label | reason |
|---|---|
| `type: Bug` | `issue_type` empty; heuristic matches `AssertionError` in body. |
| `test_module: ut` | Traceback shows a `pytest` frame with no e2e/build/infra markers. |
| `module: others` | Owning component not identifiable from traceback alone. |
| `priority: Medium` | 3 UT cases, AssertionError without crash. |

Report only — not GitHub labels, read by the calling workflow:

| field | reason |
|---|---|
| `target_component: N/A` | insufficient evidence: no local checkout provided. |
| `need_action: NEED_FIX` | `N/A` target_component. |
```
