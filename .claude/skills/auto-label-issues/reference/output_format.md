# Output

Write `agent_space/auto_label_issues/<repo_underscored>_issue_<id>/labels.md`:

```markdown
auto-label-issues: <repo>#<id>

Root cause: <=2 lines, specific, with file:line>

<In evidence-only mode add exactly one line:
Trace mode: evidence-only (no pytorch_folder provided).>

| label | reason |
|---|---|
| `issue_type: <Bug\|Task\|Feature\|Epic>` | <source: github_type, label, or heuristic, 1 line> |
| `test_module: <ut\|e2e\|build\|infrastructure\|...>` | <deciding signal from extract.json, 1 line> |
| `module: <bucket>` | <bucket-deciding signal, 1 line> |
| `<P0-P3>` | <matched rule + evidence, 1 line> |
| `dependency component: <component>` | <direct evidence, 1 line; omit row when none/null> |
| `duplicated` | Duplicate of <url> (<relevance>, <recommended_action>); omit row when no duplicate |
| `not_target` | <own_labels or duplicate:<repo>#<n>>; omit row when false |
| `target_component: <value>` | <traced fix location file:line, 1 line> |
| `need_action: <verdict>` | <deriving condition, 1 line> |
```

Rules for the table:

- `issue_type` and `test_module` come straight from `extract.json` (Step 1); no
  reference file governs them. Always emit both rows — they are never omitted.
- Emit a row only when the axis produced a value. Omit `dependency component:`,
  `duplicated`, and `not_target` rows entirely when they do not apply, but
  Step 4 must still have been run against `reference/dependnecy.md` to reach
  that `null`/`none` conclusion — an omitted row is not a skipped step.
- **Every reason is ONE line: one sentence or clause, no more than ~140
  characters.** State the deciding signal and its evidence, nothing else. No
  multi-sentence justifications, no restating the rule text, no listing the
  alternatives that were ruled out, no explaining what was not chosen.
- Every reason must cite concrete evidence: a `file:line`, an issue URL, a
  traceback frame, or a measured percentage. No bare restatements of the rule.
- `null` (insufficient evidence) is a valid outcome — emit the row with reason
  `insufficient evidence: <what was missing>` rather than guessing.
- Omitted rows need no explanation. Do not append a paragraph describing which
  rows were omitted or why.
- In evidence-only mode, emit the one-line `Trace mode:` note and cite
  `no local checkout provided` as the missing evidence in any `null` or `N/A`
  reason that a trace would have resolved.

Also print the same table to stdout.

### Brevity example

Good — root cause is 2 lines, each reason is 1 line:

```markdown
Root cause: `do_bisect()` returns a result with `.subsystem` unset when no subsystem
isolates the failure (`torch/_inductor/compiler_bisector.py:646-653`).

| label | reason |
|---|---|
| `issue_type: Bug` | `github_type` field is empty; `AssertionError` in title/body maps to Bug. |
| `test_module: ut` | Reproduce steps run `pytest test/test_...py`, no e2e/build/infra signal. |
| `module: inductor` | Fails in `torch/_inductor/compiler_bisector.py:60-72` via the torch.compile path. |
| `P2` | 3 UT cases, AssertionError without crash. |
| `target_component: pytorch` | Traced to `torch/_inductor/compiler_bisector.py:646-653`, upstream of torch-xpu-ops. |
| `need_action: NEED_FIX` | `pytorch` target_component. |
```

Bad — a paragraph of root cause, and reasons that argue the case:

```markdown
Root cause: Three cases assert that the bisector isolates a subsystem, and on
Windows it instead returns None. The mechanism is upstream and device-neutral:
BACKENDS registers pre_grad_passes, joint_graph_passes, ... [15 more lines]

| `P2` | No project Priority field is set, so the decision tree applies. Not P0
(no SIGSEGV on a Core API, no measured >7% regression). Not P1 (3 distinct
cases, not >6; no cited pass-then-fail pair). Matches P2 twice: ... |
```

### Evidence-only example

No `pytorch_folder`, and the issue evidence does not pin an owner:

```markdown
Root cause: insufficient information for root causing: no pytorch_folder provided
and issue evidence is not self-sufficient

Trace mode: evidence-only (no pytorch_folder provided).

| label | reason |
|---|---|
| `issue_type: Bug` | `github_type` empty; heuristic matches `AssertionError` in body. |
| `test_module: ut` | Traceback shows a `pytest` frame with no e2e/build/infra markers. |
| `module: others` | Owning component not identifiable from traceback alone. |
| `P2` | 3 UT cases, AssertionError without crash. |
| `target_component: N/A` | insufficient evidence: no local checkout provided. |
| `need_action: NEED_FIX` | `N/A` target_component. |
```
