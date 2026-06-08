# classify_ut blank status_xpu cases

This skill follows agent-guidelines AND extends it with blank-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows whose `Reason` is blank and whose `status_xpu` is blank. Blank XPU status means
the case has no direct XPU result in the workbook; classification must come from deep
case-existence analysis, source inspection, skip-list evidence, and targeted local runs when
needed. This subskill is also invoked for the **TBE re-verification pass** (rows with
`TBE_Reverify = True` and blank `status_xpu`).

## TBE Re-verification Context

See the **TBE Re-verification Rule** in `classify_ut/RULES.md` for the full decision flow.
Blank-specific invariants:

- `Reason TBD` is **never** flipped for a re-verified row; the two columns are independent.
- The updated `DetailReason` MUST start with `[Reverified: YYYY-MM-DD]` (before any other
  content) so a human reviewer can distinguish re-checked rows from rows left as-is.
- The `Confidence` workbook column is NOT populated for re-verified rows (it is reserved for
  `Reason TBD = True` rows; see the **Confidence Rubric & Need-Human-Check Rule** in `RULES.md`).

## Required Inputs

| Input | Source / Default |
|-------|------------------|
| Working file | The extracted target sheet (default) OR a row-filtered subset produced by `classify_ut/scripts/filter_target_rows.py` (when the user names a subset). See parent `classify_ut/SKILL.md` "Row-Level Filter" section. |
| Workbook row fields | `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`, `classname_xpu`, `name_xpu`, `status_xpu`, `Reason`, `DetailReason`, `Reason TBD` |
| Local PyTorch checkout | `$PYTORCH_SRC` (user-provided; default `$HOME/upstream/pytorch`) |
| XPU test checkout | `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu` |
| Conda environment | `pytorch_opencode_env` (default name; override via `PYTORCH_ENV`) |
| Deep case-existence workflow | `${ISSUE_TRIAGE_ROOT}/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md` |
| Canonical labels / CUDA-Only / Confidence / TBE | `classify_ut/RULES.md` (read on demand, not every row) |

## Required Tools

| Tool | Purpose |
|------|---------|
| `read` | Inspect base tests, XPU wrapper/direct files, distributed runner files, skip lists, parametrization, decorators |
| `bash` | Activate `pytorch_opencode_env`, run targeted tests or `pytest --collect-only`, inspect git refs, fetch remote files with `git show` or `gh` |
| `grep` / `rg` | Locate exact class/function/decorator names AFTER semantic analysis identifies the files to read. Do not classify from grep output alone |
| `gh` CLI | Inspect GitHub files/issues when local source or `message_xpu` references them |
| `openpyxl` | Write results; classification must come from source and execution evidence, not the workbook tool |

## Decision Routing

| `testfile_cuda` location | `status_xpu` | Workflow |
|--------------------------|--------------|----------|
| `test/distributed/**` | blank | Distributed Blank-Status Workflow below |
| `test/dynamo/**`, `test/inductor/**`, or direct PyTorch test | blank | Non-Distributed Blank-Status Workflow below (start with the direct PyTorch file) |
| Anything else | blank | Non-Distributed Blank-Status Workflow below |

Both workflows share the same **Decision Axes** below. The routing only changes which source
files to read first.

## Decision Axes

For each row, gather evidence on the following axes. The axes are independent and may be
investigated in parallel via subagents (see **Delegation Hints** below). After evidence is
gathered, walk the **Workflow** in order to assign a canonical label.

| Axis | What to read | HIGH signal | LOW signal |
|------|--------------|-------------|------------|
| **CUDA-Only API match** | `Operation/API` column of `Not applicable` sheet in `torch_xpu_ops_issues.xlsx` (via `scripts/list_not_applicable.py`) | Exact `Operation/API` match with `Issue ID` and `Category` cited | No matching entry; test clearly uses a CUDA-only API in source |
| **Base function** | `$PYTORCH_SRC/<testfile_cuda>` (use `git show` on candidate commits) | Located with file path + line range cited | Not found after thorough search of PyTorch + `third_party/torch-xpu-ops` |
| **XPU wrapper / registration** | `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu/**` (all sub-folders) and `instantiate_device_type_tests` call sites in the base file | `_xpu`-suffixed case located, OR `allow_xpu=True` present with cited file:line | No XPU wrapper / no `allow_xpu` / device list excludes XPU |
| **Known issue search** | `gh search issues` on `intel/torch-xpu-ops` and `pytorch/pytorch` | Open/closed issue found, URL cited and verified via `gh issue view`; for `skipped`-label issues verdict is anchored by a local run | No issue found after keyword variations on both repos |
| **Local verification** | `port-pytorch-tests-xpu` on a `daisyden/pytorch` branch | Executed test, stdout/stderr saved under `/tmp/opencode/<workbook>_local_verify/<case>.log`; cite branch + log path | Not run |
| **Workbook precedent** | Already-classified peer rows in the SAME sheet with the same `(testfile_cuda, classname_cuda)` OR `(testfile_cuda, base_method)` | ≥5 peers, ≥95% unanimous on one canonical `Reason` (MEDIUM by definition; still need ≥1 source-evidence axis) | Insufficient peers or mixed |

Confidence is assigned per the **Confidence Rubric & Need-Human-Check Rule** in `RULES.md`:
HIGH needs at least one verifiable issue URL, local-verification artifact, or file:line that
directly proves the verdict; MEDIUM is best-fit without those; LOW -> `Reason = "Need human check"`.

## Hard Constraints (blank-specific only)

- Blank `status_xpu` is NOT proof the case is missing. It means the row needs deep analysis.
- Do not classify by filename patterns, keyword matches, or a bulk script alone.
- Do not stop at test-file existence. Check exact test-case existence after class/device
  instantiation, parametrization, OpInfo/device/dtype filtering, and other generated-test
  arguments.
- XPU test cases may live directly in `pytorch/test/` (especially Dynamo/Inductor tests).
  Do not assume every non-distributed XPU case requires a
  `third_party/torch-xpu-ops/test/xpu/*_xpu.py` wrapper.
- CUDA graph / cudagraph rows are not `Not applicable` merely because the CUDA name contains
  `cuda`. XPU graph support exists via `_XPUGraph`, `torch.xpu.XPUGraph`, and
  `torch.accelerator.Graph`; missing/failing coverage is `To be enabled` with an XPU graph
  DetailReason.
- `Community Change` is used when the base function/case is removed/renamed/refactored/moved
  or disabled by an upstream community issue/commit. If an XPU variant exists after
  parametrization, do not call it community changes.
- **Only the "Not Applicable" sheet defines "Not applicable (CUDA-only)"**. The sole source
  of truth for CUDA-only classification is the "Not Applicable" sheet in
  `torch_xpu_ops_issues.xlsx` (accessed via `scripts/list_not_applicable.py`). If the test's
  CUDA-only API matches an `Operation/API` entry there with a cited `Issue ID` and `Category`,
  classify as `Not applicable (CUDA-only)`. Otherwise, it is `To be enabled`.
- **"Not applicable" `DetailReason` MUST include `not_target=<Issue ID>` marker**. When
  assigning `Not applicable (CUDA-only)`, the `DetailReason` MUST contain an explicit
  `not_target=<Issue ID>` marker (e.g., `not_target=3129`) for machine-grepable evidence
  linking to the "Not Applicable" sheet entry. This is in addition to citing the Issue ID
  and Category in prose. Format: `not_target=<issue_number>` placed anywhere in the
  `DetailReason` text.
- **Decorators like `@onlyCUDA`, `@skipIfNoCUDA`, `@tf32_on_and_off` are NOT sufficient for
  "Not applicable"**. These decorators just mean the test was authored for CUDA and hasn't
  been updated for XPU yet — that is an enablement gap (`To be enabled`), not a permanent
  exclusion. The only exception is if the API used by the test is explicitly listed in the
  "Not Applicable" sheet.
- **Parametrization / decorator does not include XPU → `To be enabled`**. If a test's
  decorators or parametrization (`@dtypesIfCUDA`, `@parametrize_test('device', ...)`,
  `@dtypes(…)` etc.) define device-specific behavior for CUDA but lack the XPU equivalent,
  the correct classification is `To be enabled` — the XPU team needs to add the corresponding
  decorator (`@dtypesIfXPU`, add `'xpu'` to the device list, etc.). Parametrization that
  merely omits XPU is an enablement gap, not permanent exclusion.

All other constraints — CUDA-Only Judgement Rule, `DetailReason` content rules,
`Reason TBD` / `Confidence` / `TBE_Reverify` column semantics, P1/P2/P3, Sibling-Class
Verdict Mapping, Workbook Precedent Rule, Dynamic-Skip Rule, Local Verification via
XPU Port, TBE Re-verification Rule, Confidence Rubric & Need-Human-Check Rule, and
`Failures (xpu broken)` issue-link requirement — are authoritative in
`classify_ut/RULES.md` and are NOT duplicated here. The parent
`classify_ut/SKILL.md` "Classification Rules" table maps each rule to its scope.

## User-Issued Policy Overrides

Apply on every row. All four policies (P1 Local-Retest Mandate, P2 `Not applicable` evidence
+ no-run rule, P3 JIT cluster policy, Sibling-Class Verdict Mapping, Workbook Precedent Rule)
are authoritative in `classify_ut/RULES.md` and are NOT duplicated here. The parent
`classify_ut/SKILL.md` "Classification Rules" table maps each policy to its scope.

## Optional Preparation Inputs

The parent-skill preparation steps (`classify_ut/preparation/SKILL.md`) are optional and
NOT run by default. PASS `local_result` rows are TERMINAL (`Reason = "Local Passed"`) and
this subskill must not touch them. For all other rows proceed with the deep case-existence
analysis below. See parent `classify_ut/SKILL.md` "Preparation" section for the full
provenance contract and the 60-second per-test timeout policy.

## Workflow

1. **Confirm eligibility**: `Reason` blank, `status_xpu` blank, CUDA metadata identifies
   the exact test file/class/method.
2. **Route by `testfile_cuda` location** per the **Decision Routing** table above.
3. **Check base-function existence and Community Change first** using the
   `Decision Axes` "Base function" axis (`PYTORCH_SRC` as source of truth). If the base
   function is absent, classify `Community Change` (refactors, renames, removals, moves).
4. **If base function exists**, derive the expected XPU case by replacing `_cuda` with `_xpu`
   in `name_cuda`, then verify generation through parametrization, decorators, and source.
5. **Run the other Decision Axes in parallel** when possible (see **Delegation Hints**).
   For non-distributed rows, the XPU wrapper axis is the most likely to determine the
   verdict. For distributed rows, the distributed skip-list axis is decisive.
6. **Assign a canonical label and confidence level** per the workflow's accumulated evidence.
   The full label set (with per-label evidence rules) is in `classify_ut/RULES.md` §
   "Column Definitions → Reason". Quick map: `Community Change` for base-function absence /
   rename / refactor / move; `To be enabled` for missing XPU registration or coverage;
   `Not applicable` for CUDA-only API (cite Issue ID + Category from the `Not applicable`
   sheet) or CPU-only case (`DetailReason = CPU Case`); `Failures (xpu broken)` for XPU
   implementation bugs; `Feature gap` for missing XPU features; `Local Passed` for executed
   `pytorch_opencode_env` PASS; `Need human check` (LOW) when no axis yields a confident
   category.

## Distributed Blank-Status Workflow

Distributed XPU tests do not use `*_xpu.py` wrappers; they run upstream files through
`third_party/torch-xpu-ops/test/xpu/run_distributed.py` and distributed skip dictionaries.

1. **Read active imports** in `third_party/torch-xpu-ops/test/xpu/run_distributed.py`.
2. **Read the matching skip lists**. For `release/2.12` analysis, the remote skip lists live
   on `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`:
   `test/xpu/skip_list_dist.py` and `test/xpu/skip_list_dist_local.py` (the latter is the
   verified branch filename; the file may also be referenced as `skip_list_local_dist.py`).
   For current-checkout analysis, read the local
   `third_party/torch-xpu-ops/test/xpu/skip_list_dist.py` (and `skip_list_dict_local.py`
   if it exists).
3. **Normalize skip-list keys** semantically. `../../../../test/distributed/...` points to
   upstream PyTorch distributed tests; `distributed/test_c10d_xccl.py` points to XPU-native
   standalone distributed tests. Also inspect
   `third_party/torch-xpu-ops/test/xpu/distributed/` for XPU-native files such as
   `test_c10d_xccl.py` and `test_c10d_ops_xccl.py`.
4. **Interpret entries**. File with value `None` -> whole file enabled for XPU. File with
   tuple/list -> file enabled, listed cases intentionally skipped, others run. File absent
   -> upstream file not run by `run_distributed.py` -> classify `To be enabled` with a
   `DetailReason` naming the missing file. For distributed rows only, also check
   `https://github.com/daisyden/pytorch/tree/release/2.12` or local `origin/release/2.12`
   for the upstream file/method before deciding a test was removed, renamed, or newly added.

## Non-Distributed Blank-Status Workflow

1. **Confirm base function in `$PYTORCH_SRC/<testfile_cuda>`**. If absent, classify
   `Community Change` before checking XPU wrappers.
2. **If the row points to a direct PyTorch test folder** (`test/dynamo/`, `test/inductor/`,
   etc.) and `testfile_xpu == testfile_cuda`, analyze the direct PyTorch test file first:
   read the class/method body, check device-generic vs `instantiate_device_type_tests`,
   and check whether the exact case is generated by
   parametrization/OpInfo/dtypes/decorators/arguments. If collection reports zero tests and
   source inspection shows the method/case is absent, classify `Community Change` (NOT
   `XPU test file missing`).
3. **Enumerate all plausible XPU locations** before declaring a wrapper/direct XPU file
   missing: `third_party/torch-xpu-ops/test/xpu/`, `…/extended/`, `…/nn/`, `…/functorch/`,
   `…/quantization/`, plus any other relevant sub-folders discovered from imports or
   naming.
4. **Read wrapper/direct XPU files**. Not all XPU tests use `XPUPatchForImport`; many are
   standalone copies or direct implementations. If a wrapper uses `XPUPatchForImport`,
   `XPUPatchForImport(False)` usually imports/instantiates upstream tests with XPU
   adaptations; `XPUPatchForImport(True)` can disable or alter instantiation.
5. **Inspect class definitions, imports, `instantiate_device_type_tests`,
   `instantiate_parametrized_tests`, OpInfo filters, decorators, and xfail/skip lists** to
   determine whether the exact XPU case is generated. Use targeted collection or local
   execution when source inspection is inconclusive.
6. **Map outcomes**: no XPU source generates the case but base test exists and feature
   applies to XPU -> `To be enabled`; source proves CUDA-only -> `Not applicable` with
   exact API named; source proves CUDA/base test removed/renamed/refactored ->
   `Community Change`.

## Delegation Hints

The six **Decision Axes** are independent and can be investigated in parallel:

| Parallelize | Subagent type | Why |
|-------------|---------------|-----|
| `Base function` + `XPU wrapper` + `CUDA-Only API match` | `explore` (or several in parallel) | Pure file/search work; no side effects |
| `Known issue search` | `librarian` (with `gh search issues`) | External reference lookup |
| `Local verification` | `unspecified-high` (or `build` agent) | Requires running a ported test; blocks on log artifact |
| `Workbook precedent` | `explore` | Reads already-classified peer rows in the sheet |

`Local verification` cannot run in parallel with itself (each test blocks on its own log
artifact) and should run AFTER the other axes resolve, since the port branch name and log
path both need to be cited in the final `DetailReason`. The other five axes can be
fanned out concurrently. After all subagents return, walk **Workflow** step 6 to assign
the canonical label.

### Subagent Prompt Construction

Every subagent prompt MUST include these constraints to prevent output duplication and
token waste:

- **Output each ROW line exactly once.** Do NOT emit rows incrementally (e.g., a "first
  pass" followed by a "final summary"). Collect all row verdicts in memory, then emit them
  as a single parseable block at the very end. Never re-emit a row that has already been
  emitted — even if the agent re-executes part of its plan for verification.
- **Use a single date-stamp across all rows.** If the prompt specifies a date prefix like
  `[Daisy-YYYY-MM-DD]`, the subagent must produce that exact date for every row. Do NOT
  let the subagent auto-stamp each row with `datetime.now()` in separate passes — this
  creates inconsistent dates when passes cross midnight or span different agent calls.
- **Return only the parseable output block.** Strip all tool output, commentary, "I found
  that..." narration, and chain-of-thought. If the subagent needs to explain a borderline
  verdict, encode that explanation in the DETAIL field, not in surrounding prose.

## Known Classification Examples

`blank/EXAMPLES.md` holds concrete `Reason` + `DetailReason` templates validated by prior
classification sessions. Read it only when you need a template for a specific verdict
category — do NOT load it for every blank-Reason row. Use as a template, NOT a substitute
for source inspection: re-read the cited source state before applying.

## Output Rules

- `Reason`: use the canonical labels in `classify_ut/RULES.md` § Column Definitions → `Reason`.
- `DetailReason`: be specific enough to act on. Always use full issue/PR URLs (e.g.
  `https://github.com/pytorch/pytorch/issues/NNNNN`), never bare numbers like `#NNNNN`.
  Include exact test identity, source files read, skip-list or wrapper evidence, and
  reasoning. Avoid generic `No XPU wrapper` and generic `CUDA-specific API`. All other
  content requirements (per-Reason evidence, `[Issue_TBD]` prefix, `Local Passed` artifact
  path) are defined in `RULES.md` and are NOT restated here.
- When `Reason` is `Not applicable (CUDA-only)`, `DetailReason` MUST contain a
  `not_target=<Issue ID>` marker (e.g., `not_target=3129`) per the Hard Constraints.
- For `Reason TBD = True` rows, `DetailReason` MUST start with a `[Confidence: HIGH|MEDIUM|LOW]`
  prefix per the **Confidence Rubric** in `RULES.md`. For re-verified rows
  (`TBE_Reverify = True`), start with `[Reverified: YYYY-MM-DD]` instead.
- Mark updated `Reason`, `DetailReason` cells blue
  (`PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`).

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for processed blank-status rows.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least one distributed enabled row, one distributed missing row, one
  CUDA-specific API row, one XPU-wrapper-existing row, and one community-change row when
  those categories are present.
- ZIP integrity: re-save and re-open the workbook to confirm no corruption.

## Notes

- `classify_ut/WORKFLOW.md` is a static Mermaid chart for documentation only. Do NOT load
  it for every blank-Reason row; the decision flow is already captured in the **Decision
  Routing**, **Decision Axes**, and **Workflow** sections above. Read `WORKFLOW.md` only
  when you need to update or sanity-check the chart itself.
- `classify_ut/RULES.md` is the source of truth for canonical labels, the CUDA-Only
  Judgement Rule, the Confidence Rubric, the TBE Re-verification Rule, and the User-Issued
  Policy Overrides. Reference it by name; do not paraphrase from memory.
- If a row filter was used at extract time, the working file is a subset; the
  agent edits only the subset's `Reason` and `DetailReason`. Use
  `classify_ut/scripts/apply_filtered_changes.py` to write the verdicts back to
  the extracted file (or to `.agent.xlsx`); do NOT write to either file directly.
  See parent `classify_ut/SKILL.md` "Row-Level Filter" section.
