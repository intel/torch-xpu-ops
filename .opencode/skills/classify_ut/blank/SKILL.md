# classify_ut blank status_xpu cases

This skill follows agent-guidelines AND extends it with blank-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows in any XPU UT status workbook whose `Reason` is blank and whose
`status_xpu` is blank. Blank XPU status means the case has no direct XPU result in the workbook;
classification must come from deep case-existence analysis, source inspection, skip-list evidence,
and targeted local runs when needed.

## Required Inputs

- Workbook row fields: `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`,
  `classname_xpu`, `name_xpu`, `status_xpu`, `Reason`, `DetailReason`,
  `Reason TBD`.
- Local PyTorch checkout: use the user-provided source path via `PYTORCH_SRC`; default to
  `$HOME/upstream/pytorch`. Do not hard-code private checkout paths in reusable logic.
- XPU test checkout: `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu`.
- Conda environment: `pytorch_opencode_env`. Optionally aligned to nightly torch, triton-xpu,
  and matching source via the preparation skill (`classify_ut/preparation/SKILL.md`,
  Environment Setup) — not required by default.
- Deep case-existence workflow:
  `${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`.

## Required Tools

- `read` - inspect base tests, XPU wrapper/direct files, distributed runner files, skip lists,
  parametrization, decorators, and helper source.
- `bash` - activate `pytorch_opencode_env`, run targeted tests or `pytest --collect-only`, inspect
  git refs, and fetch remote release files with `git show` or `gh`.
- `grep` / `rg` - locate exact class/function/decorator names after semantic analysis identifies the
  files that must be read. Do not classify from grep output alone.
- `gh` CLI - inspect GitHub files/issues when local source or `message_xpu` references them.
- Workbook tooling such as `openpyxl` may write results, but classification must come from source
  and execution evidence.

## Hard Constraints

- Do not classify by filename patterns, keyword matches, or a bulk script alone.
- Blank `status_xpu` is not proof that the case is missing. It means the row needs deep analysis.
- Always inspect the actual source that determines whether the XPU case exists, is generated, is
  skipped, or needs enablement.
- XPU test cases may live directly in `pytorch/test/`, especially Dynamo/Inductor tests. Do not
  assume every non-distributed XPU case requires a `third_party/torch-xpu-ops/test/xpu/*_xpu.py`
  wrapper.
- Do not stop at test-file existence. Check exact test-case existence after class/device
  instantiation, parametrization, OpInfo/device/dtype filtering, and other generated-test arguments.
- Do not use `release/2.12` for non-distributed rows.
- For distributed rows only, use release/2.12 and the remote distributed skip-list evidence described
  below.
- CUDA graph / cudagraph rows are not `Not applicable` merely because the CUDA name contains `cuda`.
  XPU graph support exists via `_XPUGraph`, `torch.xpu.XPUGraph`, and `torch.accelerator.Graph`;
  missing/failing coverage is `To be enabled` with an XPU graph DetailReason.
- A row may only be classified `Not applicable` on the CUDA-only branch when the underlying
  API/torch op appears in the `Operation/API` column of the `Not applicable` sheet of
  `torch_xpu_ops_issues.xlsx` (fetched via `classify_ut/scripts/list_not_applicable.py`).
  Read the **CUDA-Only Judgement Rule** section in `classify_ut/RULES.md` (the canonical
  method) before assigning `Not applicable` on this branch. `DetailReason` must cite the
  matching `Issue ID` and `Category`. Without an `Operation/API` match, do NOT use the
  CUDA-only branch — re-route to `To be enabled`, `Failures (xpu broken)`, `Feature gap`, or
  `Community Change`.
- `Not applicable` for CUDA-specific APIs must name the exact API in `DetailReason`, such as
  `CUDA-specific API: torch.cuda.jiterator (Issue #NNNN)`.
- `Community Change` is used when the base function/case is removed, renamed, refactored, moved, or
  disabled by an upstream community issue/commit in the source being compared. If an XPU variant
  exists after parametrization, do not call it community changes.
- Do not change `Reason TBD` after classification. Mark updated `Reason`, `DetailReason`, and
  `DetailReason` cells blue.
- For every row with `Reason TBD = True`, `DetailReason` MUST start with a confidence prefix
  `[Confidence: HIGH|MEDIUM|LOW]` per the **Confidence Rubric & Need-Human-Check Rule** in the
  parent skill. When the rubric resolves to LOW, set `Reason = "Need human check"` and explain
  in `DetailReason` which axes were checked and why each was inconclusive. Never use
  `Need human check` without first performing the full workflow.

## User-Issued Policy Overrides (mirror of parent skill — AUTHORITATIVE)

Apply these on every blank-status row. P2 and P3 below are AUTHORITATIVE and live here (they
were moved out of the parent skill). The parent rules file `classify_ut/RULES.md` carries the
full text for P1 and the **Sibling-Class Verdict Mapping**.

- **P1 Local-Retest Mandate:** Local retest is REQUIRED whenever the row has no linked issue,
  no linked PR, and no `not_target`/`wontfix` evidence — even when the source line alone would
  suggest a verdict.
- **P2 — `Not applicable` Must Carry Evidence, But Must NOT Be Run (AUTHORITATIVE):**
  - Do not execute `Not applicable` rows that already carry valid evidence (linked
    `not_target`/`wontfix` issue, owner-team scope, or a clearly CUDA-only API such as
    `torch.cuda.default_generators[0]`).
  - EVERY `Not applicable` classification MUST cite explicit evidence in `DetailReason`. A
    classification with no evidence is invalid — re-route to `To be enabled` or run locally.
- **P3 — JIT Cluster Policy (AUTHORITATIVE):** JIT (`oncall:jit`, `test_jit*`, `torch.jit.*`)
  clusters are `Not applicable` by owner-team scope. Cite the owner-team label as evidence in
  `DetailReason`. Do NOT classify JIT clusters as `Local Passed`, even when the test passes on
  XPU.
- **Sibling-Class Verdict Mapping (`instantiate_device_type_tests`):** XPU is NOT in the
  default test-base set. The XPU sibling is generated ONLY when the call site passes
  `allow_xpu=True`. Therefore: (1) if the source file uses `instantiate_device_type_tests`
  WITHOUT `allow_xpu=True`, the verdict is `To be enabled` — cite the file:line of the call
  and state that `allow_xpu=True` must be added before any XPU verdict is reachable; (2) if
  `allow_xpu=True` IS present, run the XPU sibling locally and map outcomes: PASS ⇒
  `Local Passed`, FAIL/SIGSEGV ⇒ `Failures (xpu broken)`, `skipIfXpu` ⇒ Dynamic-Skip Rule.
  `Not applicable` is NEVER the default for sibling-pattern cases — it requires its own
  evidence (linked `not_target`/`wontfix` issue, owner-team scope with CUDA-only dispatch, or
  CUDA-specific API in the test body). Workbook classname may differ from source classname
  (e.g. `InputAttrTrackingTests` ↔ `TestInputAttrTracking`); alias before lookup. See the
  parent skill **Sibling-Class Verdict Mapping** section for the full decision flow.
- **Workbook Precedent Rule (parent skill):** A non-Inductor TBD row may adopt a peer cluster's
  verdict when at least 5 same-`testfile_cuda` peers are ≥95% unanimous on one canonical Reason.
  This is a MEDIUM-confidence override that still requires at least one independent source-evidence
  axis in `DetailReason`. See the **Workbook Precedent Rule** section in `classify_ut/RULES.md`.

## Optional Preparation Inputs (Environment Setup + Local Pre-Screen)

The parent-skill preparation steps (`classify_ut/preparation/SKILL.md`) are **OPTIONAL and
NOT run by default**. This subskill works whether or not they were run:

- **If Environment Setup was run**: a provenance record at
  `~/.claude_classify_ut_session_provenance.json` records the aligned
  `source_alignment.pytorch_sha` / `source_alignment.torch_xpu_ops_sha` and `xpu_available`.
  Prefer those aligned checkouts for source analysis and cite the provenance in
  `DetailReason` for any `Local Passed` verdict.
- **If Local Pre-Screen was run**: a `local_result` column exists adjacent to `DetailReason`.
  - Rows whose `local_result` starts with `PASS;` are **TERMINALLY classified** as
    `Reason = "Local Passed"`. This subskill MUST NOT touch them — no source read, no
    sibling-class mapping, no issue search, no Oracle consult, no re-run, no
    re-classification. Skip and move on.
  - For rows whose `local_result` is `FAIL`, `ERROR`, `TIMEOUT`, `SKIP`, or `SEGFAULT`, the
    value and its log path are authoritative evidence for the eventual `DetailReason`.
- **If preparation was NOT run**: proceed directly with the deep case-existence analysis in
  this subskill (source inspection, known-issue search, targeted local runs as needed).

See the parent skill **Optional Preparation** section and
`classify_ut/preparation/SKILL.md` for the full helper-script invocation, log path
convention, and 60-second per-test timeout policy.

## Workflow

1. Confirm the row is eligible:
   - `Reason` is blank.
   - `status_xpu` is blank.
   - CUDA metadata identifies the exact test file, class, and method.
2. Check base-function existence and Community Change first:
   - Use `PYTORCH_SRC` (default `$HOME/upstream/pytorch`) as the source of truth.
   - Identify the **base function** in `testfile_cuda`: the function actually defined in the source
     that most closely generates `name_cuda` after decorators, device, dtype, OpInfo, and parameter
     suffixes are applied.
   - If that base function is absent, classify `Community Change`. This includes refactors where old
     generated CPU/CUDA-specific names are replaced by one device-parameterized base function, e.g.
     MinifierTests old `test_after_dynamo_cpu_*` / `test_after_dynamo_cuda_*` cases replaced by
     `test_after_dynamo_*(self, device)`.
   - If the base function exists, derive the expected XPU case by replacing `_cuda` with `_xpu` in
     `name_cuda`, then verify generation through parametrization, decorators, and source.
   - For Non-Inductor rows, check both direct PyTorch tests and
     `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu/**` including subfolders.
   - If the base function exists but no XPU case exists, decide why: missing XPU registration for
     supported functionality is `To be enabled`; exact CUDA-only API is `Not applicable`;
     implementation bug is `Failures (xpu broken)`; missing feature is `Feature gap`.
3. Check for Community Change regression:
   - If `last_status_xpu = passed` (test previously passed but is now blank/not run):
     a. Use `git log --oneline -20 -- <testfile_cuda>` to find recent commits.
     b. Use `git show <commit_hash> -- <testfile_cuda>` to inspect diffs.
     c. Look for: test renamed/removed, parametrization changed, class restructured,
        `instantiate_device_type_tests` altered, file moved.
     d. If guilty commit found: Reason = `Community Change`,
        DetailReason = `Community commit <hash> (<author>, <date>) - <summary>`.
      e. If no relevant commit, continue to step 4.
4. Derive missing XPU metadata only as a starting point:
   - `classname_cuda` ending in `CUDA` -> `XPU`.
   - `name_cuda` ending in `_cuda` -> `_xpu`.
   - `testfile_xpu` defaults to `testfile_cuda` when blank.
   Then verify against actual XPU source; do not trust the derived names blindly.
5. Determine whether the row is distributed:
   - If `testfile_cuda` is under `test/distributed/`, follow the distributed workflow below.
   - Otherwise follow the non-distributed workflow below.
6. Write one of the canonical outcomes:
   - `Reason = Community Change` when the base function/case is absent, renamed/refactored/moved,
      or a guilty upstream commit/issue is identified.
   - `Reason = To be enabled` for missing XPU registration, existing-but-unreported XPU cases,
     explicit XPU skips needing enablement, or missing XPU coverage for supported functionality.
   - `Reason = Not applicable` for CUDA-only APIs or backend-specific features that cannot apply to
     XPU. `DetailReason` must name the exact API/feature.
   - `Reason = Community Change` when source comparison proves the CUDA/base test was
      removed/renamed/refactored or no longer exists.

## Distributed Blank-Status Workflow

Distributed XPU tests usually do not use `*_xpu.py` wrappers. They run upstream files through
`third_party/torch-xpu-ops/test/xpu/run_distributed.py` and distributed skip dictionaries.

1. Read `third_party/torch-xpu-ops/test/xpu/run_distributed.py` to confirm active imports.
2. For release/2.12 distributed classification, first read the remote local distributed skip list:
   `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`, file
   `test/xpu/skip_list_dist_local.py`. The intended name may be described as
   `skip_list_local_dist.py`, but the verified branch filename is `skip_list_dist_local.py`.
3. Read the matching distributed skip list:
   `intel/torch-xpu-ops` branch `daisyden/distributed_2.12:test/xpu/skip_list_dist.py` when
   classifying against release/2.12, or local `third_party/torch-xpu-ops/test/xpu/skip_list_dist.py`
   for current-checkout analysis.
4. If a local override such as `skip_list_dict_local.py` exists, read it in full.
5. Normalize skip-list keys semantically:
   - `../../../../test/distributed/...` points to the upstream PyTorch distributed test.
   - `distributed/test_c10d_xccl.py` points to an XPU-native standalone distributed test.
6. Interpret entries:
   - File present with value `None` -> the whole file is enabled for XPU.
   - File present with tuple/list -> the file is enabled, but listed cases are intentionally skipped;
     all other cases run.
   - File absent -> the upstream file is not run by `run_distributed.py`; classify `To be enabled`
     with a specific DetailReason such as `Distributed file missing from remote distributed skip list`.
7. Also inspect `third_party/torch-xpu-ops/test/xpu/distributed/` for XPU-native files such as
   `test_c10d_xccl.py` and `test_c10d_ops_xccl.py`.
8. For distributed rows only, check `https://github.com/daisyden/pytorch/tree/release/2.12` or local
   `origin/release/2.12` for the upstream file and method before deciding a test was removed,
   renamed, or newly added.

## Non-Distributed Blank-Status Workflow

1. Read the local base test under `test/` and confirm the base function still exists in
   `PYTORCH_SRC`. If not, classify `Community Change` before checking XPU wrappers.
2. If `testfile_xpu` is the same as `testfile_cuda`, or the row points to a direct PyTorch test
   folder such as `test/dynamo/` or `test/inductor/`, analyze that direct PyTorch test file first:
   - Read the class and method body.
   - Check whether the class is device-generic or instantiated with `instantiate_device_type_tests`.
   - Check whether the exact case name is generated by parametrization, OpInfo, dtypes, devices,
     decorators, or other arguments.
   - Run a narrow collection command when source inspection is not enough, for example:
     ```bash
     source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}" && \
     python -m pytest --collect-only -q dynamo/test_modules.py -k test_assign_does_not_exist
     ```
    - If collection reports zero tests and source inspection shows the method/case is absent, classify
      as `Community Change`, not `XPU test file missing`.
   - If the case is collected or the direct source generates it for XPU, classify from the direct
     PyTorch source evidence. Do not require a `torch-xpu-ops/test/xpu` wrapper.
3. Enumerate all plausible XPU locations before declaring a wrapper/direct XPU file missing:
   - `third_party/torch-xpu-ops/test/xpu/`
   - `third_party/torch-xpu-ops/test/xpu/extended/`
   - `third_party/torch-xpu-ops/test/xpu/nn/`
   - `third_party/torch-xpu-ops/test/xpu/functorch/`
   - `third_party/torch-xpu-ops/test/xpu/quantization/`
   - other relevant subfolders discovered from imports or local naming.
4. Read wrapper/direct XPU files. Not all XPU tests use `XPUPatchForImport`; many are standalone
   copies or direct implementations.
5. If a wrapper uses `XPUPatchForImport`, understand its mode:
   - `XPUPatchForImport(False)` usually imports/instantiates upstream tests with XPU adaptations.
   - `XPUPatchForImport(True)` can disable or alter instantiation; inspect the surrounding code.
6. Inspect class definitions, imports, `instantiate_device_type_tests`,
   `instantiate_parametrized_tests`, OpInfo filters, decorators, and xfail/skip lists to determine
   whether the exact XPU case is generated.
7. Use targeted collection or local execution when source inspection is inconclusive. Run only the
   exact test or a narrow collect command in `pytorch_opencode_env`.
8. If no XPU source generates the case but the base test exists and the feature applies to XPU,
   classify `To be enabled` with a DetailReason naming the missing file/class/import/instantiation.
9. If local source proves the CUDA/base test no longer exists or has been renamed/refactored,
    classify `Community Change` with the exact source evidence.
10. If source proves the test is CUDA-only, classify `Not applicable` and name the exact API.

### Rows with No XPU Test Data

When a row has CUDA data but no XPU data (blank `status_xpu`, `name_xpu`, `device_xpu`), do NOT
use generic reasons like "No XPU test data" or "CUDA-only test". Instead:

1. **Read the test source** to understand what API/feature it tests.
2. **Check if the test uses device-agnostic patterns** (`GPU_TYPE`, `instantiate_device_type_tests`,
   parametrized devices). If so, the test SHOULD run on XPU -> `To be enabled`.
3. **Check if the test uses CUDA-specific APIs** (`torch.cuda.jiterator`, cuBLAS, TensorExpr CUDA
   fuser, CUDA-only decorators like `@onlyCUDA`). If so -> `Not applicable` with the exact API named.
4. **Check if the test file is in the XPU CI test list** (torch-xpu-ops test runner). If not,
   explain that the file is not included in the XPU CI runner.
5. `DetailReason` MUST always name the specific API or feature tested and WHY XPU does or does not
   support it. Never write just "CUDA-only test" or "No XPU test data".

## Known Blank-Status Classifications From This Workflow

These are examples, not substitutes for analysis. Re-check source before applying.

- Remote distributed file enabled:
  - Reason: `To be enabled`.
  - DetailReason: `Distributed file enabled in remote distributed skip list: <file>`.
  - DetailReason should name the remote skip-list files read and say that the file is registered for
    XPU through `run_distributed.py`.
- Remote distributed file missing:
  - Reason: `To be enabled`.
  - DetailReason: `Distributed file missing from remote distributed skip list: <file>`.
  - DetailReason should name checked dictionaries, release/2.12 file presence, and enabled sibling
    files when useful.
- CUDA graph / cudagraph coverage:
  - Reason: `To be enabled`.
  - DetailReason: `XPU graph coverage missing` or a similarly specific XPU graph gap.
  - DetailReason should mention that XPU graph APIs exist and identify the missing XPU test coverage.
- Jiterator blank-status rows:
  - Reason: `Not applicable`.
  - DetailReason: `CUDA-specific API: torch.cuda.jiterator`.
  - DetailReason should mention the concrete `torch.cuda.jiterator` APIs used.
- cuBLAS deterministic blank-status rows:
  - Reason: `Not applicable`.
  - DetailReason: `CUDA-specific API: cuBLAS`.
  - DetailReason should mention the cuBLAS determinism behavior and any `@onlyCUDA` evidence.
- TensorExpr CUDA fuser rows:
  - Reason: `Not applicable`.
  - DetailReason: `CUDA-specific API: TensorExpr CUDA fuser`.
- Existing XPU wrapper/direct file with generated XPU test but no XPU workbook result:
  - Reason: `To be enabled`.
  - DetailReason: `Test exists but blank: <XPU class or file>`.
  - DetailReason should name the exact XPU source file/class/function and expected XPU test name.
- Direct PyTorch test file with case-level collection evidence:
  - If the exact case is collected or generated from `pytorch/test`, classify using that direct
    source evidence. Do not require a `third_party/torch-xpu-ops/test/xpu` wrapper.
  - If the file exists but the exact method/case is absent and targeted collection runs zero tests,
    classify `Community Change` with the source and collection evidence.
  - Example: `test/dynamo/test_modules.py` exists, but `OptimizedModuleTest.test_assign_does_not_exist`
    was absent from local source, absent from `origin/release/2.12`, and `pytest --collect-only -k
    assign_does_not_exist` collected zero tests; this is not `XPU test file missing`.
- Local base test missing or method removed/refactored for a non-distributed row:
  - Reason: `Community Change`.
  - DetailReason should name the source evidence, e.g. `Base function not found in upstream
    <testfile>; function removed, renamed, or refactored`.

## Output Rules

- `Reason`: use canonical workbook labels: `To be enabled`, `Not applicable`, `Not applicable`,
  or `Community Change`.
- `DetailReason`: be specific enough to act on. Always use full issue/PR URLs
  (e.g., `https://github.com/pytorch/pytorch/issues/NNNNN`), never bare numbers like `#NNNNN`.
  Include exact test identity, source files read, skip-list or wrapper evidence, and reasoning.
  Avoid generic `No XPU wrapper` and generic `CUDA-specific API`.

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for processed blank-status rows.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least one distributed enabled row, one distributed missing row, one CUDA-specific API
  row, one XPU-wrapper-existing row, and one community-change row when those categories are present.
