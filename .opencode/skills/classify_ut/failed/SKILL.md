# classify_ut failed cases

This skill follows agent-guidelines AND extends it with failed-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows in any XPU UT status workbook whose `Reason` is blank and whose
`status_xpu` is `failed`. This skill is a sub-workflow of `classify_ut`; use it after the
base workbook preparation has initialized `Reason TBD`, filled missing XPU metadata, and
preserved the original workbook.

This subskill is also invoked for the **TBE re-verification pass**: rows where
`TBE_Reverify = True` in the `.agent.xlsx` (originally classified as `To be enabled`,
opt-in recheck) get routed here by `status_xpu`. The deep-analysis workflow is the same;
the only difference is the input row set and the `DetailReason` marker convention. See
**TBE Re-verification Context** below and the **TBE Re-verification Rule** in
`classify_ut/RULES.md`.

## TBE Re-verification Context

When the parent workflow runs the re-verification pass, it routes a row with
`TBE_Reverify = True` and `status_xpu = failed` to THIS subskill. The invocation is
identical to the failed-Reason case with one exception: the row is being re-checked, not
classified from scratch.

- The existing `DetailReason` was written by a prior pass that landed on
  `To be enabled`. The agent MUST re-read the cited source state (typically a closed
  `intel/torch-xpu-ops` or `pytorch/pytorch` issue, a stale skip, a missing wrapper, or
  a missing `allow_xpu=True`) and check whether it has changed since the prior verdict.
- The verdict may stay `To be enabled`, change to another canonical label (e.g. the
  wrapper was added and the case now runs and crashes -> `Failures (xpu broken)`), or
  be flagged `Need human check` (LOW) when the cited signal is no longer clear. Each
  outcome is written to `Reason` and `DetailReason` per the **TBE Re-verification Rule**.
- `Reason TBD` is **never** flipped to `True` for a re-verified row. A re-verified row
  stays `Reason TBD = False`.
- The updated `DetailReason` MUST start with `[Reverified: YYYY-MM-DD]` so a human
  reviewer can distinguish a row that was re-checked in this session from one that was
  left as-is from the original workbook. If the row also flips to `Need human check`,
  the LOW confidence prefix is required by the existing rubric:
  `[Reverified: YYYY-MM-DD] [Confidence: LOW] Need human check. ...`
- The `Confidence` workbook column is NOT populated for re-verified rows (it is reserved
  for `Reason TBD = True` rows; see the **Confidence Rubric & Need-Human-Check Rule** in
  `RULES.md`).

## Required Inputs

- Workbook row fields: `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`,
  `classname_xpu`, `name_xpu`, `status_xpu`, `message_xpu`, `Reason`, `DetailReason`,
  `Reason TBD`.
- Local PyTorch checkout: use the user-provided source path via `PYTORCH_SRC`; default to
  `$HOME/upstream/pytorch`. Do not hard-code private checkout paths in reusable logic.
- XPU test checkout: `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu`.
- Conda environment: `pytorch_opencode_env`. Optionally aligned to nightly torch, triton-xpu,
  and matching source via the preparation skill (`classify_ut/preparation/SKILL.md`,
  Environment Setup) — not required by default.
- GitHub issue source of truth: search both `intel/torch-xpu-ops` and `pytorch/pytorch` for known
  issues. Prefer `intel/torch-xpu-ops` for XPU implementation bugs, but do not skip PyTorch issues
  when source, message, disabled-test infrastructure, or upstream behavior is relevant.

## Required Tools

- `read` - inspect base tests, XPU test files, wrappers, skip decorators, and supporting source.
- `bash` - activate `pytorch_opencode_env`, run targeted tests, query `gh issue view`, and inspect
  git/remote files.
- `grep` / `rg` - locate exact methods, decorators, issue links, and error strings after deciding
  what must be inspected. Do not use grep hits alone as classification evidence.
- `gh` CLI - fetch issue state and body, especially for `intel/torch-xpu-ops` known issues.
- Workbook tooling such as `openpyxl` may prepare or write cells, but it must not replace
  semantic analysis.

## Hard Constraints

- Do not classify by filename, keyword, or message pattern alone.
- Do not assume every failure is an XPU bug without understanding the error and test surface.
- Always read enough code to understand what the test is validating.
- Always inspect `message_xpu`; if it is missing, empty, truncated beyond usefulness, or only says
  that a process exited, run the exact test locally to get a useful error whenever feasible.
- Local runs use `pytorch_opencode_env`. For source-aligned results, optionally run the
  preparation skill first (`classify_ut/preparation/SKILL.md`, Environment Setup) — not
  required by default.
- Avoid importing the unbuilt source-tree `torch` accidentally. When checking the installed torch
  package directly, run from outside the PyTorch checkout, for example `/tmp/opencode`. When running
  tests, use the correct test root described below.
- If a matching known issue exists, `DetailReason` must include the issue URL.
- If no known issue exists after searching both `intel/torch-xpu-ops` and `pytorch/pytorch`,
  `DetailReason` must start with `[Issue_TBD]` and include the concrete error summary.
- Do not change `Reason TBD` after classification. It records whether the original `Reason` was
  blank before analysis.
- Mark updated `Reason` and `DetailReason` cells blue; leave unrelated cells alone.
- For every row with `Reason TBD = True`, `DetailReason` MUST start with a confidence prefix
  `[Confidence: HIGH|MEDIUM|LOW]` per the **Confidence Rubric & Need-Human-Check Rule** in the
  parent skill. The `[Issue_TBD]` prefix and the `[Confidence: ...]` prefix coexist when both
  apply, e.g. `[Confidence: MEDIUM][Issue_TBD] XPU fails with RuntimeError ...`. When the rubric
  resolves to LOW, set `Reason = "Need human check"` and document which axes were checked.
- **Workbook Precedent Rule (parent skill):** A non-Inductor TBD row may adopt a peer cluster's
  verdict when at least 5 same-`testfile_cuda` peers are ≥95% unanimous on one canonical Reason.
  This is a MEDIUM-confidence override that still requires at least one independent source-evidence
  axis in `DetailReason`. See the **Workbook Precedent Rule** section in `classify_ut/RULES.md`.

## Local Run Rules

Run only targeted tests. Never run a whole suite.

- Inductor or Dynamo tests whose source is in `pytorch/test/` should be run from the PyTorch test
  folder:
  ```bash
  source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}" && \
  python dynamo/test_dicts.py DictSubclassMethodsTests.test_binop_or
  ```
- XPU wrapper/direct tests should be run from `third_party/torch-xpu-ops/test/xpu`:
  ```bash
  source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}" && \
  python test_ops_xpu.py TestCommonXPU.test_dtypes_addmm_xpu
  ```
- Distributed upstream tests generally run from `pytorch/test/` with the exact upstream file/class
  and method when feasible. If the XPU workflow requires `run_distributed.py`, first read the active
  distributed skip-list workflow from the main `classify_ut` skill.
- If the local run itself is skipped, read the decorator/helper source that caused the skip before
  deciding whether it is a feature gap, stale skip, failure, or environment limitation.

## Workflow

1. Confirm the row is eligible:
   - `Reason` is blank.
   - `status_xpu` is exactly `failed`.
   - CUDA and XPU metadata identify the exact test case.
2. Check for Community Change regression:
   - If `last_status_xpu = passed` (test previously passed but now fails):
     a. Use `git log --oneline -20 -- <testfile_cuda>` to find recent commits.
     b. Use `git show <commit_hash> -- <testfile_cuda>` to inspect diffs.
     c. Look for: test logic changed, expected values updated, new assertions added,
        API signature changed.
     d. If a community commit changed the test in a way that breaks XPU but not CUDA:
        Reason = `Community Change`,
        DetailReason = `Community commit <hash> (<author>, <date>) - <summary>`.
     e. If no relevant commit, continue to step 3.
3. Normalize `message_xpu` for readability only. Do not classify solely from text matching.
   Preserve enough of the original failure in `DetailReason`.
3. Inspect the relevant local source:
   - Base test under `test/`.
   - XPU wrapper/direct file under `third_party/torch-xpu-ops/test/xpu/**` if present.
   - Distributed skip dictionaries and remote release branch only for distributed tests, following
     the parent `classify_ut` rules.
4. Search both `intel/torch-xpu-ops` and `pytorch/pytorch` issues semantically for the concrete
   failing behavior. Use the method name, operator name, exception type, and meaningful error phrase,
   not just the file name.
5. If `message_xpu` is missing or too generic, run the exact test locally and use that result as
   evidence. If the local run passes, classify as `To be enabled`; if it fails, use the local error
   and continue issue search.
6. Decide the classification:
    - Known XPU implementation/runtime failure -> `Reason = Failures (xpu broken)`.
    - Known missing feature/API exposed by the failed run -> `Reason = Feature gap` only when the
      issue/source describes unsupported functionality rather than a broken implementation.
    - Local run passes and existing failure appears stale -> `Reason = Local Passed` (with evidence
      saved to local verify dir).
  - No known issue after search -> `Reason = Failures (xpu broken)` and `DetailReason` starts with
    `[Issue_TBD]`.
7. Write concise evidence:
   - `DetailReason` includes issue link or `[Issue_TBD]` plus the error summary, exact test
     identity, that `status_xpu` was `failed`, `message_xpu` or local-run output summary,
     and source/issue inspected.

## Known Failed-Case Classifications From This Workflow

These are examples, not a substitute for analysis. Re-check source and issue state before reusing.

- Jiterator failures:
  - Evidence: `message_xpu` says `Jiterator is only supported on CUDA and ROCm GPUs`.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/2918`.
  - Reason: `Failures (xpu broken)` for failed-status rows in this workbook; the detail names that
    jiterator is not supported on XPU.
- OpInfo dtype mismatch failures:
  - Evidence: `message_xpu` says supported dtypes for an op on XPU are incorrect or dtypes worked
    but are not listed by OpInfo.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/3574` when the issue content matches
    the op family; otherwise search more and use `[Issue TBD]` if no issue exists.
  - Reason: `Failures (xpu broken)` unless source/issue says the dtype is an intentionally missing
    feature.
- XPU compiler / `ocloc` failures:
  - Evidence: `message_xpu` contains `ocloc`, IGC initialization failure, or a Triton-to-ZEBIN
    compiler failure.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/3386` when the failure matches.
  - Reason: `Failures (xpu broken)`.
- cuBLAS/matmul deterministic failures:
  - Evidence: test name contains `test_cublas_deterministic` and issue/source confirms matmul
    deterministic behavior is tracked for XPU.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/2481` when the case matches.
  - Reason: `Failures (xpu broken)` for failed rows. For skipped rows whose underlying API
    appears in the `Operation/API` column of the `Not applicable` sheet of
    `torch_xpu_ops_issues.xlsx` (read the **CUDA-Only Judgement Rule** section in
    `classify_ut/RULES.md`), use `Not applicable` instead; otherwise use `Feature gap`
    and let the skipped-case skill handle it.

## Output Rules

- `Reason`: choose one of the workbook's canonical labels, especially `Failures (xpu broken)`,
  `Feature gap`, or `To be enabled`.
- **P2 — `Not applicable` Must Carry Evidence, But Must NOT Be Run (AUTHORITATIVE):** on the
  rare failed-status path that resolves to `Not applicable` (e.g. the cuBLAS/matmul case
  above), do not execute the row when it already carries valid evidence (linked
  `not_target`/`wontfix` issue, owner-team scope, or a clearly CUDA-only API). EVERY
  `Not applicable` classification MUST cite explicit evidence in `DetailReason`; with no
  evidence the classification is invalid — re-route to `To be enabled`/`Feature gap` or run
  locally.
- **P3 — JIT Cluster Policy (AUTHORITATIVE):** JIT (`oncall:jit`, `test_jit*`, `torch.jit.*`)
  clusters are `Not applicable` by owner-team scope. Cite the owner-team label as evidence in
  `DetailReason`. Do NOT classify JIT clusters as `Local Passed`, even when the test passes on
  XPU.
- `DetailReason`: include the full issue/PR URL when known (e.g.,
  `https://github.com/intel/torch-xpu-ops/issues/NNNN`), never bare numbers like `#NNNN`.
  Otherwise start with `[Issue_TBD]`. Extract URLs from `message_xpu` when present.
  Include exact test identity, error source, local-run result if used, and reasoning.

## Local Passed for Failed-Status Rows

When `status_xpu = failed` but the test PASSES locally in `pytorch_opencode_env`:
- Reason: `Local Passed`
- DetailReason: `Local verification passed in pytorch_opencode_env; stale failed status`
- Save evidence to `/tmp/opencode/<workbook>_local_verify/` per parent skill requirements
- This indicates the CI failure is flaky or already fixed in the current checkout

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for the processed set.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least one known-issue row and one `[Issue_TBD]` row against source/issue evidence.

## Working File

This subskill operates on the working file produced by the parent's extract +
optional filter steps. If a row filter was applied (e.g. the user said
"classify rows where `status_xpu=failed AND DetailReason=Daisy`"), the
working file is a subset with a `_source_row` column. Edit only the subset's
`Reason` and `DetailReason`; write the verdicts back to the extracted file
(or to `.agent.xlsx`) via `classify_ut/scripts/apply_filtered_changes.py`.
See parent `classify_ut/SKILL.md` "Row-Level Filter" section.
