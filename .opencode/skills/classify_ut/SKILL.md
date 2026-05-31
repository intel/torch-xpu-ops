# classify_ut

This skill follows agent-guidelines AND extends it with workbook-specific UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify blank `Reason` rows in XPU UT status workbooks by performing deep source analysis,
local verification, and known-issue searches. This skill works for ANY test-case workbook
(Non-Inductor, Inductor, or other sheets) as long as the required columns are present.

## Applicable Workbooks and Sheets

This skill applies to any workbook/sheet with these columns:
- Test identification: `testfile_cuda`, `classname_cuda`, `name_cuda`
- XPU status: `status_xpu`, `message_xpu`
- Classification: `Reason`, `DetailReason`
- Tracking: `Reason TBD`

Known sheets:
- `Non-Inductor XPU Skip` in `Non_inductor_ut_status_ww*.xlsx`
- `Cuda pass xpu skip` in `Inductor_ut_status_ww*.xlsx`
- Any similarly structured sheet from weekly UT status reports

## Inputs

- Target workbook: the `.xlsx` file provided by the user
- Source checkout for existence checks: use the user-provided path via `PYTORCH_SRC`; if none is
  provided, use `$HOME/upstream/pytorch`. Do not hard-code private checkout paths in commands or
  reusable logic.
- XPU test checkout: `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu`, where `PYTORCH_SRC` is the
  source checkout above.
- Reference workbook (optional):
  `${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/result/torch_xpu_ops_issues.xlsx`
- Deep case-existence workflow:
  `${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`
- Blank `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/blank/SKILL.md`
- Failed `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/failed/SKILL.md`
- Skipped/xfail `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/skipped/SKILL.md`

Recommended environment variables:
- `ISSUE_TRIAGE_ROOT=${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}`
- `CLASSIFY_UT_ROOT=${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}`
- `PYTORCH_SRC=${PYTORCH_SRC:-$HOME/upstream/pytorch}`
- `PYTORCH_ENV=${PYTORCH_ENV:-pytorch_opencode_env}`
- `CONDA_ACTIVATE=${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}`

## Status-specific classification skills

These subskills are authoritative for case-specific `Reason`, `DetailReason`, and
`DetailReason` decisions. Always read the matching subskill before classifying rows with that
`status_xpu` value.

| `status_xpu` | Skill | Purpose |
|--------------|-------|---------|
| blank / empty | `classify_ut/blank/SKILL.md` | Deep case-existence analysis for rows with no XPU result |
| `failed` | `classify_ut/failed/SKILL.md` | Failure-message, local-run, and known-issue analysis |
| `skipped` / `xfail` | `classify_ut/skipped/SKILL.md` | Skip-message, linked-issue, source, and local-run analysis |

Do not collapse these workflows into a single pattern-matching script. Bulk scripts may prepare
workbook columns, collect candidate rows, or apply already-reviewed decisions, but the actual
classification must follow the status-specific skill.

## Optional Preparation (environment setup + local pre-screen)

Two preparation aids live in `classify_ut/preparation/SKILL.md`:

- **Environment Setup** (formerly "Step -1"): align `pytorch_opencode_env` to the PyTorch
  XPU nightly and align the local pytorch + torch-xpu-ops checkouts to the wheel's commits.
- **Local Pre-Screen** (formerly "Step 0"): bulk-run every blank-`Reason` row and record a
  `local_result` plus captured logs.

These steps are **OPTIONAL and NOT run by default.** Run them only when the user explicitly
asks to refresh the environment or bulk pre-screen a sheet, or when a verdict genuinely
depends on a fresh, source-aligned local run. Classification (source inspection, known-issue
search, the status-specific subskills) can proceed without them.

When preparation IS run, its artifacts become authoritative inputs:

- The provenance JSON (`~/.claude_classify_ut_session_provenance.json`) may be cited in
  `DetailReason` for `Local Passed` verdicts.
- A `local_result` starting with `PASS;` is a **TERMINAL** verdict
  (`Reason = "Local Passed"`) — skip all further analysis for that row (except the JIT-cluster
  policy defined in the status-specific subskills).
- A non-`PASS` `local_result` is **evidence**, not a verdict, and should be cited by the
  status-specific subskill.

See `classify_ut/preparation/SKILL.md` for the full helper-script invocation, the manual
alternative, the provenance contract, the PASS-is-terminal rule, and all anti-patterns.

## Classification Rules

All detailed classification rules live in [`classify_ut/RULES.md`](./RULES.md). Read the relevant
rule there before assigning a `Reason`. The status-specific subskills (`blank/`, `failed/`,
`skipped/`) are authoritative for per-status decisions and reference these rules by name.

| Rule (in `classify_ut/RULES.md`) | Applies to |
|----------------------------------|-----------|
| **User-Issued Policy Overrides** (P1 Local-Retest Mandate) | Every row |
| **Sibling-Class Verdict Mapping** (`instantiate_device_type_tests`) | Rows referencing a CUDA-suffix class |
| **Column Definitions** (canonical `Reason` labels, `DetailReason`, `Reason TBD`) | Every row |
| **CUDA-Only Judgement Rule** | `Not applicable` candidates |
| **Workbook Precedent Rule** | Non-Inductor `Not applicable` clusters |
| **Dynamic-Skip Rule** | `skipped`-labeled issues |
| **Local Verification via XPU Port** | `To be enabled` / `skipped`-label local runs |
| **Confidence Rubric & Need-Human-Check Rule** | Every `Reason TBD = True` row |
| **Deep Analysis Requirements** (Case Existence, Community Change detection, disabled-test, Failures issue-link) | Every blank-Reason row |
| **Local Verification Evidence** (MANDATORY for `Local Passed`) | `Local Passed` rows |

## Tools Required

| Tool | Purpose |
|------|---------|
| `read` | Inspect test source, wrappers, skip lists, decorators |
| `bash` | Run tests locally, activate env, directory listing |
| `grep` | Find decorators, skip entries, method definitions |
| `gh` CLI | Search issues (`gh search issues`), view issue state (`gh issue view`) |
| `openpyxl` | Read/write workbook cells |
| `pytest --collect-only` | Check if test exists without running it |
| `python <test_file> -k <pattern> -v` | Run specific tests with PyTorch test runner |

## Environment Setup (OPTIONAL — see preparation skill)

Environment alignment (nightly wheel + source-tree checkout) is an **optional** preparation
aid, NOT a mandatory step, and is NOT run by default. The full recipe — both the
`update_env_from_nightly.sh` helper and the manual alternative — lives in
`classify_ut/preparation/SKILL.md`. Run it only when the user explicitly asks to refresh the
environment, or when a verdict depends on a fresh, source-aligned local run.

## Workflow

1. (Optional) Run preparation only if requested or needed: environment setup and/or local
   pre-screen, per `classify_ut/preparation/SKILL.md`. Not run by default.
2. Open the target sheet in the workbook.
3. Ensure workbook column `Reason TBD` exists.
4. Initialize `Reason TBD` from the ORIGINAL workbook's `Reason` value (not the `.agent.xlsx`):
   - if `Reason` is blank in the original, set `Reason TBD = True`
   - otherwise set `Reason TBD = False`
   - Once set, NEVER modify this column again during classification or reclassification
5. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
6. For each blank-Reason row, choose the status-specific skill:
   - blank `status_xpu` -> `classify_ut/blank/SKILL.md`
   - `status_xpu = failed` -> `classify_ut/failed/SKILL.md`
   - `status_xpu = skipped` or `xfail` -> `classify_ut/skipped/SKILL.md`
7. Execute the selected skill's deep analysis workflow. While classifying, also apply the
   cross-cutting rules defined in `classify_ut/RULES.md` whenever they are in scope: the
   **Sibling-Class Verdict Mapping**, **CUDA-Only Judgement Rule**, **Dynamic-Skip Rule**, and the
   **Workbook Precedent Rule** (authoritative override for non-Inductor `Not applicable`).
 8. Fill `Reason` and `DetailReason`. Mark cells blue.
 9. Save local verification results to `/tmp/opencode/<workbook>_local_verify/`
 10. Save output workbook as `.agent.xlsx` (do not modify original).
 11. Verify: 0 blank Reason rows remaining, ZIP integrity OK, reason counts match.

## Notes

- Save output as `.agent.xlsx`; do not modify original workbook.
- Preserve existing `Reason` and `DetailReason` unless deep analysis justifies an update.
- Mark updated cells blue using `PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`.
