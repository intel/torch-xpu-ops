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
- Tracking: `Reason TBD`, `Confidence` (added by this skill; see RULES.md)
- Re-verification: `TBE_Reverify` (added by this skill; see **TBE Re-verification Rule** in
  `RULES.md`). Only required in sessions that opt rows into the `To be enabled` recheck pass.

Known sheets:
- `Non-Inductor XPU Skip` in `Non_inductor_ut_status_ww*.xlsx`
- `Cuda pass xpu skip` in `Inductor_ut_status_ww*.xlsx`
- Any similarly structured sheet from weekly UT status reports

## Inputs

- Target workbook: the `.xlsx` file provided by the user. The preparation **Extract Target
  Sheet** step (run first) copies the target sheet into a small standalone
  `<stem>.<sheet_slug>.xlsx`; that extracted file becomes the working target for all later
  phases, while the original large workbook is left untouched as the source of record.
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

## Preparation (extract sheet — mandatory; environment setup + local pre-screen — optional)

Three preparation steps live in `classify_ut/preparation/SKILL.md`:

- **Extract Target Sheet** — **MANDATORY**: copy the single target sheet out of the large
  status workbook into its own small `.xlsx` via `scripts/extract_target_sheet.py`. Every
  later phase and step operates on this extracted file, not the original large workbook.
- **Environment Setup** (formerly "Step -1"): align `pytorch_opencode_env` to the PyTorch
  XPU nightly and align the local pytorch + torch-xpu-ops checkouts to the wheel's commits.
- **Local Pre-Screen** (formerly "Step 0"): bulk-run every blank-`Reason` row and record a
  `local_result` plus captured logs.

**Extract Target Sheet is MANDATORY** and must always be run first before any other phase or
step. **Environment Setup and Local Pre-Screen are OPTIONAL and NOT run by default** — run
them only when the user explicitly asks to refresh the environment or bulk pre-screen a sheet,
or when a verdict genuinely depends on a fresh, source-aligned local run. Classification
(source inspection, known-issue search, the status-specific subskills) can proceed without
the optional steps.

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
| **Column Definitions** (canonical `Reason` labels, `DetailReason`, `Reason TBD`, `Confidence`, `TBE_Reverify`) | Every row |
| **CUDA-Only Judgement Rule** | `Not applicable` candidates |
| **Workbook Precedent Rule** | Non-Inductor `Not applicable` clusters |
| **Dynamic-Skip Rule** | `skipped`-labeled issues |
| **Local Verification via XPU Port** | `To be enabled` / `skipped`-label local runs |
| **TBE Re-verification Rule** | Rows with `TBE_Reverify = True` (re-checking existing `To be enabled` verdicts) |
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
| `scripts/extract_target_sheet.py` | Sheet-level extract (MANDATORY first step) |
| `scripts/filter_target_rows.py` | OPTIONAL row-level filter when the user names a subset |
| `scripts/apply_filtered_changes.py` | OPTIONAL write-back when the row filter was used |

## Environment Setup (OPTIONAL — see preparation skill)

Environment alignment (nightly wheel + source-tree checkout) is an **optional** preparation
aid, NOT a mandatory step, and is NOT run by default. The full recipe — both the
`update_env_from_nightly.sh` helper and the manual alternative — lives in
`classify_ut/preparation/SKILL.md`. Run it only when the user explicitly asks to refresh the
environment, or when a verdict depends on a fresh, source-aligned local run.

## Row-Level Filter (OPTIONAL — only when the user names a subset)

When the user asks to classify a specific subset of rows (e.g. "classify rows where
`Reason='To be enabled' AND DetailReason='Daisy'`"), filter the extracted sheet to that
subset before classification. This keeps the per-row working set small and the agent
context lean.

```bash
# Filter the extracted sheet to the rows the user cares about.
python3 scripts/filter_target_rows.py <stem>.<sheet_slug>.xlsx \
    --where "Reason=To be enabled" "DetailReason=Daisy" \
    --out <stem>.<sheet_slug>.subset.xlsx
```

The filter script:

- Accepts one or more `--where "Column=value"` (or `"Column!=value"`) tokens, AND-ed.
- An empty value (`Reason=`) matches blank cells — use this for the typical
  "classify blank-Reason rows" workflow.
- Writes the matching rows to a new small workbook with a `_source_row` column
  appended at the end (1-based row number in the extracted file, used for
  write-back).
- Prints `matched rows` and `columns` on success; exits 1 with no output file
  when 0 rows match.

Run all subsequent classification steps (steps 4-11 in the **Workflow** below) on
the filtered subset instead of the full extracted file. To write the agent's edits
back to the full extracted file (or to a new `.agent.xlsx`), use the apply script:

```bash
python3 scripts/apply_filtered_changes.py <stem>.<sheet_slug>.xlsx \
    <stem>.<sheet_slug>.subset.xlsx \
    --write-columns Reason DetailReason Confidence \
    --out <stem>.<sheet_slug>.agent.xlsx
```

The apply script:

- Reads `_source_row` from the filtered file to find each row in the target.
- Copies the values of the named `--write-columns` from filtered -> target.
- Marks each updated cell blue (`ADD8E6`, override with `--blue`).
- Saves the target in place by default; use `--out` to write a new file
  (e.g. `.agent.xlsx`) without touching the source.

If the user did NOT name a subset, skip both scripts and follow the default
**Workflow** below end-to-end against the full extracted sheet.

## Workflow

1. **Extract the target sheet** (MANDATORY — always run first): run
   `scripts/extract_target_sheet.py <original.xlsx> --sheet "<Target Sheet>"` to produce the
   small single-sheet `<stem>.<sheet_slug>.xlsx`. Use that extracted file as the working
   target for every step below; leave the original large workbook untouched. See
   `classify_ut/preparation/SKILL.md`.
   If the user named a subset, follow it with a `scripts/filter_target_rows.py`
   call (see **Row-Level Filter** above) and treat the filtered file as the
   working target for steps 3-11. Step 13 (save) becomes the apply script's
   `--out` argument.
2. (Optional) Run the remaining preparation only if requested or needed: environment setup
   and/or local pre-screen against the extracted file, per `classify_ut/preparation/SKILL.md`.
   Not run by default.
3. Open the target sheet in the extracted workbook.
4. Ensure workbook column `Reason TBD` exists.
5. Initialize `Reason TBD` from the ORIGINAL workbook's `Reason` value (not the `.agent.xlsx`):
   - if `Reason` is blank in the original, set `Reason TBD = True`
   - otherwise set `Reason TBD = False`
   - Once set, NEVER modify this column again during classification or reclassification
6. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
7. **Initialize the re-verification opt-in `TBE_Reverify` column** (only when the workbook
   owner wants a `To be enabled` recheck pass this session; see the **TBE Re-verification
   Rule** in `classify_ut/RULES.md`):
   - Add the column immediately after `Confidence` (or after the last existing column if
     `Confidence` is not present).
   - For every row in the ORIGINAL workbook where `Reason = "To be enabled"`, default to
     `TBE_Reverify = False`. The agent flips individual rows to `True` to opt them in.
   - For every other row (blank Reason or any other filled Reason), set
     `TBE_Reverify = False`.
   - Once set, NEVER clear `TBE_Reverify` back to `False` for a row that was flipped to
     `True` and processed. The flag is a permanent record that reverification happened.
8. For each blank-Reason row, choose the status-specific skill:
   - blank `status_xpu` -> `classify_ut/blank/SKILL.md`
   - `status_xpu = failed` -> `classify_ut/failed/SKILL.md`
   - `status_xpu = skipped` or `xfail` -> `classify_ut/skipped/SKILL.md`
9. Execute the selected skill's deep analysis workflow. While classifying, also apply the
   cross-cutting rules defined in `classify_ut/RULES.md` whenever they are in scope: the
   **Sibling-Class Verdict Mapping**, **CUDA-Only Judgement Rule**, **Dynamic-Skip Rule**, and the
   **Workbook Precedent Rule** (authoritative override for non-Inductor `Not applicable`).
10. Fill `Reason` and `DetailReason`. Mark cells blue.
11. **Re-verification pass for `TBE_Reverify = True` rows** (only when the column exists
    and at least one row is flagged; see the **TBE Re-verification Rule**):
    - For each `TBE_Reverify = True` row, route by `status_xpu` to the same
      status-specific subskill as in step 8.
    - Re-read the cited source state (file:line, issue URL, wrapper) and check for
      changes since the prior verdict (commits, issue state, decorators).
    - Let the subskill's deep-analysis workflow produce a fresh verdict. The verdict may
      confirm `To be enabled`, change the label, or flag `Need human check`.
    - Update `DetailReason` and prepend `[Reverified: YYYY-MM-DD]`. `Reason TBD` is NOT
      modified. Mark updated cells blue.
12. Save local verification results to `/tmp/opencode/<workbook>_local_verify/`
13. **Save the output workbook as `.agent.xlsx`** (do not modify the original
    large workbook). If a row filter was used in step 1, run
    `scripts/apply_filtered_changes.py <extracted.xlsx> <subset.xlsx>
    --write-columns Reason DetailReason Confidence --out <stem>.agent.xlsx`
    to materialize the filtered edits into the official output. Otherwise, copy
    the extracted workbook to `<stem>.agent.xlsx`.
14. Verify: 0 blank Reason rows remaining, ZIP integrity OK, reason counts match. If the
    re-verification pass ran, also confirm that every `TBE_Reverify = True` row has a
    `[Reverified: ...]` marker in its updated `DetailReason` (or, if reverification
    produced no change, the prior `DetailReason` was left untouched and the cell fill
    is unchanged).

## Notes

- Save output as `.agent.xlsx`; do not modify original workbook.
- Preserve existing `Reason` and `DetailReason` unless deep analysis justifies an update.
- The re-verification pass is opt-in per row via `TBE_Reverify = True`. The agent should
  not mass-flip the column; flipping a row to `True` is a deliberate signal that the
  existing `To be enabled` verdict deserves a fresh recheck.
- Mark updated cells blue using `PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`.
