---
name: classify-ut
description: Batch classify UT test cases from an Excel sheet by running a cascaded decision flow (not_target → community_change → status_xpu → known_issue). Delegates each classification axis to a specialized skill. Outputs results to a new "agent" sheet.
---

# classify_ut

## Purpose

Given an Excel sheet with columns `testfile_cuda`, `classname_cuda`, `name_cuda`, `message_xpu`, and optionally `status_xpu`, classify **XPU test cases** by filling in `Reason` and `DetailReason` using a cascaded decision flow.

**XPU column inference**: The target test is an XPU test, not a CUDA test. The script computes XPU columns from CUDA columns:
- `testfile_xpu` = `testfile_cuda` (same file)
- `classname_xpu` = `classname_cuda` with `CUDA` suffix replaced by `XPU` (e.g. `TestStreamsCUDA` → `TestStreamsXPU`)
- `name_xpu` = `name_cuda` with `_cuda` replaced by `_xpu` (e.g. `test_foo_cuda_float32` → `test_foo_xpu_float32`)

If the spreadsheet already contains `testfile_xpu`, `classname_xpu`, `name_xpu` columns, those values are used as-is (no override).

The workflow reuses already-analyzed results when possible (same class + similar error message), avoiding redundant work. For new rows, it runs a decision cascade:

0. **Local test run** (for `status_xpu` blank only)? → `Local Passed` (if test passes locally)
1. Is it a **not-target feature**? → `Not Applicable`
2. Does it have a **community change**? → `Community Change`
3. Is `status_xpu` blank? → `To be enabled`
4. Is there a **known issue**? → `Failures (xpu broken)` / `Feature gap` / `To be enabled`
5. No known issue → `Submit Issue`

If a test is `Local Passed`, no further classification is needed — it skips Gates 1–4 entirely.

## Inputs

| Field | Required | Description |
|-------|----------|-------------|
| `excel_path` | **Yes** | Path to the .xlsx file. Expected columns: `testfile_cuda`, `classname_cuda`, `name_cuda`, `message_xpu`, `status_xpu`. May also have `Reason`, `DetailReason` from prior runs. If `testfile_xpu`, `classname_xpu`, `name_xpu` columns exist, they are used directly; otherwise XPU columns are inferred from CUDA columns (see Purpose). |
| `sheet_name` | No | Sheet name to read from. Default: first sheet. |
| `output_sheet` | No | Output sheet name. Default: `"agent"`. |

## Output

The script writes a new sheet (default name `"agent"`) to a standalone Excel file (e.g. `agent_results.xlsx`) with all original columns plus:

| Column | Description |
|--------|-------------|
| `Analyzed` | `TRUE` / `FALSE`. Whether the row was classified. |
| `Reason` | Classification result: `Not Applicable`, `Community Change`, `To be enabled`, `Failures (xpu broken)`, `Feature gap`, or `Submit Issue`. |
| `DetailReason` | Evidence string explaining the classification. |
| `ReuseSource` | If the result was reused from another row, the `name_cuda` of the source row. Else `""`. |
| `Confidence` | `High` if `DetailReason` contains exact evidence (commit hash, issue/PR URL), otherwise `Medium`. Auto-computed by the write script. |

In MERGE mode (`--merge`), rows already `Analyzed = TRUE` in the output file are preserved and skipped (unless `--force`); only rows present in `results.json` are written. In BUILD mode, rows already `Analyzed = TRUE` in the *input* sheet are carried over. See Phase 3 for mode selection.

## Prerequisites

Python 3 with `openpyxl` installed:

```bash
pip install openpyxl
```

## Scripts

This skill uses three companion scripts, all located in the sibling `scripts/` directory:

| Script | Purpose | Workspace-relative path |
|--------|---------|------------------------|
| `extract_tasks.py` | Reads the Excel, deduplicates rows, outputs `tasks.json` | `.opencode/skills/torch-xpu-ops-validation/scripts/extract_tasks.py` |
| `run_blank_test.py` | Runs `status_xpu` blank test cases locally; marks passing tests as `Local Passed` | `.opencode/skills/torch-xpu-ops-validation/scripts/run_blank_test.py` |
| `write_results.py` | Takes classification results and writes the `"agent"` sheet | `.opencode/skills/torch-xpu-ops-validation/scripts/write_results.py` |

Run all script commands from the **workspace root** (the repository checkout directory). Do not `cd` into subdirectories.

## Workflow

The classification proceeds in three phases using the scripts for Excel I/O and deduplication.

### Phase 1: Read and Deduplicate

Run `extract_tasks.py` to read the Excel, deduplicate, and output `tasks.json`:

```bash
python3 .opencode/skills/torch-xpu-ops-validation/scripts/extract_tasks.py <excel_path> [sheet_name] > tasks.json
```

The script outputs a JSON object with:
- `tasks` — Array of row objects needing classification (each has `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`, `classname_xpu`, `name_xpu`, `message_xpu`, `status_xpu`)
- `already_resolved` — Array of row objects that were already analyzed or deduplicated (same fields)
- `summary` — Counts of total, already_analyzed, deduplicated, needs_classification

**XPU column computation**: Every row in the output includes XPU fields. If the Excel already has XPU columns they are used directly; otherwise they are inferred from CUDA columns using the rules in Purpose.

**Deduplication logic** (inside the script):

1. Rows where `Analyzed == TRUE` are carried over as-is.
2. For remaining rows: if a row shares the same `classname_xpu` AND similar `message_xpu` with an analyzed row, copy its `Reason`/`DetailReason` and mark `ReuseSource`.
3. **Similar message**: messages share an operator reference (`aten::\w+`, `torch\.\w+`) OR have Levenshtein similarity > 0.7 after normalization.
4. **Critical exception — different GitHub issues are never similar**: Messages referencing different GitHub issue/PR numbers (e.g. `github.com/pytorch/pytorch/issues/179853` vs `github.com/pytorch/pytorch/issues/179897`) are NEVER considered similar, even if the surrounding text is identical. The issue/PR number is classification-critical evidence — each issue URL maps to a unique `DetailReason` that must not be reused across rows with different issues.

### Phase 2: Local Test Run (Gate 0)

Before running the cascade, run blank `status_xpu` test cases locally. If a test passes locally, it is `Local Passed` and skips all further classification.

```bash
# Run from workspace root with a conda environment that has PyTorch installed
python3 .opencode/skills/torch-xpu-ops-validation/scripts/run_blank_test.py tasks.json \
    --output results.json --log-dir test_logs --env <conda_env_name> --timeout 300
```

**Behavior**:
- Only tasks with blank `status_xpu` are run.
- Tests are grouped by file and run via `pytest`.
- A test is `Local Passed` only if `pytest` reports `1 passed` with no `FAILED` output.
- `DetailReason` is set to the local PyTorch version (e.g. `"Local test PASSED (torch 2.13.0+xpu)"`).
- Full pytest output is dumped to `test_logs/<testfile_safe_name>.log`.
- A `run_summary.log` is written to `test_logs/` with per-file results.
- The output `results.json` includes `Local Passed` entries ready for `write_results.py`.
- Tests that do not pass locally remain in the output for cascade processing.

**If `Local Passed`**:
- `Reason = "Local Passed"`
- `DetailReason = "Local test PASSED (torch <version>)"`
- Stop. Do not run Gates 1–4 for this row.

**If NOT `Local Passed`** → proceed to Phase 2 (cascade).

### Phase 3: Classify Each Task

For each row not yet classified (not `Local Passed`), run the decision cascade. Each check is a **hard gate** — if the condition is met, classification stops and the result is recorded.

> **IMPORTANT — Cascade is mandatory for every unclassified task**: Tasks not marked `Local Passed` must still run the cascade. The `Reason` and `DetailReason` fields in the `tasks.json` input may contain pre-populated values from a prior run. **These are NOT valid classification results.** The orchestrator MUST run the full cascade (Gates 1-4 via subagents) for every task in the `tasks` array. Only rows in `already_resolved` (Analyzed=TRUE or deduplicated) may bypass the cascade via Step 0 reuse.

> **IMPORTANT — `--filter-*` flags are post-hoc only**: The `--filter-reason` and `--filter-detailreason` flags on `extract_tasks.py` exist for extracting a subset of already-classified rows for review. They MUST NOT be used as a classification shortcut. If you use them to select rows, you MUST still run the full cascade on those rows. "I already know the result" is not valid — the subagents determine the result.

> **IMPORTANT — Cascade is mandatory for every task**: The `Reason` and `DetailReason` fields in the `tasks.json` input may contain pre-populated values from a prior run. **These are NOT valid classification results.** The orchestrator MUST run the full cascade (Gates 1-4 via subagents) for every task in the `tasks` array. Only rows in `already_resolved` (Analyzed=TRUE or deduplicated) may bypass the cascade via Step 0 reuse.

> **IMPORTANT — `--filter-*` flags are post-hoc only**: The `--filter-reason` and `--filter-detailreason` flags on `extract_tasks.py` exist for extracting a subset of already-classified rows for review. They MUST NOT be used as a classification shortcut. If you use them to select rows, you MUST still run the full cascade on those rows. "I already know the result" is not valid — the subagents determine the result.

> **Confidence**: `Confidence` is auto-computed by `write_results.py` based on whether `DetailReason` contains exact evidence (commit hash, issue/PR URL, PR reference). To produce `High` confidence, include specific, verifiable evidence in every `DetailReason`. Vague statements without references result in `Medium`.

---

#### Step 0: Reuse Exact Match

Before running the cascade, check if an already-resolved row (from `Phase 1` output's `already_resolved` array) has the **same `classname_xpu`** AND the **identical `message_xpu`** as this task.

**If a match is found**:
- Copy its `Reason` and `DetailReason`, but prefix both with `[Reused row#XX] ` (using the row number of the matched row if available).
- Set `ReuseSource` to the matched row's `name_xpu`.
- Skip the cascade for this row.

**If no match is found** → proceed to Gate 1.

This catches cases where `Phase 1` deduplication did not fire (messages were similar but not identical) but the agent can see they are the same error.

---

#### Gate 1: Not Target

Check whether the test is a not-target feature for XPU.

```python
task(
    subagent_type="explore",
    load_skills=["check_not_target_feature"],
    description=f"Not-target check: {name_xpu}",
    prompt=f"Check if {name_xpu} in {classname_xpu} ({testfile_xpu}) is not-target for XPU. "
           f"Error message: {message_xpu}. "
           f"Return verdict, evidence, and confidence."
)
```

**If `is_not_target == True`**:
- `Reason = "Not Applicable"`
- `DetailReason = "<reasoning> (Evidence: <evidence joined>)"`
- Stop cascade for this row.

**If `is_not_target == False`** → proceed to Gate 2.

---

#### Gate 2: Community Change

Check whether the test has a community change (upstream removal/rename).

The `check_community_change` skill requires a `PYTORCH_SRC` path (PyTorch source checkout) to inspect test files and git history. Provide it when available; if omitted, the skill may fall back to less authoritative checks.

```python
task(
    subagent_type="explore",
    load_skills=["check_community_change"],
    description=f"Community change check: {name_cuda}",
    prompt=f"Check community change for {name_cuda} in {class_name} (device=cuda). "
           f"Test file: {test_file}. "
           f"First check if CUDA is available. If yes, use --collect-only (Path A). "
           f"If not, use source inspection (Path B)."
)
```

**If `community_change == True`**:
- `Reason = "Community Change"`
- `DetailReason = classification.detail_reason` (from check_community_change output)
- Stop cascade for this row.

**If `community_change == False`** → proceed to Gate 3.

---

#### Gate 3: status_xpu Blank

Check the `status_xpu` value for this row.

**If `status_xpu` is blank/empty**:
- `Reason = "To be enabled"`
- `DetailReason = "status_xpu is blank — no known status. Awaiting enablement."`
- Stop cascade for this row.

**If `status_xpu` is NOT blank** → proceed to Gate 4.

---

#### Gate 4: Known Issue

Check whether a known issue exists for this test.

```python
task(
    subagent_type="explore",
    load_skills=["check_known_issue"],
    description=f"Known issue check: {name_xpu}",
    prompt=f"Search known issues for {name_xpu} in {classname_xpu} ({testfile_xpu}). "
           f"CUDA source: {name_cuda} in {classname_cuda} ({testfile_cuda}). "
           f"Error message: {message_xpu}."
)
```

**If `has_known_issue == True`**:
- Use the `Reason` and `DetailReason` provided directly in the subskill's output (`classification.Reason` and `classification.DetailReason`).
- Stop cascade for this row.

**If `has_known_issue == False`**:
- `Reason = "Submit Issue"`
- `DetailReason = "No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`

Stop cascade for this row.

---

### Phase 4: Write Results to Excel

`write_results.py` has two modes. Choose based on whether the output file already exists.

**BUILD (first run — output file does not exist yet):**

```bash
python3 .opencode/skills/torch-xpu-ops-validation/scripts/write_results.py <excel_path> results.json [sheet_name] --output_sheet=agent --output-excel=agent_results.xlsx
```

Reads the original input sheet, appends columns `Analyzed`, `Reason`, `DetailReason`, `ReuseSource`, `Confidence`, and writes a fresh `agent` sheet to the standalone `--output-excel` file. Per row: classified rows get `Analyzed = TRUE` with their `Reason`/`DetailReason`/`ReuseSource`; rows already `Analyzed = TRUE` in the *input* sheet are carried over; everything else gets `Analyzed = FALSE`.

**MERGE (incremental run — output file already exists, e.g. an accumulator):**

```bash
python3 .opencode/skills/torch-xpu-ops-validation/scripts/write_results.py --merge results.json --output_sheet=agent --output-excel=agent_results.xlsx
```

Updates the existing `--output-excel` file in place, touching ONLY the rows present in `results.json` (matched by CUDA identity: `testfile_cuda`/`classname_cuda`/`name_cuda`, falling back to `name_cuda` alone). Every other row — including rows analyzed by previous runs — is left untouched. Rows already `Analyzed = TRUE` are skipped unless `--force` is passed.

Both modes **auto-compute `Confidence`** (`High` if `DetailReason` matches exact-evidence patterns — GitHub issue/PR URL, commit hash, `#PR` reference — else `Medium`).

**Safety:**
- BUILD refuses to overwrite an existing `--output-excel` whose `agent` sheet contains `Analyzed = TRUE` rows that are NOT in `results.json` (a fresh build would discard them). It exits non-zero and tells you to use `--merge` (preserve prior rows) or `--force` (overwrite anyway).
- Any in-place write (MERGE, or BUILD `--force` over an existing file) first writes a `<file>.bak` backup.
- If a `results.json` entry matches multiple sheet rows by name (ambiguous, e.g. `results.json` lacks the full CUDA identity), those rows are NOT written and the script exits non-zero. To avoid this, ensure each `results.json` entry carries `testfile_cuda`, `classname_cuda`, and `name_cuda`.


## Execution

Load this skill to orchestrate the full classification. The agent follows this workflow directly:

1. Run Phase 1:
   ```
   python3 .opencode/skills/torch-xpu-ops-validation/scripts/extract_tasks.py <excel_path> [sheet_name] > tasks.json
   ```
2. **(New) Run Phase 2 — Local Test (Gate 0)**:
   ```
   python3 .opencode/skills/torch-xpu-ops-validation/scripts/run_blank_test.py tasks.json \
       --output results.json --log-dir test_logs --env <conda_env>
   ```
   This runs all `status_xpu` blank tests locally. Passing tests get `Reason = "Local Passed"` and skip the cascade.
3. For each remaining task (not `Local Passed`) in `results.json`, run the decision cascade:
   - **Gate 1**: Pass XPU identifiers (`name_xpu`, `classname_xpu`, `testfile_xpu`). Determine if the XPU test is a not-target feature.
   - **Gate 2**: Pass CUDA identifiers (`name_cuda`, `classname_cuda`, `testfile_cuda`). Check if the upstream CUDA test was removed/renamed (which also affects the XPU variant).
   - **Gate 3**: Check `status_xpu` value directly from the task data.
   - **Gate 4**: Pass XPU identifiers (`name_xpu`, `classname_xpu`, `testfile_xpu`) plus the CUDA source references. Search for known issues related to the XPU failure.
4. Accumulate all results (already_resolved + Local Passed + newly classified tasks) into `results.json`.
5. Run Phase 4. If the `--output-excel` accumulator does not exist yet, BUILD it:
   ```
   python3 .opencode/skills/torch-xpu-ops-validation/scripts/write_results.py <excel_path> results.json [sheet_name] --output_sheet=agent --output-excel=agent_results.xlsx
   ```
   If it already exists (incremental run adding to a prior accumulator), MERGE in place so earlier rows are preserved:
   ```
   python3 .opencode/skills/torch-xpu-ops-validation/scripts/write_results.py --merge results.json --output_sheet=agent --output-excel=agent_results.xlsx
   ```
6. Report summary statistics: rows total, deduplicated, local passed, classified per Reason category.

## Constraints

0. **Gate 0 (Local Test) runs before the cascade**: Always run `run_blank_test.py` before Gate 1. Tests that pass locally (`Local Passed`) skip Gates 1–4 entirely. Only tests that fail/skip/time out locally proceed to the cascade. This saves classification effort for working tests.
1. **Gate order is strict**: Always check `not_target` before `community_change`, and `community_change` before `status_xpu`. Breaking the order can produce wrong classifications.
2. **Deduplication is a speed optimization, not a classification shortcut**: Only reuse results from rows with the same class AND similar error message. Do not reuse across unrelated tests.
3. **Scripts handle all Excel I/O**: The agent should never manipulate Excel cells directly. Always use `.opencode/skills/torch-xpu-ops-validation/scripts/extract_tasks.py` and `.opencode/skills/torch-xpu-ops-validation/scripts/write_results.py`.
4. **Open issues with `skipped` label are treated as failures**: A skipped test is a broken test — classify as `Failures (xpu broken)`, not `To be enabled`.
5. **Closed issues with `not_target`/`wontfix` override Gate 1**: If Gate 1 said "not not-target" but Gate 4 finds a CLOSED `not_target` issue, the `not_target` label is authoritative. Reclassify as `Not Applicable`.
6. **`Submit Issue` means manual intervention**: These rows require a human to file a new issue. The agent should not auto-file.
7. **Never modify the original sheet**: Always write to the output sheet name (default `"agent"`).
8. **No local system assumptions**: The skill does not depend on any pre-existing conda environment, proxy configuration, or local PyTorch checkout. All dependencies are documented and installable via `pip install openpyxl`.
9. **All commands run from workspace root**: All script invocations use absolute (workspace-relative) paths. Do not `cd` into subdirectories before running scripts.
10. **`--filter-reason` and `--filter-detailreason` are post-hoc only**: These flags select rows by their OUTPUT classification columns. They MUST NOT be used as a classification shortcut. If you use them for subset extraction, you MUST still run the cascade via subagents on those rows. The pre-populated Reason/DetailReason values are from a prior run — cascade results override them.
11. **Ignore pre-populated Reason/DetailReason in task data**: The `tasks` array entries may contain `Reason` and `DetailReason` from a previous run. Do not copy them to the output. Always run the cascade gates (via subagents for Gates 1, 2, 4; directly for Gate 3). The `already_resolved` array handles all legitimate reuses.
12. **Task data fields not to be read during classification**: During Phase 2 cascade processing, treat `Reason` and `DetailReason` as write-only outputs. Read `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`, `classname_xpu`, `name_xpu`, `message_xpu`, and `status_xpu`. Ignore all other fields. Use XPU fields for Gate 1 and Gate 4; use CUDA fields for Gate 2.
13. **Mandatory Evidence in DetailReason**: The `DetailReason` MUST contain explicit evidence to support the classification. For `check_known_issue`, this means a valid GitHub Issue URL or number. For `check_community_change`, this means a commit hash or PR number where the change occurred. For `check_not_target_feature`, list the corresponding `intel/torch-xpu-ops` `not_target`/`wontfix` issue number (e.g., `intel/torch-xpu-ops#3127`) whenever one exists; otherwise cite a specific code snippet (e.g., an `@onlyCUDA` decorator at `file:line`) or API documentation reference. Vague statements without evidence are invalid.
14. **Never rebuild over an existing accumulator**: When `--output-excel` already contains analyzed rows from prior runs, use `--merge` (not a plain BUILD) so those rows are preserved. A plain BUILD is only for creating the file the first time. The script enforces this (BUILD aborts rather than discarding prior `Analyzed = TRUE` rows), but choose the correct mode up front. For unambiguous merge matching, every `results.json` entry MUST carry `testfile_cuda`, `classname_cuda`, and `name_cuda`.

## Version

- v3.0.0 - 2026-06-17 - Added Gate 0 (Local Test) via `run_blank_test.py`. Blank `status_xpu` tests are run locally before the cascade. Passing tests get `Reason = "Local Passed"` and skip further classification. Added `run_blank_test.py` script and updated workflow Phases (2→local test, 3→cascade, 4→write results). New Constraint 0.
- v2.3.0 - 2026-06-14 - `check_not_target_feature`: require listing the corresponding `intel/torch-xpu-ops` `not_target`/`wontfix` issue number in `evidence` whenever one exists (new Step 6 "Attach Issue Number" + Strict Constraint 6), even for verdicts reached via the Not-Applicable sheet (Step 3) or implementation analysis (Step 5). Updated Constraint 13 to prefer issue numbers for not_target evidence.
- v2.2.0 - 2026-06-14 - Added incremental `--merge` mode to `write_results.py` for safely updating an existing accumulator in place (only touches rows in `results.json`, preserves all prior rows). Added a destructive-rebuild guard (BUILD aborts rather than discarding prior `Analyzed = TRUE` rows; override with `--force`), automatic `.bak` backup on in-place writes, and ambiguity detection. Documented BUILD vs MERGE in Phase 3 and added Constraint 14.
- v2.1.0 - 2026-06-10 - Added Constraint 10-12: explicit rules against using `--filter-reason`/`--filter-detailreason` as classification shortcut, ignoring pre-populated Reason/DetailReason in tasks, and restricting readable task fields. Added IMPORTANT callouts at Phase 2 entry. Added WARNING stderr messages in `extract_tasks.py` for `--filter-reason`/`--filter-detailreason` to discourage misuse.
- v2.1.0 - 2026-06-10 - Added Constraint 10-12: explicit rules against using `--filter-reason`/`--filter-detailreason` as classification shortcut, ignoring pre-populated Reason/DetailReason in tasks, and restricting readable task fields. Added IMPORTANT callouts at Phase 2 entry.
- v2.0.0 - 2026-06-10 - Rewritten: fixed script paths to `.opencode/skills/torch-xpu-ops-validation/scripts/`, removed Phase 0 conda setup (Intel-specific infra), removed `PYTORCH_SRC` default, removed local system dependencies.

## See Also

- `run_blank_test.py` — Gate 0: runs blank `status_xpu` tests locally, marks passing tests as `Local Passed`
- `check_not_target_feature` — Gate 1: determines if a test is CUDA-only / not applicable for XPU
- `check_community_change` — Gate 2: determines if a test was removed/renamed upstream
- `check_known_issue` — Gate 4: searches for known issues in pytorch/pytorch and intel/torch-xpu-ops
