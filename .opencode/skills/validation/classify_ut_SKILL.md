---
name: classify-ut
description: Batch classify UT test cases from an Excel sheet by running a cascaded decision flow (not_target → community_change → status_xpu → known_issue). Uses scripts/extract_tasks.py for Excel reading + deduplication and scripts/write_results.py for writing the output sheet. Delegates each classification axis to a specialized skill. Outputs results to a new "agent" sheet.
---

# classify_ut

## Purpose

Given an Excel sheet with columns `testfile_cuda`, `classname_cuda`, `name_cuda`, and `message_xpu`, classify each row by filling in `Reason` and `DetailReason` using a cascaded decision flow.

The workflow reuses already-analyzed results when possible (same class + similar error message), avoiding redundant work. For new rows, it runs a decision cascade:

1. Is it a **not-target feature**? → `Not Applicable`
2. Does it have a **community change**? → `Community Change`
3. Is `status_xpu` blank? → `To be enabled`
4. Is there a **known issue**? → `Failures (xpu broken)` / `Feature gap` / `To be enabled`
5. No known issue → `Submit Issue`

## Inputs

| Field | Required | Description |
|-------|----------|-------------|
| `excel_path` | **Yes** | Path to the .xlsx file. Expected columns: `testfile_cuda`, `classname_cuda`, `name_cuda`, `message_xpu`, `status_xpu`. May also have `Reason`, `DetailReason` from prior runs. |
| `sheet_name` | No | Sheet name to read from. Default: first sheet. |
| `output_sheet` | No | Output sheet name. Default: `"agent"`. |
| `PYTORCH_SRC` | No | PyTorch source checkout path. Default: `$HOME/upstream/pytorch`. |

## Output

The script writes a new sheet (default name `"agent"`) to the same Excel file with all original columns plus:

| Column | Description |
|--------|-------------|
| `Analyzed` | `TRUE` / `FALSE`. Whether the row was classified. |
| `Reason` | Classification result: `Not Applicable`, `Community Change`, `To be enabled`, `Failures (xpu broken)`, `Feature gap`, or `Submit Issue`. |
| `DetailReason` | Evidence string explaining the classification. |
| `ReuseSource` | If the result was reused from another row, the `name_cuda` of the source row. Else `""`. |
| `Confidence` | `High` if `DetailReason` contains exact evidence (commit hash, issue/PR URL), otherwise `Medium`. Auto-computed by the write script. |

Rows where `Analyzed` was already `TRUE` from a prior run are left untouched.

## Workflow

The classification uses two scripts (in `scripts/`) for Excel I/O and deduplication:
- `scripts/extract_tasks.py` — reads the Excel, deduplicates rows, outputs `tasks.json`
- `scripts/write_results.py` — takes classification results and writes the `"agent"` sheet

The AI agent processes each unique row requiring classification, running the decision cascade using the specialized skills.

### Phase 0: Setup Conda Environment

```bash
bash .opencode/skills/validation/scripts/setup_env.sh
conda activate classify_ut
```

### Phase 1: Read and Deduplicate

Run `scripts/extract_tasks.py` to read the Excel, deduplicate, and output `tasks.json`:

```bash
python3 scripts/extract_tasks.py <excel_path> [sheet_name] > tasks.json
```

The script outputs a JSON object with:
- `tasks` — Array of row objects needing classification (each has `testfile_cuda`, `classname_cuda`, `name_cuda`, `message_xpu`, `status_xpu`)
- `already_resolved` — Array of row objects that were already analyzed or deduplicated
- `summary` — Counts of total, already_analyzed, deduplicated, needs_classification

**Deduplication logic** (inside the script):

1. Rows where `Analyzed == TRUE` are carried over as-is.
2. For remaining rows: if a row shares the same `classname_cuda` AND similar `message_xpu` with an analyzed row, copy its `Reason`/`DetailReason` and mark `ReuseSource`.
3. **Similar message**: messages share an operator reference (`aten::\w+`, `torch\.\w+`) OR have Levenshtein similarity > 0.7 after normalization.

### Phase 2: Classify Each Task

For each row in `tasks.json`, run the decision cascade. Each check is a **hard gate** — if the condition is met, classification stops and the result is recorded.

> **Confidence**: `Confidence` is auto-computed by `write_results.py` based on whether `DetailReason` contains exact evidence (commit hash, issue/PR URL, PR reference). To produce `High` confidence, include specific, verifiable evidence in every `DetailReason`. Vague statements without references result in `Medium`.

---

#### Step 0: Reuse Exact Match

Before running the cascade, check if an already-resolved row (from `Phase 1` output's `already_resolved` array) has the **same `classname_cuda`** AND the **identical `message_xpu`** as this task.

**If a match is found**:
- Copy its `Reason` and `DetailReason`.
- Set `ReuseSource` to the matched row's `name_cuda`.
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
    description=f"Not-target check: {name_cuda}",
    prompt=f"Check if {name_cuda} in {class_name} ({test_file}) is not-target for XPU. "
           f"Error message: {message_xpu}. "
           f"Return verdict, evidence, and confidence."
)
```

**If `is_not_target == True`**:
- `Reason = "Not Applicable"`
- `DetailReason = evidence from check_not_target_feature`
- Stop cascade for this row.

**If `is_not_target == False`** → proceed to Gate 2.

---

#### Gate 2: Community Change

Check whether the test has a community change (upstream removal/rename).

```python
task(
    subagent_type="explore",
    load_skills=["check_community_change"],
    description=f"Community change check: {name_cuda}",
    prompt=f"Check community change for {name_cuda} in {class_name} (device=cuda). "
           f"Test file: {test_file}. PYTORCH_SRC={PYTORCH_SRC}. "
           f"First check if CUDA is available. If yes, use --collect-only (Path A). "
           f"If not, use source inspection (Path B)."
)
```

**If `community_change == True`**:
- `Reason = "Community Change"`
- `DetailReason = evidence from check_community_change (classification.detail_reason)`
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
    description=f"Known issue check: {name_cuda}",
    prompt=f"Search known issues for {name_cuda} in {class_name}. "
           f"Test file: {test_file}. "
           f"Error message: {message_xpu}."
)
```

**If `has_known_issue == True`**:

Examine the matched issues. For the **highest relevance** match:

| Issue State | Labels | `Reason` | `DetailReason` |
|-------------|--------|----------|----------------|
| OPEN | `bug` or no label | `Failures (xpu broken)` | Issue title + URL |
| OPEN | `feature` / `enhancement` | `Feature gap` | Issue title + URL |
| OPEN | `skipped` | `Failures (xpu broken)` | Issue title + URL (skipped tests are broken) |
| CLOSED | `not_target` / `wontfix` | `Not Applicable` | Issue title + URL (override Gate 1) |
| CLOSED | other or no relevant label | `To be enabled` | Issue title + URL (likely fixed) |

**If `has_known_issue == False`**:
- `Reason = "Submit Issue"`
- `DetailReason = "No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`

Stop cascade for this row.

---

### Phase 3: Write Results to Excel

Run `scripts/write_results.py` to write all classification results to the output sheet:

```bash
python3 scripts/write_results.py <excel_path> results.json [sheet_name] [--output_sheet=agent]
```

The script:
1. Reads the original data from the input sheet.
2. Appends columns: `Analyzed`, `Reason`, `DetailReason`, `ReuseSource`, `Confidence`.
3. For each row:
   - If it was classified (from Phase 2 or deduplicated in Phase 1) → fill `Analyzed = TRUE`, fill `Reason`, `DetailReason`, `ReuseSource`.
   - If it was already `Analyzed = TRUE` before → leave untouched.
   - Otherwise → `Analyzed = FALSE`.
4. **Auto-computes `Confidence`**: `High` if `DetailReason` matches patterns for exact evidence (GitHub issue URL, PR URL, commit hash, `#PR` reference). Else `Medium`.
5. Writes to a new sheet (default name `"agent"`).
6. Does **not** modify the original sheet.



## Execution

Load this skill to orchestrate the full classification. The agent follows this workflow directly:

1. Run Phase 1: `python3 .opencode/skills/validation/scripts/extract_tasks.py <excel_path> [sheet_name] > tasks.json`. This produces a JSON file with `tasks` (rows needing classification) and `already_resolved` (deduplicated rows).
2. For each task in `tasks.json`, run the decision cascade:
   - **Gate 1**: Delegate to `check_not_target_feature` skill.
   - **Gate 2**: Delegate to `check_community_change` skill.
   - **Gate 3**: Check `status_xpu` value directly from the task data.
   - **Gate 4**: Delegate to `check_known_issue` skill.
3. Accumulate all results (already_resolved + newly classified tasks) into `results.json`.
4. Run Phase 3: `python3 .opencode/skills/validation/scripts/write_results.py <excel_path> results.json [sheet_name]`.
5. Report summary statistics: rows total, deduplicated, classified per Reason category.

## Constraints

1. **Gate order is strict**: Always check `not_target` before `community_change`, and `community_change` before `status_xpu`. Breaking the order can produce wrong classifications.
2. **Deduplication is a speed optimization, not a classification shortcut**: Only reuse results from rows with the same class AND similar error message. Do not reuse across unrelated tests.
3. **Scripts handle all Excel I/O**: The agent should never manipulate Excel cells directly. Always use `scripts/extract_tasks.py` and `scripts/write_results.py`. Script paths are relative to `.opencode/skills/validation/`.
4. **Open issues with `skipped` label are treated as failures**: A skipped test is a broken test — classify as `Failures (xpu broken)`, not `To be enabled`.
5. **Closed issues with `not_target`/`wontfix` override Gate 1**: If Gate 1 said "not not-target" but Gate 4 finds a CLOSED `not_target` issue, the `not_target` label is authoritative. Reclassify as `Not Applicable`.
6. **`Submit Issue` means manual intervention**: These rows require a human to file a new issue. The agent should not auto-file.
7. **Never modify the original sheet**: Always write to the output sheet name (default `"agent"`).

## Version

- v1.0.0 - 2026-06-09 - Initial classify_ut skill with Python script I/O and cascaded classification.
- v1.1.0 - 2026-06-09 - Extracted scripts to `scripts/extract_tasks.py` and `scripts/write_results.py`; removed embedded script.

## See Also

- `check_not_target_feature` — Gate 1: determines if a test is CUDA-only / not applicable for XPU
- `check_community_change` — Gate 2: determines if a test was removed/renamed upstream
- `check_known_issue` — Gate 4: searches for known issues in pytorch/pytorch and intel/torch-xpu-ops
