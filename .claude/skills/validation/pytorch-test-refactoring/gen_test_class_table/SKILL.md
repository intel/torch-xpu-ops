---
name: gen_test_class_table
description: Generate test_class_classification_merged.xlsx - a workbook of every PyTorch test class classified as device_unrelated / device_agnostic / device_specific, joined with the Device-Generic Refactoring tracker and UT_Upstream_Status columns, and optionally annotated with local XPU test results. Use when asked to build, regenerate, or refresh the test-class classification table / merged tracker spreadsheet used to drive pytorch-test-refactoring work.
---

# Generate Test Class Classification Table

Produce `test_class_classification_merged.xlsx`: a single spreadsheet with one row
per PyTorch test class, its device-dependency classification, and the tracker /
UT-status metadata that drives the pytorch-test-refactoring workflow.

## Pipeline

Three ordered stages. Stage 3 is optional (only when local XPU results exist).

```
scan_all_test_classes.py   ->  test_class_classification.xlsx        (stage 1)
merge_tracker.py           ->  test_class_classification_merged.xlsx (stage 2)
add_intel_columns.py       ->  (annotate the merged xlsx in place)   (stage 3, optional)
```

| Stage | Script | Reads | Writes / Effect |
|-------|--------|-------|-----------------|
| 1 | `scripts/scan_all_test_classes.py` | PyTorch `test/` tree | `test_class_classification.xlsx` (file, class, category, xpu_enabled) |
| 2 | `scripts/merge_tracker.py` | stage-1 xlsx + tracker xlsx + UT status xlsx | `test_class_classification_merged.xlsx` (adds tracker + UT columns + Merged) |
| 3 | `scripts/add_intel_columns.py` | merged xlsx + summary.tsv + committed.tsv + known_issues.tsv | appends Intel Status / Intel Result / Known Issue columns in place |

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| PyTorch `test/` dir | `$HOME/daisyden/upstream/test` (or `--pytorch-test`) | Source tree scanned in stage 1 |
| Tracker xlsx | `.opencode/skills/validation/pytorch-test-refactoring/Device-Generic Refactoring Test Class Tracker.xlsx` | Multi-sheet refactor tracker |
| UT status xlsx | `.opencode/skills/validation/pytorch-test-refactoring/UT_Upstream_Status.xlsx` | `test_all_classes` sheet |
| Output dir | current working dir (use `agent_space_xpu/`) | Where the xlsx files land |

Paths are overridable by CLI flags or env vars (`PYTORCH_TEST_DIR`,
`OUT_CLASSIFICATION_XLSX`, `TRACKER_XLSX`, `UT_STATUS_XLSX`, `OUT_MERGED_XLSX`).
Never hardcode a user's absolute paths into the scripts - pass them at call time.

## Output columns (merged xlsx)

`file, class, category, xpu_enabled, needs_refactor, priority, refactor_poc,`
`refactoring_notes, tracker_sheet, owner, ut_priority, status, Q2, Merged`

With stage 3, three more: `Intel Status, Intel Result, Known Issue`.

## Classification (stage 1)

Hierarchical: **device_specific > device_agnostic > device_unrelated**.

- **device_specific**: file named `test_{cuda,mps,xpu,rocm,mtia,tpu,lazy}.py`; imports
  `TEST_CUDA/TEST_MPS/TEST_XPU/TEST_ROCM/TEST_MTIA`; or references `torch.{cuda,mps,xpu}.` APIs.
- **device_agnostic**: imports/calls `instantiate_device_type_tests`; uses
  `@ops/@dtypes/@dtypesIf*`; imports `common_methods_invocations`/`common_dtype`; or
  references `TEST_PRIVATEUSE1/TEST_ACCELERATOR`/`torch.accelerator.`.
- **device_unrelated**: none of the above (CPU-only).

`xpu_enabled` is filled only for `device_agnostic` classes: `True` when their
`instantiate_device_type_tests(...)` call passes `allow_xpu=True`, else `False`.

## Usage

Run from the torch-xpu-ops repo root. Emit artifacts into `agent_space_xpu/`
(git-ignored). Use the project `.venv` python if present.

```bash
SKILL=.opencode/skills/validation/pytorch-test-refactoring/gen_test_class_table
OUT=agent_space_xpu
mkdir -p "$OUT"

# Stage 1: classify every test class
python "$SKILL/scripts/scan_all_test_classes.py" \
    --pytorch-test "$HOME/daisyden/upstream/test" \
    --out "$OUT/test_class_classification.xlsx"

# Stage 2: merge tracker + UT status -> merged xlsx
# Add --no-pr-check to skip gh network calls (Merged column left blank).
python "$SKILL/scripts/merge_tracker.py" \
    --src "$OUT/test_class_classification.xlsx" \
    --out "$OUT/test_class_classification_merged.xlsx"

# Stage 3 (optional): annotate with local XPU results / commits / known issues
python "$SKILL/scripts/add_intel_columns.py" \
    --xlsx "$OUT/test_class_classification_merged.xlsx" \
    --summary "$OUT/phase2_logs/summary.tsv" \
    --committed /tmp/committed_pairs.tsv
```

## Stage-3 input formats

- `summary.tsv` - TSV with header including `file, class, gate, xpu_passed,
  xpu_failed, xpu_skipped`. `gate` is one of `PASS`, `TIMEOUT`,
  `FAIL-unexplained`, `FAIL-no-xpu-rows`.
- `committed.tsv` - `<file_relpath>\t<class>` per line; marks rows committed to
  the fork branch (`--branch-link`).
- `known_issues.tsv` - `<file_relpath>\t<class>\t<issue_text>` per line.

Any missing stage-3 input simply leaves that column blank; stage 3 is safe to
skip entirely if there are no local results.

## Join semantics

- **Tracker** is left-joined by `(basename(file), class)` (also tries a
  filename-less `("", class)` fallback). Sheets scanned: `generic`, `profiler`,
  `distributed`, `dynamo`, `cuda-specific`. First sheet wins on duplicate keys.
- **UT status** is joined by full relative path `(file, class)` first, then by
  `(basename, class)`.
- Unmatched rows keep the joined columns blank - expected for the long tail.

## Verify

1. Row parity: merged row count == stage-1 row count == number of test classes.
   ```bash
   python -c "import openpyxl,sys; \
   a=openpyxl.load_workbook('agent_space_xpu/test_class_classification.xlsx').active.max_row; \
   b=openpyxl.load_workbook('agent_space_xpu/test_class_classification_merged.xlsx').active.max_row; \
   print('stage1',a,'merged',b); sys.exit(0 if a==b else 1)"
   ```
2. Header check: merged headers match the 14 (or 17 with stage 3) column names above.
3. Match rates printed by stage 2 (`Tracker matched`, `UT status matched`) are non-zero.

## Notes

- `openpyxl` is required. Stage 1 falls back to CSV if it is missing; stages 2-3
  hard-require it.
- Stage 2 calls `gh api` to resolve PR merge status embedded in tracker notes.
  Use `--no-pr-check` when offline or when `gh` is unauthenticated.
- The tracker and UT-status xlsx live inside this skill family's parent
  directory; keep them current before regenerating.
