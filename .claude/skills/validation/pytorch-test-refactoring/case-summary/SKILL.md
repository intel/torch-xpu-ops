---
name: case-summary
description: Annotate test_class_classification_merged.xlsx with a per-class test-case count (via pytest --collect-only) and add a summary sheet reporting category distribution and xpu_enabled coverage by both class count and case count, including a ut_priority P0-P3/blank subset. Use when asked to summarize the merged classification table, count test cases per class, or compute device-category / XPU-enablement percentages weighted by test-case volume.
---

# Case Summary

Extend `test_class_classification_merged.xlsx` (from the `gen_test_class_table`
skill) with real per-class test-case counts and a `summary` sheet of category /
XPU-enablement statistics, computed both by class count and by case count.

## Pipeline

Two ordered stages.

```
collect_case_counts.py  ->  case_counts.tsv                         (stage 1)
summarize_merged.py     ->  merged xlsx + case_count col + summary   (stage 2)
```

| Stage | Script | Reads | Writes / Effect |
|-------|--------|-------|-----------------|
| 1 | `scripts/collect_case_counts.py` | merged xlsx (file list) + PyTorch `test/` tree | `case_counts.tsv` (file, class, case_count) + errors tsv |
| 2 | `scripts/summarize_merged.py` | merged xlsx + `case_counts.tsv` | adds `case_count` column + `summary` sheet in place |

## Stage 1: collect case counts

Runs `pytest --collect-only -q` on every unique file in the merged xlsx from a
neutral CWD (`/tmp`) so the *installed* torch resolves, not the source checkout.
Collected node IDs (`file.py::ClassXPU::test_...`) are parsed; the per-device
class suffix (`CPU`/`CUDA`/`XPU`/`Meta`/`HPU`/`MPS`/`MTIA`/`Lazy`/`PrivateUse1`)
is stripped to recover the generic class name stored in the xlsx, then counts
are aggregated per `(file, generic_class)`.

Collection is slow (~30-45 min for the full PyTorch test tree) and some files
fail to collect (distributed / inductor harness requirements, import errors).
Run it in tmux so it survives the bash timeout:

```bash
tmux new-session -d -s casecollect \
  "cd $(pwd) && python3 SKILL_DIR/scripts/collect_case_counts.py --timeout 150 \
   > agent_space_xpu/collect.log 2>&1; echo DONE_EXIT_\$? >> agent_space_xpu/collect.log"
# poll: tail -f agent_space_xpu/collect.log
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--xlsx` | `agent_space_xpu/test_class_classification_merged.xlsx` | Source of the file list |
| `--test-root` | `$HOME/daisyden/upstream/test` | PyTorch checkout `test/` dir |
| `--out` | `agent_space_xpu/case_counts.tsv` | Per-class counts |
| `--errors` | `agent_space_xpu/case_counts_errors.tsv` | Files that failed collection |
| `--timeout` | `180` | Per-file collection timeout (seconds) |

Files failing collection are logged with a status (`timeout`, `missing`,
`error(rc=N)`) and simply get a blank `case_count` downstream - they contribute
0 to case totals. A nonzero final exit (`DONE_EXIT_1`) is normal: it reflects
the last per-file pytest return code, not a script failure. Confirm success by
the `Wrote N rows` line in the log.

## Stage 2: summarize

Adds a `case_count` column keyed by `(file, class)`, then writes a `summary`
sheet with four analyses:

| Block | Weight | Scope |
|-------|--------|-------|
| **#1** | class count | all classes |
| **#2** | class count | `ut_priority` in {P0, P1, P2, P3, blank} |
| **#4a** | case count | all classes |
| **#4b** | case count | `ut_priority` in {P0, P1, P2, P3, blank} |

Each block reports the `category` distribution (device_unrelated /
device_agnostic / device_specific) with counts + percentages, and the
`xpu_enabled` True/False split *within* device_agnostic. The priority subset
keeps only P0-P3 and blank rows (excludes `by default enabled`, `Not Target`,
`Done`).

```bash
python3 SKILL_DIR/scripts/summarize_merged.py \
    --xlsx agent_space_xpu/test_class_classification_merged.xlsx \
    --case-counts agent_space_xpu/case_counts.tsv
```

The script also prints the full summary to stdout and is idempotent - rerunning
replaces the `case_count` column values and the `summary` sheet.

## Usage

```bash
SKILL=.opencode/skills/validation/pytorch-test-refactoring/case-summary
XLSX=agent_space_xpu/test_class_classification_merged.xlsx

# Stage 1 (long-running; use tmux as shown above)
python3 "$SKILL/scripts/collect_case_counts.py" \
    --xlsx "$XLSX" --test-root "$HOME/daisyden/upstream/test" --timeout 150

# Stage 2
python3 "$SKILL/scripts/summarize_merged.py" \
    --xlsx "$XLSX" --case-counts agent_space_xpu/case_counts.tsv
```

## Verify

1. `case_count` column present and populated for most rows:
   ```bash
   python3 -c "import openpyxl; ws=openpyxl.load_workbook('$XLSX').active; \
   hdr=[ws.cell(1,c).value for c in range(1,ws.max_column+1)]; \
   cc=hdr.index('case_count')+1; \
   have=sum(1 for r in range(2,ws.max_row+1) if isinstance(ws.cell(r,cc).value,int)); \
   print('populated', have, 'of', ws.max_row-1)"
   ```
2. `summary` sheet exists; block #1 category counts sum to the total row count,
   block #2 to the priority-subset row count.
3. Every `xpu_enabled` True+False pair sums to that block's `device_agnostic` total.

## Reference results

`results/` holds a completed run for reference (regenerate to refresh):

| File | Description |
|------|-------------|
| `results/test_class_classification_merged.xlsx` | Merged table with the `case_count` column and the `summary` sheet populated |
| `results/case_counts.tsv` | Stage-1 per-`(file, class)` case counts (3141 rows, ~500k cases) |
| `results/case_counts_errors.tsv` | Files that failed collection (90: timeouts, `rc=2`, `rc=5`) |

## Notes

- Requires `openpyxl` and a `pytest`/`torch` install with the target accelerator.
  Collecting from the PyTorch source root fails (`No module named
  'torch.version'`) - the script deliberately runs collection from `/tmp`.
- Case totals only count classes present in the merged xlsx; device variants are
  merged under the generic class name. Files failing collection undercount their
  categories (notably device_specific for distributed/inductor). Re-collect the
  failing subset with an appropriate harness if exact case totals are required.
- Emit all artifacts into `agent_space_xpu/` (git-ignored scratch space).
