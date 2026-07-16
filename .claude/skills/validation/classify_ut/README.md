# classify-ut

Batch-classify XPU unit-test cases from an Excel sheet using a cascaded
decision flow, and (with your approval) file GitHub issues for genuine
failures. This README is a quick-start entry point; the full, authoritative
procedure lives in [`SKILL.md`](./SKILL.md) in this same directory - read it
before making any changes to the workflow.

## Quick start

Run it via the `/ulw-loop` command (or any agent invocation), pointing at the
skill file and giving it your Excel sheet, filter, and environment:

```
/ulw-loop use .opencode/skills/validation/classify_ut/SKILL.md classify <excel file> sheet 'XPU skipped only Inductor' and Reason is blank cases, conda_env is cond_env1 pytorch_folder is ~/pytorch
```

Concrete example:

```
/ulw-loop use .opencode/skills/validation/classify_ut/SKILL.md classify /home/user/data/release-2.13.xlsx sheet 'XPU skipped only Inductor' and Reason is blank cases, conda_env is pytorch_opencode_env pytorch_folder is ~/daisy_pytorch
```

### Command parameters

| Placeholder | Meaning | Default if omitted |
|---|---|---|
| `<excel file>` | Absolute path to the `.xlsx` sheet to classify | required |
| `sheet '<name>'` | Sheet name inside the workbook | first sheet |
| `Reason is blank cases` (or other filter) | Which rows to process - typically "Reason is blank", optionally narrowed further (e.g. "and testfile_cuda is test/inductor/foo.py") | all rows needing classification |
| `conda_env is <name>` | Conda environment with PyTorch (XPU) installed | `pytorch_opencode_env` |
| `pytorch_folder is <path>` | Local `pytorch/pytorch` checkout the tests live in | `$HOME/daisy_pytorch` |

If the conda env or pytorch checkout don't exist yet, the `prepare-env` skill
bootstraps them automatically via `scripts/setup_env.sh` (creates the env with
`python=3.10`, installs the XPU nightly wheel + `pytest-timeout`, clones and
pins the checkout). See `SKILL.md`'s Execution Step 0.

## What it does

For every matching row, `classify-ut` runs a decision cascade and assigns a
`Reason`:

1. **Gate 0 - Local test** (blank `status_xpu` only): passes locally ->
   `Local Passed`.
2. **Gate 1 - Not-target**: CUDA-only / out-of-scope for XPU -> `Not Applicable`.
3. **Gate 2 - Community change**: test removed/renamed upstream ->
   `Community Change`.
4. **Gate 3 - Blank status**: `status_xpu` still blank -> `To be enabled`.
5. **Gate 4 - Known issue**: already tracked in `pytorch/pytorch` or
   `intel/torch-xpu-ops` -> `Failures (xpu broken)` / `Feature gap` /
   `To be enabled`.
6. **Gate 5 - Enablement** (skipped tests only, no known issue): feasible to
   enable -> `To be enabled`; otherwise -> `Submit Issue`.
7. No known issue (non-skipped) -> `Submit Issue`.

Rows sharing the same test class and a similar error message are deduplicated
and classified together (one cascade run, reused across all matching rows) -
see `SKILL.md`'s Phase 1/Step 0 for the exact similarity rule.

`Submit Issue` rows are **never filed automatically**. They are drafted,
checked against existing issues for duplicates, and presented to you for
explicit per-item approval before anything is created on GitHub.

## Output

- A new `agent` sheet is written to a standalone accumulator file (e.g.
  `agent_results.xlsx`), never to the original input file. Re-running the
  same accumulator merges in new rows without disturbing prior ones.
- Every row gets `Analyzed`, `Reason`, `DetailReason`, `ReuseSource`,
  `Confidence`.
- Full audit trail (session log, per-gate JSON, pytest logs, issue drafts) is
  written to `agent_space/` at the workspace root (git-ignored).

## Prerequisites

- Python 3 with `openpyxl` and `pytest-timeout` installed in the target conda
  env (installed automatically by `scripts/setup_env.sh`).
- `gh` CLI authenticated, for known-issue search and (approved) issue filing.

## Reference

- [`SKILL.md`](./SKILL.md) - the full, versioned procedure. This README
  summarizes it; `SKILL.md` is authoritative.
- `../scripts/` - `extract_tasks.py`, `run_blank_test.py`, `write_results.py`,
  `setup_env.sh`, `attach_not_target_evidence.py`.
- `../check_not_target_feature/`, `../check_community_change/`,
  `../check_known_issue/`, `../check_enablement_feasibility/` - the four
  cascade-gate sub-skills.
- `../prepare_env/` - session environment setup.
- `../ut_follow_up/` - the confirm-gated PR/issue filing agent for
  `Submit Issue` rows.

Note: the classify-ut workflow lives entirely under
`.opencode/skills/validation/`. The top-level `README.md` at the repository
root describes the unrelated torch-xpu-ops PyTorch operator project itself
and was left untouched by this file.
