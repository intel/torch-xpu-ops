# classify_ut preparation (OPTIONAL)

This skill follows agent-guidelines AND extends it with optional preparation steps for UT
classification. Always apply agent-guidelines rules including the mandatory post-write commit
protocol.

## Steps in this skill

1. **Extract Target Sheet** (run first) — copy the single target sheet out of the large
   status workbook into its own small `.xlsx`. Every later preparation step AND the entire
   parent classification workflow then operate on this extracted file, never the original.
2. **Environment Setup** (formerly "Step -1") — OPTIONAL.
3. **Local Pre-Screen** (formerly "Step 0") — OPTIONAL.

Step 1 SHOULD be run first whenever the source workbook is large or multi-sheet (the normal
case): the weekly status workbooks are too big to re-open and re-save in every phase. Steps 2
and 3 are **preparation aids, not mandatory steps** — by default they are **NOT run**. Invoke
them only when the user explicitly asks to refresh the environment or to bulk pre-screen a
sheet, or when a verdict genuinely depends on a fresh source-aligned local run.

The parent `classify_ut/SKILL.md` workflow can proceed (source inspection, known-issue
search, status-specific subskills) without running steps 2-3. When those steps ARE run,
their artifacts (provenance JSON, `local_result` column, captured logs) become authoritative
inputs that downstream classification should cite.

## Extract Target Sheet (run first)

The weekly UT status workbooks carry many sheets and thousands of rows, so re-opening and
re-saving the whole file in every phase is slow. The first preparation action copies ONLY the
target sheet into a new, small workbook. All subsequent phases and steps — environment setup,
local pre-screen, and the full parent classification workflow — operate on this extracted file.

### Extractor script (authoritative)

```
python3 .opencode/skills/classify_ut/scripts/extract_target_sheet.py \
    <original_status_workbook.xlsx> --sheet "<Target Sheet Name>"
```

The script:

- Loads the original workbook once, deletes every sheet except the target, and saves the
  result to `<original_stem>.<sheet_slug>.xlsx` next to the original (override with `--out`).
- Preserves the target sheet exactly (cell values, fills, column widths).
- Never modifies the original workbook.
- Errors out, listing available sheet names, if the target sheet does not exist.
- Prints the extracted file path plus its row/column counts.

### Downstream contract (MANDATORY once extracted)

- The extracted single-sheet `.xlsx` is THE target workbook for every later phase and step.
  Use its path wherever a workbook argument is required (local pre-screen below, and the
  parent workflow's open/classify/save steps).
- Do NOT re-open the original large workbook for classification; keep it only as the
  untouched source of record.
- The final `.agent.xlsx` output is derived from the extracted file (still "do not modify the
  original"), so it likewise contains only the target sheet.

## When to run preparation

Run **Environment Setup** when:
- The user explicitly asks to update/align the environment.
- A `Local Passed` verdict (or any local-run evidence) is going to be produced and you need
  the wheel + source checkouts aligned to the same commit.

Run **Local Pre-Screen** when:
- The user explicitly asks to bulk pre-screen a sheet's blank-`Reason` rows.
- You want a fast double-check of many rows before per-row analysis.

If neither condition holds, skip this skill entirely.

## Environment Setup (formerly Step -1)

When run, the active Python environment is aligned with the PyTorch XPU nightly **and the
local pytorch + torch-xpu-ops source checkouts are aligned to the exact commits that produced
the installed wheel**. This is the only way to guarantee that source analysis, local
pre-screen, and DetailReason citations all reflect the same revision of the code.

The env-update helper takes no arguments — the PyPI XPU nightly index is the single source of
truth.

### Env-update helper (authoritative)

```
bash .opencode/skills/classify_ut/scripts/update_env_from_nightly.sh
```

The helper performs, in order:

1. Activates `pytorch_opencode_env` (`ENV_NAME` overrides) and runs all `pip` commands
   from `/tmp` so a workspace torch source checkout cannot shadow the installed wheel.
2. `pip install --pre --upgrade torch pytorch-triton-xpu --index-url
   https://download.pytorch.org/whl/nightly/xpu` (`INDEX_URL` / `PACKAGES` override the
   defaults; `PIP_FORCE_REINSTALL=1` adds `--force-reinstall --no-deps` when the env
   already has a newer locally built torch).
3. Imports torch from `/tmp` to read `torch.__version__`, `torch.version.git_version`,
   and `torch.xpu.is_available()`.
4. **Aligns `${PYTORCH_SRC:-/home/daisyden/upstream/pytorch}` to
   `torch.version.git_version`** — the pytorch commit baked into the installed wheel.
   If the working tree is dirty, the helper auto-stashes (`git stash push
   --include-untracked` with a `classify_ut auto-stash <UTC>` message) and prints the
   `git -C <repo> stash pop` restore hint. Then `git fetch origin` (if the SHA isn't
   local) and `git checkout --detach <sha>`.
5. **Aligns `${XPU_OPS_SRC:-${PYTORCH_SRC}/third_party/torch-xpu-ops}` to the SHA in
   `${PYTORCH_SRC}/third_party/xpu.txt`** — the torch-xpu-ops commit pinned by the
   pytorch revision just checked out. Same stash-then-detach flow as step 4.
6. Writes a session provenance record to
   `~/.claude_classify_ut_session_provenance.json` (`PROVENANCE_FILE` overrides).

### Manual alternative (no helper script)

If the helper is unavailable, the same alignment can be done manually:

```bash
# Select and update PyTorch source to main. Use the user's requested checkout if provided.
export PYTORCH_SRC="${PYTORCH_SRC:-$HOME/upstream/pytorch}"
cd "$PYTORCH_SRC"
git fetch origin main && git checkout main && git pull --rebase origin main

# Update torch-xpu-ops submodule to main
cd third_party/torch-xpu-ops
git fetch origin main && git checkout main && git pull --rebase origin main
cd ../..
```

```bash
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}"

# Install latest nightly torch for XPU
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

# Install latest nightly triton-xpu
pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu --pre pytorch-triton-xpu --dest /tmp/opencode/triton_whl
pip install --root-user-action=ignore /tmp/opencode/triton_whl/pytorch_triton_xpu-*.whl
```

```bash
python -c "import torch; print(f'torch={torch.__version__}, xpu_available={torch.xpu.is_available()}')"
python -c "import triton; print(f'triton={triton.__version__}')"
```

If the environment is already up-to-date from a recent run in the same session, skip this setup.

### Provenance contract

When Environment Setup is run, the provenance file SHOULD be referenced by the agent in
session notes and may be cited in `DetailReason` for `Local Passed` verdicts. It encodes:

- `pypi_index_url`, `packages_installed`, `versions` — exactly which torch +
  pytorch-triton-xpu bits were used.
- `source_alignment.pytorch_sha` — value of `torch.version.git_version`, also the HEAD of
  the local pytorch checkout after alignment.
- `source_alignment.torch_xpu_ops_sha` — content of `third_party/xpu.txt` at that pytorch
  commit, also the HEAD of the local torch-xpu-ops checkout after alignment.
- `xpu_available` — SHOULD be `true` before any local-run evidence is produced.

### Anti-patterns

- Re-using a stale provenance file from a previous session when you DID choose to run
  preparation. If you rely on a fresh local run, re-run Environment Setup first — the
  nightly index advances daily, and `third_party/xpu.txt` may have moved with it.
- Manually `pip install`-ing torch without then aligning the pytorch + torch-xpu-ops
  source trees. Source analysis against a mismatched checkout will cite code that is
  not in the wheel under test.
- Discarding the auto-stash without inspecting it. The stash is the only record of
  pre-session local edits; restore it with `git -C <repo> stash pop` after the
  classification session ends if those edits were intentional.

## Local Pre-Screen (formerly Step 0)

When run, this performs a bulk local pre-screen of every blank-`Reason` row. This is a
**double-check** of the workbook data — the script does NOT modify any test code; it only
runs the corresponding XPU test and records what it observed.

### Applicable sheets

- `XPU skipped only Non-Inductor`
- `XPU skipped only Inductor`

### Runner script (authoritative)

Run:

```
source /home/daisyden/miniforge3/bin/activate pytorch_opencode_env
python .opencode/skills/classify_ut/scripts/run_blank_local_prescreen.py <workbook.xlsx>
```

The script:

- Adds a `local_result` column adjacent to `DetailReason` if missing.
- For each blank-`Reason` row, resolves the test file by trying
  `$PYTORCH_SRC/<testfile_cuda>` (the upstream pytorch test path, with the full
  subpath preserved) first, then falling back to
  `third_party/torch-xpu-ops/test/xpu/[<subdir>/]<basename>_xpu.py` only when
  the upstream file does not exist. The XPU mirror filename has an `_xpu`
  suffix appended to the basename. Pytorch-first order prevents basename
  collisions across `test/` subdirectories (e.g. `test/export/test_sparse.py`
  vs `test/test_sparse.py`).
- Derives the pytest node id from `classname_xpu` / `name_xpu` when populated, otherwise
  swaps the `CUDA` suffix on `classname_cuda` for `XPU` and uses `name_cuda`.
- Runs `pytest -v` with a 60-second per-test timeout (override via `PYTEST_PER_TEST_TIMEOUT`),
  capturing the full output to `${LOG_DIR:-/tmp/opencode/<workbook_basename>_local_verify}/`.
- Writes `local_result = <STATUS>;<absolute log path>` with `<STATUS>` in
  `{PASS, FAIL, ERROR, TIMEOUT, SKIP, SEGFAULT}`.
- On `PASS`, also writes `Reason = "Local Passed"` and
  `DetailReason = "[Confidence: HIGH] Local pre-screen PASS on pytorch_opencode_env (<location>); log: <path>"`,
  and marks all touched cells blue (`ADD8E6`). `Reason TBD` is never modified.
- Is resume-safe: rows whose `local_result` cell is already populated are skipped on re-run.
- Backs the workbook up to `<workbook>.bak_step0` before the first save and checkpoints
  every 25 rows.

### Downstream contract (only when Local Pre-Screen was run)

#### PASS is a TERMINAL verdict — no further analysis

When `local_result` starts with `PASS;`, the row is **fully classified by the pre-screen
alone**. The agent MUST:

- Treat `Reason = "Local Passed"` and the cited log path as the final answer.
- SKIP all status-specific subskills (`blank/`, `failed/`, `skipped/`) for this row.
- SKIP source inspection, sibling-class analysis, issue/PR searches, and
  `instantiate_device_type_tests` mapping for this row.
- NOT re-run the test, NOT delegate, NOT consult Oracle, NOT modify `Reason`,
  `DetailReason`, `Reason TBD`, or the cell fill for this row.

A green local run on the source-aligned env is the strongest possible evidence the
workbook can carry. Adding further analysis on top of a PASS cannot improve the verdict
and risks overwriting it. **Stop at PASS.**

The only exception is the policy-driven non-PASS verdict:

- **P3 JIT**: even if the JIT test PASSes locally, the verdict remains `Not applicable`
  by owner-team scope. The pre-screen runner does not run JIT clusters (they are excluded
  upstream of the pre-screen); if a JIT row somehow reaches PASS here, override to
  `Not applicable` with the owner-team evidence.

#### Non-PASS values are EVIDENCE, not verdicts

For every blank-`Reason` row whose `local_result` is NOT `PASS`, the value (`FAIL`,
`ERROR`, `TIMEOUT`, `SKIP`, or `SEGFAULT`) plus its log path is authoritative pre-screen
evidence and SHOULD be cited in the eventual `DetailReason` produced by the status-specific
subskill. These rows still require the normal further analysis.

### Anti-patterns

- Treating a `local_result` of `FAIL` / `ERROR` / `TIMEOUT` / `SKIP` / `SEGFAULT` as a
  classification verdict. Those values are evidence, not a `Reason`.
- Performing **any** further analysis (source read, sibling-class mapping, issue search,
  Oracle consult, subskill routing) on a row whose `local_result` starts with `PASS;`.
  PASS is terminal.
- Re-running a test whose `local_result` is already `PASS`. Use the existing log; the
  classification is final.
- Writing `Reason = "Local Passed"` without citing the captured log path.
- Overwriting an existing `Reason = "Local Passed"` with any other verdict (including
  `Not applicable`) on the basis of post-hoc source analysis. The only legitimate
  override is P3 JIT, and the runner is expected to exclude JIT clusters before they
  reach PASS.
