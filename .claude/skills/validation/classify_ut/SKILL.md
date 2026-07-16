---
name: classify-ut
description: Batch classify UT test cases from an Excel sheet by running a cascaded decision flow (not_target → community_change → status_xpu → known_issue → enablement_analysis). Delegates each classification axis to a specialized skill. Outputs results to a new "agent" sheet.
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
1. Is it a **not-target feature**? → `Not Applicable` (the not-target check returns `false` for any missing file/method so the row falls through to Gate 2; it also deterministically backfills a tracking issue link into the evidence when one exists)
2. Does it have a **community change**? → `Community Change`
3. Is `status_xpu` blank? → `To be enabled`
4. Is there a **known issue**? → `Failures (xpu broken)` / `Feature gap` / `To be enabled`
5. **For skipped tests** (`status_xpu = "skipped"`) without a known issue: is XPU **enablement feasible**? → `To be enabled` (if feasible, with method documented) / `Submit Issue` (if not feasible)
6. No known issue (and not a skipped test) → `Submit Issue`

If a test is `Local Passed`, no further classification is needed — it skips Gates 1–5 entirely.

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
| `Reason` | Classification result: `Not Applicable`, `Community Change`, `To be enabled`, `Failures (xpu broken)`, `Feature gap`, `Submit Issue`, or `Submit PR`. `Submit PR` is set in Phase 4 when the ut-follow-up agent fixes a test-code bug and submits a PR; `Submit Issue` when an issue is filed (or submission was skipped). |
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
| `extract_tasks.py` | Reads the Excel, deduplicates rows, outputs `tasks.json` | `.opencode/skills/validation/scripts/extract_tasks.py` |
| `run_blank_test.py` | Runs `status_xpu` blank test cases locally; marks passing tests as `Local Passed` | `.opencode/skills/validation/scripts/run_blank_test.py` |
| `attach_not_target_evidence.py` | Deterministic bounded lookup that finds a tracking issue (closed `not_target` / open `skipped`) for a `Not Applicable` test and returns its link | `.opencode/skills/validation/scripts/attach_not_target_evidence.py` |
| `write_results.py` | Takes classification results and writes the `"agent"` sheet | `.opencode/skills/validation/scripts/write_results.py` |

Run all script commands from the **workspace root** (the repository checkout directory). Do not `cd` into subdirectories.

## Workflow

The classification proceeds in three phases using the scripts for Excel I/O and deduplication.

### Phase 1: Read and Deduplicate

Run `extract_tasks.py` to read the Excel, deduplicate, and output `tasks.json`:

```bash
python3 .opencode/skills/validation/scripts/extract_tasks.py <excel_path> [sheet_name] > tasks.json
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
# Run from workspace root with a conda environment that has PyTorch installed.
# --pytorch-root points at the pytorch checkout the tests live in (the folder
# prepared by setup_env.sh); it defaults to $PYTORCH_FOLDER or the cwd.
mkdir -p agent_space/test_logs
python3 .opencode/skills/validation/scripts/run_blank_test.py tasks.json \
    --output results.json --log-dir agent_space/test_logs --env <conda_env_name> --timeout 600 \
    --pytorch-root <pytorch_folder>
```

**Behavior**:
- Only tasks with blank `status_xpu` are run.
- Tests are grouped by file and run via `pytest`.
- **Per-case timeout**: each pytest invocation runs with `--timeout 600 --timeout_method=thread` (the `pytest-timeout` plugin, matching CI's own convention - see AGENTS.md), so a single hanging test is interrupted by pytest itself rather than relying solely on an external process kill. The outer subprocess watchdog (`--timeout` + a small buffer) is a safety net only. Requires `pytest-timeout` to be installed in the target env (installed by `setup_env.sh`); if missing, the script fails fast with an install instruction rather than installing it itself.
- A test is `Local Passed` only if `pytest` reports `1 passed` with no `FAILED` output.
- `DetailReason` is set to the local PyTorch version (e.g. `"Local test PASSED (torch 2.13.0+xpu)"`).
- Full pytest output is dumped to `agent_space/test_logs/<testfile_safe_name>.log`.
- A `run_summary.log` is written to `agent_space/test_logs/` with per-file results.
- The output `results.json` includes `Local Passed` entries ready for `write_results.py`.
- Tests that do not pass locally remain in the output for cascade processing.

**If `Local Passed`**:
- `Reason = "Local Passed"`
- `DetailReason = "Local test PASSED (torch <version>)"`
- Stop. Do not run Gates 1–4 for this row.

**If NOT `Local Passed`** → proceed to Phase 2 (cascade).

### Phase 3: Classify Each Task

For each row not yet classified (not `Local Passed`), run the decision cascade. Each check is a **hard gate** — if the condition is met, classification stops and the result is recorded.

> **IMPORTANT — Cascade is mandatory for every unclassified task**: Tasks not marked `Local Passed` must still run the cascade. The `Reason` and `DetailReason` fields in the `tasks.json` input may contain pre-populated values from a prior run. **These are NOT valid classification results.** The orchestrator MUST run the full cascade (Gates 1-4 via subagents) for every task in the `tasks` array. Only rows in `already_resolved` (Analyzed=TRUE or deduplicated) may bypass the cascade via Step 0 reuse. The `--filter-reason`/`--filter-detailreason` flags on `extract_tasks.py` are post-hoc row-selection only — never a classification shortcut; rows they select still require the full cascade.

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

#### Gates 1, 2, 4, 5: Delegated Checks

Gates 1, 2, 4, and 5 each delegate to a subskill via a common template, then map
its verdict to a `Reason`/`DetailReason`. **Gate 3 is a direct check (no
delegation) — see below.** Run the gates in strict order; each is a hard gate
(stop the cascade the moment one fires). Gate 5 is entered only from Gate 4.

**Canonical delegation template** (fill in the per-gate `load_skills`, `identifiers`, and `extra prompt` from the table):

```python
task(
    subagent_type="explore",
    load_skills=[<skill>],
    description=f"<gate> check: {name_xpu}",
    prompt=f"<identifiers>. Error message: {message_xpu}. <extra prompt> "
           f"Return the subskill's verdict JSON."
)
```

| Gate | `load_skills` | Identifiers to pass | Extra prompt |
|------|---------------|---------------------|--------------|
| 1 Not Target | `check-not-target-feature` | `{name_xpu}` in `{classname_xpu}` (`{testfile_xpu}`) | `PYTORCH_SRC={pytorch_folder}.` |
| 2 Community Change | `check-community-change` | `{name_cuda}` in `{classname_cuda}` (`{testfile_cuda}`), device=cuda | `PYTORCH_SRC={pytorch_folder}. conda_env={conda_env}. If CUDA is available use --collect-only (Path A, via conda run -n {conda_env}); else use source inspection (Path B).` |
| 4 Known Issue | `check-known-issue` | `{name_xpu}` in `{classname_xpu}` (`{testfile_xpu}`), CUDA source `{name_cuda}` in `{classname_cuda}` (`{testfile_cuda}`) | (none) |
| 5 Enablement | `check-enablement-feasibility` | `{name_xpu}` in `{classname_xpu}` (`{testfile_xpu}`), CUDA source `{name_cuda}` in `{classname_cuda}` (`{testfile_cuda}`) | `PYTORCH_SRC={pytorch_folder}. conda_env={conda_env}. status_xpu: {status_xpu}. Return enablement verdict, skip mechanism, and required changes.` |

Gates 1 and 4 do no environment-dependent execution (static source/git
inspection and `gh` issue search only), so they are not passed `conda_env`.
Gates 2 and 5 may run `pytest`/`import torch` in the caller's env, so they
receive `conda_env={conda_env}` and must invoke Python via
`conda run -n {conda_env} ...` rather than a bare `python3`.

**Verdict mapping** (on the flagged verdict, set `Reason`/`DetailReason` and STOP the cascade; otherwise fall through as shown):

| Gate | Verdict field | On TRUE → `Reason` / `DetailReason` | On FALSE → next |
|------|---------------|--------------------------------------|-----------------|
| 1 | `is_not_target` | `Not Applicable` / `"<reasoning> (Evidence: <evidence joined>)"` | Gate 2 |
| 2 | `community_change` | `Community Change` / `classification.detail_reason` | Gate 3 |
| 4 | `has_known_issue` | use subskill's `classification.Reason` and `classification.DetailReason` verbatim | see Gate 4 note |
| 5 | `enablement_feasible` | `To be enabled` / `classification.DetailReason` | `Submit Issue` / `classification.DetailReason` |

**Gate 1 note**: the `check-not-target-feature` skill's Step 6 deterministically backfills a tracking issue link (e.g. `intel/torch-xpu-ops#4179`) into `evidence` when one exists, so `Not Applicable` verdicts carry a supporting link without a separate orchestrator step.

**Gate 4 — MANDATORY, no shortcut**: Every row that reaches Gate 4 (`not_target == False`, `community_change == False`, `status_xpu` non-blank) MUST have the `check-known-issue` delegation actually executed and its output recorded in `agent_space/gate4_known_issue.json`. You may NOT infer `has_known_issue = False` from an earlier gate's reasoning, prior triage text, or "it looks like a failure" — the only valid source is a real `check-known-issue` result for that row. Applying the `has_known_issue == False` default without a delegation is a hard violation. Before Phase 5, assert every such row has an entry in `gate4_known_issue.json`; if any is missing, run Gate 4 for it before writing. When `has_known_issue == False`: if `status_xpu == "skipped"` → proceed to **Gate 5**; otherwise (`"failed"`/other non-blank) → `Reason = "Submit Issue"`, `DetailReason = "No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`, stop.

**Gate 5 — entry, nuance, logging**: Enter only when `has_known_issue == False` AND `status_xpu == "skipped"`. If the subskill output contains verification evidence (e.g. "Verified: test passes on XPU after removing @skipXPU"), include it in `DetailReason`. **"Verified passing" does NOT make it `Local Passed`** — `Local Passed` is only for tests that pass with NO code changes; any test needing a change (removing a skip, editing the harness) to pass is `To be enabled` with the verification documented in `DetailReason`. Save the subskill output to `agent_space/gate5_enablement.json` (appended per batch) and log the delegation to `agent_space/session_log.txt`.

---

#### Gate 3: status_xpu Blank (direct check, no delegation)

- **If `status_xpu` is blank/empty**: `Reason = "To be enabled"`, `DetailReason = "status_xpu is blank — no known status. Awaiting enablement."`, stop.
- **If `status_xpu` is NOT blank** → proceed to Gate 4.

---

### Phase 4: Hand Off `Submit Issue` Rows to the ut-follow-up Agent

Run this **before** the Excel write (Phase 5) so any returned PR/issue links
are written into `results.json` and land in the output in a single pass.

Collect every classified row whose `Reason == "Submit Issue"`. If there are
none, skip this phase and go to Phase 5.

If there is at least one, invoke the dedicated `ut-follow-up` agent. The
agent attempts a test-code fix (submitted as a **PR**) or files an **issue**,
and **requires explicit user confirmation before creating any PR or issue** —
classify-ut never files anything itself and never approves on the user's behalf.

```python
task(
    subagent_type="ut-follow-up",
    description=f"Submit handoff for {len(submit_rows)} Submit Issue rows",
    prompt=(
        "Input mode: from classify-ut. "
        f"Reuse this session's environment (do NOT bootstrap): "
        f"conda_env={conda_env}, pytorch_folder={pytorch_folder}. "
        "The following rows were classified as "
        "'Submit Issue' (no known issue, not enablable). For each row, either "
        "(a) fix a genuine test-code bug and submit the fix as a PR, or "
        "(b) file an issue to intel/torch-xpu-ops for an infra/backend failure. "
        "Group rows that share an error signature; cross-reference existing "
        "issues/PRs to avoid duplicates; present every PR/issue draft for the "
        "user's per-item approval and do NOT submit without it. "
        "Return the JSON array defined by the Return Contract (one entry per "
        "row with outcome=pr|issue|skipped and the url).\n\n"
        f"Rows (JSON): {submit_rows_json}"
    )
)
```

`conda_env` and `pytorch_folder` are the session values established in the
Execution preamble (Step 0); passing them lets the ut-follow-up agent reuse
the exact same environment and checkout instead of bootstrapping its own.

`submit_rows_json` is the list of Submit Issue rows, each carrying
`name_cuda`, `classname_cuda`, `testfile_cuda`, `name_xpu`, `classname_xpu`,
`testfile_xpu`, `message_xpu`, and `status_xpu`. The CUDA identity fields are
included so the agent can key its return entries back to the exact rows.

**Apply the returned results to `results.json`** (matched by CUDA identity):

| Agent `outcome` | `Reason` | `DetailReason` |
|---|---|---|
| `pr` | `Submit PR` | `"Fix submitted: <url>"` (the PR link) |
| `issue` | `Submit Issue` | `"Issue submitted: <url>"` (the issue link) |
| `skipped` | `Submit Issue` (unchanged) | keep prior DetailReason; append `"(submission skipped: <summary>)"` |

`write_results.py` auto-computes `Confidence = High` when `DetailReason`
contains a GitHub issue/PR URL, so filed rows are upgraded automatically.

**Logging**: Append the handoff to `agent_space/session_log.txt`
(`subagent: ut-follow-up | task: submit N rows | file_refs: <urls>`) and
save the agent's returned JSON array to
`agent_space/phase5_submit_issues.json`.

This phase is **draft-and-confirm only**: the agent must not create any PR or
issue without the user approving each draft. classify-ut's role ends at routing
the rows, recording the outcome links, and updating `Reason`/`DetailReason`.

---

## Phase 5: Write Results to Excel

`write_results.py` has two modes. Choose based on whether the output file already exists.

**BUILD (first run — output file does not exist yet):**

```bash
python3 .opencode/skills/validation/scripts/write_results.py <excel_path> results.json [sheet_name] --output_sheet=agent --output-excel=agent_results.xlsx
```

Reads the original input sheet, appends columns `Analyzed`, `Reason`, `DetailReason`, `ReuseSource`, `Confidence`, and writes a fresh `agent` sheet to the standalone `--output-excel` file. Per row: classified rows get `Analyzed = TRUE` with their `Reason`/`DetailReason`/`ReuseSource`; rows already `Analyzed = TRUE` in the *input* sheet are carried over; everything else gets `Analyzed = FALSE`.

**MERGE (incremental run — output file already exists, e.g. an accumulator):**

```bash
python3 .opencode/skills/validation/scripts/write_results.py --merge results.json --output_sheet=agent --output-excel=agent_results.xlsx
```

Updates the existing `--output-excel` file in place, touching ONLY the rows present in `results.json` (matched by CUDA identity: `testfile_cuda`/`classname_cuda`/`name_cuda`, falling back to `name_cuda` alone). Every other row — including rows analyzed by previous runs — is left untouched. Rows already `Analyzed = TRUE` are skipped unless `--force` is passed.

Both modes **auto-compute `Confidence`** (`High` if `DetailReason` matches exact-evidence patterns — GitHub issue/PR URL, commit hash, `#PR` reference — else `Medium`).

**Safety:**
- BUILD refuses to overwrite an existing `--output-excel` whose `agent` sheet contains `Analyzed = TRUE` rows that are NOT in `results.json` (a fresh build would discard them). It exits non-zero and tells you to use `--merge` (preserve prior rows) or `--force` (overwrite anyway).
- Any in-place write (MERGE, or BUILD `--force` over an existing file) first writes a `<file>.bak` backup.
- If a `results.json` entry matches multiple sheet rows by name (ambiguous, e.g. `results.json` lacks the full CUDA identity), those rows are NOT written and the script exits non-zero. To avoid this, ensure each `results.json` entry carries `testfile_cuda`, `classname_cuda`, and `name_cuda`.


## Execution

Load this skill to orchestrate the full classification. Run the phases in order;
each command and its details live in the referenced Workflow/Phase section above
(do not duplicate them here).

0. **Session setup — delegate to the `prepare-env` skill (run once).** Establish
   `conda_env` and `pytorch_folder` by dispatching the `prepare-env` skill as a
   subagent; do NOT inline the bootstrap here. It resolves the two values
   (defaults `pytorch_opencode_env` and `$HOME/daisy_pytorch`), verifies the
   conda env imports torch with XPU and that `<pytorch_folder>/.git` exists, and
   bootstraps once via `setup_env.sh` when either is missing/broken.

   ```python
   task(
       subagent_type="explore",
       load_skills=["prepare-env"],
       description="Session setup: establish conda env + pytorch checkout",
       prompt=(
           "Prepare the XPU UT classification environment. "
           f"conda_env={conda_env}, pytorch_folder={pytorch_folder}. "
           "Verify the conda env imports torch with XPU and that the pytorch "
           "folder is a git checkout; bootstrap via setup_env.sh if either is "
           "missing or broken. Return the prepare-env output JSON "
           "(conda_env, pytorch_folder, torch_version, xpu_available, "
           "bootstrapped, status)."
       )
   )
   ```

   Read `conda_env` and `pytorch_folder` from the request first, then pass them
   in the prompt above (or omit for the defaults). If the returned `status` is
   `fatal`, log `[FATAL]` to `agent_space/session_log.txt` and stop the session.
   Otherwise take the returned `conda_env`/`pytorch_folder` as the session
   values, `export PYTORCH_FOLDER=<pytorch_folder>`, reuse both for every step,
   and pass them to the ut-follow-up agent in Phase 4.
1. **Phase 1** — `extract_tasks.py` → `tasks.json` (see Phase 1).
2. **Phase 2 / Gate 0** — `run_blank_test.py --env <conda_env> --pytorch-root <pytorch_folder>`; passing tests get `Local Passed` and skip the cascade (see Phase 2).
3. **Phase 3 cascade** — for each non-`Local Passed` row, run Step 0 reuse then Gates 1→2→3→4→5 in strict order (see Phase 3). Pass XPU identifiers to Gates 1/4/5 (with `PYTORCH_SRC=<pytorch_folder>` for Gates 1/2/5) and CUDA identifiers to Gate 2. Also pass `conda_env=<conda_env>` to Gates 2 and 5 (they may run `import torch` / `pytest`); Gates 1 and 4 need no env (static inspection / `gh` only).
4. Accumulate all results (already_resolved + Local Passed + newly classified) into `results.json`.
5. **Phase 4 — Submit handoff (BEFORE the Excel write)** — route any `Reason == "Submit Issue"` rows to the `ut-follow-up` agent and map returned outcomes back into `results.json` (see Phase 4).
6. **Phase 5 — Write Results** — BUILD if the `--output-excel` accumulator does not exist yet, else MERGE in place (see Phase 5).
7. Report summary statistics: rows total, deduplicated, local passed, classified per Reason category (including `Submit PR` and `Submit Issue`).

## Constraints

### Logging & Audit Trail (MANDATORY)

Every step of the classification pipeline MUST produce persistent logs. No silent steps. All logs go into `agent_space/` (the git-ignored scratch directory at the repo root).

0. **Session log in `agent_space/session_log.txt`**: Before any work begins, create a timestamped session log file at `agent_space/session_log.txt`. Append to it throughout the session. Each entry must follow this format:
   ```
   [YYYY-MM-DD HH:MM:SS] <step description> | subagent: <skill_name> | task: <brief description> | file_refs: <comma-sep file/issue refs>
   ```
   Example entries:
   ```
   [2026-06-25 10:30:00] Gate 4 known-issue check | subagent: check-known-issue | task: test_comprehensive_nn_functional_adaptive_max_pool1d_xpu_bfloat16 | file_refs: test_decomp_xpu.py, intel/torch-xpu-ops#3890
   [2026-06-25 10:35:00] Gate 1 not-target check | subagent: check-not-target-feature | task: test_streams_xpu | file_refs: test_streams_xpu.py
   [2026-06-25 11:00:00] Gate 0 local test run | subagent: N/A (script) | task: run_blank_test.py --output results.json | file_refs: test_logs/test_decomp_xpu.log
   [2026-06-25 11:30:00] Phase 4 write results | subagent: N/A (script) | task: write_results.py --merge | file_refs: agent_results.xlsx
   ```
   **Every subagent invocation MUST be logged** — this is the audit trail for what was called and why.

1. **Gate-specific log files**: Each gate MUST save results to a separate file:
   - `agent_space/prepare_env.json` — the `prepare-env` skill's returned output JSON (`conda_env`, `pytorch_folder`, `torch_version`, `xpu_available`, `bootstrapped`, `status`) for Session setup (Step 0)
   - `agent_space/gate0_local_test.log` — pytest output for Gate 0 local test runs (appended per batch)
   - `agent_space/gate1_not_target.json` — combined results of all not-target checks
   - `agent_space/gate2_community_change.json` — combined results of all community change checks
   - `agent_space/gate4_known_issue.json` — combined results of all known issue searches
   - `agent_space/gate5_enablement.json` — combined results of all enablement analyses
   - `agent_space/na_evidence_backfill.json` — tracking-issue link lookups performed by the `check-not-target-feature` skill's Step 6 backfill for `Not Applicable` verdicts (matched issue link per row, or empty result)
   - `agent_space/phase1_dedup.json` — output of `extract_tasks.py`
   - `agent_space/phase4_write.log` — output of `write_results.py`
   - `agent_space/phase5_submit_issues.json` — ut-follow-up agent return array (per-row `outcome`=`pr`/`issue`/`skipped` with PR/issue URLs) for `Submit Issue` rows

2. **Detailed pytest logs for local tests (Gate 0)**: When running `run_blank_test.py`, ensure `--log-dir` points to `agent_space/test_logs/`. Every pytest run MUST produce a per-file log saved to this directory. These logs are the sole evidence for `Local Passed` classification. A `run_summary.log` MUST also be written.

3. **Subagent delegation log**: Every time a subagent is dispatched (especially for Gates 1, 2, 4), record the call immediately after dispatching:
   ```
   Delegated: <gate_name> | subagent_type: <type> | load_skills: [<skills>] | task_count: <N> | batch_key: <class_name or file_name>
   ```
   This is distinct from the session log — it's a real-time call log to track what was launched in parallel.

3a. **Session setup (`prepare-env`) MUST be logged**: The Step 0 `prepare-env`
   delegation is a subagent invocation and MUST be logged like any other. Before
   any pipeline work:
   - Append a `session_log.txt` entry, e.g.
     `[YYYY-MM-DD HH:MM:SS] Session setup | subagent: prepare-env | task: establish conda_env=<conda_env>, pytorch_folder=<pytorch_folder> (bootstrapped=<bool>) | file_refs: prepare_env.json`
   - Add the matching real-time delegation-log line:
     `Delegated: session-setup | subagent_type: explore | load_skills: [prepare-env] | task_count: 1 | batch_key: <conda_env>`
   - Save the skill's returned output JSON to `agent_space/prepare_env.json`.
   A `status = "fatal"` return MUST additionally be logged as
   `[FATAL] prepare-env: environment bootstrap failed — halting session` before
   stopping.

### Error Handling — Fatal Conditions

4. **Fatal: no `torch` package for local test run (Gate 0)**: If `run_blank_test.py` fails because `torch` is not importable (`ModuleNotFoundError: No module named 'torch'`, or `pytest` exits with `torch`-related import errors), do NOT attempt to install it or find workarounds. Immediately:
   - Append to `session_log.txt`: `[FATAL] Gate 0 local test: torch not available — halting session`
   - Print: `[FATAL] No torch package available for local test execution. This requires a conda environment with PyTorch installed. Stopping session. Please set up the environment and re-run.`
   - Stop all further work in this session.

5. **Fatal: subagent model unavailable**: If a subagent call (for any gate) fails with a model-related error (e.g., "model not found", "rate limit exhausted", "quota exceeded", "insufficient credits"), do NOT retry or fallback to manual processing. Immediately:
   - Append to `session_log.txt`: `[FATAL] Subagent <type> unavailable for <gate>: <error> — halting session`
   - Print: `[FATAL] Subagent <type> encountered a fatal error: <error>. Cannot continue classification without this capability. Stopping session. Please check subagent availability and re-run.`
   - Stop all further work in this session.

6. **Fatal: unrecoverable script error**: If `extract_tasks.py`, `run_blank_test.py`, or `write_results.py` exits with a non-zero code that indicates an unrecoverable problem (file not found, corrupted Excel, invalid schema — NOT a recoverable condition like "output already exists, use --merge"), do NOT attempt to manually patch the Excel or work around the error. Immediately:
   - Append to `session_log.txt`: `[FATAL] Script <name> failed: <error> — halting session`
   - Print the full error.
   - Stop all further work in this session.

7. **Non-fatal errors**: For transient errors (network timeout during `gh search issues`, pagination limit reached, individual test timeout during `run_blank_test.py`), log the issue to `agent_space/session_log.txt` with `[WARN]` prefix, and continue. Document the limitation in `DetailReason` when applicable (e.g. `"Known issue search skipped due to network error"`).

### Classification Constraints

8. **Gate 0 (Local Test) runs before the cascade**: Always run `run_blank_test.py` before Gate 1. Tests that pass locally (`Local Passed`) skip Gates 1–5 entirely. Only tests that fail/skip/time out locally proceed to the cascade. This saves classification effort for working tests.
9. **Gate order is strict**: Always check `not_target` before `community_change`, and `community_change` before `status_xpu`. Breaking the order can produce wrong classifications. A removed test file/method is handled by Gate 1 returning `false` (never `Not Applicable`) so the row falls through to Gate 2, whose deterministic file-existence fast path classifies it as `Community Change`.
10. **Deduplication is a speed optimization, not a classification shortcut**: Only reuse results from rows with the same class AND similar error message. Do not reuse across unrelated tests.
11. **Scripts handle all Excel I/O**: The agent should never manipulate Excel cells directly. Always use `.opencode/skills/validation/scripts/extract_tasks.py` and `.opencode/skills/validation/scripts/write_results.py`.
12. **Open issues with `skipped` label are treated as failures**: A skipped test is a broken test — classify as `Failures (xpu broken)`, not `To be enabled`.
13. **Closed issues with `not_target`/`wontfix` override Gate 1**: If Gate 1 said "not not-target" but Gate 4 finds a CLOSED `not_target` issue, the `not_target` label is authoritative. Reclassify as `Not Applicable`. Conversely, every `Not Applicable` verdict MUST carry a supporting GitHub issue link when one exists — the `check-not-target-feature` skill's Step 6 backfills this deterministically (via `attach_not_target_evidence.py`). Do not leave a `Not Applicable` row for a `skipped`/`failed` test without an issue link unless that deterministic `not_target`/`skipped` body search genuinely returned no match.
14. **`Submit Issue` rows are routed to the ut-follow-up agent (Phase 4), which may resolve them as a PR or an issue, confirm-gated**: classify-ut hands all `Submit Issue` rows to the `ut-follow-up` agent BEFORE the Excel write. The agent fixes test-code bugs as a **PR** or files an **issue**, and returns a link per row. classify-ut records the link in `DetailReason` and sets `Reason = "Submit PR"` (PR returned) or `Reason = "Submit Issue"` (issue returned, or submission skipped). classify-ut itself MUST NOT file anything, and the agent MUST NOT create any PR/issue without explicit per-item user approval. No silent auto-filing.
15. **Never modify the original sheet**: Always write to the output sheet name (default `"agent"`).
16. **All commands run from workspace root**: All script invocations use absolute (workspace-relative) paths. Do not `cd` into subdirectories before running scripts. Note: `run_blank_test.py` still runs pytest inside the pytorch checkout internally via `--pytorch-root` — you pass that folder as a flag, you do not `cd` into it.
17. **Ignore pre-populated Reason/DetailReason; `--filter-*` is post-hoc only**: The `tasks` array entries may carry `Reason`/`DetailReason` from a prior run — these are NOT valid results. Do not copy them to the output; always run the cascade gates (subagents for Gates 1, 2, 4; directly for Gate 3). `already_resolved` handles all legitimate reuses. The `--filter-reason`/`--filter-detailreason` flags select rows by their OUTPUT columns for subset extraction only; rows they select still require the full cascade.
18. **Task data fields not to be read during classification**: During Phase 2 cascade processing, treat `Reason` and `DetailReason` as write-only outputs. Read `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`, `classname_xpu`, `name_xpu`, `message_xpu`, and `status_xpu`. Ignore all other fields. Use XPU fields for Gate 1 and Gate 4; use CUDA fields for Gate 2.
19. **Mandatory Evidence in DetailReason**: Every `DetailReason` MUST contain explicit, verifiable evidence (GitHub issue/PR URL or number, commit hash, or a specific `file:line` code citation) as required by each gate's own section. Vague statements without evidence are invalid.
20. **Never rebuild over an existing accumulator**: When `--output-excel` already contains analyzed rows from prior runs, use `--merge` (not a plain BUILD) so those rows are preserved. A plain BUILD is only for creating the file the first time. The script enforces this (BUILD aborts rather than discarding prior `Analyzed = TRUE` rows), but choose the correct mode up front. For unambiguous merge matching, every `results.json` entry MUST carry `testfile_cuda`, `classname_cuda`, and `name_cuda`.
21. **Gate 5 (Enablement Analysis) is only for skipped tests**: Only enter Gate 5 when `has_known_issue == False` AND `status_xpu == "skipped"`. Non-skipped tests (e.g. `status_xpu == "failed"`) without a known issue go directly to `Submit Issue` — do NOT run Gate 5 on them.

## Version

- v3.7.0 - 2026-07-05 - `run_blank_test.py` Gate 0 now enforces a per-test-case timeout via the `pytest-timeout` plugin (`--timeout <SECONDS> --timeout_method=thread`, matching CI's own convention per AGENTS.md) instead of relying solely on the outer `subprocess.run` watchdog. The outer watchdog is kept as a safety net (`--timeout` value + 60s buffer) so pytest-timeout's own thread-based timeout gets a chance to report gracefully first. `check_environment()` now fails fast with an install instruction if `pytest-timeout` is missing (does not auto-install, per this project's fail-fast dependency policy). Default `--timeout` changed 300 -> 600 to match CI. Updated the Phase 2 example command and behavior notes.
- v3.6.0 - 2026-07-05 - Extracted Session setup (conda env + pytorch checkout bootstrap via `setup_env.sh`) into a standalone `prepare-env` skill. Execution Step 0 now delegates to `prepare-env` as a subagent (`task(load_skills=["prepare-env"])`) which returns the resolved `conda_env`/`pytorch_folder` for the session, instead of running the bootstrap inline. Added `prepare-env` to See Also. No change to gate order, cascade, or verdict mappings.
- v3.5.5 - 2026-07-01 - Aligned the `PYTORCH_SRC` and `conda_env` contracts across the cascade (no logic change to gate order/verdicts). **PYTORCH_SRC**: Gate 5 now passes `PYTORCH_SRC={pytorch_folder}` (previously omitted, so enablement ran against cwd); downstream skills referencing `$PYTORCH_SRC` in shell (`check-not-target-feature`, `check-community-change`, `check-community-change-source-inspection`) each begin with an explicit `export PYTORCH_SRC=...` step so the variable expands; `check-enablement-feasibility` input renamed `pytorch_src` → `PYTORCH_SRC` and given an export/path-resolution step. **conda_env**: the delegated-checks table now passes `conda_env={conda_env}` to Gates 2 and 5 (the only gates that run `import torch`/`pytest`); Gate 2's Step 3 Path A device check and `pytest --collect-only` now run via `conda run -n {conda_env}` instead of a bare `python3` (a missing env previously made Path A silently fall through to Path B source inspection). Gates 1 and 4 remain env-free (static inspection / `gh` only).
- v3.5.4 - 2026-07-01 - Documentation-only slimming (no logic change): consolidated the four near-identical delegated-gate sections (Gates 1, 2, 4, 5) into a single "Delegated Checks" section with one canonical `task(...)` template plus a per-gate delegation table (skill, identifiers, extra prompt) and a verdict-mapping table; kept the non-templatable rules as callouts (Gate 1 Step-6 backfill, Gate 4 MANDATORY-no-shortcut + fallthrough, Gate 5 entry/nuance/logging). Gate 3 (direct check) split into its own short section. Normalized Gate 2's prompt identifiers to `{classname_cuda}`/`{testfile_cuda}` and added the `PYTORCH_SRC` the skill already required. Gate order, identifiers, verdict fields, and all Reason/DetailReason mappings are unchanged.
- v3.5.3 - 2026-07-01 - Documentation-only slimming (no logic change): removed duplicated Phase 3 IMPORTANT callouts and merged the `--filter-*` rule into the cascade callout; shrank the Execution section to an ordered checklist that points to the Workflow phases instead of re-listing every bash command; merged Constraints 17+18 (pre-populated Reason / `--filter-*`) into one and trimmed Constraint 20 (Mandatory Evidence) to a pointer, renumbering the trailing constraints; removed a duplicate v2.1.0 version entry.
- v3.5.2 - 2026-07-01 - Removed Gate 0.7 (deterministic bash existence pre-check). The missing-test → Community Change decision is now handled purely by the cascade: `check-not-target-feature` returns `is_not_target=false` for any missing file/method (enforced by its strengthened Missing-Test Guard), so the row falls through to Gate 2, whose Step 1.5 file-existence fast path classifies it as `Community Change`. Removed the Gate 0.7 section, Gate 1 precondition, Execution bullet, `gate07_existence.json` log entry, and reverted Constraint 9.
- v3.5.1 - 2026-07-01 - Moved the "Not Applicable Evidence Backfill" logic out of the orchestrator and into the `check-not-target-feature` skill's Step 6, so the tracking-issue link (`attach_not_target_evidence.py`) is attached inside Gate 1 whenever it returns `Not Applicable`. Removed the standalone backfill subsection and Execution bullet; Gate 1 handling, Purpose cascade, and Constraint 13 now reference the skill-owned backfill.
- v3.5.0 - 2026-07-01 - Added the non-terminal "Not Applicable Evidence Backfill" (via new `attach_not_target_evidence.py`) that appends a tracking-issue link (e.g. `intel/torch-xpu-ops#4179`) to `Not Applicable` verdicts lacking one, without changing the verdict. (Note: this version also added a Gate 0.7 existence pre-check that was subsequently removed in v3.5.2.)
- v3.4.0 - 2026-06-30 - Submit handoff now runs as Phase 4 (BEFORE the Excel write, which becomes Phase 5) so returned links land in `results.json` in one pass. The `ut-follow-up` agent returns a per-row outcome (`pr`/`issue`/`skipped`) with a link; classify-ut records the link in `DetailReason` and sets `Reason = "Submit PR"` (test-code fix submitted as a PR) or `Reason = "Submit Issue"` (issue filed). Updated Constraint 14, Execution steps 5-7, and See Also.
- v3.3.0 - 2026-06-30 - Added Phase 5: `Submit Issue` rows are routed to the `ut-follow-up` agent to prepare confirm-gated GitHub issue drafts. Reworded Constraint 14 from "no auto-file" to "agent-assisted, confirm-gated filing" (classify-ut never files; agent never POSTs without per-issue user approval). Added `phase5_submit_issues.json` log and `ut-follow-up` See Also entry.
- v3.2.0 - 2026-06-25 - Extracted Gate 5 into standalone `check_enablement_feasibility` subskill with JSON output. Gate 5 now delegates to subagent instead of inline analysis. Added logging & audit trail constraints (0-3), fatal error handling constraints (4-7). Added `check_enablement_feasibility` to See Also. Renumbered constraints 0-15 → 8-22.
- v3.1.0 - 2026-06-25 - Added Gate 5 (Enablement Analysis for Skipped Tests). Skipped tests (`status_xpu = "skipped"`) without a known issue now undergo deep source code analysis to determine XPU enablement feasibility. Feasible tests get `Reason = "To be enabled"` with the enablement method; infeasible tests get `Reason = "Submit Issue"`. Updated description, Purpose cascade, Gate 4, Execution workflow, Constraints 0 and 15.
- v3.0.0 - 2026-06-17 - Added Gate 0 (Local Test) via `run_blank_test.py`. Blank `status_xpu` tests are run locally before the cascade. Passing tests get `Reason = "Local Passed"` and skip further classification. Added `run_blank_test.py` script and updated workflow Phases (2→local test, 3→cascade, 4→write results). New Constraint 0.
- v2.3.0 - 2026-06-14 - `check_not_target_feature`: require listing the corresponding `intel/torch-xpu-ops` `not_target`/`wontfix` issue number in `evidence` whenever one exists (new Step 6 "Attach Issue Number" + Strict Constraint 6), even for verdicts reached via the Not-Applicable sheet (Step 3) or implementation analysis (Step 5). Updated Constraint 13 to prefer issue numbers for not_target evidence.
- v2.2.0 - 2026-06-14 - Added incremental `--merge` mode to `write_results.py` for safely updating an existing accumulator in place (only touches rows in `results.json`, preserves all prior rows). Added a destructive-rebuild guard (BUILD aborts rather than discarding prior `Analyzed = TRUE` rows; override with `--force`), automatic `.bak` backup on in-place writes, and ambiguity detection. Documented BUILD vs MERGE in Phase 3 and added Constraint 14.
- v2.1.0 - 2026-06-10 - Added Constraint 10-12: explicit rules against using `--filter-reason`/`--filter-detailreason` as classification shortcut, ignoring pre-populated Reason/DetailReason in tasks, and restricting readable task fields. Added IMPORTANT callouts at Phase 2 entry. Added WARNING stderr messages in `extract_tasks.py` for `--filter-reason`/`--filter-detailreason` to discourage misuse.
- v2.0.0 - 2026-06-10 - Rewritten: adjusted script paths, removed Phase 0 conda setup (Intel-specific infra), removed `PYTORCH_SRC` default, removed local system dependencies.

## See Also

- `prepare-env` — Session setup: establishes/bootstraps the conda env and pytorch checkout (via `setup_env.sh`) and returns `conda_env`/`pytorch_folder`
- `run_blank_test.py` — Gate 0: runs blank `status_xpu` tests locally, marks passing tests as `Local Passed`
- `check-not-target-feature` — Gate 1: determines if a test is CUDA-only / not applicable for XPU
- `check-community-change` — Gate 2: determines if a test was removed/renamed upstream
- `check-known-issue` — Gate 4: searches for known issues in pytorch/pytorch and intel/torch-xpu-ops
- `check-enablement-feasibility` — Gate 5: deep source code analysis for skip mechanism and XPU enablement feasibility
- `ut-follow-up` (agent) — Phase 4: fixes test-code bugs as a confirm-gated PR or files a confirm-gated issue for `Submit Issue` rows, returning the link recorded in `DetailReason` (`Reason` becomes `Submit PR` or `Submit Issue`)
