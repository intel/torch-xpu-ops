---
name: xpu-nightly-ci-fix
description: >
  Orchestrator for fixing XPU nightly CI failures in batch. Takes a CI failure
  report, reproduces each failure, triages root cause, applies fixes, and
  generates a summary report. Uses fix/ leaf skills for all core logic.
---

# Nightly CI Fix — Orchestrator

Processes a batch of nightly CI failures. Each failure runs through the same
`fix/reproduce` → `fix/triage` → `fix/implement` → `fix/verify` pipeline
independently. All detailed fix logic lives in the `fix/` leaf skills — this
skill owns the batch scheduling, branch strategy, commit format, and summary
report.

> **Before starting:** Read the `## Working Principles` section of `CLAUDE.md`.
> State which principles apply to this task before proceeding.

## Inputs

- A nightly failure report: email, test list, or log snippet containing:
  - Failing test names (file, class, method)
  - PyTorch commit hash (`ci_commit`)
  - Report date

## Required: Initialize Todo List

Immediately after parsing the failure report, create a TodoWrite list:

```
- [ ] Step 0: Ensure PyTorch checkout
- [ ] Step 1: Parse report — extract ci_commit, date, failing test list
- [ ] Step 2: Reproduce <failure_1> with `fix/reproduce` (ci_commit, test command)
- [ ] Step 3: Reproduce <failure_2> with `fix/reproduce` (ci_commit, test command)
      ... (one entry per failure)
- [ ] Step 4-6: Fix <failure_1> (checkout fix branch -> triage → implement → verify → commit)
- [ ] Step 4-6: Fix <failure_2> (checkout fix branch -> triage → implement → verify → commit)
      ... (one entry per failure)
- [ ] Step 7: Generate summary report
```

Mark a fix item `completed` only after the test actually passes and is
committed. Never skip to Step 7.

## Step 0: Ensure PyTorch checkout

Check `agent_space_xpu/pytorch/`:
```bash
ls agent_space_xpu/pytorch/ 2>/dev/null || echo "NOT FOUND"
```

If not found, clone:
```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git \
  agent_space_xpu/pytorch
git -C agent_space_xpu/pytorch submodule update --init --recursive
```

If found, fetch latest:
```bash
git -C agent_space_xpu/pytorch fetch origin
```

## Step 1: Parse the failure report

Extract:
- `report_date` (e.g. `20260608`)
- `ci_commit` — pytorch commit hash; use `origin/main` if absent
- Failing test list: group by test file/module


## Step 2-3: Reproduce each failure

For each failure, call `fix/reproduce` with:
- `reproducer_command` — the CI test command
- `ci_commit` — from Step 1
- `pytorch_dir` — `agent_space_xpu/pytorch/`

Interpret output per failure:

| Output | Action |
|--------|--------|
| `REPRODUCED` | Continue to Step 4 for this failure |
| `NOT_REPRODUCED` | Mark in summary: "already fixed"; skip to next failure |
| `NEEDS_HUMAN` | Mark in summary: "needs_human: cannot verify (+ blocker)"; skip to next failure |

## Step 4: Triage each reproduced failure

before calling `fix/triage`, checkout to a new fix branch for this failure:

```bash
git checkout -b fix-<report_date_issuenumber>  # e.g. fix-20260608
```

Call `fix/triage` with the failure description and error log.

| Verdict | Action |
|---------|--------|
| `IMPLEMENTING` | Continue to Step 5 (domain skill already loaded) |
| `NEEDS_HUMAN` | Mark in summary: "needs human (+ reason)"; skip to next failure |

## Step 5: Implement each fix

Call `fix/implement` with:
- `triage_result` from Step 4
- `pytorch_dir` — `agent_space_xpu/pytorch/`
- `allow_skip=true` — nightly-ci-fix may add `@skipIfXpu` with tracking issue
  when implementation is out of scope for a nightly fix
- `commit_message_template`:
  ```
  [xpu][fix] <short description>

  ## Motivation
  <why this fix is needed>

  ## Solution
  <what was changed and CUDA alignment if applicable>

  ## Test plan
  <how it was verified>

  Note: This commit was authored with AI assistance.
  ```

## Step 6: Verify and commit each fix

Call `fix/verify` with:
- `refined_command` from Step 3 (`fix/reproduce` output)
- `pytorch_dir` — `agent_space_xpu/pytorch/`
- `changed_files` from Step 5
- `run_before_after_diff=true`
- `run_lint=true`

| Output | Action |
|--------|--------|
| `PASSED` | Commit (one fix per commit); mark in summary: "fixed (commit: <hash>)" |
| `FAILED` | Re-triage with failure context, then re-implement (see fix loop below) |
| `NEEDS_HUMAN` | Mark in summary: "needs_human: cannot verify after fix"; skip to next failure |

**Fix loop** (max 3 attempts total, counting the first verify):

```
attempt N (starting at 1 after the first fix/verify returns FAILED):
  1. Call fix/triage again, passing:
     - original failure description
     - fix/verify failure_output from this attempt
     - fix/verify suggestion from this attempt
     - prior fix strategy (so triage knows what was already tried)
  2. Call fix/implement with the new triage_result.
  3. Call fix/verify again.
     - PASSED → commit, exit loop.
     - FAILED and attempt < 3 → increment attempt, repeat from step 1.
     - FAILED and attempt == 3 → exit loop.
     - NEEDS_HUMAN → exit loop.
```

If loop exits without `PASSED`, mark in summary: "needs human (fix loop exhausted after 3 attempts)"; record each attempt's `failure_output` and `suggestion` in the summary under a "Fix Attempts" subsection; skip to next failure.

Commit after each verified fix:
```bash
git -C agent_space_xpu/pytorch add <changed_files>
git -C agent_space_xpu/pytorch commit -m "<commit_message>"
```

If the fix was a skip (`fix/implement` output has `skip_added: yes`):
- Record the tracking issue URL in the summary under "Needs Human" (skip added).
- The tracking issue already contains the root cause and fix strategy from triage.

Each fix is one commit. Do not batch multiple fixes into one commit.

## Step 7: Generate summary report

Write to `agent_space_xpu/summary_<report_date>.md`:

```markdown
# Nightly CI Fix Summary — <report_date>

PyTorch commit: <ci_commit>
Total failures: N | Fixed: X | Skipped (already fixed): Y | Needs human: Z | Cannot verify: W

## Status at a Glance

| Failure | Status | Commit | Notes |
|---------|--------|--------|-------|
| test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu | Fixed | abc1234 | tolerance: 1e-5→1e-4 |
| test_nn_xpu.py::TestNNXPU::test_conv3d_groups | Needs human | — | missing kernel, tracking: #1234 |
| test_sparse_xpu.py::TestSparseXPU::test_mm | Already fixed | — | passes on nightly |

## Fixed

### test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- Root cause: tolerance too tight
- Fix: increased atol 1e-5 → 1e-4 to match CUDA
- Commit: abc1234
- AR: submit PR to pytorch/pytorch

## Needs Human

### test_nn_xpu.py::TestNNXPU::test_conv3d_groups
- Root cause: missing XPU kernel for grouped conv3d
- Decision: skip added with tracking issue intel/torch-xpu-ops#1234
- AR: prioritize kernel implementation

### test_foo_xpu.py::TestFooXPU::test_bar (fix loop exhausted)
- Fix Attempts:
  - Attempt 1: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
  - Attempt 2: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
  - Attempt 3: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
- AR: manual investigation required

## Already Fixed / Cannot Verify

...
```

## Critical rules

- **Never cherry-pick** upstream fixes. Rebase (`git rebase origin/main`) instead.
- **Always rebuild after rebase or branch switch** before running tests.
- **Fix in torch-xpu-ops?** Use the dev override from `CLAUDE.md` "Commit Pin
  & Development Override": clone your torch-xpu-ops branch into
  `agent_space_xpu/pytorch/third_party/torch-xpu-ops/`, then update the pin
  so CMake's checkout becomes a no-op:
  ```bash
  cd agent_space_xpu/pytorch/third_party/torch-xpu-ops
  git checkout <your-fix-branch>
  git rev-parse HEAD > agent_space_xpu/pytorch/third_party/xpu.txt
  ```
  Do NOT commit `xpu.txt`. Then rebuild:
  ```bash
  pip install -e . -v --no-build-isolation
  ```
  Run from `agent_space_xpu/pytorch/`. For pure C++ changes that do not touch
  codegen (no new ops, no dispatch registration changes), `ninja -C
  agent_space_xpu/pytorch/build` can be used to speed up incremental C++
  compilation, but `pip install -e .` is always safe and required whenever in
  doubt.
- Each failure is independent — one failure's `CANNOT_VERIFY` does not block
  others.
- One fix per commit.
