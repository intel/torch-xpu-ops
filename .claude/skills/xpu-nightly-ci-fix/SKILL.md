---
name: xpu-nightly-ci-fix
description: Analyze nightly CI test failures and fix XPU test cases. Use when the user provides CI failure reports, nightly status emails, failing test names, or asks to "fix nightly failures", "analyze CI failures", or "debug XPU tests". Handles triaging, reproducing, root cause analysis, and applying fixes with verification.
---
> **Before starting:** Read the `## Working Principles` section of `AGENTS.md` at the repository root.
> Then explicitly state which principles apply to this task and how you will follow them.
> Do not proceed until you have done this.

# Nightly CI Test Fixing for XPU

Analyze CI nightly test failure reports and fix failing XPU test cases on PyTorch.

## Quick Start

Provide a nightly failure report (email, test list, or log snippet). The skill will:
1. Extract failing tests and PyTorch commit
2. Reproduce each failure locally
3. Categorize by root cause
4. Apply fixes aligned with CUDA reference
5. Verify all fixes individually
6. Generate summary report

```
I have a nightly CI failure report from 2026-06-08. Here are the failing tests:
- test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- test_nn_xpu.py::TestNNXPU::test_relu_xpu
PyTorch commit: abc123def
```

## Required: Initialize Todo List Before Starting

**Immediately after reading this skill and parsing the failure report, use TodoWrite to create the
following items before doing any other work.** Do not skip or merge steps. Each failing test group
gets its own reproduce/fix/verify trio.

```
- [ ] Step 0: Ensure PyTorch checkout exists and is up to date
- [ ] Step 1: Parse report — extract commit, date, failing test list
- [ ] Step 2a: Checkout origin/main and rebuild PyTorch
- [ ] Step 2b: Reproduce <test_group_1> — confirm failure
- [ ] Step 2b: Reproduce <test_group_2> — confirm failure
      ... (one entry per distinct failing test or group)
- [ ] Step 3: Analyze root cause for each failure
- [ ] Step 4+5: Fix + verify <test_group_1> (run test, confirm pass, lint, commit)
- [ ] Step 4+5: Fix + verify <test_group_2> (run test, confirm pass, lint, commit)
      ... (one entry per fix)
- [ ] Step 6: Generate summary report
```

Only mark a fix item `completed` after the test actually passes. Never skip directly to Step 6.

## Prerequisites

- XPU hardware available and oneAPI environment configured
- A local PyTorch checkout at `agent_space_xpu/pytorch/` (see Step 0 below)
- PyTorch built from source with XPU support (see `AGENTS.md` Build section and `/xpu-build-pytorch` skill)

## Step 0: Ensure PyTorch Checkout Exists

Check whether `agent_space_xpu/pytorch/` already exists. If not, clone it before proceeding.

```bash
ls agent_space_xpu/pytorch/ 2>/dev/null || echo "NOT FOUND"
```

If **not found**, clone with a partial (blobless) clone to save time and disk space:

```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git agent_space_xpu/pytorch
cd agent_space_xpu/pytorch
git submodule update --init --recursive
```

If **already found**, verify the remote is correct and fetch latest:

```bash
git -C agent_space_xpu/pytorch remote get-url origin  # should be pytorch/pytorch
git -C agent_space_xpu/pytorch fetch origin
```

All subsequent steps run from `agent_space_xpu/pytorch/` as the working directory.

## Step 1: Parse the Failure Report

- Extract `report_date` from the report
- Extract `commit_id` if present; otherwise use `origin/main`
- Extract failing tests (test file, class, method name)
- Group failures by test file / module

## Step 2: Reproduce Locally

All commands below run from `agent_space_xpu/pytorch/`.

1. **Checkout the target commit:**

   Always use `origin/main` by default, even if the report includes a commit ID.
   Only use a specific commit if the user explicitly requests it.
   ```bash
   git fetch origin main
   git checkout origin/main
   ```

2. **Create a fix branch:**
   ```bash
   git checkout -b fix-<report_date>  # e.g. fix-20260608
   ```

3. **Build PyTorch** (clean rebuild for accurate reproduction):
   ```bash
   python setup.py clean
   pip install -e . -v --no-build-isolation
   ```
   For environment setup and building PyTorch, see `AGENTS.md` Build section and the `/xpu-build-pytorch` skill.

4. **Run each failing test:**
   ```bash
   python <test_file> -k <test_name> 2>&1 | tail -80
   ```

5. **Confirm the failure reproduces** before proceeding to Step 3.

**Branch strategy:** `fix-<report_date>` is a **local working branch only** — not a single upstream PR. Each independent fix is one focused commit. When submitting upstream, each commit becomes a **separate PR** to `pytorch/pytorch`. Fixes in `torch-xpu-ops` kernel code require a separate PR to `intel/torch-xpu-ops`. Step 6 tracks which fix maps to which PR.

## Step 3: Analyze and Categorize

**First:** Use `git log` to check when the test was added. If recently added, check the introducing commit/PR to see if XPU support is required — then skip to Step 4.

**Otherwise**, categorize the root cause:

| Category | Description | Typical Fix Location |
|----------|-------------|---------------------|
| XPU backend bug | Backend implementation issue | `torch/_inductor/` or `third_party/torch-xpu-ops/` |
| Tolerance too tight | Numeric precision mismatch | Increase atol/rtol to match CUDA |
| Skip decorator stale | Test now passes on XPU | Remove `@skipIfXpu` or `@expectedFailure` |
| Upstream regression | New upstream code changed behavior XPU relied on. Apply XPU-side fix aligned with upstream intent using CUDA as reference (e.g., `intel/torch-xpu-ops#3809`). If the upstream commit is itself a bug (CUDA also broken), file issue in `pytorch/pytorch` + add temporary `@skipIfXpu` with tracking issue. Do not add bypasses deviating from CUDA behavior. | XPU-side fix in `torch-xpu-ops` or test file |
| Test infrastructure | Environment, import, or setup issue | Test setup/config files |

## Step 4: Fix

Read the corresponding CUDA implementation in `pytorch/aten/src/ATen/native/cuda/` to understand expected behavior.

- **Newly added test:** Try to enable XPU support. If not feasible (missing kernel, blocked feature), skip with `@skipIfXpu` and a tracking issue.
- **Regression:** Find the guilty commit (`git log --oneline -20 -- <file>`). Apply an XPU-side fix aligned with upstream intent; document any CUDA divergence in comments.
- **Unknown root cause:** Compare with CUDA/ROCm backend behavior to identify the issue.

## Step 5: Verify and Commit

1. Run the fixed test and confirm it passes:
   ```bash
   python <test_file> -k <test_name> 2>&1 | tail -80
   ```
2. Run the full test file to check for regressions
3. Lint:
   ```bash
   spin fixlint
   ```
4. Commit (one fix per commit):
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

## Step 6: Generate Summary Report

Write to `agent_space_xpu/summary_<report_date>.md`:

```markdown
# Nightly CI Fix Summary — 2026-06-08

PyTorch commit: abc123def
Total failures: 15 | Fixed: 12 | Skipped: 2 | Investigating: 1

## Status at a Glance

| Failure | Local fix | PR submitted | CI unblocked |
|---------|-----------|--------------|--------------|
| test_ops_xpu.py::...::test_add_xpu | YES (commit abc1234) | NO | NO |
| test_nn_xpu.py::...::test_conv3d_groups | YES (commit def5678) | NO | NO |
| Windows wheel-py3_*-xpu-test | N/A (infra) | N/A | NO — needs manual log investigation |

**<one-line summary of overall status, e.g. "Nothing pushed to pytorch/pytorch yet.">**

---

## Fixed Tests

### test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- Root cause: Tolerance too tight
- Fix: Increased atol 1e-5 → 1e-4 to match CUDA
- Commit: fix-add-tolerance-20260608
- AR: Submit PR to pytorch/pytorch

[... more entries ...]

## Skipped Tests (with tracking issues)

### test_nn_xpu.py::TestNNXPU::test_conv3d_groups
- Root cause: Missing XPU kernel for grouped conv3d
- Decision: Skip — implementation ~3-5 days, beyond nightly scope
- Fix: Added @skipIfXpu with issue reference
- Tracking issue: intel/torch-xpu-ops#1234
- AR: Prioritize kernel implementation in next sprint
```

## Critical Rules

### Build Discipline

- **Always rebuild after rebase or branch switch.** After `git rebase`, `git checkout`, or any commit-base change, rebuild before running tests. Without rebuilding, C++ extensions and generated code are stale — test results will be completely unreliable (segfaults, wrong pass/fail, masked issues).
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk after the CI commit, rebase onto latest trunk (`git rebase origin/main`) instead.
- **Fix in `torch-xpu-ops`?** Update `xpu.txt` to your local HEAD **before rebuilding** so CMake's checkout becomes a no-op:
  ```bash
  git rev-parse HEAD > <pytorch_root>/third_party/xpu.txt
  # Do NOT commit xpu.txt — local-only override
  pip install -e . -v --no-build-isolation
  ```
  After verification, submit PR to `intel/torch-xpu-ops`, then update the pin in `pytorch/pytorch` once merged.

### Verification

- **Run EVERY failing test case individually** after fixing. Do not skip any case or assume one representative case is sufficient. Run all cases explicitly, batch by batch if needed.
- Always reproduce before fixing.

### C++ / Header Changes

- Editable installs resolve Python from source but C++ headers from `torch/include/`. After editing a C++ header, **manually copy** it to the installed include path.

### Code Style

- Match upstream CUDA tolerances when adjusting XPU tolerances
- Remove unused imports when removing skip decorators
- Keep commits focused: one fix per commit
- Scratch files go in `agent_space_xpu/` (git-ignored)

## Advanced Usage

For AOT Inductor C++ compile error diagnosis and other advanced debug patterns, see [reference.md](reference.md).
