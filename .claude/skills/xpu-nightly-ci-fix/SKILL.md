---
name: xpu-nightly-ci-fix
description: Analyze nightly CI test failures and fix XPU test cases. Use when the user provides CI failure reports, nightly status emails, failing test names, or asks to "fix nightly failures", "analyze CI failures", or "debug XPU tests". Handles triaging, reproducing, root cause analysis, and applying fixes with verification.
---
> Please also read and refer to `AGENTS.md` at the repository root for general behavioral guidelines before proceeding.

> **Planned refactor:** This skill will be split into `xpu-issue-fix` (single-test reproduce/fix/verify) and `xpu-nightly-ci-fix` (report parsing + batch orchestration + summary). The current monolithic structure is intentional for immediate use; refactoring tracked as a follow-up.

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

## Prerequisites

- PyTorch built from source with XPU support (see `AGENTS.md` Build section)
- `.env` configured for the oneAPI environment
- **Always `source .env` before any Python/torch command**

## Step 1: Parse the Failure Report

- Extract `commit_id` and `report_date` from the report
- Extract failing tests (test file, class, method name)
- Group failures by test file / module

## Step 2: Reproduce Locally

Work from the PyTorch root directory.

1. **Fetch and checkout the target commit:**
   ```bash
   git fetch origin main
   git checkout <commit_id>  # use origin/main if no commit_id in report
   ```

2. **Create a fix branch:**
   ```bash
   git checkout -b fix-<report_date>  # e.g. fix-20260608
   ```

3. **Build PyTorch:**
   ```bash
   source .env
   python setup.py clean
   pip install -e . -v --no-build-isolation
   ```

4. **Run each failing test:**
   ```bash
   source .env && python <test_file> -k <test_name> 2>&1 | tail -80
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
| Upstream regression | New upstream code changed behavior XPU relied on | XPU-side fix aligned with upstream intent; skip only if upstream itself is buggy (see Critical Rules) |
| Test infrastructure | Environment, import, or setup issue | Test setup/config files |

All categories are grounded in real observed nightly failures. A single failure may map to more than one category.

## Step 4: Fix

Read the corresponding CUDA implementation in `pytorch/aten/src/ATen/native/cuda/` to understand expected behavior.

- **Newly added test:** Try to enable XPU support. If not feasible (missing kernel, blocked feature), skip with `@skipIfXpu` and a tracking issue.
- **Regression:** Find the guilty commit (`git log --oneline -20 -- <file>`). Apply an XPU-side fix aligned with upstream intent; document any CUDA divergence in comments.
- **Unknown root cause:** Compare with CUDA/ROCm backend behavior to identify the issue.

## Step 5: Verify and Commit

1. Run the fixed test and confirm it passes:
   ```bash
   source .env && python <test_file> -k <test_name> 2>&1 | tail -80
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

Write to `agent_space/summary_<report_date>.md`:

```markdown
# Nightly CI Fix Summary — 2026-06-08

PyTorch commit: abc123def
Total failures: 15 | Fixed: 12 | Skipped: 2 | Investigating: 1

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

## Examples

See Step 6 above for a complete summary report example. For AOT Inductor C++ compile error diagnosis, see [reference.md](reference.md).

## Critical Rules

### Build Discipline

- **Always rebuild after rebase or branch switch.** After `git rebase`, `git checkout`, or any commit-base change, rebuild before running tests. Without rebuilding, C++ extensions and generated code are stale — test results will be completely unreliable (segfaults, wrong pass/fail, masked issues).
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk after the CI commit, rebase onto latest trunk (`git rebase origin/main`) instead.
- **Fix in `torch-xpu-ops`?** Update `xpu.txt` to your local HEAD **before rebuilding** so CMake's checkout becomes a no-op:
  ```bash
  git rev-parse HEAD > <pytorch_root>/third_party/xpu.txt
  # Do NOT commit xpu.txt — local-only override
  source .env && pip install -e . -v --no-build-isolation
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
- Scratch files go in `agent_space/` (git-ignored)

### Upstream Regression Fix Approach

Apply an **XPU-side fix** to align with the upstream change's intent, using the CUDA implementation as reference (e.g., `intel/torch-xpu-ops#3809`). If the upstream commit is itself a bug (CUDA also broken), file an issue in `pytorch/pytorch` and add a temporary `@skipIfXpu` with a tracking issue. Do not add bypasses that deviate from CUDA behavior without justification.

## Requirements

- PyTorch built from source with XPU support (see `AGENTS.md` Build section)
- Build verified: `torch.xpu.is_available()` returns `True`
- Scratch files in `agent_space/` (git-ignored)

## Advanced Usage

For AOT Inductor C++ compile error diagnosis and other advanced debug patterns, see [reference.md](reference.md).
