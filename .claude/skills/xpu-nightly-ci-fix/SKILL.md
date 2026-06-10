---
name: xpu-nightly-ci-fix
description: Analyze nightly CI test failures and fix XPU test cases. Use when the user provides CI failure reports, nightly status emails, failing test names, or asks to "fix nightly failures", "analyze CI failures", or "debug XPU tests". Handles triaging, reproducing, root cause analysis, and applying fixes with verification.
---
> Please also read and refer to `AGENTS.md` at the repository root for general behavioral guidelines before proceeding.

> **Planned refactor:** This skill will be split into `xpu-issue-fix` (single-test reproduce/fix/verify) and `xpu-nightly-ci-fix` (report parsing + batch orchestration + summary). The current monolithic structure is intentional for immediate use; refactoring tracked as a follow-up.

# Nightly CI Test Fixing for XPU

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

## Step 1: Parse the Failure Report

- Extract PyTorch commit_id and report_date
- Extract failing tests (test file, class, method name)
- Group failures by test file / module

## Step 2: Reproduce Locally

### Build PyTorch

Follow the **Build** section in `AGENTS.md`. Confirm `torch.xpu.is_available()` returns `True` before proceeding.

### Fetch and Checkout

- Fetch and checkout the PyTorch commit from the report; if none provided, use origin/main
- Create a local branch: `fix-YYYYMMDD` (e.g., `fix-20260608`)

**Branch strategy:** `fix-YYYYMMDD` is a **local working branch only** — not a single upstream PR. Each independent fix is one focused commit. When submitting upstream, each commit becomes a **separate PR** to `pytorch/pytorch`. Fixes in `torch-xpu-ops` kernel code require a separate PR to `intel/torch-xpu-ops`. Step 7 tracks which fix maps to which PR.

### Run Tests to Reproduce

```bash
source build_pytorch.env && pytest <test_file>::<TestClass>::<test_method> -xvs
```

Confirm each failure reproduces before proceeding.

## Step 3: Analyze and Categorize

For each failure, determine root cause by answering:

1. **When was the test added?** `git log --diff-filter=A --oneline -- path/to/test_file.py` — recent (< 7 days) → likely new feature needing XPU support
2. **Does CUDA have this test?** `rg "def test_<name>" test/` — if yes, compare CUDA vs XPU behavior
3. **What's the error type?**
   - `NotImplementedError` / `RuntimeError: not implemented` → missing XPU kernel
   - `AssertionError` with tolerance → precision/tolerance mismatch
   - `AttributeError` / `ImportError` → test infrastructure issue
   - Numerical mismatch → compare with CUDA kernel logic

### Root Cause Categories

All categories below are grounded in real observed nightly failures. A single failure may map to more than one category.

1. **XPU backend bug** — Fix in `torch/_inductor/` or `third_party/torch-xpu-ops/`
2. **Tolerance too tight** — Increase atol/rtol to match CUDA tolerances
3. **Skip decorator stale** — Remove `@skipIfXpu` or `@expectedFailure` if test now passes
4. **New feature needs XPU support** — Implement XPU kernel aligned with CUDA
5. **Upstream regression** — A recent PyTorch commit changed behavior XPU relied on. Preferred response: apply an **XPU-side fix** to align with the upstream change's intent, using the CUDA implementation as reference (e.g., `intel/torch-xpu-ops#3809`). If the upstream commit is itself a bug (CUDA also broken), file an issue in `pytorch/pytorch` and add a temporary `@skipIfXpu` with a tracking issue. Do not add bypasses that deviate from CUDA behavior without justification.
6. **Test infrastructure** — Environment, import, or setup issue

## Step 4: Fix

For each failure, read the corresponding CUDA implementation in `pytorch/aten/src/ATen/native/cuda/` to understand expected behavior, then present options:

**Option A — Fix/implement:**
- Newly added test: implement XPU kernel in `third_party/torch-xpu-ops/src/` aligned with CUDA logic, or fix CUDA-specific test assumptions
- Regression: apply XPU-side fix aligned with the upstream change's intent; if XPU must diverge from CUDA, document why in comments

**Option B — Skip with justification:**
- Add skip decorator with tracking issue reference
- File tracking issue with: root cause, missing feature/blocker, estimated effort
- Example:
  ```python
  @skipIfXpu  # TODO(#1234): Missing grouped conv3d kernel for XPU
  def test_conv3d_groups(self):
      ...
  ```
- Commit just the skip to unblock nightly; investigate separately

**Recommendation:** Provide your assessment of which option fits. Prefer Option A when effort is reasonable. For regressions, prefer Option A unless the root cause is unclear. Let the engineer decide.

If you cannot identify the guilty commit: compare with CUDA/ROCm backend, read upstream PyTorch code, check recent commits for related changes.

## Step 5: Verify (CRITICAL — Do Not Skip)

For **every** test in the initial failure report:

1. Run the individual test:
   ```bash
   source build_pytorch.env && pytest path/to/test_file.py::TestClass::test_method -xvs
   ```
2. Confirm it passes (exit code 0)
3. Run related tests in the same class:
   ```bash
   pytest path/to/test_file.py::TestClass -xvs
   ```
4. Run linter:
   ```bash
   spin fixlint
   ```

Run each test individually. Do not batch or skip any test — verifying one representative test is not sufficient.

## Step 6: Commit Changes

One focused commit per fix. Commit message template:

```
[xpu][fix] <one line summary (max 72 chars)>

## Motivation
<why this fix is needed — link to nightly failure or issue>

## Solution
<how the fix works — mention CUDA alignment if applicable>

## Test Plan
pytest test/xpu/test_foo_xpu.py::test_bar -xvs

Note: This commit was authored with AI assistance.
```

## Step 7: Generate Summary Report

Write to `agent_space/summary_YYYYMMDD.md`:

```markdown
# Nightly CI Fix Summary — 2026-06-08

PyTorch commit: abc123def
Total failures: 15 | Fixed: 12 | Skipped: 2 | Investigating: 1

## Fixed Tests

### test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- Root cause: Tolerance too tight (Category 2)
- Fix: Increased atol 1e-5 → 1e-4 to match CUDA
- Commit: fix-add-tolerance-20260608
- AR: Submit PR to pytorch/pytorch

[... more entries ...]

## Skipped Tests (with tracking issues)

### test_nn_xpu.py::TestNNXPU::test_conv3d_groups
- Root cause: Missing XPU kernel for grouped conv3d (Category 4)
- Decision: Skip — implementation ~3-5 days, beyond nightly scope
- Fix: Added @skipIfXpu with issue reference
- Tracking issue: intel/torch-xpu-ops#1234
- AR: Prioritize kernel implementation in next sprint
```

## Examples

See Step 7 above for a complete summary report example. For advanced build safety and AOT Inductor debug tips, see [reference.md](reference.md).

## Best Practices

### Decision Making
- Present both options (implement / skip) with effort estimates; let the engineer decide
- Prefer fixing over skipping when effort is reasonable
- When skipping: always file a tracking issue and document effort estimate

### Verification
- Always reproduce before fixing
- Match upstream CUDA tolerances when adjusting XPU tolerances
- Remove unused imports when removing skip decorators

### Commit Hygiene
- One fix per commit
- **Never cherry-pick** upstream fixes into the fix branch — rebase onto latest trunk instead (`git rebase origin/main`)

### Build Safety
- **Always rebuild after `git rebase` or `git checkout`** — stale C++ extensions produce unreliable test results
- **Fix in `torch-xpu-ops`?** Update `xpu.txt` to your local HEAD **before rebuilding** so CMake's checkout becomes a no-op:
  ```bash
  git rev-parse HEAD > <pytorch_root>/third_party/xpu.txt
  # Do NOT commit xpu.txt — local-only override
  source build_pytorch.env && pip install -e . -v --no-build-isolation
  ```
  After verification, submit PR to `intel/torch-xpu-ops`, then update the pin in `pytorch/pytorch` once merged.

## Requirements

- PyTorch built from source with XPU support (see `AGENTS.md` Build section)
- Build verified: `torch.xpu.is_available()` returns `True`
- Scratch files in `agent_space/` (git-ignored)

## Advanced Usage

For AOT Inductor C++ compile error diagnosis and other advanced nightly CI debugging patterns, see [reference.md](reference.md).
