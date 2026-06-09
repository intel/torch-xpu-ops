---
name: xpu-nightly-ci-fix
description: Analyze nightly CI test failures and fix XPU test cases. Use when the user provides CI failure reports, nightly status emails, failing test names, or asks to "fix nightly failures", "analyze CI failures", or "debug XPU tests". Handles triaging, reproducing, root cause analysis, and applying fixes with verification.
---
> Please also read and refer to `AGENTS.md` at the repository root for general behavioral guidelines before proceeding.

# Nightly CI Test Fixing for XPU

## When to Use

- The user provides a CI nightly failure report (email content, test list, or log snippets).
- The goal is to analyze, triage, reproduce, and fix multiple failing XPU tests in PyTorch, then output a summary report.

## Quick Start

Provide a nightly failure report (email, test list, or log snippet). The skill will:
1. Extract failing tests and PyTorch commit
2. Reproduce each failure locally
3. Categorize by root cause
4. Apply fixes aligned with CUDA reference
5. Verify all fixes individually
6. Generate summary report

Example invocation:
```
I have a nightly CI failure report from 2026-06-08. Here are the failing tests:
- test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- test_nn_xpu.py::TestNNXPU::test_relu_xpu
PyTorch commit: abc123def
```

## Step 1: Parse the Failure Report

- Parse the PyTorch commit_id and report_date
- Extract the list of failing tests (test file, class, method name)
- Group failures by test file / module

## Step 2: Reproduce Locally

### Verify Prerequisites

Before building, verify the environment:
```bash
# Confirm in PyTorch root directory
test -f setup.py || echo "ERROR: Not in PyTorch root"

# Verify oneAPI environment
which icpx || echo "ERROR: oneAPI compiler not found"
```

### Fetch and Checkout

- Fetch latest PyTorch origin main if commit_id is provided in previous step, checkout to it; else use origin main
- Create a new branch for the fix, branch name: `fix-YYYYMMDD` (e.g., `fix-20260608`)

### Build PyTorch

**Example `build_pytorch.env`** (adjust paths for your system):
```bash
export TORCH_XPU_ARCH_LIST=pvc
export USE_XPU=1
export USE_CUDA=0

# Adjust these paths to your local oneAPI installation
source /opt/intel/oneapi/compiler/latest/env/vars.sh
# Check if oneapi-vars.sh is present, sometimes it's located directly in oneapi/
# source /opt/intel/oneapi/setvars.sh

# pti is optional depending on the scenario
# source /opt/intel/oneapi/pti/latest/env/vars.sh
```

Build PyTorch (**IMPORTANT**: Read warnings below before building):
```bash
source build_pytorch.env
python setup.py clean
pip install -e . -v --no-build-isolation
```

**Build warnings:**
- After any `git rebase` or `git checkout`, you MUST rebuild
- Editable installs have stale C++ headers — manually copy edited headers to `torch/include/`
- Delete PCH cache after header changes: `rm -rf /tmp/torchinductor_$USER/precompiled_headers/`

### Run Tests to Reproduce

Run each failing test on this machine:
```bash
source build_pytorch.env && pytest <test_file>::<TestClass>::<test_method> -xvs
```

Confirm the failure reproduces.

## Step 3: Analyze and Categorize

For each failure, determine the root cause category by answering these diagnostic questions:

### Diagnostic Questions

1. **When was the test added?**
   - Run: `git log --diff-filter=A --oneline -- path/to/test_file.py`
   - If within last 7 days → likely new feature needs XPU support

2. **Does CUDA have this test?**
   - Search PyTorch repo: `rg "def test_<name>" test/`
   - If yes → compare CUDA vs XPU behavior
   - If no → check if test is XPU-specific

3. **What's the error type?**
   - `NotImplementedError` / `RuntimeError: not implemented` → missing XPU kernel
   - `AssertionError` with tolerance → precision/tolerance issue
   - `AttributeError` / `ImportError` → test infrastructure
   - Numerical mismatch → compare with CUDA kernel logic

### Root Cause Categories

1. **XPU backend bug** — Fix in `torch/_inductor/` or `third_party/torch-xpu-ops/`
2. **Tolerance too tight** — Increase atol/rtol to match CUDA tolerances
3. **Skip decorator stale** — Remove `@skipIfXpu` or `@expectedFailure` if the test now passes
4. **New feature needs XPU support** — Implement XPU kernel aligned with CUDA
5. **Upstream regression** — New PyTorch commit broke XPU; needs XPU-specific workaround
6. **Test infrastructure** — Environment, import, or setup issue

## Step 4: Fix

### For Newly Added Tests

Analyze why the test fails for XPU and provide the engineer with options:

1. **Understand the failure:**
   - Missing XPU kernel implementation?
   - Test has CUDA-specific assumptions?
   - Requires hardware-specific feature not available on XPU?
   - **Read the corresponding CUDA implementation** in `pytorch/aten/src/ATen/native/cuda/` to understand expected behavior

2. **Present options to engineer:**

   **Option A: Implement XPU support**
   - Implement XPU kernel in `third_party/torch-xpu-ops/src/` aligned with CUDA logic
   - Fix test assumptions to work on XPU (in pytorch repo)
   - Estimated effort: [your assessment]
   - Benefits: Full XPU feature parity
   
   **Option B: Skip the test (with proper justification)**
   - Use appropriate skip decorator with clear reason
   - File a tracking issue in `intel/torch-xpu-ops` with:
     - Root cause analysis
     - Missing feature or blocker
     - Estimated implementation effort
   - Add TODO comment with issue link
   - Example:
     ```python
     @skipIfXpu  # TODO(#1234): Missing grouped conv3d kernel for XPU
     def test_conv3d_groups(self):
         ...
     ```
   
   **Recommendation:** [Your analysis of which option makes sense for this specific case]

3. **Let the engineer decide** based on:
   - Nightly CI urgency
   - Implementation complexity
   - Feature priority
   - Available resources

### For Regression Tests

1. **Find the guilty commit:**
   ```bash
   git log --oneline --all -- path/to/test_file.py
   ```

2. **Read the corresponding CUDA kernel** in `pytorch/aten/src/ATen/native/cuda/` to understand expected behavior

3. **Analyze the regression:**
   - What changed in the guilty commit?
   - Does CUDA still work? (verify CUDA behavior didn't change)
   - Is this an XPU-specific issue or framework-wide?

4. **Present options to engineer:**

   **Option A: Fix the XPU implementation**
   - Apply XPU-specific fix that **aligns with CUDA implementation** logic, tolerances, and dtypes
   - If XPU must diverge from CUDA, document the reason in code comments
   - Estimated effort: [your assessment]
   
   **Option B: Temporarily skip while investigating**
   - If root cause is unclear or fix is complex
   - Add skip decorator with issue reference
   - File tracking issue with regression details
   - Commit just the skip to unblock nightly, investigate separately
   
   **Recommendation:** [Your analysis - usually prefer Option A for regressions unless very complex]

### General Analysis

If you cannot identify the guilty commit, start analysis:
- Compare with CUDA/ROCm backend implementation
- Read upstream PyTorch code to understand expected semantics
- Check recent PyTorch main commits for related changes
- Try to find the root cause

Then present options as above.

## Step 5: Verify (CRITICAL — Do Not Skip)

For EVERY test from the initial failure report:

1. **Run the individual test:**
   ```bash
   source build_pytorch.env
   pytest path/to/test_file.py::TestClass::test_method -xvs
   ```

2. **Verify it passes** (exit code 0)

3. **Run related tests in the same file:**
   ```bash
   pytest path/to/test_file.py::TestClass -xvs
   ```

4. **Run linter:**
   ```bash
   spin fixlint
   ```

5. **Analyze the output log and fix any linting issues**

**Do NOT batch verification** — run each test individually and confirm pass before moving to next.

## Step 6: Commit Changes

Stage and commit changes. The commit message must follow this template:

```
[xpu][fix] <one line summary (max 72 chars)>

## Motivation
<why this fix is needed — link to issue or describe the nightly failure>

## Solution
<how the fix works — mention CUDA alignment if applicable>

## Test Plan
```bash
pytest test/xpu/test_foo_xpu.py::test_bar -xvs
```

Note: This commit was authored with AI assistance.
```

## Step 7: Generate Summary Report

For all the failing tests provided in the initial CI report, write a structured summary to `agent_space/summary_YYYYMMDD.md`.

Each entry in the summary should include:
- Test name / module
- Root cause and category (from Step 3)
- Resolution or fix applied (with commit reference if a change was made)
- AR: Action required (e.g., submit PR, rebase, none)

**Example `agent_space/summary_20260608.md`:**

```markdown
# Nightly CI Fix Summary — 2026-06-08

PyTorch commit: abc123def
Total failures: 15
Fixed: 12
Skipped: 3 (with tracking issues)
Still investigating: 0

## Fixed Tests

### test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- Root cause: Tolerance too tight (Category 2)
- Fix: Increased atol from 1e-5 to 1e-4 to match CUDA
- Commit: fix-add-tolerance-20260608
- AR: Submit PR #3895

### test_nn_xpu.py::TestNNXPU::test_relu_xpu
- Root cause: Skip decorator stale (Category 3)
- Fix: Removed @skipIfXpu decorator, test now passes
- Commit: fix-remove-skip-relu-20260608
- AR: Submit PR #3895

[... more entries ...]

## Skipped Tests (with tracking issues)

### test_nn_xpu.py::TestNNXPU::test_conv3d_groups
- Root cause: Missing XPU kernel for grouped conv3d (Category 4)
- Decision: Skip with tracking issue (implementation estimated 3-5 days, beyond nightly scope)
- Fix: Added @skipIfXpu decorator with issue reference
- Tracking issue: intel/torch-xpu-ops#1234
- AR: Prioritize kernel implementation in next sprint

### test_inductor_xpu.py::TestInductorXPU::test_advanced_feature
- Root cause: XPU backend missing feature X (Category 1)
- Decision: Skip temporarily while investigating (complex issue, needs design discussion)
- Fix: Added @skipIfXpu with TODO
- Tracking issue: intel/torch-xpu-ops#1235
- AR: Schedule design review meeting
```

## Environment

- Source `build_pytorch.env` before running any test or build
- Build: `pip install -e . -v --no-build-isolation`
- Scratch files go in `agent_space/` (git-ignored)

## Best Practices

See `AGENTS.md` "Working Principles" section for coding philosophy.

Additional nightly CI-specific practices:

### Decision Making
- **Present options, let engineer decide:** For each failure, analyze and present both "implement" and "skip" options with effort estimates
- **Prefer fixing over skipping:** When effort is reasonable, recommend implementing the fix
- **Skip with justification:** When skipping is necessary (complex implementation, blocked by other work), always:
  - File a tracking issue with root cause analysis
  - Add skip decorator with issue reference
  - Document estimated implementation effort
  - Add to "Skipped Tests" section in summary report

### Verification
- Always reproduce before fixing
- **After fixing, run EVERY failing test from the report individually to verify it passes.** Do not skip any test or assume that verifying one representative test is sufficient. Run all tests explicitly, batch by batch if needed, and confirm each one passes.
- Match upstream CUDA tolerances when adjusting XPU tolerances
- Remove unused imports when removing skip decorators

### Commit Hygiene
- Keep commits focused: one fix per commit
- **Never cherry-pick** upstream fixes into the fix branch. If a fix already landed on trunk after the CI commit, rebase the fix branch onto the latest trunk (`git rebase origin/main`) instead.

### Build Safety
- **Always rebuild after rebase or branch switch.** After `git rebase`, `git checkout`, or any operation that changes the commit base, you MUST rebuild (`source build_pytorch.env && pip install -e . -v --no-build-isolation`) before running any tests. Without rebuilding, the installed C++ extensions and generated code are stale and test results are **completely unreliable** — they may segfault, produce wrong pass/fail results, or mask real issues. Never trust test results from a stale build.
- Editable installs resolve Python from source but C++ headers from the installed location (`torch/include/`). After editing a C++ header, **manually copy** it to the installed include path.
- Delete the PCH cache (`/tmp/torchinductor_$USER/precompiled_headers/`) after modifying any header under `torch/csrc/inductor/cpp_wrapper/` — stale precompiled headers mask the fix.
- For C++ compile errors in AOT Inductor generated code (`CppCompileError`), read the generated `.wrapper.cpp` error message carefully — the root cause is usually in the **codegen ordering** in `cpp_wrapper_cpu.py` (e.g. a function used before its definition is emitted). Check `write_wrapper_decl()` and `generate_input_output_runtime_checks()` ordering.

## Requirements

- PyTorch built from source with XPU support
- `build_pytorch.env` file configured for the oneAPI environment
