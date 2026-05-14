---
name: xpu-issues-triaging
description: >
  Instructions for issue triaging. Works for both pytorch and torch-xpu-ops repos.
---

# Triage & Fix XPU Issue

## Step 0: Triage the Issue

1. **Check issue labels.** If the issue has the label **"task"**, it is a
   tracking issue — do NOT make any code changes or create a PR. Simply
   acknowledge the task and stop. Then leave the comment in the issue on why you stopped (e.g., "This is a tracking issue, so no code changes will be made.").

2. **Classify the issue type.** Determine which category the issue falls into:
   - **A) XPU kernel / operator bug** — failure in XPU-specific operator code.
   - **B) PyTorch core bug** — failure in device-agnostic or framework code
     that happens to surface on XPU.
   - **C) CUDA UT porting issue** — a test originally written for CUDA was
     ported to XPU and fails due to porting gaps (see [Step 1b](#step-1b-cuda-ut-porting-issues)).

3. **Obtain PyTorch source for cross-reference.** If you are in the
   `torch-xpu-ops` repo, clone or fetch the PyTorch repo so you can inspect
   the upstream code:
   ```bash
   git clone --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch
   ```
   Use this checkout to compare CUDA kernels, check upstream fixes, and
   verify device-agnostic code paths. Reference files from `/tmp/pytorch/`
   throughout the remaining steps.

4. **Check which repo you're in:** `basename $(git rev-parse --show-toplevel)`
   - `torch-xpu-ops` → XPU kernel/operator code (files under `src/`)
   - `pytorch` → PyTorch core code (files under `torch/`, `aten/`, `test/`, `c10/`)

## Step 1: Investigate Before Coding

### Check if already fixed upstream
Before writing any fix, check whether the issue has already been resolved
on the PyTorch **main** branch:

1. Search the PyTorch repo for recent commits touching the relevant file(s)
   or function(s).
2. Search PyTorch GitHub issues / PRs for the same error message or test name.
3. If a fix already exists on PyTorch main, report that in your summary and
   **do not duplicate the fix** in torch-xpu-ops.

### Decide the right repo for the fix
- If the root cause is in **device-agnostic / framework code** (e.g.,
  `torch/`, `aten/src/ATen/`, `c10/`), the fix belongs in **pytorch**, not
  torch-xpu-ops. Do NOT submit a PR to torch-xpu-ops for such issues.
  Comment on the original issue to explain the root cause and where the fix belongs.
- If the root cause is in **XPU-specific kernel or dispatch code** (e.g.,
  `src/ATen/native/xpu/`), the fix belongs in **torch-xpu-ops**.

### Construct a reproducer
Extract the reproducer from the issue description. If absent, construct one
from the failed test name and error context. The reproducer should be a
standalone Python script or pytest command. Do NOT run the reproducer locally
(no XPU hardware is available in this environment).

### Step 1b: CUDA UT Porting Issues

If the issue is caused by a **CUDA unit test ported to XPU**, follow this
dedicated workflow:

1. **Locate the original CUDA test** — find the corresponding test in the
   PyTorch repo (usually under `test/`). Identify the CUDA-specific
   assertions, tolerances, dtypes, or device assumptions.
2. **Create a proper reproducer** — write a minimal reproducer that mirrors
   the CUDA test's logic but targets `device="xpu"`. Include the same
   inputs, dtypes, and expected outputs as the CUDA version.
3. **Diff CUDA vs XPU behavior** — compare what the CUDA test expects with
   what XPU produces. Note differences in:
   - Supported dtypes
   - Numerical tolerances (`atol` / `rtol`)
   - Operator dispatch paths
   - Device-specific decorators or skip conditions in the test
4. **Root cause** — determine whether the failure is due to:
   - Missing XPU kernel implementation → fix in `src/`
   - Incorrect test porting (wrong tolerance, missing dtype) → fix in
     `test/` (pytorch repo)
   - Genuine behavioral difference requiring XPU-specific handling
5. Continue to Step 2 with the root cause.

## Step 2: Propose the Fix

This is **issue-driven development** — the fix must address the root cause described in the issue,
not merely make a single reproducer pass.

Key principles:

- **Minimal changes** — fix only what's broken.
- **Align with CUDA** — when the feature does not depend on hardware-specific
  information (e.g., warp size, shared memory layout), the XPU kernel
  implementation should match the CUDA implementation's logic, tolerances,
  and behavior. Use the CUDA kernel as the reference.
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`.
- **Stay in your repo** — if in pytorch, don't modify `third_party/*`; if in
  torch-xpu-ops, only modify files in torch-xpu-ops.
- **Never modify unrelated files.**
- **Issue-driven, not reproducer-driven** — ensure the fix addresses the
  underlying issue. A reproducer passing is necessary but not sufficient;
  verify that the root cause is resolved and related code paths are correct.


## HARD RULES
- NEVER make code changes or create PRs for issues labeled **"task"**.
- NEVER submit a torch-xpu-ops PR for a bug whose root cause is in pytorch.
- NEVER add skip decorators. FIX the test.
- NEVER modify files outside your repo scope.
- NEVER modify unrelated files.

## Output
At the end, output:
```
### Agent Summary
- **Issue type:** <kernel bug / pytorch core bug / CUDA UT porting / task>
- **Fix repo:** <pytorch / torch-xpu-ops / N/A (already fixed or task)>
- **What I found:** <root cause in one sentence>
- **What I changed:** <bullet list of files, or "None" for task issues>
- **CUDA alignment:** <how the fix aligns with CUDA, or "N/A">
- **Open questions / risks:** <concerns or "None">
```
