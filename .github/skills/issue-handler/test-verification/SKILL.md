---
name: test-verification
description: >
  Verify whether a discovered CI failure still reproduces locally.
  Resolves test paths from issue descriptions to correct local paths,
  builds a runnable command, executes it, and reports the result.
---

# Test Verification

## Your Task
Given a raw test reference from a CI failure issue, resolve it to a correct
local command, run it, and report whether the bug still reproduces.

The **PyTorch Directory** is provided in the prompt (e.g. `~/pytorch`).
All references below use `$PYTORCH_DIR` — substitute the actual path from the prompt.

## Step 1: Read the Issue

Read the issue body (provided in the prompt). Extract the test reference from:
- **Reproducer** section — a bash/pytest/python command
- **Failed Tests** section — test paths like `file.py::Class::method`
- **CI metadata** — format like `op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,test_method`

Identify the **raw test file path** and the **test selector** (class, method, `-k` filter).

## Step 2: Resolve the Path

The working directory is always `$PYTORCH_DIR/`. All test paths must be relative to it.

### Path Mapping Rules

1. **`test/xpu/*.py`** → The actual file is at `third_party/torch-xpu-ops/test/xpu/*.py`
   - Example: `test/xpu/test_sparse_xpu.py` → `third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py`

2. **Bare filename** (e.g., `test_ops.py` with no directory) → Search the filesystem:
   ```bash
   find $PYTORCH_DIR/ -name "test_ops.py" -not -path "*/.git/*" -not -path "*/build/*" -not -path "*/.venv/*" -not -path "*__pycache__*" | head -20
   ```
   - If the issue is about XPU and multiple matches exist, prefer files under `third_party/torch-xpu-ops/` or `test/xpu/`
   - If only one match → use it

3. **Upstream pytorch test paths** (e.g., `test/nn/test_embedding.py`, `test/inductor/test_cuda_repro.py`) → Use as-is, they are relative to `$PYTORCH_DIR/`

4. **CI metadata format** — convert dots to slashes:
   - `third_party.torch-xpu-ops.test.xpu.test_ops_xpu` → `third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py`
   - The class and method after the last module component become the test selector

5. **Paths starting with `third_party/torch-xpu-ops/`** → Use as-is

### Validation
After resolving, **verify the file exists**:
```bash
ls -la $PYTORCH_DIR/<resolved_path>
```
If the file does not exist, report `CANNOT_VERIFY` with reason.

## Step 3: Validate the Test Selector

If the command has a test class or method selector (e.g., `::TestSparseAnyXPU::test_gradcheck_mm_...`):

### Dynamic Test Classes (PyTorch convention)
Many XPU/CUDA/CPU test classes are **dynamically generated** via `instantiate_device_type_tests`.
- `TestSparseAnyXPU` is generated from base class `TestSparseAny`
- `TestCommonXPU` is generated from `TestCommon`
- The pattern: `<BaseClass>` + device suffix (`XPU`, `CUDA`, `CPU`)

To validate:
```bash
# Check if the base class exists in the file
grep "class TestSparseAny" $PYTORCH_DIR/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py
# Check if instantiate_device_type_tests is called
grep "instantiate_device_type_tests" $PYTORCH_DIR/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py
```
If the base class exists AND `instantiate_device_type_tests` is present, the dynamic class is valid.

### Static Test Classes
For non-dynamic classes, just grep for the class name directly.

### `-k` Filters
For `-k` filters with specific test names, you cannot easily validate without running.
Use `--collect-only` to check:
```bash
cd $PYTORCH_DIR && pytest --collect-only -q "<file>" -k "<filter>" 2>&1 | tail -5
```
If `0 items / no tests collected` → the filter matches nothing → report `CANNOT_VERIFY`.

## Step 4: Build and Run the Command

### Environment Setup
**MANDATORY** — run these before ANY test or import of torch:
```bash
source ~/intel/oneapi/setvars.sh --force 2>/dev/null
source $PYTORCH_DIR/.venv/bin/activate
```
Without this, `torch.xpu.is_available()` returns False and XPU tests collect 0 items.

All commands run from `$PYTORCH_DIR/`.

### Command Formats
- **pytest**: `pytest -xvs "<resolved_path>::Class::method"`
- **python unittest**: `python <resolved_path> Class.method`
- Keep the original command format (pytest vs python) — just fix the paths

### Run
Execute the resolved command. Timeout: 10 minutes.

## Step 5: Interpret Results

### PASSED (bug is fixed)
- pytest exit code 0 with `N passed` and NO `xfailed`, NO `all skipped`
- unittest exit code 0 with `Ran N tests... OK` and actual tests ran

### FAILED (bug still exists)
- pytest exit code 1 with test failures
- unittest with `FAILED` or `ERROR`
- Any actual test failure output

### CANNOT_VERIFY
- `collected 0 items` / `no tests ran` — test selector matches nothing
- File not found on disk
- All tests skipped (can't confirm either way)
- All tests xfailed (expected failure = bug still present, report as FAILED)
- Timeout

**Important: @skipIfXpu / @xfailIfXpu decorators.** If ALL tests are skipped because of `@skipIfXpu` or similar decorators, first **remove the skip decorator** from the test, then re-run. The skip decorator is often the bug itself — the issue wants to make the test pass on XPU, not keep it skipped. After removing the skip, report the actual test result (PASSED or FAILED). Only report CANNOT_VERIFY if you cannot run the test for other reasons (file not found, 0 collected, timeout).

## Output

You MUST output EXACTLY this JSON block as the LAST thing in your response.
No text after the JSON block.

```json
{
  "status": "PASSED" | "FAILED" | "CANNOT_VERIFY",
  "refined_command": "the exact command that was run (with corrected paths)",
  "original_command": "the raw test reference from the issue",
  "reason": "one sentence explaining the result",
  "output_tail": "last 30 lines of test output"
}
```

## HARD RULES
- NEVER commit or push anything. Only do verification.
- NEVER skip the file existence check.
- If you cannot resolve the path, report CANNOT_VERIFY. Do NOT guess.
- If 0 tests are collected, report CANNOT_VERIFY, NEVER report PASSED.
- If all tests are xfailed, report FAILED (xfail = bug still exists).
- If all tests are skipped by `@skipIfXpu`, remove the decorator and re-run. You MAY modify test files ONLY to remove skip/xfail decorators — revert after running.
- TIME BUDGET: Complete within 3 minutes. If stuck, report CANNOT_VERIFY.
