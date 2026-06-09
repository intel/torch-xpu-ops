---
name: test-verification
description: >
  Run a test and report whether a bug still reproduces (before a fix) or is
  resolved (after a fix). In interactive mode you usually already have the
  command and just run it; in pipeline mode you resolve a raw CI reference to a
  local command first.
---

# Test Verification

> **Execution mode:** this skill behaves differently in interactive (default)
> vs pipeline mode. See [../references/execution-modes.md](../references/execution-modes.md).
> For environment activation and build commands, see
> [../references/environment-setup.md](../references/environment-setup.md).

## When to use this skill

Run a test command and report whether the bug still reproduces (before a fix) or
is resolved (after a fix). How much work you do depends on the mode:

- **Interactive mode (default):** you already have a concrete command (the user
  or the issue gave it to you, or `issue-fix` is calling you with one). Skip
  path resolution and go straight to the [Quick path](#quick-path-interactive).
- **Pipeline mode:** you only have a raw CI reference (CI metadata, a bare
  filename, or a `test/xpu/...` path) that must be resolved to a real local path
  and selector first. Follow [Full resolution](#full-resolution-pipeline).

`$PYTORCH_DIR` is the local PyTorch checkout (e.g. `~/pytorch`); all test paths
are relative to it. If it or the Python environment is missing, ask for it or
derive it before proceeding.

## Quick path (interactive)

When you already have a runnable command:

1. **Activate the environment** — see
   [../references/environment-setup.md](../references/environment-setup.md)
   (MANDATORY; without it XPU tests collect 0 items).
2. **Run the command as given**, from `$PYTORCH_DIR/`. Fix only obvious path
   typos; do not redesign it. If you changed C++/SYCL code, rebuild first
   (see environment-setup).
3. **Interpret and report** — see [Interpreting results](#interpreting-results)
   and tell the user/caller in plain language. No JSON required.

If the given command turns out to be a raw/unresolved CI reference, drop into
[Full resolution](#full-resolution-pipeline) for just the resolution you need.

## HARD RULES
- NEVER commit or push anything. Only do verification.
- NEVER skip the file existence check (full-resolution path).
- If you cannot resolve the path, report CANNOT_VERIFY. Do NOT guess.
- If 0 tests are collected, report CANNOT_VERIFY, NEVER report PASSED.
- If all tests are xfailed, report FAILED (xfail = bug still exists).
- If all tests are skipped by `@skipIfXpu`, remove the decorator and re-run. You MAY modify test files ONLY to remove skip/xfail decorators — revert after running.
- TIME BUDGET: Complete within 5 minutes. If stuck, report CANNOT_VERIFY.

<!-- Below are pipeline mode only -->

## Full resolution (pipeline)

Use this when the input is a raw CI reference rather than a runnable command.

### Step 1: Read the issue

Extract the test reference from one of:
- **Reproducer** section — a bash/pytest/python command
- **Failed Tests** section — test paths like `file.py::Class::method`
- **CI metadata** — e.g. `op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,test_method`

Identify the **raw test file path** and the **test selector** (class, method, `-k` filter).

### Step 2: Resolve the path

The working directory is always `$PYTORCH_DIR/`. All test paths must be relative to it.

1. **`test/xpu/*.py`** → actual file is `third_party/torch-xpu-ops/test/xpu/*.py`
   - Example: `test/xpu/test_sparse_xpu.py` → `third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py`
2. **Bare filename** (e.g. `test_ops.py`) → search the filesystem:
   ```bash
   find $PYTORCH_DIR/ -name "test_ops.py" -not -path "*/.git/*" -not -path "*/build/*" -not -path "*/.venv/*" -not -path "*__pycache__*" | head -20
   ```
   - For XPU issues with multiple matches, prefer files under `third_party/torch-xpu-ops/` or `test/xpu/`; if only one match → use it.
3. **Upstream pytorch paths** (e.g. `test/nn/test_embedding.py`) → use as-is, relative to `$PYTORCH_DIR/`.
4. **CI metadata** — convert dots to slashes:
   - `third_party.torch-xpu-ops.test.xpu.test_ops_xpu` → `third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py`
   - The class and method after the last module component become the test selector.
5. **Paths starting with `third_party/torch-xpu-ops/`** → use as-is.

**Validate the file exists:**
```bash
ls -la $PYTORCH_DIR/<resolved_path>
```
If it does not exist, report `CANNOT_VERIFY` with a reason.

### Step 3: Validate the test selector

For a class/method selector (e.g. `::TestSparseAnyXPU::test_gradcheck_mm_...`):

**Dynamic test classes** — many XPU/CUDA/CPU classes are generated via
`instantiate_device_type_tests` (`<BaseClass>` + device suffix; e.g.
`TestSparseAnyXPU` from `TestSparseAny`). Validate with:
```bash
grep "class TestSparseAny" $PYTORCH_DIR/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py
grep "instantiate_device_type_tests" $PYTORCH_DIR/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py
```
If the base class exists AND `instantiate_device_type_tests` is present, the
dynamic class is valid. For static classes, grep the class name directly.

**`-k` filters** — you cannot easily validate without running. Use
`--collect-only`:
```bash
cd $PYTORCH_DIR && pytest --collect-only -q "<file>" -k "<filter>" 2>&1 | tail -5
```
If `0 items / no tests collected` → the filter matches nothing → `CANNOT_VERIFY`.

### Run

Activate the environment (see
[../references/environment-setup.md](../references/environment-setup.md)), then
run from `$PYTORCH_DIR/`:
- **pytest**: `pytest -xvs "<resolved_path>::Class::method"`
- **python unittest**: `python <resolved_path> Class.method`

Keep the original command format (pytest vs python) — just fix the paths.
Timeout: 10 minutes.

## Interpreting results

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

**`@skipIfXpu` / `@xfailIfXpu` decorators.** If ALL tests are skipped because of
`@skipIfXpu` or similar, first **remove the skip decorator** and re-run — the
skip is often the bug itself (the issue wants the test to pass on XPU). Report
the actual result afterward. Only report `CANNOT_VERIFY` for other reasons
(file not found, 0 collected, timeout).

## Output (pipeline mode)

In **interactive mode**, report the result conversationally — no JSON needed.

In **pipeline mode**, output EXACTLY this JSON block as the LAST thing in your
response, with no text after it:

```json
{
  "status": "PASSED" | "FAILED" | "CANNOT_VERIFY",
  "refined_command": "the exact command that was run (with corrected paths)",
  "original_command": "the raw test reference from the issue",
  "reason": "one sentence explaining the result",
  "output_tail": "last 30 lines of test output"
}
```

## Issue-body status (pipeline mode only)

In interactive mode (default), report the verification result to the
user/orchestrator and do not write to the issue body. See
[../references/execution-modes.md](../references/execution-modes.md) for the full
contract.

This skill runs at two points: verifying the failure reproduces (before triage)
and confirming the fix (after `issue-fix`). In pipeline mode this stage owns the
`<!-- agent:verification-log -->` slot and the "Fix verified locally" Action
Item (check it only on a post-fix `PASSED`).
