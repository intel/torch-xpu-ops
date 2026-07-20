---
name: verify-xpu-test
description: Locally verify Intel XPU test enablement on an XPU host after the develop-xpu-test skill has extended a test class instantiation to include "xpu" and/or widened op_db DecorateInfo device_type entries to ('cuda', 'xpu'). Use to confirm the newly enabled XPU test variants run (pass/skip/xfail) cleanly, CUDA-only cells still behave, decorator parity updates (largeTensorTest and dtype decorators) are present for XPU, and to flag any widened expectedFailure that unexpectedly passes on XPU and must be reverted. Runs pytest for the target class and affected test_ops.py surfaces in the XPU conda environment.
---

# Verify XPU Test

Locally verify the XPU enablement produced by the `develop-xpu-test` skill. This
skill runs the affected tests on an XPU host and reports whether the enablement
is sound: XPU variants run cleanly, CUDA-only cells are unchanged, decorator
parity edits are present, and any widened `expectedFailure` that unexpectedly
*passes* on XPU is flagged for revert.

This skill does **not** edit source. It runs tests and reports. Any revert it
recommends is applied by the caller (`develop-xpu-test`) or the user.

## When to Use

- Immediately after `develop-xpu-test` has:
  - extended `instantiate_device_type_tests(..., only_for=("cuda", "xpu"), allow_xpu=True)`
    (or added a `HAS_GPU` guard), and/or
  - mirrored CUDA-only decorators to XPU in the test file (for
    `@largeTensorTest(..., "cuda")` and CUDA dtype decorators), and/or
  - widened `DecorateInfo(device_type='cuda')` entries to
    `device_type=('cuda', 'xpu')` in
    `torch/testing/_internal/common_methods_invocations.py`.
- Any time you need to confirm on the XPU host that an enabled test class or an
  affected `test_ops.py` op surface behaves as expected on XPU.

## Inputs

| Field | Required | Description |
|-------|----------|-------------|
| Test file path | For class verification | Absolute path to the enabled test file (e.g. `<repo>/test/<file>.py`). |
| Class name | For class verification | The enabled test class (e.g. `TestSDPAGpuOnly`). |
| op name(s) | For op_db verification | The op(s) whose `DecorateInfo` entries were widened (drives the `test_ops.py` filter). |
| conda env | No | Conda env with PyTorch (XPU). Default: `pytorch_opencode_env`. |

## Tools Used

- **bash**: activate the XPU conda env and run pytest.
- **grep / Read**: inspect pytest output for XPU rows and outcomes.

## Workflow

### Step 0: Diff scope check — no unrelated code touched (HARD GATE)

Before running any test, inspect the pending diff and confirm it is limited to
the target test class. Enablement for one test class must never modify op_db
entries (or any other code) that belong to a *different* test class or test name.

```bash
cd <repo>
git diff --stat
git diff -- torch/testing/_internal/common_methods_invocations.py
git diff -- test/<file>.py
```

Checks:

1. **Enumerate the target class's generic test names.** Read the target test
   class body and list its test methods (the `@ops(...)`-decorated methods and
   any plain `test_*` methods). Use the base class name (e.g. `TestComplexTensor`),
   not the device-suffixed instantiated name.

2. **Every changed `DecorateInfo` must be in scope.** For each `+`/`-` line in
   the `common_methods_invocations.py` diff, confirm the enclosing `DecorateInfo`
   references the **target class name** AND one of its **generic test names**.
   Any changed `DecorateInfo` that references a different test class
   (`TestForeach`, `TestUnaryUfuncs`, `TestInductorOpInfo`, `TestSparseCSR`,
   `TestCommon`, `TestConsistency`, ...) or a test name the target class does not
   run is **out of scope**.

3. **No unrelated files.** The diff should touch only the target test file and
   (optionally) `common_methods_invocations.py`. Any other changed source file is
   out of scope unless the user explicitly asked for it. Ignore untracked
   dev-only paths such as `third_party/torch-xpu-ops/`.

**Decision:**

- **In-scope only** (or op_db unchanged) → proceed to Step 1.
- **Any out-of-scope change found** → set overall verdict to **out-of-scope
  changes** (a failing verdict). Do NOT proceed to run tests as if verified.
  Report the exact out-of-scope `DecorateInfo` lines / files and instruct the
  caller (`develop-xpu-test`) to revert them so the diff is limited to the
  target class. Only after the diff is re-scoped should verification continue.

Rationale: enabling `TestComplexTensor` must not widen `TestForeach` /
`TestUnaryUfuncs` / etc. op_db entries. If the target class has no matching
op_db entries at all, the correct diff has `common_methods_invocations.py`
unchanged.

### Step 1: Sanity — confirm XPU is available

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp

python -c "import torch; print('cuda:', torch.cuda.is_available(), ' xpu:', torch.xpu.is_available())"
# Expect xpu: True on the XPU validation host.
```

If `torch.xpu.is_available()` returns `False`, the env is broken (wrong wheel,
missing driver, etc.) — fix the env; do NOT skip verification.

### Step 2: Run the enabled test class

```bash
# Run the enabled class; XPU rows should pass or skip cleanly.
python -m pytest <repo>/test/<file>.py -v -k "<ClassName>" --tb=short | tee /tmp/enable_xpu.txt

# Targeted XPU-only filter for fast iteration:
python -m pytest <repo>/test/<file>.py -v -k "<ClassName> and xpu" --tb=short | tee /tmp/enable_xpu.xpu.txt
```

### Step 3: Run the affected `op_db` / `test_ops.py` surfaces

For op_db changes, run the relevant `test_ops.py` surface for each affected op
on XPU, e.g.:

```bash
python -m pytest <repo>/test/test_ops.py -v -k "<op_name> and xpu" --tb=short | tee /tmp/enable_xpu.ops.txt
```

### Step 4: Evaluate against pass criteria

Pass criteria:
- **At least one XPU row is exercised**:
  `grep -E 'xpu.*PASSED|xpu.*SKIPPED|xpu.*XFAIL' /tmp/enable_xpu.xpu.txt`.
  An empty XPU axis means `only_for=("cuda", "xpu")` / `allow_xpu=True` is
  missing or not taking effect — report this as a failure of the enablement.
- **No unexplained new XPU failures.** A registered `expectedFailure` widened to
  `('cuda', 'xpu')` should now show the XPU variant as an expected failure
  (`XFAIL`), not a hard `FAILED`/`ERROR`.
- **CUDA-only cells are unchanged.** They skip on the XPU host because
  `torch.cuda.is_available()` is False there — that is expected, not a
  regression.

### Step 5: Verify decorator parity edits in the test file

Check that requested XPU decorator parity edits are present in the modified test
file.

1. **`largeTensorTest` parity check**
   - If the file has `@largeTensorTest("20GB", "cuda")`, it must also have
     `@largeTensorTest("20GB", "xpu")`.
   - More generally, for each CUDA large-tensor decorator that was mirrored by
     develop-xpu-test, confirm the XPU version exists with the same size scope.

2. **dtype decorator parity check**
   - If a CUDA dtype decorator is present (`@dtypeIfCuda(...)` or
     `@dtypesIfCUDA(...)`) and was mirrored for XPU by develop-xpu-test,
     confirm the XPU counterpart exists with the same dtype scope.

3. **Import check for XPU dtype decorator**
   - Confirm the test file includes the requested import:

   ```python
   from torch.testing._internal.common_device_type import dtypeIfXpu
   ```

   (or an equivalent grouped import line that includes `dtypeIfXpu`).

### Step 6: Flag entries to revert

If a widened `expectedFailure` does **not** actually fail on XPU (i.e. the op
passes on XPU, showing as `XPASS`/`PASSED` where an xfail was expected), that
entry should not have been widened for that op. Flag it for revert: the caller
must revert that single `DecorateInfo` entry back to `device_type='cuda'` and
leave it CUDA-only. Do not leave a widened `expectedFailure` that
unexpectedly-passes on XPU.

## Output

Return a concise verification report:

- Diff scope check: **in scope** (only target class touched / op_db unchanged) or
  **out-of-scope changes** — list any changed `DecorateInfo`/file that references
  a different test class or test name.
- XPU rows exercised: yes/no (with a count).
- New XPU failures/errors: list any (test id + short reason), or "none".
- Decorator parity checks: pass/fail for `largeTensorTest`, dtype parity, and
  `dtypeIfXpu` import.
- Widened `expectedFailure` entries that unexpectedly pass on XPU (must revert):
  list `(op, test_class, test_name)`, or "none".
- CUDA-only cells: "unchanged (skipped on XPU host)" or list any regressions.
- Overall verdict: **verified** / **out-of-scope changes** / **needs revert** /
  **enablement not effective**.

## Constraints

1. **Read-only on source.** This skill runs tests only; it does not edit the
   test file or `common_methods_invocations.py`. Recommend reverts; the caller
   applies them.
2. **Diff scope is a hard gate.** If the diff touches any `DecorateInfo` for a
   test class/name outside the target class, or any unrelated source file, the
   verdict is **out-of-scope changes** and the enablement is not verified until
   the diff is re-scoped to the target class (op_db unchanged when no entry
   matches).
3. **Run from `/tmp`** to avoid a local pytorch checkout shadowing the conda
   env's installed torch.
4. **Do not install packages.** If `torch` is not importable, report the broken
   env and stop — do not attempt to install or work around it.
5. **CUDA numerics are CI's job.** The XPU host has no CUDA; CUDA-only cells
   skipping is expected and is not a failure. This gate proves XPU enablement
   works and did not break CUDA-cell skip predicates — not CUDA correctness.

## See Also

- `develop-xpu-test` — produces the enablement edits this skill verifies, and
  applies any revert this skill recommends.
- `submit-xpu-test-pr` — packages the verified edits into a confirm-gated draft PR.
