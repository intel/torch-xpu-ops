---
name: port-to-xpu-ops
description: Port PyTorch unit tests to torch-xpu-ops for XPU backend coverage. Use when copying/migrating tests from pytorch/test to third_party/torch-xpu-ops/test/xpu. Covers two approaches (direct copy and hook override), enforces hook-body parity with upstream (no assertions/checkers/branches removed), requires a recursive word-level diff audit before commit, and requires filing intel/torch-xpu-ops issues with a Context section linking back to the PR for any XPU-side failure. Follows agent-guidelines for atomic commits and semantic analysis.
---

# Port PyTorch Tests to torch-xpu-ops

This skill guides the porting of PyTorch unit tests to torch-xpu-ops repo for XPU backend coverage.

## Skill Integration

**This skill follows agent-guidelines AND extends it with specific constraints.**

Always apply agent-guidelines rules including:
- Mandatory post-write commit protocol (ask user before committing)
- Deep semantic analysis instead of pattern matching
- Atomic commits for each ported test
- All constraints defined in agent-guidelines

## Related Skills
- **agent-guidelines** (REQUIRED): Must be loaded first for behavior rules
- **submit_ut_issues** (REQUIRED when filing issues): Defines the issue template, labels, GitHub API workflow, and the mandatory Context section that cross-links the porting PR
- **at-dispatch-v2**: For C++ kernel type dispatch work
- **add-uint-support**: For unsigned integer type additions

## When to Use This Skill

Use when:
- Adding XPU test coverage for pytorch operators
- Migrating CUDA-only tests to XPU
- Creating XPU counterparts for existing pytorch tests
- Fixing missing tests identified in CI/test collection

## Key Concepts

### torch-xpu-ops Test Structure

```
third_party/torch-xpu-ops/
├── test/xpu/
│   ├── test_meta_xpu.py           # XPU meta dispatch tests
│   ├── test_transformers_xpu.py   # XPU transformer tests
│   ├── test_modules_xpu.py        # XPU module tests
│   └── xpu_test_utils.py          # Test utilities & XPUPatchForImport
```

### Porting Approaches

There are **two porting approaches** depending on test complexity:

#### Approach 1: Direct Copy (NO XPUPatchForImport)

Used when copying entire test file with minimal modifications.

**Examples:** `test_transformers_xpu.py`, `test_meta_xpu.py`

**Characteristics:**
- Standalone test file copied from pytorch/test
- No XPUPatchForImport usage
- Imports pytorch test code directly
- Modifies instantiations for XPU enablement

**Pattern:**
```python
# Example instantiation modification
instantiate_device_type_tests(
    TestSDPACudaOnly, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)
```

#### Approach 2: Hook Override (WITH XPUPatchForImport)

Used when selectively overriding specific tests/methods without copying entire files.

**Examples:** `test_modules_xpu.py`

**Characteristics:**
- Uses `XPUPatchForImport(False)` context
- Imports only for method override access
- Overrides specific tests/methods after import

**Pattern:**
```python
from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_modules import TestModule

# Override specific methods
TestModule._test_gradients_helper = _gradients_helper
TestModule.test_multiple_device_transfer = _test_multiple_device_transfer
```

## Workflow Steps

### Step 1: Find CUDA Test and Map to XPU

1. Locate the CUDA test in pytorch repo:
   ```bash
   grep "test_name" test/*.py  # Find source test
   ```

2. Map CUDA test naming to XPU by changing suffix:
   ```
   test_name_cuda -> test_name_xpu
   ```

3. Check if XPU version already exists:
   ```bash
   grep "test_name_xpu" third_party/torch-xpu-ops/test/xpu/*.py
   ```

### Step 2: Update/Create XPU Test in torch-xpu-ops

If test exists in `third_party/torch-xpu-ops/test/xpu/`:
- Edit the existing test file directly
- Keep changes localized to torch-xpu-ops repo
- No `skip_list_common.py` change is required for an existing entry.

If test does not exist:
- Copy from pytorch/test to torch-xpu-ops/test/xpu/
- Add `_xpu` suffix to file and class names
- Use appropriate porting approach (see Porting Approaches)
- **Register the new file in `test/xpu/skip_list_common.py` (see Step 2b)**. The op_ut CI runner does not auto-discover `test/xpu/*.py`; it iterates `skip_dict`. Skipping this step means the new tests silently never run in CI even though local pytest discovers them.

### Step 2b: Register New File in `skip_list_common.py` (only when Step 2 created a new file)

Local pytest collection is necessary but NOT sufficient — the CI op_ut suite enumerates files exclusively through `skip_dict` in `test/xpu/skip_list_common.py`. New files that are not added are silently dropped from CI coverage (this was the root cause of intel/torch-xpu-ops PR #3427's first run landing zero coverage of the new file).

1. Open `third_party/torch-xpu-ops/test/xpu/skip_list_common.py`.
2. Append at the end of `skip_dict` (before the closing `}`). The key is the file path **relative to `test/xpu/`**, matching sibling entries:

```python
skip_dict = {
    ...
    "functorch/test_aotdispatch_xpu.py": None,
    "<subdir>/test_<module>_xpu.py": None,   # use None when no per-test skips needed
}
```

3. If the port intentionally requires per-test skips (Approach 2 with hook-driven xfail/skip is preferred over `skip_dict` tuples; see Approach 2), use a tuple of substrings only when the alternative is filing infrastructure issues that cannot be resolved on this PR. Match existing styles such as `test_fake_tensor_xpu.py` and add a comment with the tracking-issue URL.
4. Verify the file parses and the entry exists:

```bash
python3 -c "from importlib.util import spec_from_file_location, module_from_spec; \
  s = spec_from_file_location('s', 'third_party/torch-xpu-ops/test/xpu/skip_list_common.py'); \
  m = module_from_spec(s); s.loader.exec_module(m); \
  k = '<subdir>/test_<module>_xpu.py'; \
  assert k in m.skip_dict, f'{k} not registered'; print('OK', k, '->', m.skip_dict[k])"
```

5. Stage `skip_list_common.py` together with the new test file in the same atomic commit (Step 7). A commit that adds the test file without the registration is incorrect.

### Step 3: Run Test in Conda Environment

```bash
# Activate pytorch conda environment
source ~/miniforge3/bin/activate pytorch_opencode_env

# Run from torch-xpu-ops test directory with junit output
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu
pytest -v --junit-xml=test_<name>_xpu.xml dynamo/test_<name>_xpu.py

# For specific test filtering
pytest -v --tb=short dynamo/test_<name>_xpu.py -k "test_pattern"
```

### Step 4: Analyze Why Test Does Not Run

Common reasons for tests not discovering or running:

1. **Missing backend in platform list** - Add missing backends
2. **OpInfo dtypesIf condition** - Patch dtypesIf for XPU (see Step 5)
3. **Device restriction decorators** - Extend to include XPU
4. **Skip decorators** - Adjust for XPU-specific conditions
5. **Platform check failures** - Review `PLATFORM_*` variables

### Step 5: Enable Test with Appropriate Solution

#### Solution A: Add Backend to Platform List

```python
# Add missing backend for XPU
if TEST_XPU and SDPBackend.MISSING not in PLATFORM_SPECIFIC_SDPA:
    PLATFORM_SPECIFIC_SDPA.append(SDPBackend.MISSING)
```

#### Solution B: Patch OpInfo dtypesIfCUDA

When `SM53OrLater` or similar conditions exclude XPU:

```python
# Patch at module top
bf16 = torch.bfloat16
_ops = ["op1", "op2"]
for _op in op_db:
    if _op.name in _ops:
        for _dtype_list in [_op.dtypesIfCUDA, _op.dtypesIfXPU, _op.dtypesIf.get("xpu")]:
            if _dtype_list is not None and bf16 not in _dtype_list:
                _dtype_list.add(bf16)
```

**Constraint:** Mutate in place with `.add()`, never use direct assignment for OpInfo dtypes.

#### Solution C: Update Device-Specific Checks

Ensure CUDA-specific checks include device parameter:
```python
# Before (CUDA only)
if backend == SDPBackend.X and randomness == "different":

# After (CUDA + XPU)
if backend == SDPBackend.X and randomness == "different" and device == "cuda":
```

### Step 6: Check Intel torch-xpu-ops Issues for Known Failures

If test fails after enabling:

1. Search intel/torch-xpu-ops GitHub issues:
   - https://github.com/intel/torch-xpu-ops/issues

2. Check for similar documented issues:
   - Keyword: test name or error pattern
   - Label: `xpu`, `backend`, specific operator

3. If known issue found:
   - Document issue URL in commit/PR
   - Enable test anyway (will fail with tracked limitation)
   - Do NOT add skip unless directed

### Step 7: Verify and Commit

```bash
# Verify test discovers
python -m pytest test_vmap_xpu.py --collect-only 2>&1 | grep "test_name"

# Run and verify behavior
python -m pytest test_vmap_xpu.py -k "test_name_xpu" -v --tb=short

# Review diff
git diff third_party/torch-xpu-ops/test/xpu/

# Commit with descriptive message
git add third_party/torch-xpu-ops/test/xpu/test_name_xpu.py
git commit -m "XPU: Enable test_name_xpu tests

Enables XPU coverage for:
- test_name_xpu_variant1
- test_name_xpu_variant2

Solution: [rief description of change]
Reference: https://github.com/intel/torch-xpu-ops/issues/XXXX
"
```

## Key Files Reference

### xpu_test_utils.py Utilities

| Component | Purpose |
|-----------|---------|
| `XPUPatchForImport` | Main class for importing pytorch tests to XPU context |
| `dtypesIfXPUMock` | Mocks dtypesIfCUDA -> dtypesIfXPU translation |
| `align_supported_dtypes` | Aligns op_db dtypes for XPU hardware capabilities |
| `onlyXPU` | XPU device restriction decorator |
| `skipXPU` | XPU skip decorator |

## Dtype Alignment: Core Pattern

**Critical Discovery:** CUDA tests use `@dtypesIfCUDA` decorators for GPU-specific dtypes. XPU tests MUST have equivalent `@dtypesIfXPU` decorators to enable the same dtype coverage.

### The Pattern

Each test with `@dtypesIfCUDA(X, Y, Z)` needs a corresponding `@dtypesIfXPU(X, Y, Z)`:

```python
# CUDA source test_nn.py:
@dtypesIfCUDA(torch.half, torch.float)
def test_softmax_results(self, device, dtype):
    ...

# XPU aligned test_nn_xpu.py:
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,  # MUST import dtypesIfXPU
    instantiate_device_type_tests,
    # ... other imports
)

@dtypesIfCUDA(torch.half, torch.float)
@dtypesIfXPU(torch.half, torch.float)  # ALIGN WITH CUDA!
def test_softmax_results(self, device, dtype):
    ...
```

### Step-by-Step Dtype Alignment

1. **Find CUDA dtypesIfCUDA decorator** for the test:

   ```bash
   grep -B1 "def test_name" test/test_nn.py | grep "@dtypesIfCUDA"
   # Example output: @dtypesIfCUDA(torch.half, torch.float, torch.double)
   ```

2. **Check XPU file** for existing `@dtypesIfXPU`:

   ```bash
   grep -A1 "@dtypesIfXPU" third_party/torch-xpu-ops/test/xpu/test_nn_xpu.py
   ```

3. **Add/update @dtypesIfXPU** to match @dtypesIfCUDA exactly:

   ```python
   # Before (WRONG - missing dtypes):
   @dtypesIfCUDA(torch.half, torch.float, torch.double)
   @dtypesIfXPU(torch.half)  # WRONG!

   # After (CORRECT - aligned with CUDA):
   @dtypesIfCUDA(torch.half, torch.float, torch.double)
   @dtypesIfXPU(torch.half, torch.float, torch.double)  # CORRECT!
   ```

4. **Verify import** - ensure `dtypesIfXPU` is imported:

   ```python
   from torch.testing._internal.common_device_type import (
       dtypes,
       dtypesIfCUDA,
       dtypesIfXPU,  # Add this import
       # ... other imports
   )
   ```

5. **Collect tests** to verify dtype variants appear:

   ```bash
   python -m pytest test_nn_xpu.py -k "test_name_xpu" --collect-only 2>&1 | grep "Function"
   # Should see: test_name_xpu_float16, test_name_xpu_float32, etc.
   ```

### Common Dtype Gaps Found

| CUDA dtypesIfCUDA | XPU Required dtypesIfXPU |
|-------------------|-------------------------|
| `torch.half` | `torch.half` (float16) |
| `torch.float` | `torch.float` (float32) |
| `torch.double` | `torch.double` (float64) |
| `torch.bfloat16` | `torch.bfloat16` (bfloat16) |
| `torch.complex128` | `torch.complex128` |
| Combinations | Match exactly |

### Validation Checklist

After adding @dtypesIfXPU decorators:

- [ ] `dtypesIfXPU` imported in test file header
- [ ] Each `@dtypesIfCUDA(A, B, C)` has corresponding `@dtypesIfXPU(A, B, C)`
- [ ] Test collection shows expected dtype variants: `test_name_xpu_float16`, `test_name_xpu_bfloat16`, etc.
- [ ] Tests pass (or fail with documented known limitations)

### XPUPatchForImport Initialization

```python
class XPUPatchForImport:
    def __init__(self, patch_test_case=True) -> None:
        test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../../test"
        )
        # Patches: onlyCUDA, dtypesIfCUDA, onlyNativeDeviceTypes, etc.
```

### Method `_enter__` Key Patches

```python
# Key patches in __enter__
common_device_type.onlyCUDA = common_device_type.onlyXPU
common_device_type.skipXPU = _skipXPU
common_device_type.dtypesIfCUDA = get_dtypesIf_mock("cuda")
common_device_type.dtypesIfXPU = get_dtypesIf_mock("xpu")
```

## Common Issues & Solutions

### Issue 1: Test Not Discovered

**Symptom:** Test exists in pytorch but not in torch-xpu-ops collection. Two distinct flavors:
- (a) `pytest --collect-only test/xpu/<file>` finds nothing or errors out, **or**
- (b) local `pytest --collect-only` works fine but the file produces no `op_ut_with_all.<module>.xml` artifact in CI (i.e. CI never ran the file even though local does).

**Check:**
1. **For flavor (b) first:** confirm the file's relative path is a key in `test/xpu/skip_list_common.py`'s `skip_dict`. The op_ut CI runner enumerates only files listed there — anything not in the dict is silently skipped at the suite level. This is the most common cause of "added a new file, CI is green, but the new tests never ran".
2. Check if test exists in torch-xpu-ops test directory.
3. OpInfo skip decorators or device restrictions.
4. dtypesIf condition not including XPU.

**Fix:**
- Flavor (b): add the file to `skip_dict` per Step 2b.
- Flavor (a): apply Steps 2-5 above.

### Issue 2: dtypesIf Setter Error

**Symptom:** `AssertionError: Expected _dispatch_dtypes or None`

**Cause:** Direct assignment bypasses OpInfo property setter

**Fix:** Mutate in place using `.add()`:
```python
_dtype_list.add(torch.bfloat16)
```

### Issue 3: Backend Not Supported on XPU (Known Limitation)

**Symptom:** Test fails with unknown backend or no viable backend error

**Example:** `RuntimeError: No viable backend for scaled_dot_product_attention was found`

**Root cause:** XPU lacks cuDNN hardware/software for certain backends

**Decision:**
- If backend is known unsupported on XPU → enable test anyway (tracks limitation)
- Check intel/torch-xpu-ops issues for documented cases

**Implementation:**
```python
# Force add unsupported backend to platform list
if TEST_XPU and SDPBackend.UNSUPPORTED not in PLATFORM_LIST:
    PLATFORM_LIST.append(SDPBackend.UNSUPPORTED)

# Remove skip that blocks XPU testing:
# if device == "xpu" and backend == SDPBackend.UNSUPPORTED:
#     raise unittest.SkipTest("...")  # REMOVE this
```

### Issue 4: Import Errors After Copying

**Symptom:** `ImportError` or `NameError` during test collection after copy

**Root cause analysis path:**
1. Check pytorch source test folder for same import:
   ```bash
   ls pytorch/test/<subdir>/test_*.py | head -10
   grep "def requires_gpu\|requires_gpu =" pytorch/test/<subdir>/*.py | head -10
   ```
2. Check existing xpu tests in same folder for reference:
   ```bash
   head -30 test/xpu/<subdir>/test_<other>_xpu.py
   ```

**Common patterns and fixes:**

#### Pattern: Missing requires_cuda/requires_gpu
Original pytorch may use these but they may not be available in installed torch.

**Fix:** Define locally using device availability:
```python
import torch
import unittest

# Define requires_gpu locally (not available in installed pytorch)
requires_gpu = unittest.skipUnless(
    torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available()),
    "requires cuda or xpu"
)
```

#### Pattern: Cross-file imports (from . import sibling_module)
When source test imports sibling modules with relative imports.

**Fix:** Add path to pytorch test source folder:
```python
from pathlib import Path
import sys

# Path: test/xpu/dynamo/file.py -> test/dynamo/ is 5 levels up
PYTORCH_TEST_PATH = str(Path(__file__).resolve().parents[5] / "test" / "dynamo")
if PYTORCH_TEST_PATH not in sys.path:
    sys.path.insert(0, PYTORCH_TEST_PATH)

from test_functions import *  # Now will find pytorch source
```

**Rule**: Use `parents[5]` for `test/xpu/<subdir>/` to reach `test/<subdir>/` in pytorch.

#### Pattern: requires_gpu_and_triton
Use from triton_utils if available:
```python
from torch.testing._internal.triton_utils import requires_gpu_and_triton
```

## Template Outline

### For Direct Copy Test

```python
# third_party/torch-xpu-ops/test/xpu/test_name_xpu.py
"""
XPU specific tests for test_name.
Modified from pytorch/test/test_name.py
"""

import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, instantiate_device_type_tests
)
from xpu_test_utils import XPUPatchForImport

# Patch op_db if needed for dtype coverage
for _op in op_db:
    if _op.name in ["op1", "op2"]:
        for _dtype_list in [_op.dtypesIfCUDA, _op.dtypesIfXPU, _op.dtypesIf.get("xpu")]:
            if _dtype_list is not None and torch.bfloat16 not in _dtype_list:
                _dtype_list.add(torch.bfloat16)

# Import source test
with XPUPatchForImport(False):
    from test_name import TestNameClass

# XPU instantiation
instantiate_device_type_tests(
    TestNameClass, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
```

### For Hook Override Test

```python
# third_party/torch-xpu-ops/test/xpu/test_name_xpu.py
"""
XPU specific test overrides for test_name.
Uses XPUPatchForImport for selective overrides.
"""

import torch
from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_name import TestNameClass

def _xpu_test_method(self, device):
    # XPU implementation
    ...

TestNameClass.test_method = _xpu_test_method
```

## Case Study: NestedTensor SDPA Tests

### Background

Tests like `test_sdpa_with_packed_in_proj` use nested tensors with SDPA. These tests may have CUDA-specific conditions:

```python
# NestedTensor SDPA test has PLATFORM check
@unittest.skipIf(
    not PLATFORM_SUPPORTS_FUSED_ATTENTION,
    "Platform doesn't support flash or mem-efficient attention",
)
```

### Challenge

XPU may lack support for certain features even when dtype alignment is correct. This results in tests that:
1. **Discover successfully** (dtype variants found)
2. **Fail at runtime** with "No viable backend" error

### Approach: Enable-But-Track-Fail

For XPU limitations (not bugs):

1. **Enable the test** - don't add skip just because XPU fails
2. **Document the known limitation** - reference intel/torch-xpu-ops issue
3. **Track the gap** - let CI show where kernel support is missing

```python
# Example: SDPA with packed in_proj on XPU has known limitation
@skipXPUIf(
    True,  # Always skip due to known limitation
    "XPU nestedtensor SDPA requires fused attention kernel support"
)
# OR use @expectedFailureXPU decorator
```

### Decision Matrix

| Situation | Action |
|-----------|--------|
| Type bug (wrong logic) | Fix the test implementation |
| Missing dtype coverage | Add @dtypesIfXPU to align with CUDA |
| Missing kernel on XPU | Enable anyway, document limitation, reference issue |
| XPU根本没实现功能 | Add skipXPUIf with clear message |

## Parity Requirement: Hook Bodies Must Match Upstream

**MANDATORY:** Any hook function or function copied from `pytorch/test/...` into `third_party/torch-xpu-ops/test/xpu/...` MUST preserve the exact logic of the original. In particular:

- **Do NOT remove any assertion** (`self.assertEqual`, `self.assertTrue`, `self.assertRaises*`, `assert_allclose`, `assertExpectedInline`, `assertIn`, `assertRegex`, etc.).
- **Do NOT remove any checker call** (`self.assertNoLogs`, `self.assertWarns*`, `self.assertNotWarns`, `assertNoOpResolved`, etc.).
- **Do NOT remove platform branches** (`if torch.version.hip:`, `if IS_WINDOWS:`, `if not TEST_WITH_ROCM:`, etc.). Substitute the inner CUDA-specific values rather than dropping the branch.
- **Do NOT silently weaken `assertExpectedInline(...)` to `assertIn(...)`**. If the upstream baseline cannot match on XPU, keep the upstream check, let it fail, and file an issue (see "Submitting Issues for XPU Failures" below).
- **Do NOT drop helper assertions inside loops**, e.g. comparisons against CPU/reference, dtype/shape sanity checks, gradient checks, or numerics tolerances.

**Allowed substitutions only:**
- `cuda` → `xpu` (`.cuda()` → `.xpu()`, `device="cuda"` → `device="xpu"`, `ProfilerActivity.CUDA` → `ProfilerActivity.XPU`).
- `torch.cuda.X` → `torch.xpu.X` when an exact XPU equivalent exists.
- `cudaLaunchKernel` → `xpuLaunchKernel` (or the actual XPU runtime event name once known).
- `TEST_CUDA` → `torch.xpu.is_available()` only when the decorator is the gating condition; do NOT use this as a way to drop branches.
- Comments referring to "CUDA" → "XPU" only when the comment is purely descriptive.

**Forbidden:**
- Replacing a real upstream check with a smoke check (`assertIn("aten::", trace)` instead of `assertExpectedInline(trace, expected)`).
- Removing `setUp` / `tearDown` logic.
- Removing nested classes or helper functions used only inside the hook.
- "Simplifying" away device-list iteration (e.g. collapsing `for device in ['cpu', 'xpu']` to `for device in ['xpu']`) without explicit justification.

**Required justification format if any logic IS dropped:**
A code comment immediately above the divergence stating:
1. What was removed.
2. Why it cannot run on XPU.
3. The intel/torch-xpu-ops issue tracking the gap.

Example:
```python
# XPU gap: upstream asserts cudaLaunchKernel but XPU profiler emits
# many Level Zero events per aten op. Tracked in intel/torch-xpu-ops#3483.
# Keep upstream assertExpectedInline anyway so the test fails loudly
# until parity lands.
self.assertExpectedInline(actual_traces, expected)
```

## Audit & Word-Level Diff (MANDATORY at end of porting work)

Before considering a port complete, run a recursive body-text audit comparing every ported hook to its upstream counterpart and emit word-level diffs.

### Required tools

Two helper scripts to drop in `agent_space/` or `/tmp/`:

1. **Strict body comparator** — uses `ast.unparse` recursively (so nested classes/methods are included), normalizes `cuda → xpu` and quote style, then runs `difflib.unified_diff`:

   ```python
   # /tmp/strict_body_diff.py (sketch)
   import ast, difflib, re
   def normalize(src):
       src = re.sub(r"\bcuda\b", "xpu", src)
       src = re.sub(r"\bCUDA\b", "XPU", src)
       src = re.sub(r"cudaLaunchKernel", "xpuLaunchKernel", src)
       src = src.replace("'", '"')
       return src
   def body_text(func):
       return "\n".join(ast.unparse(s) for s in func.body)
   ```

2. **Word-level diff emitter** — for every `(hook_func, upstream_method)` pair, emit a markdown file under `agent_space/per_function_diffs/<file>.diff.md` using `[-removed-]{+added+}` syntax:

   ```python
   # /tmp/diff_xpu_vs_upstream.py (sketch)
   from difflib import ndiff
   def word_diff(a, b):
       # Tokenize on whitespace, run ndiff, render with [-...]{+...+}
       ...
   ```

### Audit workflow

1. Inventory every hook function across all newly-added/modified XPU test files (`def _test_*`, `def _xpu_*`).
2. Locate each hook's upstream counterpart in `pytorch/test/...` (search by un-prefixed name, e.g. `_test_foo` → `test_foo`).
3. For each pair, run the strict body comparator. Any non-empty diff goes to manual classification:
   - **Cosmetic** (quote style, `ast.unparse` artifacts, dropped no-op `__init__`, return-type annotations): no action.
   - **Intentional XPU-specific override**: must already have a justification comment (see Parity Requirement). If missing, add one.
   - **Real divergence** (missing assertion, removed branch, weakened check): **fix it before commit**.
4. Emit word-level diffs to `agent_space/per_function_diffs/`. One file per ported test file. Word diff uses `[-old-]{+new+}` for any tokens that differ after normalization.
5. Write a summary table to `agent_space/per_function_diffs/AUDIT_RESULTS.md` with one row per hook: file, hook name, upstream name, verdict (Cosmetic / Intentional / Fixed / Bug).
6. The audit is the gate: if any hook's verdict is "Bug", fix and re-run before committing.

### Audit deliverables checklist

- [ ] `agent_space/per_function_diffs/` populated with one `*.diff.md` per modified XPU test file.
- [ ] `agent_space/per_function_diffs/AUDIT_RESULTS.md` with verdicts for every hook.
- [ ] Zero "Bug" verdicts remaining (all fixed).
- [ ] Every "Intentional" verdict has an inline justification comment in the source code referencing the tracking issue.

## Submitting Issues for XPU Test Failures

When a ported test fails on XPU and the failure is **not** a porting bug (i.e. the port itself is faithful but XPU lacks the underlying capability or numeric behavior), file an issue on `intel/torch-xpu-ops` BEFORE committing the port.

**Follow the `submit_ut_issues` skill** for the full submission workflow (template, labels, GitHub API calls, deep-analysis patterns).

In addition to the requirements in `submit_ut_issues`, every issue filed during porting work MUST include the **Context** section described in `submit_ut_issues/SKILL.md` ("Context Section: When Required"). Concretely the Context section must:

1. Name the porting PR by number (`PR #NNNN`) and include the full PR URL.
2. State the current PR state for the failing test (`skipped`, `failing`, or `enabled-but-failing`) and the file path holding the workaround.
3. State what becomes possible once the issue is resolved (skip can be removed / assertion will pass without further changes).

After filing, follow the `submit_ut_issues` "After Filing" steps: reference the issue URL in the relevant commit message, add an inline code comment next to the skip / weakened check (`# Tracked in intel/torch-xpu-ops#NNNN`), and ensure the issue is listed in the PR description's tracking section.

### When to file (porting-specific guidance)

File an issue if:
- Ported hook runs end-to-end but fails an assertion due to XPU runtime behavior (event names, distribution uniformity, error messages, tolerances).
- An XPU equivalent of a CUDA-only helper is missing (`CUDARngStateHelper`, blockwise scaled_mm, etc.).
- A CUDA-only feature is invoked by the test body and there is no XPU analog yet.

Do NOT file an issue for:
- A bug introduced by the port itself (fix the port instead — see Parity Requirement).
- A pure environment / build problem on the local machine.
- A known issue already tracked — comment on the existing issue with the new test name and PR cross-link instead of opening a duplicate.

## Checklist Before Commit

- [ ] CUDA test located and mapped to XPU naming (_cuda -> _xpu)
- [ ] XPU test updated/created in third_party/torch-xpu-ops/test/xpu/
- [ ] **For new files: registered in `test/xpu/skip_list_common.py` `skip_dict` (Step 2b) — CI op_ut suite does not auto-discover files**
- [ ] `dtypesIfXPU` import verified
- [ ] @dtypesIfXPU aligned with CUDA @dtypesIfCUDA
- [ ] **Hook body parity verified: no assertions / checkers / branches removed vs upstream**
- [ ] **Any intentional divergence has an inline justification comment + tracking-issue reference**
- [ ] Test runs in pytorch_opencode_env conda environment
- [ ] Test discovery verified (--collect-only)
- [ ] Failure analyzed: Is it missing dtypes, XPU limitation, or a bug?
- [ ] Intel torch-xpu-ops issues checked for similar cases
- [ ] **For every non-port-bug failure: issue filed on intel/torch-xpu-ops with a Context section linking back to this PR**
- [ ] Solution implemented and test re-run
- [ ] Known limitations documented with issue references
- [ ] **Audit & word-level diff completed (see "Audit & Word-Level Diff")**
- [ ] **`agent_space/per_function_diffs/AUDIT_RESULTS.md` updated; zero "Bug" verdicts**
- [ ] Atomic commit created
- [ ] PR description prepared (if upstream contribution); links to all tracking issues

## Boundaries

**This skill covers:**
- Porting CUDA tests to XPU by mapping _cuda suffix to _xpu
- Editing existing tests in third_party/torch-xpu-ops/test/xpu/
- Running tests in pytorch_opencode_env conda environment
- Analyzing and resolving missing test discovery
- Aligning XPU @dtypesIfXPU decorators with CUDA @dtypesIfCUDA pattern
- Enforcing hook-body parity with upstream (no assertions / checkers / branches removed)
- Running the recursive word-level diff audit (`agent_space/per_function_diffs/`)
- Delegating to the `submit_ut_issues` skill for filing intel/torch-xpu-ops issues, while requiring its mandatory Context section to cross-link the porting PR for every non-port-bug XPU failure
- Enabling tests with known limitations tracked via intel/torch-xpu-ops issues
- Restarting unsupported backends (like CUDNN_ATTENTION) for tracking
- Handling nestedtensor/SDPA tests with XPU-specific constraints

**This skill does NOT cover:**
- C++ kernel implementation
- Operator registration
- Build system changes
- Documentation writing
- Pattern-matching specific test files (each case is unique)
