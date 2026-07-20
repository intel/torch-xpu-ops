---
name: check-enablement-feasibility
description: Analyze whether a skipped XPU test can be enabled on Intel GPU by inspecting skip mechanisms, XPU operator support, and required code changes. Returns enablement verdict with method description.
---

# `check-enablement-feasibility`

## Objective
Determine whether a skipped test (`status_xpu = "skipped"`) can be enabled on XPU and, if so, what code change is required. This is Gate 5 of the UT classification cascade, entered only when `has_known_issue == False` AND `status_xpu == "skipped"`.

## Inputs
- `test_file` — Test file path relative to pytorch root (e.g. `test/test_ops.py`)
- `class_name` — Test class name (XPU variant, e.g. `TestDecompXPU`)
- `test_name` — Test method name (XPU variant, e.g. `test_comprehensive_nn_functional_adaptive_max_pool1d_xpu_bfloat16`)
- `test_name_cuda` — CUDA test method name (e.g. `test_comprehensive_nn_functional_adaptive_max_pool1d_cuda_bfloat16`)
- `class_name_cuda` — CUDA test class name (e.g. `TestDecompCUDA`)
- `error_message` — The XPU error message or traceback (if available)
- `status_xpu` — Must be `"skipped"`
- `PYTORCH_SRC` — Absolute path to the pytorch source root. This is the
  `pytorch_folder` the calling agent already prepared; use it as given. **Do NOT
  set up or activate any environment** (no `setup_env.sh`). If omitted, fall back
  to the caller's checkout / current directory. All `test_file` / `test_file_cuda`
  inputs are relative to this root; export it and resolve paths against it before
  running the commands below.
- `conda_env` — The env the calling agent established; use it (e.g. via
  `conda run -n "${conda_env}" pytest ...`) for any verification run.

## Output Format
Return this JSON object:

```python
{
    "enablement_feasible": bool,
    "skip_mechanism": {
        "type": str,          # e.g. "@skipXPU", "@onlyCUDA", "@skipCUDAIfNoMagma", "@skipIf", "unittest.skipIf", "@deviceCountAtLeast", "decorator list", "skip in CI script", "other"
        "location": str,      # file:line where the skip is defined
        "code_snippet": str   # the exact decorator line(s)
    },
    "enablement_method": {
        "feasible": bool,
        "description": str,   # What change is needed to enable the test on XPU
        "required_changes": [str],  # List of specific actions, e.g. ["Remove @skipXPU from test_foo_xpu_float32 in test_ops_xpu.py:42", "Add XPU to @dtypes decorator in test_ops.py:120"]
        "estimated_effort": "trivial" | "moderate" | "complex" | "infeasible",
        "verification_hint": str  # How to verify the enablement (e.g. "Run pytest test/xpu/test_foo_xpu.py -k test_foo_xpu_float32")
    },
    "blockers": [str],        # List of reasons why enablement is NOT feasible (empty if feasible)
    "xpu_support_status": {
        "operator_name": str,     # The aten/torch operator involved (if identifiable)
        "has_xpu_kernel": bool | None,  # Does the operator have an XPU kernel? None if unknown
        "xpu_kernel_source": str | None # File path in torch-xpu-ops if found
    },
    "classification": {
        "Reason": "To be enabled" | "Submit Issue",
        "DetailReason": str
    }
}
```

## Deep Analysis Workflow

### 0. Export `PYTORCH_SRC` and resolve paths

Export the caller-provided path once so path resolution is unambiguous, then
resolve the (relative) `test_file` / `test_file_cuda` inputs against it. Do NOT
set up or activate any environment:

```bash
export PYTORCH_SRC="<pytorch_folder the caller provided>"   # falls back to cwd if unset
test_file="$PYTORCH_SRC/${test_file}"
test_file_cuda="$PYTORCH_SRC/${test_file_cuda}"
```

All `grep`/search commands below assume `${test_file}` / `${test_file_cuda}`
have been resolved to absolute paths under `$PYTORCH_SRC`.

### 1. Mandatory Input Scrubbing
- **Ignore** any pre-existing `Reason` or `DetailReason` from the task input. Do not carry them forward.
- **Never read the Excel file directly.**
- `status_xpu` MUST be `"skipped"`. If it is not, return `enablement_feasible = False` with `classification.Reason = "Submit Issue"` and note the status mismatch.

### 2. Locate the Skip Mechanism

Find where the test is skipped on XPU. Search in this order:

```bash
# 1. Search in the XPU test file first (if test_file references an xpu dir like test/xpu/)
grep -n -B 5 "def ${test_name}" "${test_file}"
grep -n -B 5 "${test_name}" "${test_file}"

# 2. Search in the CUDA test file (the original test, which may have decorators)
grep -n -B 5 "def ${test_name_cuda}" "${test_file_cuda}"

# 3. Search for class-level decorators
grep -n -B 10 "class ${class_name}" "${test_file}"

# 4. Search for skip decorators by pattern in both files
grep -n "skip.*XPU\|XPU.*skip\|onlyCUDA\|skipCUDA\|skipIf.*xpu\|deviceCountAtLeast" "${test_file}"
```

Classify the skip mechanism:
- **`@skipXPU`** — Direct XPU skip. Usually trivially removable.
- **`@onlyCUDA`** — Test restricted to CUDA only. May require refactoring to support XPU.
- **`@skipCUDAIfNoMagma`** / **`@skipCUDAIfNoCudnn`** — CUDA-specific skip. Check if the underlying feature is relevant on XPU.
- **`@skipIf`** — Conditional skip. Check the condition.
- **`@deviceCountAtLeast`** — Multi-device test. XPU may not support multi-device in the same way.
- **`unittest.skipIf`** / **`unittest.skip`** — Python skip. Check the condition.
- **Decorator list in CI script** — Skip may be in a CI config file (e.g. `test/xpu/run_test_with_skip.py`), not in source code.

### 3. Analyze Enablement Feasibility

For each skip type, evaluate feasibility:

**`@skipXPU`**:
- Usually trivially removable.
- Check if the skip is stale: look at git log for when it was added.
- If the test passes when `@skipXPU` is removed → `feasible = True`, `trivial`.
- If removing `@skipXPU` causes a real failure → investigate the actual error.

**`@onlyCUDA`**:
- The test is restricted to CUDA device. XPU enablement requires:
  - Changing `@onlyCUDA` to a device-agnostic decorator, OR
  - Adding a separate XPU variant of the test.
- Check if the test body uses CUDA-specific APIs (`torch.cuda.*`, `Stream`, `Event`, etc.).
- If the test body is device-agnostic → feasible (moderate effort — refactor decorator).
- If the test body calls CUDA-specific APIs that have no XPU equivalent → `infeasible`.

**`@skipIf` with CUDA condition**:
- Check if the same skip condition applies to XPU.
- Example: `@skipIf(not TEST_WITH_ROCM, "test requires rocm")` — XPU is not ROCm, so enablement is unrelated.
- Example: `@skipIf(not TEST_CUDA, "CUDA not available")` — check `TEST_XPU` equivalent.

**`@deviceCountAtLeast`**:
- XPU multi-device support varies. Check if the test actually needs multiple GPUs or just parametrizes over device count.
- If the test is fundamentally about CUDA multi-GPU behavior → may be infeasible.

**CI script skip**:
- Check `test/xpu/run_test_with_skip.py` or equivalent.
- If the skip is a blanket skip of the whole test class or file, removing it may expose individual test failures.

### 4. Determine XPU Operator Support

If the error message or test body references an operator (e.g. `aten::adaptive_max_pool1d`, `torch.nn.functional.adaptive_max_pool1d`):

```bash
# Check if the operator has an XPU kernel
grep -r "adaptive_max_pool1d" src/ATen/native/xpu/ --include="*.cpp" --include="*.h" -l 2>/dev/null || echo "Not found in xpu native directory"

# Check the dispatch registration
grep "adaptive_max_pool1d" aten/src/ATen/native/native_functions.yaml 2>/dev/null | head -5
```

If the operator does NOT have an XPU kernel:
- Feasibility depends on whether implementing one is in scope.
- Check if there are existing XPU kernels for related operators.
- Default to `feasible = True` with `estimated_effort = "complex"` if the operator is a standard PyTorch op.

### 5. Classification Synthesis

Apply this EXACT mapping:

| Feasible | `classification.Reason` | `classification.DetailReason` |
|----------|------------------------|-------------------------------|
| `True` | `To be enabled` | `"Enablement analysis: [skip_mechanism]. [enablement_method.description]. Estimated effort: [trivial/moderate/complex]. (Source: [file:line references])"` |
| `False` | `Submit Issue` | `"No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test, and enablement analysis indicates the test cannot be enabled on XPU because: [blockers]. Submit a new issue with the error details."` |

If feasibility was verified by actually running the test after applying the change (e.g., removing `@skipXPU` and confirming PASS), note this in `DetailReason` e.g. `"Verified: test passes on XPU after removing @skipXPU"`.

**Important**: "Verified passing" does NOT make it `Local Passed`. `Local Passed` is only for tests that pass WITH NO code changes. If a test requires any code change to pass, it is `To be enabled` — with the verification result documented in `DetailReason`.

### Complete Example Output

For a trivially removable `@skipXPU`:

```json
{
    "enablement_feasible": true,
    "skip_mechanism": {
        "type": "@skipXPU",
        "location": "test/xpu/test_ops_xpu.py:142",
        "code_snippet": "@skipXPU  # Remove after XPU supports adaptive_max_pool1d"
    },
    "enablement_method": {
        "feasible": true,
        "description": "Remove stale @skipXPU decorator — the test passes on XPU without it.",
        "required_changes": ["Remove @skipXPU line (test/xpu/test_ops_xpu.py:142)"],
        "estimated_effort": "trivial",
        "verification_hint": "pytest test/xpu/test_ops_xpu.py -k test_comprehensive_nn_functional_adaptive_max_pool1d_xpu_bfloat16"
    },
    "blockers": [],
    "xpu_support_status": {
        "operator_name": "adaptive_max_pool1d",
        "has_xpu_kernel": true,
        "xpu_kernel_source": "src/ATen/native/xpu/sycl/AdaptiveMaxPoolingKernels.cpp"
    },
    "classification": {
        "Reason": "To be enabled",
        "DetailReason": "Enablement analysis: @skipXPU at test/xpu/test_ops_xpu.py:142. Remove stale @skipXPU decorator — test passes on XPU without it. Estimated effort: trivial."
    }
}
```

For an infeasible `@onlyCUDA` using CUDA-specific APIs:

```json
{
    "enablement_feasible": false,
    "skip_mechanism": {
        "type": "@onlyCUDA",
        "location": "test/test_ops.py:1200",
        "code_snippet": "@onlyCUDA\ndef test_cudnn_rnn_xpu_float32(self, device):"
    },
    "enablement_method": {
        "feasible": false,
        "description": "",
        "required_changes": [],
        "estimated_effort": "infeasible",
        "verification_hint": ""
    },
    "blockers": [
        "Test uses cuDNN-specific API (torch.backends.cudnn.*) which has no XPU equivalent",
        "cuDNN RNN implementation is NVIDIA GPU-specific; XPU uses different RNN backend"
    ],
    "xpu_support_status": {
        "operator_name": null,
        "has_xpu_kernel": null,
        "xpu_kernel_source": null
    },
    "classification": {
        "Reason": "Submit Issue",
        "DetailReason": "No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test, and enablement analysis indicates the test cannot be enabled on XPU because: Test uses cuDNN-specific API (torch.backends.cudnn.*) which has no XPU equivalent; cuDNN RNN implementation is NVIDIA GPU-specific. Submit a new issue with the error details."
    }
}
```

## Strict Constraints (ZERO TOLERANCE)

1. **Default to Feasible**: If the skip is `@skipXPU` and the test is a standard PyTorch operator test, default to `feasible = True` unless there is specific evidence the underlying operator is missing on XPU.
2. **Evidence Required**: Every classification MUST include `file:line` references for the skip mechanism. Vague claims like "the test has a skip decorator" are invalid.
3. **Tool Use**: Use `bash` (grep, git log), `read`, `grep`. Prefer source inspection over guessing.
4. **No Blind Copies**: Do not copy input classification columns. Evaluate from scratch.
5. **No Excel Reads**: Never read the input Excel file directly. Rely only on the passed task parameters.
6. **Honest Blockers**: If enablement is infeasible, list specific blockers with evidence (e.g. "cuDNN-specific API", "CUDA-only hardware feature"). Do not fabricate blockers.
7. **`@onlyCUDA` is NOT automatically infeasible**: `@onlyCUDA` means the test was written for CUDA only, not that it cannot work on XPU. Check the test body for actual CUDA-specific APIs before concluding infeasibility.
