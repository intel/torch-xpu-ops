---
name: xpu-bug-investigation
description: >
  Debug and investigate XPU operator failures, crashes, or incorrect results.
  Use when triaging a bug report, writing a reproducer, bisecting a regression,
  or diagnosing XPU-specific issues like sync bugs, fallback problems, or
  dtype mismatches.
---

# XPU Bug Investigation

This Skill guides you through the full lifecycle of investigating an XPU operator bug: reproduce → isolate → root-cause → fix → verify.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Step 1: Write a minimal reproducer

Every investigation starts with a reproducer. Create `test/repro/test_<description>.py`:

```python
import torch
import pytest


def test_op_failure():
    """Minimal reproducer for <describe the bug>."""
    device = "xpu"
    # Minimal tensor setup that triggers the issue
    x = torch.randn(4, 4, device=device)
    # The operation that fails
    result = torch.some_op(x)
    # Expected behavior
    expected = torch.some_op(x.cpu()).to(device)
    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Reproducer requirements:**
- File in `test/repro/` with `test_` prefix
- Contains pytest-style `def test_...()` functions
- Uses `xpu` device explicitly
- Runnable via `pytest test/repro/test_<description>.py`
- Minimal — strip away everything not needed to trigger the bug

---

## Step 2: Classify the failure

Run the reproducer and classify the result:

### Category A: Fallback warning (op not natively implemented)
```
UserWarning: Aten Op fallback from XPU to CPU happens.
```
→ The op is routed through `XPUFallback.template` to CPU. This is a **missing implementation**, not a bug.
→ See the `xpu-fallback-reduction` skill.

### Category B: Incorrect results (numerical mismatch)
```
AssertionError: Tensor-likes are not close!
```
→ Likely a kernel bug: wrong dtype promotion, accumulation precision, non-contiguous handling, or semantic mismatch with CPU/CUDA.

### Category C: Runtime error
```
RuntimeError: ...
```
→ Could be: unsupported dtype, shape validation, or dispatch failure. Check error message.

### Category D: Crash / segfault
→ Likely: out-of-bounds access, 32-bit index overflow, null pointer, or sync bug. Run with:
```bash
PYTORCH_DEBUG_XPU_FALLBACK=1 TORCH_SHOW_CPP_STACKTRACES=1 python test/repro/test_<description>.py
```

### Category E: Hang / deadlock
→ Likely: stream synchronization issue or distributed backend deadlock. Check stream usage.

---

## Step 3: Check operator registration status

Determine whether the op is natively implemented or falling back:

```bash
# Check if the op is in the fallback list
grep "op_name" src/ATen/native/xpu/XPUFallback.template

# Check if the op is in xpu_functions.yaml
grep "op_name" yaml/xpu_functions.yaml

# Use the check_ops tool for a comprehensive view
python tools/check_ops.py
```

Also verify which dispatch path is taken:
```python
import torch
torch.xpu.is_available()  # Confirm XPU is visible
x = torch.randn(2, 2, device="xpu")
# Enable debug fallback to see which ops fall back
import os
os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"
```

---

## Step 4: Compare against CPU/CUDA reference

For numerical mismatches, compare the XPU result against CPU (and CUDA if available):

```python
import torch

x_cpu = torch.randn(4, 4)
x_xpu = x_cpu.to("xpu")

result_cpu = torch.some_op(x_cpu)
result_xpu = torch.some_op(x_xpu)

# Compare
print(f"Max diff: {(result_cpu - result_xpu.cpu()).abs().max()}")
print(f"CPU result:\n{result_cpu}")
print(f"XPU result:\n{result_xpu.cpu()}")
```

For parity investigation, also inspect the upstream CUDA implementation in `pytorch/pytorch` at `aten/src/ATen/native/cuda/` — do not rely on memory.

---

## Step 5: Narrow down the root cause

### For numerical mismatches:

1. **Check dtypes**: test FP32, BF16, FP16 separately. Is only one dtype wrong?
2. **Check layouts**: test contiguous vs non-contiguous vs channels-last
3. **Check shapes**: test small, large, empty, scalar, broadcasted
4. **Check accumulation**: is the op doing reduction? Check if `opmath_type` is used for half precision
5. **Check edge cases**: NaN, Inf, zero, negative

### For crashes:

1. **Check indexing**: is `int` used where `int64_t` is needed?
2. **Check tensor size**: does the crash only happen for large tensors?
3. **Check non-contiguous**: does the crash only happen for strided tensors?

### For wrong dispatch:

1. Is the op in `yaml/xpu_functions.yaml`?
2. Does `yaml/native/native_functions.yaml` have the correct dispatch entries?
3. Is the op still listed in `XPUFallback.template`'s explicit fallback list?

---

## Step 6: Bisect the regression (if applicable)

If the op used to work and regressed, use the bisect workflow:

### Manual bisect:
```bash
git log --oneline --since="2 weeks ago" -- src/ATen/native/xpu/
# Find the suspect commit range
git bisect start
git bisect bad HEAD
git bisect good <known-good-commit>
# Test at each step with the reproducer
python test/repro/test_<description>.py
git bisect good  # or: git bisect bad
```

### CI bisect workflow:
The repo has `.github/workflows/bisect_search.yml` for automated bisect on CI hardware.
It accepts:
- `search_commits`: e.g., `pytorch=old/new,xpu-ops=old/new`
- `search_case`: e.g., `pytest test/repro/test_<description>.py`
- `search_check`: `accuracy`, `performance`, or test suite name

---

## Step 7: Fix and verify

1. **Apply the fix** in the relevant kernel file(s)
2. **Run the reproducer** to confirm it passes
3. **Run related existing tests** to ensure no regressions:
   ```bash
   python test/xpu/run_test_with_skip.py test/xpu/test_ops_xpu.py -k "test_<op_name>"
   ```
4. **Check multiple dtypes and layouts** in your reproducer
5. **Move the reproducer** from a scratch script to `test/repro/test_<description>.py` if not already there

---

## Common root causes

| Symptom | Likely cause | Where to look |
|---------|-------------|---------------|
| Results differ from CPU only for BF16/FP16 | Missing `opmath_type` accumulation | SYCL kernel functor |
| Crash on large tensors | 32-bit index overflow | Kernel loop variables, `OffsetCalculator` usage |
| Wrong results for non-contiguous | Missing stride handling | Kernel indexing, `TensorIterator` usage |
| Op returns zeros | Output tensor not written | Check kernel launch, verify dispatch reaches kernel |
| Op silently falls back to CPU | Missing YAML entry or still in fallback list | `xpu_functions.yaml`, `XPUFallback.template` |
| Results differ only for inplace variant | `out=` / inplace aliasing issue | Host wrapper, kernel output pointer |
| Intermittent wrong results | Race condition / sync bug | Atomic usage, stream ordering |

---

## Diagnostic environment variables

| Variable | Effect |
|----------|--------|
| `PYTORCH_DEBUG_XPU_FALLBACK=1` | Print every fallback op name |
| `PYTORCH_ENABLE_XPU_FALLBACK=1` | Allow CPU fallback (default: error) |
| `PYTORCH_XPU_FALLBACK_OP=op1,op2` | Force specific ops to CPU fallback |
| `TORCH_SHOW_CPP_STACKTRACES=1` | Show C++ stack traces in Python errors |

---

## Checklist

- [ ] Minimal reproducer written in `test/repro/test_<description>.py`
- [ ] Failure classified (fallback / numerical / crash / hang)
- [ ] Operator registration status verified
- [ ] Compared against CPU/CUDA reference
- [ ] Root cause identified (dtype / indexing / dispatch / sync)
- [ ] Fix applied and reproducer passes
- [ ] Existing tests still pass
- [ ] Multiple dtypes and layouts tested
