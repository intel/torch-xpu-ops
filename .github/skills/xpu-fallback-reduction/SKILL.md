---
name: xpu-fallback-reduction
description: >
  Identify and implement missing native XPU operators that currently fall back
  to CPU. Use when reducing CPU fallback coverage, implementing operators from
  the XPUFallback.template list, or when the user mentions fallback, missing
  operators, or CPU fallback elimination.
---

# XPU Fallback Reduction

This Skill guides you through the systematic process of identifying fallback operators and implementing them natively to reduce CPU fallback coverage.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Background: How XPU fallback works

When an operator has no native XPU implementation, PyTorch handles it via `src/ATen/native/xpu/XPUFallback.template`:

1. **Default behavior** (no env var): raises `TORCH_CHECK_NOT_IMPLEMENTED` error
2. **With `PYTORCH_ENABLE_XPU_FALLBACK=1`**: silently copies tensors to CPU, runs the CPU implementation, copies results back to XPU
3. **Explicit fallback list**: ops in the `fallback_list` vector always fall back to CPU (even without the env var)

The goal is to eliminate fallback by implementing native SYCL kernels for all supported ops.

---

## Step 1: Identify fallback operators

### Method A: Check the explicit fallback list

Ops explicitly routed to CPU fallback in `XPUFallback.template`:

```bash
# Show all explicitly listed fallback ops
grep -A 50 "std::vector<std::string> fallback_list" src/ATen/native/xpu/XPUFallback.template
```

Current explicit fallback ops include linear algebra ops like `cholesky`, `linalg_eig`, `linalg_qr`, `_linalg_svd`, `_efficient_attention_forward`, etc.

### Method B: Run the check_ops tool

```bash
python tools/check_ops.py
```

This compares registered XPU ops (from build output) against the full op catalog to find gaps.

### Method C: Runtime detection

```bash
PYTORCH_DEBUG_XPU_FALLBACK=1 PYTORCH_ENABLE_XPU_FALLBACK=1 python your_script.py 2>&1 | grep "falling back"
```

This prints every op that falls back to CPU during execution.

### Method D: Check xpu_functions.yaml vs upstream

Compare `yaml/xpu_functions.yaml` against `pytorch/pytorch`'s `aten/src/ATen/native/native_functions.yaml` to find ops with CUDA implementations but no XPU entry.

---

## Step 2: Prioritize which ops to implement

### Priority criteria

| Priority | Criteria |
|----------|---------|
| **P0** | Ops blocking model training/inference (shows up in real workloads) |
| **P1** | Ops causing test failures or CI skips |
| **P2** | Ops with CUDA kernels that are straightforward to port |
| **P3** | Rarely used ops or ops requiring complex library backends (oneMKL, oneDNN) |

### Routing decisions

Before implementing, determine where the op belongs:

| Op type | Where to implement |
|---------|-------------------|
| Elementwise, pointwise, reduction | `torch-xpu-ops` (this repo) — SYCL kernel |
| Linear algebra (BLAS, LAPACK) | `torch-xpu-ops` via `src/ATen/native/xpu/mkl/` — oneMKL calls |
| Convolution, matmul (performance-critical) | Upstream PyTorch via oneDNN XPU — NOT this repo |
| Attention ops | Upstream PyTorch or dedicated library — usually NOT this repo |
| Simple compound ops | Consider `CompositeExplicitAutograd` dispatch (device-agnostic) |

**Do not implement in this repo** what belongs in upstream PyTorch or a library backend.

---

## Step 3: Implementation workflow

For each fallback op, follow this workflow:

### 3.1: Find the upstream CUDA implementation

```bash
# In a pytorch/pytorch checkout:
grep -r "op_name" aten/src/ATen/native/cuda/
grep -r "op_name" aten/src/ATen/native/native_functions.yaml
```

### 3.2: Port the kernel

Use the `xpu-cuda-to-sycl-porting` skill for translation guidance:
- Create SYCL kernel files in `src/ATen/native/xpu/sycl/`
- Create host wrapper in `src/ATen/native/xpu/`

Use the `xpu-kernel-implementation` skill for file structure guidance.

### 3.3: Register in YAML

Use the `xpu-yaml-registration` skill:
- Add to `yaml/xpu_functions.yaml`
- Add dispatch entries to `yaml/native/native_functions.yaml` (if Pattern B)

### 3.4: Remove from fallback

Edit `src/ATen/native/xpu/XPUFallback.template`:
- Remove the op from `fallback_list` vector

```cpp
// BEFORE:
std::vector<std::string> fallback_list = {
    "cholesky",
    "cholesky.out",
    "my_op",           // <-- remove this
    "my_op.out",       // <-- and this
    ...
};

// AFTER:
std::vector<std::string> fallback_list = {
    "cholesky",
    "cholesky.out",
    ...
};
```

### 3.5: Add tests

- Add a regression test in `test/regressions/test_<op>.py`
- Verify the op passes in `test/xpu/test_ops_xpu.py`
- If the op was causing test skips, remove the skip from the appropriate `skip_list_*.py`

### 3.6: Verify no fallback

```python
import os
os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"
os.environ["PYTORCH_ENABLE_XPU_FALLBACK"] = "0"  # default — errors on fallback

import torch
x = torch.randn(4, 4, device="xpu")
result = torch.my_op(x)  # Should NOT print fallback warning or error
```

---

## Step 4: Performance validation

**Critical**: After implementing an op natively, measure performance. If the XPU implementation is not significantly faster than CPU fallback, reconsider the implementation — fallback may be the better trade-off.

### 4.1: Create a performance benchmark

Create `test/microbench/my_op.py` (or add to an existing group file):

```python
import torch
import time


def benchmark_my_op():
    """Compare XPU native vs CPU fallback performance."""
    
    # Test shapes and dtypes
    shapes = [(256, 256), (1024, 1024), (4096, 4096)]
    dtypes = [torch.float32, torch.bfloat16]
    
    results = {}
    
    for shape in shapes:
        for dtype in dtypes:
            x_cpu = torch.randn(*shape, dtype=dtype)
            x_xpu = x_cpu.to("xpu")
            
            # Warmup
            for _ in range(10):
                torch.my_op(x_xpu)
            torch.xpu.synchronize()
            
            # Time XPU native
            start = time.perf_counter()
            for _ in range(100):
                torch.my_op(x_xpu)
            torch.xpu.synchronize()
            xpu_time = time.perf_counter() - start
            
            # Time CPU fallback (force via copy)
            x_cpu_copy = x_cpu.clone()
            start = time.perf_counter()
            for _ in range(100):
                torch.my_op(x_cpu_copy)
            cpu_time = time.perf_counter() - start
            
            # Speedup ratio
            speedup = cpu_time / xpu_time
            results[f"{shape}_{dtype}"] = {
                "xpu_ms": xpu_time * 10,  # per-call ms
                "cpu_ms": cpu_time * 10,
                "speedup": speedup,
            }
            
            print(f"Shape {shape} / {dtype}:")
            print(f"  XPU:  {xpu_time*10:.3f} ms (per call)")
            print(f"  CPU:  {cpu_time*10:.3f} ms (per call)")
            print(f"  Speedup: {speedup:.1f}x")
            print()
    
    return results


if __name__ == "__main__":
    benchmark_my_op()
```

### 4.2: Interpret the results

**Good performance**:
- XPU is **2-3x faster or more** than CPU → native implementation is justified ✅
- Example: `cholesky` on CPU: 50 ms, on XPU: 15 ms (3.3x speedup)

**Marginal performance**:
- XPU is **1.2-2x faster** → consider the maintenance cost:
  - ✅ Keep if: frequently used op, important for models
  - ❌ Reconsider if: rarely used, high kernel complexity

**Poor performance**:
- XPU is **slower than CPU** or **only 1.1x faster** → remove the native implementation
  - CPU fallback + H2D/D2H overhead is acceptable
  - Kernel optimization is likely not worth the effort
  - Library backend (oneMKL, oneDNN) may be better

**Very large performance gap**:
- XPU is **10-100x faster** → indicates a good SYCL/oneMKL implementation
- Example: dense linear algebra with oneMKL

### 4.3: Platform-specific considerations

Different Intel GPU architectures have different characteristics:

| Platform | Consideration |
|----------|---------------|
| **Arc (DG2)** | Good SYCL support; expect 2-5x speedups for most ops |
| **Meteor Lake** | Smaller GPU; memory bandwidth limited; smaller speedups acceptable (1.5-2x) |
| **Battlemage** | Similar to Arc; aim for 2-3x+ speedups |
| **Linear algebra** (all platforms) | oneMKL often gives 3-10x speedup over CPU |

### 4.4: Decision tree

```
XPU performance vs CPU fallback?
│
├─ XPU is 2-3x+ faster
│  └─ ✅ KEEP native implementation — fallback removal justified
│
├─ XPU is 1.2-2x faster
│  ├─ Frequently used (shows up in real workloads)?
│  │  └─ YES → ✅ KEEP (user experience matters)
│  │  └─ NO  → ⚠️  Consider removing — maintenance cost may not be worth it
│  │
│  └─ Complex kernel (many special cases)?
│     └─ YES → ⚠️  Reconsider — maintenance burden vs benefit
│     └─ NO  → ✅ KEEP if clean implementation
│
├─ XPU is slower or only 1.1x faster
│  └─ ❌ REMOVE native implementation
│     └─ Option A: Leave only CPU fallback (simple)
│     └─ Option B: Check if oneMKL/oneDNN backend would help
│
└─ XPU is very inconsistent (varies by shape)
   └─ ⚠️  Investigate — kernel may need tuning for specific tensor sizes
```

### 4.5: Actions based on results

**If keeping the implementation:**
1. Document the expected speedup in PR description
2. Add the microbench file to the repo for CI performance tracking
3. Include performance evidence in the PR

**If removing the implementation:**
1. Remove native kernel files from `src/ATen/native/xpu/sycl/`
2. Remove host wrapper from `src/ATen/native/xpu/`
3. Remove YAML entries from `yaml/xpu_functions.yaml`
4. Leave op on fallback list in `XPUFallback.template` (or use explicit fallback in `native_functions.yaml`)
5. Document why fallback is acceptable in commit message

**If optimizing further:**
1. Profile the kernel: use `SYCL_DEVICE_FILTER` and Intel GPU profiling tools
2. Check memory bandwidth utilization vs compute utilization
3. Adjust work-group size, loop unrolling, or memory access patterns
4. Benchmark again and repeat

---

## Step 5: Batch implementation strategy

When implementing multiple ops:

1. **Group by kernel family** — ops sharing the same kernel pattern can share infrastructure:
   - All unary ops share `Loops.h` → `gpu_kernel(iter, UnaryFunctor())`
   - All binary ops share `Loops.h` → `gpu_kernel(iter, BinaryFunctor())`
   - All reductions share `Reduce.h`

2. **Implement .out variant first** — `.Tensor` and `_.Tensor` variants often delegate to `.out`

3. **Test incrementally** — verify each op individually before moving to the next

---

## Step 6: MKL-backed operators

Some linear algebra ops should use oneMKL instead of custom SYCL kernels:

```
src/ATen/native/xpu/mkl/
├── BatchLinearAlgebra.cpp   # batched matrix operations
├── BlasImpl.cpp             # BLAS routines (gemm, gemv, etc.)
├── SpectralOps.cpp          # FFT operations
```

For these ops:
1. Check if oneMKL supports the operation
2. Implement using oneMKL API calls (not raw SYCL)
3. Use the existing `mkl/` directory infrastructure
4. Link against `ONEMKL` via `cmake/ONEMKL.cmake`

---

## Checklist

- [ ] Fallback op identified and upstream CUDA implementation located
- [ ] Routing decision made (this repo vs upstream vs library)
- [ ] SYCL kernel implemented (or oneMKL call for linalg ops)
- [ ] Host wrapper created
- [ ] YAML registration complete (all overloads)
- [ ] Op removed from `XPUFallback.template` fallback list
- [ ] Regression test added
- [ ] Skip list entry removed (if applicable)
- [ ] Verified no fallback warning at runtime
- [ ] **Performance benchmark created and results analyzed**
- [ ] **XPU speedup is 2x+ or decision made to keep/remove based on use frequency**
- [ ] Tested across relevant dtypes and layouts
