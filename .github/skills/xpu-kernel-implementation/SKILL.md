---
name: xpu-kernel-implementation
description: >
  Implement a new XPU/SYCL kernel operator for the torch-xpu-ops repository.
  Use when adding a new operator, implementing a missing native kernel,
  or porting an upstream PyTorch operator to the XPU backend with SYCL.
---

# XPU Kernel Implementation

This Skill guides you through implementing a new native XPU operator with SYCL kernels.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Step 1: Identify the operator

Before writing code, determine:

1. **Which operator** to implement (e.g., `lerp.Tensor`, `batch_norm`)
2. **Where it currently lives**: check `src/ATen/native/xpu/XPUFallback.template` fallback list, or check if it's unregistered
3. **Upstream reference**: locate the CUDA implementation in `pytorch/pytorch` at `aten/src/ATen/native/cuda/`
4. **Registration style**: dispatch-stub vs direct registration (see Step 4)

Use `tools/check_ops.py` to inspect current operator registration status.

---

## Step 2: Determine the file structure

Every XPU kernel needs up to 3 files, depending on the registration pattern:

### Pattern A: Dispatch-stub operators (preferred for TensorIterator-based ops)

```
src/ATen/native/xpu/
├── OpName.cpp              # Host wrapper — registers dispatch stubs
src/ATen/native/xpu/sycl/
├── OpNameKernels.cpp       # SYCL kernel implementation
├── OpNameKernels.h         # Header declaring kernel functions
```

### Pattern B: Direct registration operators

```
src/ATen/native/xpu/
├── OpName.cpp              # Host wrapper — implements op_xpu() functions directly
src/ATen/native/xpu/sycl/
├── OpNameKernels.cpp       # SYCL kernel implementation
├── OpNameKernels.h         # Header declaring kernel functions
```

### Naming conventions

- Host wrapper filename: matches the logical operator group (e.g., `Lerp.cpp`, `BatchNorm.cpp`, `SoftMax.cpp`)
- SYCL kernel files: append `Kernels` suffix (e.g., `LerpKernels.cpp`, `LerpKernels.h`)
- If multiple kernels logically group under one op file (e.g., all activations), use the group name (e.g., `ActivationGeluKernel.cpp`)

---

## Step 3: Write the SYCL kernel header

Create `src/ATen/native/xpu/sycl/OpNameKernels.h`:

```cpp
/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void my_op_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
```

**Rules:**
- Use `#pragma once` (never `#ifndef` guards)
- Namespace: `at::native::xpu`
- Mark all public kernel functions with `TORCH_XPU_API`
- Include only what the declarations need (usually `TensorIterator.h` or `<ATen/core/Tensor.h>`)

---

## Step 4: Write the SYCL kernel implementation

Create `src/ATen/native/xpu/sycl/OpNameKernels.cpp`:

### For TensorIterator-based pointwise ops (most common):

```cpp
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/OpNameKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MyOpFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    // kernel logic
    return a + b;
  }
};

void my_op_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "my_op_xpu",
      [&] {
        gpu_kernel(iter, MyOpFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
```

### Kernel functor pattern

SYCL requires named functor types (no device lambdas in many cases). Use structs:

```cpp
template <typename scalar_t>
struct MyOpFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  scalar_t operator()(scalar_t self_val, scalar_t other_val) const {
    opmath_t self_opmath = static_cast<opmath_t>(self_val);
    opmath_t other_opmath = static_cast<opmath_t>(other_val);
    return static_cast<scalar_t>(self_opmath + other_opmath);
  }
};
```

If the functor needs runtime state (e.g., a scalar parameter), add a constructor and private member:

```cpp
template <typename scalar_t>
struct MyOpWithScalarFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  scalar_t operator()(scalar_t self_val) const {
    return static_cast<scalar_t>(
        static_cast<opmath_t>(self_val) * alpha_);
  }

  MyOpWithScalarFunctor(opmath_t alpha) : alpha_(alpha) {}

 private:
  opmath_t alpha_;
};
```

### Key utilities

| Utility | Header | Use |
|---------|--------|-----|
| `gpu_kernel` | `sycl/Loops.h` | Pointwise element-wise ops via TensorIterator |
| `gpu_kernel_with_scalars` | `sycl/Loops.h` | Ops that accept scalar inputs |
| Atomics | `sycl/Atomics.h` | Atomic operations on device memory |
| Reductions | `sycl/Reduce.h` | Reduction kernels |
| Group algorithms | `sycl/SYCLGroupAlgorithm.h` | Subgroup/workgroup collective ops |
| pSTL | `sycl/pstl/PSTLFunctions.h` | Parallel STL scan, sort, etc. |
| Offset calculator | `sycl/OffsetCalculator.h` | Multi-dimensional index math |
| Memory access | `sycl/MemoryAccess.h` | Vectorized loads/stores |
| Random | `sycl/Philox4x32.h` | Philox PRNG for distributions |
| Kernel utils | `sycl/KernelUtils.h` | Work-group/item size helpers |
| Launch utils | `sycl/LaunchUtils.h` | Launch configuration |

---

## Step 5: Write the host wrapper

Create `src/ATen/native/xpu/OpName.cpp`:

### Pattern A: Dispatch-stub registration

```cpp
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/OpHeader.h>           // upstream header that declares the stub
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/OpNameKernels.h>

namespace at::native {

REGISTER_XPU_DISPATCH(op_stub_name, &xpu::my_op_kernel);

} // namespace at::native
```

### Pattern B: Direct registration (named `op_xpu` functions)

```cpp
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/OpNameKernels.h>
#include <comm/xpu_aten.h>

namespace at::native {

Tensor my_op_xpu(const Tensor& self, const Tensor& other) {
  auto output = at::empty_like(self);
  xpu::my_op_kernel(output, self, other);
  return output;
}

Tensor& my_op_xpu_out(const Tensor& self, const Tensor& other, Tensor& out) {
  xpu::my_op_kernel(out, self, other);
  return out;
}

} // namespace at::native
```

### How to choose

- **Dispatch stub** (Pattern A): use when the upstream PyTorch op declares a `DispatchStub` in its header (most elementwise, unary, binary ops). Simpler — just register the kernel function pointer.
- **Direct registration** (Pattern B): use when the op needs custom host-side logic (tensor allocation, shape checks, multi-kernel orchestration), or when the upstream op uses `dispatch:` entries in `native_functions.yaml`.

---

## Step 6: Register in YAML

See the `xpu-yaml-registration` skill for full details. At minimum:

1. Add the operator overloads to `yaml/xpu_functions.yaml` under the `supported:` section
2. If the op uses direct registration (Pattern B), also add dispatch entries in `yaml/native/native_functions.yaml`
3. If the op was in the fallback list in `XPUFallback.template`, remove it from there

---

## Step 7: Add tests

1. Create a regression test in `test/regressions/test_<op>.py` for specific behavior validation
2. Verify the op passes in existing `test/xpu/test_ops_xpu.py` (opinfo-based tests)
3. For complex ops, add a microbenchmark in `test/microbench/`

---

## Common mistakes

1. **Forgetting the `.h` file** — every SYCL kernel `.cpp` needs a corresponding `.h` declaring its functions with `TORCH_XPU_API`
2. **Wrong namespace** — kernels use `at::native::xpu`, host wrappers use `at::native`
3. **Missing overloads** — if the op has `.Tensor`, `_.Tensor`, and `.out` variants, implement all three
4. **Not removing from fallback** — after implementing natively, remove the op from `XPUFallback.template`'s fallback list
5. **Using lambdas as SYCL kernels** — SYCL often requires named functor types, not lambdas
6. **Missing `opmath_type`** — for half/bfloat16, compute in higher precision using `at::opmath_type<scalar_t>`
7. **Forgetting to include the license header** — all files need the Intel copyright block

---

## Checklist

- [ ] Upstream CUDA implementation inspected for parity
- [ ] SYCL kernel `.h` created with `TORCH_XPU_API` declarations
- [ ] SYCL kernel `.cpp` created with correct functor pattern
- [ ] Host wrapper `.cpp` created with correct registration pattern
- [ ] `yaml/xpu_functions.yaml` updated (all overloads)
- [ ] `yaml/native/native_functions.yaml` updated (if Pattern B)
- [ ] Op removed from `XPUFallback.template` fallback list (if applicable)
- [ ] Tests added or existing tests verified
- [ ] `opmath_type` used for half/bfloat16 precision
- [ ] License header on all new files
