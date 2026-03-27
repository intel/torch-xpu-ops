# Dependency Analysis: Issues #3174, #3175, #3176

This document analyzes the dependency characteristics of three issues filed against the
`intel/torch-xpu-ops` repository. Each issue is examined for its root dependency on
**driver**, **compiler**, **oneDNN**, **oneMKL**, **oneAPI / SYCL**, **Triton**, or
**third-party** components.

---

## Issue #3174 — Accuracy failure of `test_Conv2d_groups_nobias`

### Summary

`TestConvolutionNN.test_Conv2d_groups_nobias` fails with an accuracy error when running
grouped 2D convolution (no bias) on XPU. The test exercises `torch.nn.Conv2d` with
`groups > 1` and no bias term.

### Environment

- PyTorch: 2.10.0a0+gita60d9e1  
- Hardware: Intel Data Center GPU Max 1100 (8 × 49 GB)  
- Driver: `libze1` 1.21.9.0, `intel-opencl-icd` 25.18.33578.38  
- oneAPI toolchain: `dpcpp-cpp-rt` 2025.2.1, `intel-sycl-rt` 2025.2.1  
- oneMKL: `onemkl-sycl-blas` 2025.2.0

### Dependency: oneDNN

**Category: oneDNN**

The XPU path for `torch.nn.Conv2d` dispatches to an oneDNN convolution primitive. Grouped
convolution (`groups > 1`, no bias) is handled by a dedicated oneDNN group-convolution
code path that has historically exhibited numerical precision differences compared to the
CPU reference implementation, particularly for certain input shapes and floating-point
accumulation orders.

**Evidence:**

- `test_Conv2d_groups_nobias` exclusively exercises the grouped convolution path. The
  failure mode is an accuracy discrepancy (not a runtime error or crash), which is the
  signature behavior of oneDNN's internal floating-point reduction ordering differing
  from PyTorch's reference.
- The XPU Conv2d kernel delegates to oneDNN's `convolution_forward` primitive. Any
  accuracy gap between XPU and the CPU reference is owned by oneDNN's implementation of
  group convolution.
- The `onemkl-sycl-blas` 2025.2.0 and `dpcpp-cpp-rt` 2025.2.1 versions present in the
  environment are the same oneAPI release branch used by oneDNN, confirming that the full
  oneDNN/oneAPI stack is active.

**Resolution path:** Investigate oneDNN group-convolution precision for the specific
input configuration used by the test (channel count, group count, kernel size). A fix
may require relaxing the test tolerance for XPU, adding a precision hint to the oneDNN
descriptor, or filing a bug report against oneDNN.

---

## Issue #3175 — `sampled_addmm` device mismatch in Triton sparse tests

### Summary

Nine variants of `TestSparseCompressedTritonKernelsXPU.test_triton_sampled_addmm_block_size_*`
fail across block sizes 16, 32, 64 and dtypes `float16`, `bfloat16`, `float32` with:

```
ValueError: sampled_addmm(): all inputs are expected to be on the same GPU device.
```

### Environment

- PyTorch: 2.12.0a0+git4cce831  
- Hardware: Intel Data Center GPU Max 1100 (8 × 49 GB)  
- Driver: `libze1` 1.24.0.0, `intel-opencl-icd` 25.18.33578.42  
- Triton XPU: `triton-xpu` 3.7.0+git307748db  
- oneAPI toolchain: `dpcpp-cpp-rt` 2025.3.2, `intel-sycl-rt` 2025.3.2  
- oneMKL: `onemkl-sycl-blas` 2025.3.1

### Dependency: Triton

**Category: Triton**

`TestSparseCompressedTritonKernelsXPU` is explicitly a Triton kernel test class. The
`sampled_addmm` operation in this test suite is implemented as a block-sparse Triton
kernel (not through the oneDNN or standard ATen path). The error message
`"all inputs are expected to be on the same GPU device"` is raised inside PyTorch's
device-consistency check before or inside the Triton kernel dispatch path.

**Evidence:**

- The failing tests are all inside `TestSparseCompressedTritonKernelsXPU`, a class whose
  entire purpose is to validate Triton-based sparse kernels on XPU.
- The test name explicitly includes `triton_sampled_addmm`, confirming the Triton kernel
  code path is exercised.
- The device-mismatch error occurs at all block sizes and all three dtypes, indicating a
  systematic issue with how the Triton-based sparse `sampled_addmm` places or queries the
  device index of its sparse CSR inputs on XPU (e.g., the sparse tensor's `crow_indices`,
  `col_indices`, or `values` tensors may be created with an inconsistent device ID when
  going through the Triton kernel plumbing on XPU).
- `triton-xpu` 3.7.0 is the runtime Triton XPU backend in use. The version upgrade from
  3.5.x to 3.7.x is a potential regression vector.

**Resolution path:** Investigate how the Triton-based `sampled_addmm` kernel in
`test_sparse_csr.py` (PyTorch upstream) constructs its sparse CSR input tensors when
running on XPU. Ensure that sparse tensor components (`crow_indices`, `col_indices`,
`values`) and the dense operands all carry the same XPU device ID. If the issue is a
device-index mismatch introduced in `triton-xpu` 3.7.0, a compatibility fix or version
pin may be needed.

---

## Issue #3176 — `_scaled_dot_product_attention` device mismatch in Triton sparse tests

### Summary

Nine variants of
`TestSparseCompressedTritonKernelsXPU.test_triton_scaled_dot_product_attention_block_size_*`
fail across block sizes 16, 32, 64 and dtypes `float16`, `bfloat16`, `float32` with:

```
ValueError: _scaled_dot_product_attention(): all inputs are expected to be on the same GPU device.
```

### Environment

- PyTorch: 2.12.0a0+git4cce831  
- Hardware: Intel Data Center GPU Max 1100 (8 × 49 GB)  
- Driver: `libze1` 1.24.0.0, `intel-opencl-icd` 25.18.33578.42  
- Triton XPU: `triton-xpu` 3.7.0+git307748db  
- oneAPI toolchain: `dpcpp-cpp-rt` 2025.3.2, `intel-sycl-rt` 2025.3.2  
- oneMKL: `onemkl-sycl-blas` 2025.3.1

### Dependency: Triton

**Category: Triton**

This issue is the direct counterpart of issue #3175 in the domain of sparse block
attention. `TestSparseCompressedTritonKernelsXPU.test_triton_scaled_dot_product_attention_*`
uses Triton kernels to perform sparse block-structured scaled dot-product attention (SDPA).
The same device-consistency check failure occurs when the Triton XPU backend is invoked.

**Evidence:**

- All failing tests reside in `TestSparseCompressedTritonKernelsXPU`, a class dedicated
  to Triton-based sparse kernels.
- The test name explicitly includes `triton_scaled_dot_product_attention`, confirming the
  Triton execution path.
- The identical device-mismatch error fires at every combination of block size and dtype,
  mirroring the pattern seen in issue #3175 and pointing to the same root cause: device
  index inconsistency in the sparse tensor components when dispatched through the Triton
  XPU kernel path.
- Both issues (#3175 and this one) were filed from the same environment with the same
  `triton-xpu` 3.7.0 version, reinforcing that the regression was introduced in or
  around that release.

**Resolution path:** Apply the same investigation as for issue #3175. Check device
placement of the sparse block attention mask (BSR matrix) and the query/key/value tensors
inside the Triton sparse SDPA kernel. Ensure that the `crow_indices`, `col_indices`, and
`values` of the block-sparse mask as well as `Q`, `K`, `V` are all assigned to the same
XPU device index before the Triton kernel is launched.

---

## Summary Table

| Issue | Title | Failing Test | Primary Dependency | Secondary Dependencies |
|-------|-------|--------------|-------------------|----------------------|
| #3174 | Accuracy failure of `test_Conv2d_groups_nobias` | `TestConvolutionNN.test_Conv2d_groups_nobias` | **oneDNN** | oneAPI / SYCL (toolchain) |
| #3175 | `sampled_addmm()` all inputs expected on same GPU device | `TestSparseCompressedTritonKernelsXPU.test_triton_sampled_addmm_block_size_*` | **Triton** | oneAPI / SYCL (toolchain) |
| #3176 | `_scaled_dot_product_attention()` all inputs expected on same GPU device | `TestSparseCompressedTritonKernelsXPU.test_triton_scaled_dot_product_attention_block_size_*` | **Triton** | oneAPI / SYCL (toolchain) |

### Key Observations

1. **oneDNN accuracy (#3174):** Grouped 2D convolution without bias uses oneDNN's
   group-convolution primitive. The accuracy gap is caused by oneDNN's internal
   floating-point reduction ordering and is independent of the Triton or driver stack.
   The fix lives in the oneDNN interaction layer (tolerance tuning or precision hints).

2. **Triton sparse device mismatch (#3175, #3176):** Both failures occur exclusively in
   Triton-based sparse kernel tests and share the same error message pattern, strongly
   suggesting a common root cause in how `triton-xpu` 3.7.0 handles XPU device identity
   for sparse CSR tensors. The device-mismatch may be a regression introduced in the
   `triton-xpu` 3.7.0 release or an incompatibility between the new `triton-xpu` version
   and the PyTorch sparse-Triton bridge for XPU.

3. **No driver or oneMKL dependency for #3175/#3176:** The errors occur before any
   kernel execution (device-check validation), so the Intel GPU driver and oneMKL are
   not involved in the failure mechanism.

4. **No compiler dependency for any of the three issues:** None of the three failures
   involve a compilation step (IGC, DPC++, or Triton compile-time failure). They are
   all runtime-level failures.
