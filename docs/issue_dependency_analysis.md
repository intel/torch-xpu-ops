# Issue Dependency Analysis

This document analyzes which issues (from the 90-issue list) have dependencies on
**driver**, **compiler** (Intel GPU Compiler / Triton backend), **oneDNN**, **oneMKL**,
**oneAPI / SYCL**, **Triton**, or **third-party** components.

An issue is included in a category when:
- It carries the corresponding `dependency component: <name>` label, **or**
- The error trace or issue description explicitly identifies that component as the root cause.

Issues can appear in multiple categories if they depend on more than one external component.

---

## 1. Driver Dependencies

**Dependency label:** `dependency component: driver`

These issues are blocked by or exhibit behavior caused by the Intel GPU driver (Level-Zero / OpenCL). Resolution typically requires a driver update, driver bug report, or a torch-xpu-ops workaround that avoids the problematic driver code path.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_crossref_backward_no_amp_to_sparse_xpu_float32 | Labeled `dependency component: driver`. Fatal segfault in a driver-level code path during sparse tensor operations. |
| [#2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprog | Labeled `dependency component: driver`. Numerical accuracy discrepancy in compiled subgraphs rooted in driver-level precision behavior. |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprog | Labeled `dependency component: driver`. Same class of accuracy failure as #2611. |
| [#2623](https://github.com/intel/torch-xpu-ops/issues/2623) | [upstream_ut] AssertionError: Tensor-likes are not close! in test_torchinductor | Labeled `dependency component: driver`. Tensor value mismatch attributed to driver-level numerical precision. |
| [#2625](https://github.com/intel/torch-xpu-ops/issues/2625) | [upstream_ut] AssertionError: Tensor-likes are not close! in test_torchinductor | Labeled `dependency component: driver`. Same class as #2623. |

**Total: 5 issues**

---

## 2. Compiler Dependencies (Intel GPU Compiler / IGC + Triton XPU Backend)

These issues are caused by the Intel GPU Compiler (IGC) or the Triton XPU compiler backend (pytorch-triton-xpu). The failures occur during GPU kernel compilation (JIT compilation, SPIR-V generation, or PassManager processing) rather than at runtime.

> **Note:** The repository uses `dependency component: Triton` for issues involving the Triton XPU compiler backend. Pure IGC failures often appear under the `dependency component: driver` label since the driver stack includes IGC.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | Labeled `dependency component: Triton`. Underlying failure is `InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_subgroup_matrix_multiply_accumulate'` — the Triton XPU compiler generates SPIR-V with an extension not supported by the current compiler/driver. |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut] AssertionError not raised in test_cuda_repro | Error trace: `triton/backends/intel/compiler.py` → `pm.run(mod, 'make_ttgir')` → `RuntimeError: PassManager::run failed`. The Intel Triton compiler backend cannot process a specific IR pattern. |
| [#3003](https://github.com/intel/torch-xpu-ops/issues/3003) | RuntimeError: PassManager::run failed | Identical compiler failure: `triton/backends/intel/compiler.py` → `pm.run(mod, 'make_ttir')` → `RuntimeError: PassManager::run failed`. Mixed-order reduction kernel triggers an unsupported IR transformation in the Intel Triton backend. |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Error in `torch/_inductor/codegen/triton.py`: `_get_min_elements_per_thread` asserts that direct `float8_e4m3fn` → `float8_e5m2` conversion is unsupported in the Triton XPU codegen path. |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpectedly found | XPU Triton codegen inserts a redundant `.to(tl.float16)` (Triton language cast) for `argmax` with float16 input. This is a Triton codegen generation bug specific to the XPU backend. |

**Total: 5 issues**

---

## 3. oneDNN Dependencies

These issues are caused by limitations or behaviors of the Intel oneDNN library, which underpins XPU implementations of GEMM, convolution, attention, and related operators.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | Title and description explicitly cite oneDNN. Accuracy gap in `mv`, `addmv`, `addbmm`, `addmm`, `baddbmm` ops due to oneDNN internal floating-point precision. Labeled `dependency component: oneDNN` in triage. |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape in test_addmm_errors_xpu_float32 | Error message is `oneDNN: matmul received incorrect shape`. oneDNN raises its own error before the expected PyTorch assertion, showing the dependency on oneDNN error propagation behavior. |
| [#2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is not supported in oneDNN | Error: `RuntimeError: Long is not supported in oneDNN`. oneDNN does not support `int64` dtype for matmul/conv ops. Affects all matmul-family tests with `int64` inputs. |
| [#2541](https://github.com/intel/torch-xpu-ops/issues/2541) | [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | Error: `RuntimeError: could not construct a memory descriptor using strides`. oneDNN cannot accept the non-standard stride pattern produced for certain `einsum` inputs. |

**Total: 4 issues**

---

## 4. oneMKL Dependencies

These issues involve the Intel oneMKL library, which provides DFT (FFT), BLAS, LAPACK, and sparse BLAS operations for XPU. The failures stem from oneMKL not supporting certain data types (e.g., `float16` for DFT).

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / torch.float16 | Error: `RuntimeError: Unsupported dtype Half` / `Unsupported dtype torch.float16`. FFT operations (`fft`, `fft2`, `fftn`, etc.) on XPU call oneMKL DFT, which does not support `float16`. All 24+ FFT schema/ops tests with float16 fail due to this oneMKL limitation. |
| [#2673](https://github.com/intel/torch-xpu-ops/issues/2673) | [Bug Skip]: AssertionError: The supported dtypes for _refs.fft.fft2 on device type xpu are incorrect! | Same oneMKL DFT `float16` limitation. XPU's FFT dtype capabilities (excluding `float16`) don't match the OpInfo entries that claim `float16` support, causing dtype alignment failures across all FFT ops. |

**Total: 2 issues**

---

## 5. oneAPI / SYCL Runtime Dependencies

These issues depend on Intel oneAPI runtime components (SYCL runtime, Level-Zero Unified Runtime, `sycl_ext_oneapi_*` extension availability) or require changes to the XPU device API surface that maps to oneAPI functionality.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Labeled `dependency component: oneAPI`. The `XpuDeviceProperties` class lacks the `major`/`minor` compute capability attributes available in CUDA. Adding these attributes requires mapping oneAPI device query APIs. |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut] RuntimeError: sycl_ext_oneapi_work_group_scratch_memory feature not available for SYCL Graph | Error: `RuntimeError: ... sycl_ext_oneapi_work_group_scratch_memory feature not available`. The `sycl_ext_oneapi_work_group_scratch_memory` SYCL extension cannot be used within a SYCL Graph context, which is a current oneAPI runtime limitation. |

**Total: 2 issues**

---

## 6. Triton Dependencies

These issues involve the Triton XPU backend (pytorch-triton-xpu) at the runtime, autotuning, or kernel dispatch level (as opposed to pure compilation failures, which are listed under §2 Compiler). The failures occur during Triton kernel execution, autotuning result checking, or Triton-specific test infrastructure.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#2552](https://github.com/intel/torch-xpu-ops/issues/2552) | [upstream_ut] AssertionError: Incorrect result from choice TritonTemplateCaller | Error: `AssertionError: Incorrect result from choice TritonTemplateCaller(...)`. The Triton-based autotuning template produces numerically incorrect results for INT8 weight-only quantization matmul on XPU. |
| [#2553](https://github.com/intel/torch-xpu-ops/issues/2553) | [upstream_ut] AssertionError: Scalars are not equal! in test_cuda_select_algorithm | Triton-based autotuning matcher (`woq_matcher_count`) returns 0 instead of 1, indicating the Triton WOQ kernel is not being selected for the given input shapes on XPU. |
| [#2558](https://github.com/intel/torch-xpu-ops/issues/2558) | [upstream_ut] subprocess.CalledProcessError with test_triton_interpret | Test `TRITON_INTERPRET=1` mode fails with exit status 1 on XPU. Triton interpreter mode is not functioning correctly for the XPU backend. |
| [#2692](https://github.com/intel/torch-xpu-ops/issues/2692) | [upstream_ut] AssertionError: False is not true in TestArgumentCloneAndRestore | `test_triton_heuristics.py` tests for Triton argument cloning/restoring fail on XPU. The XPU Triton heuristics do not perform the expected buffer copy for contiguous, non-contiguous, and offset arguments. |
| [#2621](https://github.com/intel/torch-xpu-ops/issues/2621) | [upstream_ut] AssertionError: 'device-side assert' not found | `test_minifier_isolate.py`: XPU Triton produces assertion failures with `AssertHandler::printMessage` format instead of the expected `device-side assert` string format, due to differences in Triton XPU vs CUDA assertion reporting. |
| [#3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | `test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU` — sparse CSR/CSC Triton kernel consistency tests fail across many dtypes, indicating XPU-specific issues in the Triton sparse kernel implementations. |

**Total: 6 issues**

---

## 7. Third-Party Component Dependencies

These issues depend on components outside the torch-xpu-ops repository itself: upstream PyTorch, TorchAO (quantization toolkit), or other external packages.

| Issue | Title | Dependency Evidence |
|-------|-------|---------------------|
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | Labeled `dependency component: community`. Intermittent failure in `test_expanded_weights_xpu.py`. The bug originates in upstream community code behavior (`input.grad` is unexpectedly `None`), not in torch-xpu-ops kernel code. |
| [#2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | Issue body states: "Depend on https://github.com/pytorch/pytorch/pull/169039/files". Resolution requires an upstream PyTorch change to support `share_memory` for XPU nested tensors before the test can pass. |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test_quant_api.py AssertionError: SQNR -2.90625 is too low | Labeled `module: ao`. Tests the TorchAO quantization library (`Int4WeightOnlyConfig`) on XPU. The very low SQNR indicates a quantization accuracy bug that is specific to the TorchAO package's XPU path, not torch-xpu-ops itself. |

**Total: 3 issues**

---

## Cross-Category Summary

Some issues appear in multiple categories because they involve more than one external dependency.

| Category | Count | Issues |
|----------|-------|--------|
| Driver | 5 | #1962, #2611, #2613, #2623, #2625 |
| Compiler (IGC / Triton XPU backend) | 5 | #2329, #2554, #2888, #3003, #3006 |
| oneDNN | 4 | #1893, #2245, #2255, #2541 |
| oneMKL | 2 | #2615, #2673 |
| oneAPI / SYCL | 2 | #2800, #3142 |
| Triton (runtime/autotuning) | 6 | #2552, #2553, #2558, #2621, #2692, #3166 |
| Third-party | 3 | #2436, #2447, #2578 |
| **Total unique issues with external dependencies** | **27** | |
| **Issues with no external dependency (pure XPU impl work)** | **63** | |

> Note on counting: Each issue is counted only in the single most specific category. #2329 is labeled `dependency component: Triton` but is listed under "Compiler" because its root cause is a compiler-level SPIR-V extension gap (`SPV_INTEL_subgroup_matrix_multiply_accumulate`), not a Triton runtime issue. Summing the counts per category gives 5+5+4+2+2+6+3 = **27 unique issues**. The remaining 63 of the 90 tracked issues have no external component dependency and require only torch-xpu-ops internal engineering work.

---

## Observations

1. **Triton is the most impactful category**: 11 issues (combining §2 Compiler + §6 Triton Runtime) are blocked by the Triton XPU backend, spanning compiler pass failures, codegen bugs, autotuning accuracy, and kernel consistency.

2. **oneDNN dtype limitations drive multiple failures**: The `int64` and memory descriptor limitations (#2255, #2541) stem from oneDNN design constraints rather than implementation gaps and may require fallback paths rather than oneDNN fixes.

3. **Driver accuracy issues are hard to work around**: The 5 driver-labeled issues (#2611, #2613, #2623, #2625, #1962) require either driver updates or XPU-side numerical workarounds.

4. **oneMKL FFT float16 limitation**: Issues #2615 and #2673 both stem from oneMKL DFT not supporting `float16`. The correct fix is either to remove `float16` from the XPU FFT OpInfo dtype list or to implement a software fallback path.

5. **oneAPI SYCL extensions**: Issue #3142 is blocked by a SYCL runtime feature (`sycl_ext_oneapi_work_group_scratch_memory`) that is not available within SYCL Graphs. This may be resolved by a future oneAPI runtime update.

6. **The majority of issues (63/90) have no external dependency**: They require torch-xpu-ops internal work — implementing missing kernels, fixing accuracy bugs, or aligning test/API behavior.
