# Issue Classification Report

This document classifies the listed issues into three levels based on the degree of intervention required.

## Classification Levels

- **Level 1**: No additional information or human engineer intervention required. Issues that are already resolved, have a clear "wontfix" decision, or are duplicates already tracked elsewhere.
- **Level 2**: Additional information required but no human engineer intervention needed. Issues that need more data (e.g., reproduction confirmation, version verification, intermittent failure patterns) before a decision can be made, but the resolution path does not require writing new code.
- **Level 3**: Relies on human engineers to resolve. Issues that require engineering design, code implementation, bug fixes, or active investigation into complex failures.

---

## Level 1 — No Additional Information or Human Engineer Intervention Required

**5 issues**

These issues have already been resolved, have a firm "wontfix" decision in place, or are duplicates being tracked under a different issue. No new action is needed.

| Issue | Title | Basis for Grading |
|-------|-------|-------------------|
| [#1972](https://github.com/intel/torch-xpu-ops/issues/1972) | Segmentfault test cases | **Already CLOSED** (`state: closed, state_reason: completed`). The segfault cases were resolved (all items checked off in the issue body). No further intervention needed. |
| [#2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it is cuda specific | Labeled **`wontfix`** and **`skipped`**. This test is CUDA-specific and cannot be adapted for XPU. The decision not to fix it has been made and the test is in the skip list. No additional action is required. |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not aligned with cuda | Labeled **`duplicate`**. This issue is being tracked under another issue. No independent resolution is needed here. |
| [#2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Labeled **`wontfix`** and **`skipped`**. The team has made a deliberate decision not to implement these ops (`aten::_dyn_quant_pack_4bit_weight`, `aten::narrow_copy`, `aten::_histogramdd_bin_edges`) for XPU. The affected tests are in the skip list. No further action required. |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | Labeled **`duplicate`**. Tracked as a duplicate of another issue related to Triton/inductor SPV extension support. No separate tracking needed here. |

---

## Level 2 — Additional Information Required but No Human Engineer Intervention Needed

**3 issues**

These issues require gathering more information (reproduction data, environment details, test statistics) before they can be properly addressed. Once that information is available, the resolution path does not require writing new code—it may simply be confirming behavior, updating skip list entries based on fresh data, or waiting for an external component update.

| Issue | Title | Basis for Grading |
|-------|-------|-------------------|
| [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test] EmbeddingBag device test failures | Labeled **`skipped`** and **`ut_upstream`**. The failures occur only in the upstream CI environment but cannot be reproduced locally. The TODO items reference upstream PRs awaiting workaround removal, but the root cause is not yet fully identified. More information about the CI vs. local environment differences is needed before any code change can be justified. |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | Labeled **`random`** and **`dependency component: community`**. This is an intermittent (non-deterministic) failure that depends on community code behavior. The `random` label indicates the failure is inconsistent across runs. More test data is needed to confirm reproducibility and determine whether this is a genuine XPU bug or a flaky upstream test, before any engineering fix can be designed. |
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | Labeled **`random`** and **`skipped`**. The issue is classified as a random/intermittent failure related to softmax numerical tolerance. More reproduction data is needed to determine if this is a genuine numerical precision issue (requiring a fix) or simply test flakiness. No code change is warranted until the issue can be consistently reproduced. |

---

## Level 3 — Relies on Human Engineers to Resolve

**82 issues**

All remaining issues require active engineering work: implementing missing XPU kernels, fixing accuracy/correctness bugs, adding new XPU feature support, resolving hardware/driver dependency issues, or debugging complex inductor/compiler failures.

| Issue | Title | Basis for Grading |
|-------|-------|-------------------|
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | Accuracy gap between XPU and reference implementations for `mv`, `addmv`, `addbmm`, `addmm`, `baddbmm`. Requires investigation of oneDNN precision behavior and potential XPU kernel or tolerance tuning. |
| [#1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_crossref_backward_no_amp_to_sparse_xpu_float32 | Fatal segfault in a driver-dependent code path (`dependency component: driver`). Requires driver-level investigation and potentially a bug report to driver team or a workaround in XPU code. |
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoderLayer | TransformerEncoderLayer produces `inf`/`nan` values for float16/float32/float64 inputs on XPU. Requires engineer to trace the numerical error and fix the underlying attention or normalization implementation. |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with CUDA enabled | Tests from `TestPoolingNN` assert CUDA availability. Requires adding proper XPU guards/skips or providing XPU-equivalent test implementations. |
| [#2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_size_comparison_no_recompile | XPU triggers two recompilations instead of one, indicating a guard/recompile policy difference between XPU and CUDA. Requires investigation of Dynamo guard logic for XPU. |
| [#2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requests and CUDA alignment | Feature tracking issue for FP8/MXFP8 op support. Requires engineers to implement `_scaled_grouped_mm` and other FP8 GEMM operations for XPU as part of PyTorch 2.12 deliverables. |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape in test_addmm_errors_xpu_float32 | oneDNN raises an error for an invalid matmul shape before the expected assertion check, causing the test to fail. Requires fixing the error message propagation or adding a prior shape validation step in the XPU addmm path. |
| [#2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is not supported in oneDNN | oneDNN does not support `int64` for matmul/conv ops. Requires adding proper dtype guards or fallback paths in XPU op implementations. |
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test/xpu/test_decomp.py | `aten::_flash_attention_forward` decomp test fails. Requires completing the XPU implementation of flash attention decomposition. |
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | `aten::_efficient_attention_forward` is not available from the CPU backend (used in meta/symbolic dispatch). Requires implementing proper XPU efficient attention support and meta kernel. |
| [#2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | `_refs.linspace` and `_refs.logspace` fail for XPU because `aten.copy.default` is not supported in TorchRefs mode. Requires engineer to add the missing ref implementation. |
| [#2289](https://github.com/intel/torch-xpu-ops/issues/2289) | [upstream_ut] tensordot dtypes is not aligned with cuda | XPU `tensordot` supports additional integer dtypes (`int8`, `uint8`) not listed in OpInfo. The OpInfo dtype table needs to be updated to reflect actual XPU capabilities. |
| [#2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with OpInfo | Multiple ops (`einsum`, `inner`, `mm`, `nn.functional.embedding_bag`, `nn.functional.linear`) have XPU dtype capabilities that don't match their OpInfo entries. Requires updating OpInfo entries or restricting supported dtypes. |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | `logaddexp` is not implemented for complex dtypes on XPU. Requires implementing complex number support for this op or adding proper dtype error messages. |
| [#2382](https://github.com/intel/torch-xpu-ops/issues/2382) | [upstream_ut] refs.vdot got MetadataMismatchError | `_refs.vdot` produces a conjugate mismatch error for complex types on XPU. Requires fixing conjugation handling in the XPU vdot or refs implementation. |
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2447](https://github.com/intel/torch-xpu-ops/issues/2447) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced by pytorch test sample input updates | After a pytorch upstream commit changed test sample inputs for `conv_transpose`, the XPU dtype tests began failing. Requires updating XPU op dtype definitions or skip entries to match new test expectations. |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2532](https://github.com/intel/torch-xpu-ops/issues/2532) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2536](https://github.com/intel/torch-xpu-ops/issues/2536) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2541](https://github.com/intel/torch-xpu-ops/issues/2541) | [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | oneDNN cannot construct a memory descriptor for the given stride pattern in `einsum`. Requires adding stride normalization or contiguous copy logic before passing tensors to oneDNN. |
| [#2552](https://github.com/intel/torch-xpu-ops/issues/2552) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2553](https://github.com/intel/torch-xpu-ops/issues/2553) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2558](https://github.com/intel/torch-xpu-ops/issues/2558) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2565](https://github.com/intel/torch-xpu-ops/issues/2565) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2566](https://github.com/intel/torch-xpu-ops/issues/2566) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut] CppCompileError: C++ compile error | AOTInductor C++ wrapper is missing `aoti_torch_xpu_fn_square` declaration. Custom ops need XPU-specific AOTI runtime function registration. Requires engineering to add the missing XPU AOTI function. |
| [#2611](https://github.com/intel/torch-xpu-ops/issues/2611) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2620](https://github.com/intel/torch-xpu-ops/issues/2620) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2621](https://github.com/intel/torch-xpu-ops/issues/2621) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2623](https://github.com/intel/torch-xpu-ops/issues/2623) | (upstream_ut issue, driver dependency) | Labeled `dependency component: driver`. Accuracy issue dependent on driver behavior. Requires driver investigation and potential XPU-side workaround. |
| [#2624](https://github.com/intel/torch-xpu-ops/issues/2624) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2625](https://github.com/intel/torch-xpu-ops/issues/2625) | (upstream_ut issue, driver dependency) | Labeled `dependency component: driver`. Accuracy issue dependent on driver behavior. Requires driver investigation and potential XPU-side workaround. |
| [#2645](https://github.com/intel/torch-xpu-ops/issues/2645) | (upstream_ut, missing kernel) | Missing XPU kernel for an op. Requires implementing the kernel for XPU. |
| [#2673](https://github.com/intel/torch-xpu-ops/issues/2673) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2692](https://github.com/intel/torch-xpu-ops/issues/2692) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2694](https://github.com/intel/torch-xpu-ops/issues/2694) | (upstream_ut issue) | Based on issue pattern, requires engineering investigation and fix. |
| [#2695](https://github.com/intel/torch-xpu-ops/issues/2695) | [upstream_ut] NoValidChoicesError: No choices exist for backend | Inductor auto-tune has no valid choices for `aten.mm.dtype` on XPU (missing backend support). Requires adding `ATEN` or XPU-specific gemm choices in the inductor configuration/code. |
| [#2696](https://github.com/intel/torch-xpu-ops/issues/2696) | [upstream_ut] RuntimeError: Expected to find "(262144, 0, 512, 1" | The XPU codegen does not produce the expected tensor stride pattern for `_scaled_dot_product_efficient_attention`. Requires investigation into XPU inductor lowering of efficient attention. |
| [#2697](https://github.com/intel/torch-xpu-ops/issues/2697) | [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | Same class of failure as #2696; XPU codegen doesn't match expected stride patterns. Requires inductor lowering fix. |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Flash attention on XPU does not support head dimensions other than {64, 96, 128, 192}. Requires expanding supported head dimensions in the XPU flash attention kernel. |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | `torch.xpu.device.__init__` is in Dynamo's skip list, causing tracing failures. Requires adding `@torch._dynamo.dont_skip_tracing` decorator or fixing the trace_rules for XPU device context managers. |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: DispatchStub: missing kernel for xpu | XPU kernel for `ldexp` (and related `_ldexp_stub`) is missing. Requires implementing the XPU kernel for `aten::ldexp`. |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | The `driver` dictionary in SVD tests doesn't include an XPU entry, causing a KeyError. Requires adding XPU to the driver dictionary or updating the test to handle XPU properly. |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | `XpuDeviceProperties` is missing a `major` attribute that CUDA device properties have. Requires adding this attribute to the XPU device properties class. |
| [#2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_attention issues | Multiple issues: causal attention mask conflict, `MetadataMismatchError` in fake tensor propagation, and assertion error in export tests. Requires fixing XPU flash attention implementation to properly handle causal masking and metadata. |
| [#2805](https://github.com/intel/torch-xpu-ops/issues/2805) | max_clock_rate need xpu support | `get_device_tflops` calls `nvidia-smi` which doesn't exist on XPU systems. Requires implementing an XPU-compatible `max_clock_rate` function using XPU-specific APIs. |
| [#2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | `CompiledAOTI` in PyTorch inductor raises `unsupported device type xpu`. Requires adding XPU support to the `CompiledAOTI` output code path. |
| [#2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison failed: Decimal != Decimal('0') | XPU does not flush denormal floats to zero (FTZ) as expected in `test_not_disabling_ftz_yields_zero`. Requires investigating XPU FTZ behavior and either implementing it or updating the test's XPU expectations. |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU | `_flash_attention_forward` tries to dispatch to CPU (which doesn't implement it) when called on XPU tensors in certain meta/schema check contexts. Requires completing the XPU meta kernel for flash attention. |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | XPU Triton codegen does not support direct `float8_e4m3fn` → `float8_e5m2` conversion. Requires either adding conversion support or adding a decomposition to convert through fp32. |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, 0, 512, 1" | Same class of failure as #2696. XPU efficient attention lowering in inductor doesn't produce expected strided tensor. Requires XPU-specific inductor lowering fix. |
| [#2956](https://github.com/intel/torch-xpu-ops/issues/2956) | Exception: The values for attribute 'stride()' do not match in test_comprehensive_nn_functional_linear_xpu_float16 | XPU `linear` op returns tensors with incorrect strides compared to expected (24,8,1 vs 12,4,1). Requires investigating XPU linear output layout and fixing stride computation. |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_compile | DTensor compile test produces a different graph structure on XPU than expected (different `from_local` representation). Requires updating XPU DTensor compile behavior or the test expectation. |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_max_autotune | XPU produces `nan` values in the linear + cross-entropy loss max-autotune path. Requires investigation of the XPU gemm autotuning producing numerically incorrect results. |
| [#2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_libdevice' | XPU environment doesn't have `torch._inductor.config.eager_numerics.use_pytorch_libdevice`. Requires adding this config key to XPU inductor configuration. |
| [#3003](https://github.com/intel/torch-xpu-ops/issues/3003) | RuntimeError: PassManager::run failed | Triton XPU backend's PassManager fails for a mixed-order reduction kernel. Requires investigation into Triton XPU compiler to identify the unsupported IR pattern. |
| [#3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | The XPU `_record_memory_history` function signature does not match CUDA's. The CUDA version accepts bool but XPU requires specific string/int arguments. Requires updating the XPU memory history recording API to match CUDA's interface. |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpectedly found | XPU Triton codegen inserts an unnecessary `.to(tl.float16)` conversion for `argmax` with float16 input. Requires fixing the XPU Triton codegen to avoid redundant type casts for integer output reductions. |
| [#3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | `test_flash_attention_dynamic` expects 2 compilation frames but XPU produces 3, indicating an extra recompilation. Requires investigation into XPU flash attention dynamic shape handling. |
| [#3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut] Two NestedTensor issues with flash attention | XPU flash attention does not support `NestedTensorXPU` backend (missing dispatch), and warning messages differ from expected. Requires implementing NestedTensor dispatch for XPU flash attention. |
| [#3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut] AssertionError: AssertionError not raised | `test_math_backend_high_precision` expects tensors to differ but XPU produces matching results. Suggests XPU has higher precision than expected or uses a different precision path. Requires investigation of XPU math backend precision settings. |
| [#3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut] AssertionError: RuntimeError not raised | `test_invalid_fused_inputs_invalid_dtype_kernel1` expects a `RuntimeError` for invalid dtype inputs but XPU doesn't raise it. Requires adding dtype validation in the XPU SDPA kernel. |
| [#3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut] AssertionError: UserWarning not triggered | XPU SDPA backward does not emit the expected determinism warning. Requires adding the appropriate `UserWarning` in the XPU SDPA backward path. |
| [#3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut] AssertionError: Output should not contain NaNs! | XPU flash attention produces `nan` for large bf16 inputs. Requires investigating numerical stability in XPU flash attention for bf16, potentially adding max-value overflow protection. |
| [#3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut] NotImplementedError: _scaled_dot_product_efficient_attention_backward | XPU doesn't implement `aten::_scaled_dot_product_efficient_attention_backward`. Test expects a specific error message but gets a different one (XPU not implemented). Requires implementing the XPU backward for efficient attention or updating the error message. |
| [#3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut] fused_kernels_nested_broadcasting: RuntimeError: No available kernel | XPU raises "No available kernel" for nested tensor broadcasting in fused attention. Requires adding XPU kernel support for nested tensor SDPA with broadcasting. |
| [#3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut] RuntimeError: scaled_dot_product_attention: nested tensors must be contiguous | XPU's `scaled_dot_product_attention` requires nested tensors to be contiguous (stricter than expected). Requires either relaxing this constraint or adding a contiguous copy for non-contiguous nested tensors. |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut] AssertionError: False is not true (test_disable_fastpath) | The MHA fastpath is not being called on XPU when it should be. Requires investigating XPU multi-head attention fastpath enablement conditions. |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut] RuntimeError: expected scalar type Half but found Float | XPU `_transformer_encoder_layer_fwd` doesn't handle autocast properly, mixing float16 and float32. Requires fixing XPU autocast support in transformer encoder layer. |
| [#3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | XPU flash attention does not support non-zero dropout. Requires implementing dropout support in XPU flash attention. |
| [#3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Same as #2698 but in transformers tests. XPU flash attention doesn't support head_dim=32. Requires expanding the supported head dimension set in the XPU flash attention kernel. |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut] RuntimeError: sycl_ext_oneapi_work_group_scratch_memory feature not available for SYCL Graph | XPU SYCL Graph extension doesn't support a required work-group scratch memory feature. This is a `dependency component: oneAPI`/driver limitation. Requires a driver update or implementing a fallback that doesn't use this SYCL extension within SYCL graphs. |
| [#3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: _scaled_dot_product_efficient_attention_backward not implemented | Same as #3131. XPU lacks the backward implementation for `_scaled_dot_product_efficient_attention`. Requires implementing the backward pass or redirecting to an alternative implementation. |
| [#3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | XPU's sparse CSR add operation returns int64 index tensors where int32 is expected. Requires investigating index type selection in the XPU sparse CSR add implementation. |
| [#3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | Sparse compressed Triton kernel consistency tests fail for both SparseCSR and SparseCSC formats across many dtypes. Requires investigating and fixing XPU sparse Triton kernel consistency issues. |

---

## Summary

| Level | Count | Description |
|-------|-------|-------------|
| Level 1 | 5 | Already resolved, wontfix decisions made, or duplicate issues — no action needed |
| Level 2 | 3 | Intermittent/environment-specific failures needing more data before action |
| Level 3 | 82 | Require active engineering work: kernel implementation, bug fixes, feature development |
| **Total** | **90** | |

### Level 1 Summary (5 issues)
Issues #1972 (closed), #2164 (wontfix, CUDA-specific), #2253 (duplicate), #2309 (wontfix, unsupported ops), and #2329 (duplicate) require no further intervention. The first is already resolved; the rest have deliberate decisions in place with tests added to the skip list.

### Level 2 Summary (3 issues)
Issues #2295 (CI-only failures needing environment analysis), #2436 (random failure needing reproducibility data), and #3033 (intermittent softmax tolerance needing more test statistics) need additional information before an action can be taken, but do not require writing new production code once the information is gathered.

### Level 3 Summary (82 issues)
The vast majority of issues require human engineers to:
- **Implement missing XPU kernels** (e.g., `ldexp`, `logaddexp` complex, efficient attention backward)
- **Fix accuracy/correctness bugs** (e.g., oneDNN accuracy, TransformerEncoder inf/nan, NaN in flash attention bf16)
- **Add new feature support** (e.g., FP8/MXFP8 ops, NestedTensor SDPA, flash attention dropout, extended head dims)
- **Resolve inductor/compiler issues** (e.g., PassManager failures, AOTI C++ compile errors, stride mismatches, codegen bugs)
- **Address hardware/driver dependencies** (e.g., SYCL Graph scratch memory, driver accuracy issues, missing XPU device properties)
- **Fix test/API compatibility** (e.g., device properties `major` attribute, `_xpu_recordMemoryHistory` signature, Dynamo skip rules for XPU context managers)
