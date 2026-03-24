# Intel torch-xpu-ops Issue Tracker Summary

> Auto-generated summary of issue tracking categories.

## 2. Accuracy / Numerical Mismatches

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace op | 🟢 Open | PawelSwider2000 | module: ut, skipped, bug_fix_stage5 |
| [#2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test failures when test_compare_cpu atol and rtol changed | 🟢 Open | pbielak | skipped |
| [#2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Object comparison failed: torch.float32 != torch.float64 | 🟢 Open | etaf | skipped, ut_upstream |
| [#2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: Tensor-likes are not close! in functorch/test_vmap.py | 🟢 Open | tszulist-hbn | skipped |
| [#2630](https://github.com/intel/torch-xpu-ops/issues/2630) | [upstream_ut]  AssertionError: Scalars are not equal! | 🟢 Open | jmamzax | skipped, port_from_skiplist |
| [#2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] AssertionError: Scalars are not close! / Tensor-likes are not close! | 🟢 Open | jmamzax | skipped, bug_fix_stage5 |
| [#2594](https://github.com/intel/torch-xpu-ops/issues/2594) | [upstream_ut] backend_expected comparison failed in test/nn/test_convolution.py | 🟢 Open | tszulist-hbn | module: ut, ut_upstream |
| [#2593](https://github.com/intel/torch-xpu-ops/issues/2593) | [upstream_ut] AssertionError: Tensor-likes are not close! in test/nn/test_convolution.py | 🟢 Open | tszulist-hbn | module: ut, ut_upstream |
| [#2568](https://github.com/intel/torch-xpu-ops/issues/2568) | [Bug Skip]: [regression] AssertionError: Scalars are not close! & RuntimeError: scatter_reduce_kernel_sum | 🟢 Open | kdrozd-dev | skipped |
| [#2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: Scalars are not equal! (test_storage_use_count) | 🟢 Open | PawelSwider2000 | skipped |
| [#2378](https://github.com/intel/torch-xpu-ops/issues/2378) | [Bug Skip]: test_compare_cpu_nn_functional_huber_loss_xpu_float16 accuracy check failed | 🔴 Closed | BBBela | module: ut, skipped, bug_fix_stage4 |
| [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut] nn/test_embedding.py AssertionError: Tensor-likes are not close! | 🟢 Open | CuiYifeng, yucai-intel | module: inductor, skipped, ut_upstream |
| [#2269](https://github.com/intel/torch-xpu-ops/issues/2269) | Exception: Jacobian mismatch for output 0 with respect to input 0 | 🔴 Closed | BBBela | duplicate, Ready for merge, module: ut, skipped, bug_fix_stage4 |
| [#2267](https://github.com/intel/torch-xpu-ops/issues/2267) | Accuracy failures in test/xpu/test_decomp_xpu.py | 🔴 Closed | PawelSwider2000 | module: ut, skipped, bug_fix_stage5 |
| [#2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | 🟢 Open | pbielak | skipped, bug_fix_stage4 |
| [#2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are not close! in test/functorch/test_ops.py | 🟢 Open | BBBela | skipped, bug_fix_stage3 |
| [#2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: RuntimeError not raised (histc/mean dtype cast) | 🟢 Open | Silv3S | module: ut, skipped, ut_upstream |
| [#2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qkv_nested_xpu_float32 failed: AssertionError: Scalars are not equal! | 🟢 Open | PawelSwider2000 | Accuracy, module: ut, skipped |
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoderLayer | 🟢 Open | yucai-intel | skipped |
| [#1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Tensor-likes are not equal or close! | 🟢 Open | gplutop7 | hw: PVC, module: ut, skipped, bug_fix_stage4 |
| [#2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scalars are not equal! with test_prune_configs_over_shared_memory_limit | 🟢 Open | hoshibara | dependency component: oneAPI, module: inductor, module: ut, ut_upstream |
| [#2552](https://github.com/intel/torch-xpu-ops/issues/2552) | [upstream_ut]  AssertionError: Incorrect result from choice TritonTemplateCaller | 🟢 Open | daisyden, xiaowangintel | module: inductor, skipped, ut_upstream |
| [#2705](https://github.com/intel/torch-xpu-ops/issues/2705) | [upstream_ut]  RuntimeError: expected scalar type Float but found Half | 🟢 Open | Silv3S | skipped, ut_upstream, bug_fix_stage5 |
| [#2704](https://github.com/intel/torch-xpu-ops/issues/2704) | [upstream_ut]  AssertionError: AssertionError not raised (test_math_backend_high_precision) | 🟢 Open | kdrozd-dev | skipped, ut_upstream, bug_fix_stage5 |
| [#1895](https://github.com/intel/torch-xpu-ops/issues/1895) | div_trunk_rounding accuracy gap on float64 in test_ops_xpu.py | 🟢 Open | kdrozd-dev | skipped, bug_fix_stage5 |

## 3. Regression Trackers / Batch Skip Reports

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT failures 2026-3-13 | 🟢 Open | jenniew | skipped |
| [#2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3-2 ut failures | 🟢 Open | jmamzax | skipped |
| [#2897](https://github.com/intel/torch-xpu-ops/issues/2897) | [Bug Skip]: [UT] failed cases 2026-2-18 | 🟢 Open | jmamzax | skipped |
| [#2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py contains several failed ops | 🟢 Open | PawelSwider2000 | skipped |
| [#2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] failed cases 2026-2-2 | 🟢 Open | jmamzax | skipped |
| [#2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2026-1-22 | 🟢 Open | AKloniecki | skipped |
| [#2726](https://github.com/intel/torch-xpu-ops/issues/2726) | [Bug Skip]: [regression]AssertionError: dtypes redefinition for xpu in test_reductions_xpu.py | 🟢 Open | gplutop7 | skipped |
| [#2708](https://github.com/intel/torch-xpu-ops/issues/2708) | [Bug Skip] some cases of test_decomp_xpu.py failed | 🟢 Open | astachowiczhabana | module: ut, skipped |
| [#2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025-11-28 | 🟢 Open | AKloniecki | module: ut, skipped, bug_fix_stage4 |
| [#2266](https://github.com/intel/torch-xpu-ops/issues/2266) | PyTorch upstream introduced new issues in Nov 03 | 🔴 Closed | Silv3S | skipped, bug_fix_stage5 |
| [#2264](https://github.com/intel/torch-xpu-ops/issues/2264) | PyTorch upstream introduced new issues in oct 31 | 🟢 Open | Silv3S | skipped, bug_fix_stage5 |
| [#2218](https://github.com/intel/torch-xpu-ops/issues/2218) | [Bug Skip]: New failures 2025-10-27 | 🟢 Open | CuiYifeng | module: ut, skipped |
| [#2188](https://github.com/intel/torch-xpu-ops/issues/2188) | [Bug Skip]: new failures in 2025-10-17 | 🟢 Open | jmamzax | skipped, bug_fix_stage5 |
| [#2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing test cases caused by PyTorch changes | 🟢 Open | AKloniecki | skipped |
| [#2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly Wheel [2026-02-24] | 🟢 Open | Silv3S | skipped |
| [#2279](https://github.com/intel/torch-xpu-ops/issues/2279) | [Bug Skip]:  Complex cases got failed with 2025-11-03 pytorch | 🔴 Closed | Silv3S | module: ut, skipped |
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test/xpu/test_decomp.py | 🟢 Open | LuFinch | module: ut, skipped |
| [#2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced by pytorch test sample input updates | 🟢 Open | daisyden | skipped |
| [#2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success of test_cpu_gpu_parity_nn_ConvTranspose3d_xpu_complex32 | 🟢 Open | gplutop7 | skipped |
| [#3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel test_decomp_xpu.HasDecompTest,test_has_decomposition | 🟢 Open | — | skipped, random |
| [#2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'NoneType' object has no attribute 'clone' | 🟢 Open | pponikox | skipped, bug_fix_stage5 |
| [#2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed: Unexpected success | 🟢 Open | — | skipped, port_from_skiplist |
| [#2435](https://github.com/intel/torch-xpu-ops/issues/2435) | [upstream_ut]  AssertionError: RuntimeError not raised | 🟢 Open | — | skipped |

## 5. Operator Not Implemented for XPU

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedError: Could not run 'aten::flip' with arguments from the 'QuantizedXPU' backend | 🟢 Open | Silv3S | skipped, bug_fix_stage5 |
| [#2711](https://github.com/intel/torch-xpu-ops/issues/2711) | [upstream_ut]  NotImplementedError: The operator 'aten::miopen_rnn' is not currently implemented for the XPU device | 🔴 Closed | Silv3S | skipped, ut_upstream |
| [#2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could not create a primitive descriptor for the deconvolution forward propagation in functorch/test_vmap.py | 🟢 Open | tszulist-hbn | skipped, ut_upstream |
| [#2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] RuntimeError: false INTERNAL ASSERT FAILED. DispatchStub: missing kernel for xpu | 🟢 Open | xiaowangintel | skipped |
| [#2624](https://github.com/intel/torch-xpu-ops/issues/2624) | [upstream_ut]  RuntimeError: Unsupported dtype Half | 🟢 Open | CuiYifeng | module: inductor, skipped, ut_upstream |
| [#2616](https://github.com/intel/torch-xpu-ops/issues/2616) | [Bug Skip]: New failures RuntimeError: MKL FFT doesn't support tensor of type | 🔴 Closed | CuiYifeng | skipped |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | 🟢 Open | CuiYifeng | module: ut, skipped |
| [#2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: The operator 'aten::_cudnn_rnn' is not currently implemented for the XPU device | 🟢 Open | Silv3S | wontfix, module: ut, skipped |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | 🟢 Open | CuiYifeng | module: ut, skipped |
| [#2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend | 🟢 Open | Silv3S | Ready for merge, ut_upstream, bug_fix_stage5 |
| [#2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | 🟢 Open | daisyden | wontfix, module: op impl, skipped |
| [#2826](https://github.com/intel/torch-xpu-ops/issues/2826) | Title: [upstream_ut]  NotImplementedError: "upsample_bicubic2d_xpu" not implemented for 'Byte' | 🟢 Open | pbielak | skipped, ut_upstream |
| [#2833](https://github.com/intel/torch-xpu-ops/issues/2833) | test_nn upsampling issue (4) | 🔴 Closed | Silv3S | skipped |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | 🟢 Open | xiaowangintel | skipped |
| [#2673](https://github.com/intel/torch-xpu-ops/issues/2673) | [Bug Skip]: AssertionError: The supported dtypes for _refs.fft.fft2 on device type xpu are incorrect! | 🟢 Open | CuiYifeng | skipped |
| [#2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a primitive descriptor for the deconvolution forward propagation primitive. in test/functorch/test_ops.py | 🟢 Open | wpietka | skipped, bug_fix_stage5 |
| [#2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in softmax/unsampling/nonzero | 🟢 Open | CuiYifeng, yucai-intel | module: ut, skipped |

## 6. Sparse Tensor Operations

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | 🟢 Open | — | module: ut, ut_upstream |
| [#2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py need to be ported to enable for Intel GPU for test_sparse and test_sparse_csr cases | 🟢 Open | — | skipped |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 RuntimeError: could not create a primitive descriptor for the matmul primitive. | 🟢 Open | jenniew, CuiYifeng | module: ut, skipped |
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | 🟢 Open | jenniew | module: ut, skipped |
| [#2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_dense_addmm_meta_xpu meet unexpected warning | 🟢 Open | — | skipped |
| [#2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_ meet ValueError: all inputs are expected to be on the same GPU device. | 🟢 Open | — | skipped |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | 🟢 Open | jenniew | skipped |
| [#2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_subgroup_matrix_multiply_accumulate' | 🟢 Open | — | duplicate, module: dependency bug, dependency component: Triton, skipped |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | 🟢 Open | jenniew | skipped, ut_upstream |
| [#2213](https://github.com/intel/torch-xpu-ops/issues/2213) | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm meet RuntimeError: `ccol_indices[..., 0] == 0` is not satisfied | 🟢 Open | jenniew | skipped, ut_upstream |
| [#2212](https://github.com/intel/torch-xpu-ops/issues/2212) | test/test_sparse.py::TestSparseAnyXPU::test_constructor_autograd_SparseCSC_xpu meet python core dump | 🟢 Open | CuiYifeng | skipped, ut_upstream |
| [#2211](https://github.com/intel/torch-xpu-ops/issues/2211) | test/test_sparse.py::TestSparseXPU::test_sparse_addmm and test_sparse_matmul meet NotImplementedError: Could not run 'aten::addmm' with arguments from the 'SparseXPU' backend. | 🟢 Open | jenniew | skipped, ut_upstream |
| [#2209](https://github.com/intel/torch-xpu-ops/issues/2209) | test_sparse.py::TestSparseAnyXPU::test_binary_operation meet RuntimeError: expected row_indices to be a contiguous tensor per batch | 🟢 Open | CuiYifeng, yucai-intel | skipped |
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is not supported | 🟢 Open | jenniew | duplicate, ut_upstream |
| [#2109](https://github.com/intel/torch-xpu-ops/issues/2109) | "worker 'gw0' crashed while running sparce_coo cases on PVC in CI | 🟢 Open | — | module: ut, skipped |
| [#1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_crossref_backward_no_amp_to_sparse_xpu_float32 | 🟢 Open | jenniew, mengfei25 | dependency component: driver, module: ut, skipped |

## 7. Random / Intermittent Failures

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | 🟢 Open | chunhuanMeng | skipped, random |
| [#2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW10 | 🟢 Open | — | hw: PVC, skipped, random |
| [#2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW09 | 🟢 Open | — | skipped, random |
| [#2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 2026WW05 | 🟢 Open | AKloniecki | skipped, random |
| [#2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW04 | 🟢 Open | — | skipped, random |
| [#2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 2026WW03 | 🟢 Open | Silv3S | skipped, bug_fix_stage5, random |
| [#2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | 🟢 Open | — | skipped, random |
| [#2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | 🟢 Open | wpietka | skipped, random |
| [#2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 2025-12-17 | 🟢 Open | — | skipped, random |
| [#2481](https://github.com/intel/torch-xpu-ops/issues/2481) | Two test_matmul_cuda.py cases failed in torch-xpu-ops CI but passed in local machine. | 🟢 Open | BBBela | skipped |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: 'NoneType' object has no attribute 'clone' | 🟢 Open | daisyden | skipped, random |
| [#2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_size_comparison_no_recompile | 🟢 Open | guangyey | skipped |
| [#2143](https://github.com/intel/torch-xpu-ops/issues/2143) | Random accuracy failure in "_index_put_impl_" | 🔴 Closed | CuiYifeng | module: ut |

## 8. CUDA Compatibility / Porting Gaps

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: Tried to instantiate dummy base class CUDAGraph | 🟢 Open | wincent8 | skipped, port_from_skiplist |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torch not compiled with CUDA enabled | 🟢 Open | daisyden | skipped, port_from_skiplist |
| [#2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  AssertionError: RuntimeError not raised | 🟢 Open | PatrykWilczewski | skipped, bug_fix_stage5, port_from_skiplist |
| [#2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: False is not true | 🟢 Open | Silv3S | skipped, port_from_skiplist |
| [#2520](https://github.com/intel/torch-xpu-ops/issues/2520) | [upstream_ut]  TypeError: type torch.cuda.FloatTensor not available. Torch not compiled with CUDA enabled. | 🔴 Closed | Silv3S | skipped |
| [#2495](https://github.com/intel/torch-xpu-ops/issues/2495) | [upstream_ut]  AssertionError: Torch not compiled with CUDA enabled | 🔴 Closed | pbielak | skipped |
| [#2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: False is not true | 🟢 Open | PatrykWilczewski | skipped, bug_fix_stage5 |
| [#2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it is cuda specific | 🟢 Open | daisyden | wontfix, skipped |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with CUDA enabled | 🟢 Open | daisyden | module: ut, skipped |
| [#2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with "AssertionError: Torch not compiled with CUDA enabled " | 🟢 Open | — | module: ut |
| [#1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: CUDA not available | 🟢 Open | etaf | module: ut |
| [#2334](https://github.com/intel/torch-xpu-ops/issues/2334) | [upstream_ut] AssertionError: Legacy XPU profiling is not supported. Requires use_kineto=True on XPU devices. | 🟢 Open | daisyden | module: inductor, module: ut, ut_upstream |
| [#2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn isinstance() check | 🟢 Open | Silv3S | skipped |

## 12. Hardware-Specific (BMG/LNL/PVC/Windows)

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 && cur_target < n_classes` failed'  not found in 'PYTORCH_API_USAGE torch.python | 🟢 Open | — | os: Windows, module: ut |
| [#2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] New failed test cases and 2.9 also failed but pvc passed | 🟢 Open | kdrozd-dev | os: Windows, hw: BMG, module: ut |
| [#2661](https://github.com/intel/torch-xpu-ops/issues/2661) | [release/2.10][Windows][BMG] New failed test cases but 2.9 passed | 🔴 Closed | kdrozd-dev | os: Windows, hw: BMG, module: ut |
| [#2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] New failed test cases | 🟢 Open | — | os: Windows, hw: BMG, module: ut |
| [#2299](https://github.com/intel/torch-xpu-ops/issues/2299) | [Windows][LNL][BMG]PyTorch 2.9 UT regression summary. | 🟢 Open | xuhancn, libohao1201 | os: Windows, hw: LNL, hw: BMG, module: ut |
| [#1819](https://github.com/intel/torch-xpu-ops/issues/1819) | [BMG-Windows][PT2.8]Torch-xpu-ops UT got TypeError | 🟢 Open | kdrozd-dev | os: Windows, hw: BMG, skipped |
| [#2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | 🟢 Open | — | os: Windows, module: ut, skipped_windows |

## 13. Crash / Segfault / Hang / Device Lost

| Issue | Title | Status | Assignee(s) | Labels |
|-------|-------|--------|-------------|--------|
| [#2990](https://github.com/intel/torch-xpu-ops/issues/2990) | [Bug Skip]: test.regressions.test_tril.TestSimpleBinary,test_tril got UR_RESULT_ERROR_DEVICE_LOST | 🟢 Open | BBBela | hw: BMG, skipped |
| [#2757](https://github.com/intel/torch-xpu-ops/issues/2757) | [BMG] test_tril got UR_RESULT_ERROR_DEVICE_LOST | 🟢 Open | — | hw: BMG, module: ut |
| [#2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault when running test_torch.TestTorch and test_torch.TestTorchDeviceType at the same tiem. | 🟢 Open | astachowiczhabana | skipped |
| [#2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) ; Runtime | 🟢 Open | wincent8 | skipped |
| [#1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu-ops UT][ww10] 2 coredump | 🟢 Open | mengfei25 | dependency component: driver, module: ut |
| [#1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang on BMG Ubuntu | 🟢 Open | Stonepia, mengfei25 | bug, client, os: Ubuntu, hw: BMG, dependency component: driver, module: ut |
