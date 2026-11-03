#!/bin/bash
# Sourced by run_ut.sh for all P0 groups/platforms. Paths are relative to each test working directory.

# Known P0 failures skipped by exact pytest node ID, grouped by tracking issue
# or failure reason. Paths are relative to the directory each test runs from;
# run_pytest maps them to pytest's rootdir-relative node IDs.
PYTEST_DESELECTS=(
    # https://github.com/intel/torch-xpu-ops/issues/1624 (open): [DONT CLOSE] Known UT Issue list
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_float64"
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_float16"
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_float8_e5m2"
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_float32"
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_bfloat16"
    "test/xpu/distributed/test_c10d_xccl.py::ProcessGroupXCCLTest::test_nan_assert_float8_e4m3fn"
    # https://github.com/intel/torch-xpu-ops/issues/1951 (open): Functionality issues in TestCommon.test_out.
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_ormqr_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_triangular_solve_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_cholesky_inverse_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_geqrf_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/1973 (open): AssertionError: Scalars or Tensor-likes are not equal or close!
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_state_dict_with_cuda_params_Muon_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/2249 (closed): NotImplementedError: The operator 'aten::_histogramdd_bin_edges' is not currently implemen
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_view_replay_histogramdd_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestMathBitsXPU::test_neg_view_histogramdd_xpu_float64"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_operator_histogramdd_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/2253 (open): the supported dtypes are not align with cuda
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_dtypes_histogramdd_xpu"
    # https://github.com/intel/torch-xpu-ops/issues/2254 (closed): NotImplementedError detected int test_ops.py
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_noncontiguous_samples_histogramdd_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_variant_consistency_eager_histogramdd_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_warning_histogramdd_xpu"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_histogramdd_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/2257 (open): Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_tan_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_tanh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_acosh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_asin_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_round_decimals_3_xpu_bfloat16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_tan_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal_tanh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large__refs_acosh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large__refs_asinh_xpu_complex128"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large__refs_asinh_xpu_complex32"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large__refs_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal__refs_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_nn_functional_tanhshrink_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_polygamma_polygamma_n_1_xpu_float16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_polygamma_polygamma_n_2_xpu_float16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_polygamma_polygamma_n_3_xpu_float16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_polygamma_polygamma_n_4_xpu_float16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_acos_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_acosh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_asin_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal_round_decimals_3_xpu_bfloat16"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_normal__refs_nn_functional_tanhshrink_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large_acosh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large_asinh_xpu_complex128"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large_asinh_xpu_complex32"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_large_asinh_xpu_complex64"
    "test/xpu/test_unary_ufuncs_xpu.py::TestUnaryUfuncsXPU::test_reference_numerics_extremal__refs_nn_functional_tanhshrink_xpu_complex64"
    # https://github.com/intel/torch-xpu-ops/issues/2285 (closed): Support efficient attention
    "test/xpu/test_ops_xpu.py::TestFakeTensorXPU::test_fake_crossref_backward_amp_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_view_replay_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_variant_consistency_eager_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_warning_torch_ops_aten__efficient_attention_forward_xpu"
    "test/xpu/test_ops_xpu.py::TestFakeTensorXPU::test_fake_crossref_backward_no_amp_torch_ops_aten__efficient_attention_forward_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_dtypes_torch_ops_aten__efficient_attention_forward_xpu"
    # https://github.com/intel/torch-xpu-ops/issues/2722 (closed): [Bug Skip]: NotImplementedError: Could not run 'aten::flip' with arguments from the 'Quant
    "test/xpu/test_shape_ops_xpu.py::TestShapeOpsXPU::test_flip_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/2853 (open): [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU.
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_torch_ops_aten__flash_attention_forward_xpu_float16"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_warning_torch_ops_aten__flash_attention_forward_xpu"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_dtypes_torch_ops_aten__flash_attention_forward_xpu"
    # https://github.com/intel/torch-xpu-ops/issues/2919 (closed): [release/2.14] [XPU][upstream_ut][COW] Fix materialization in remaining TestCompositeCompl
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv1d_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv2d_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv3d_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv_transpose1d_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv_transpose2d_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_conv_transpose3d_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/3082 (open): multithread support in distributed
    "test/distributed/fsdp/test_fsdp_misc.py::TestFSDPMiscWorldSize1::test_training_device_mismatch_errors"
    # https://github.com/intel/torch-xpu-ops/issues/3267 (closed): New failed test cases 2026-04-06
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_hook_skips_existing_state_amsgrad_True_Adam_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_numerics_AdamW_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_numerics_Adam_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_state_init_amsgrad_False_AdamW_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_state_init_amsgrad_False_Adam_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_state_init_amsgrad_True_AdamW_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_state_init_amsgrad_True_Adam_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_hook_skips_existing_state_amsgrad_False_AdamW_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_hook_skips_existing_state_amsgrad_False_Adam_xpu_float32"
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_fused_mixed_precision_hook_skips_existing_state_amsgrad_True_AdamW_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/3791 (open): [Driver 26.19] New UT failures in XPU topk-related tests
    "test/inductor/test_torchinductor.py::GPUTests::test_topk_xpu"
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_compare_cpu_topk_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/4861 (closed): [release/2.14] test_optim_xpu.py::TestOptimRenewedXPU::test_state_dict_cross_device_Muon_x
    "test/xpu/test_optim_xpu.py::TestOptimRenewedXPU::test_state_dict_cross_device_Muon_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/5027 (closed): test_comprehensive_combinations_xpu_float16 fails with Tensor-likes not close (numeric mis
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoXPU::test_comprehensive_combinations_xpu_float16"
    # https://github.com/intel/torch-xpu-ops/issues/5047 (closed): [release/2.14] [LNL] test_ops_xpu.py::TestCommonXPU::test_python_ref__refs_true_divide_xpu
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_python_ref__refs_true_divide_xpu_complex32"
    # https://github.com/intel/torch-xpu-ops/issues/5107 (open): [BMG] test_profiler_tree_xpu experimental_tree_with_stack stack frame mismatch (wait vs _b
    "test/profiler/test_profiler_tree.py::TestProfilerTree::test_profiler_experimental_tree_with_stack_and_torch_dispatch"
    "test/profiler/test_profiler_tree.py::TestProfilerTree::test_profiler_experimental_tree_with_stack_and_torch_function"
    "test/profiler/test_profiler_tree.py::TestProfilerTree::test_profiler_experimental_tree_with_stack_and_modules"
    # https://github.com/intel/torch-xpu-ops/issues/5454 (open): [Bug Skip]: [BMG] upstream_inductor UT failures (130 cases)
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoXPU::test_comprehensive_nn_functional_max_pool3d_xpu_float32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool3d_cpu_float32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool2d_cpu_int32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool3d_cpu_int32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool3d_cpu_float16"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool2d_cpu_float16"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool2d_cpu_float32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool2d_cpu_float64"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoXPU::test_comprehensive_nn_functional_max_pool3d_xpu_float64"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool3d_cpu_float64"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool2d_cpu_int64"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::test_comprehensive_nn_functional_max_pool3d_cpu_int64"
    # https://github.com/intel/torch-xpu-ops/issues/5484 (open): test_update_user_managed_buffer_xpu fails: constant-buffer allocation not reflected by XPU
    "test/inductor/test_aot_inductor.py::AOTInductorTestDualWrapper::test_update_user_managed_buffer_xpu"
    "test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleGpu::test_update_user_managed_buffer_xpu"
    # No tracking issue: AOTInductor package C++ compile failures
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_cpu::test_compile_with_exporter"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_cpu::test_compile_with_exporter_weights"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_with_exporter"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_with_exporter_weights"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_cpu::test_compile_after_package"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_cpu::test_compile_after_package_static"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_cpu::test_compile_standalone_cos"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_after_package_static"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_standalone_cos"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_after_package"
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackageCpp_xpu::test_compile_after_package_multi_arch"
    # No tracking issue: AssertionError: Scalars are not equal! (load package in fresh subprocess)
    "test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleCpu::test_aoti_load_package_in_fresh_subprocess_cpu"
    "test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleGpu::test_aoti_load_package_in_fresh_subprocess_xpu"
    "test/inductor/test_aot_inductor.py::AOTInductorTestDualWrapper::test_aoti_load_package_in_fresh_subprocess_xpu"
    # No tracking issue: AssertionError: Tensor-likes are not close!
    "test/inductor/test_torchinductor.py::TritonCodeGenTests::test_topk_decompose_sort_ops"
    # No tracking issue: custom op schema not found when loading package in fresh subprocess
    "test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleCpu::test_python_custom_op_load_package_error_in_fresh_subprocess_cpu"
    # No tracking issue: script, not a pytest test: fixture 'x' not found
    "test/profiling/triton_xpu_ops_time.py::test"
    # Additional failures observed in the 2.15 BMG CI run 40892572474.
    # https://github.com/intel/torch-xpu-ops/issues/5860
    "test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoXPU::test_comprehensive_nn_functional_interpolate_bicubic_xpu_float16"
    # https://github.com/intel/torch-xpu-ops/issues/5452 (feature: #5542)
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_cow_input_nn_functional_grid_sample_xpu_float32"
    "test/xpu/test_ops_xpu.py::TestCompositeComplianceXPU::test_forward_ad_nn_functional_grid_sample_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/6011
    "test/inductor/test_torchinductor_opinfo.py::TestInductorNextafterXPU::test_nextafter_noncontiguous_False_xpu_float32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorNextafterXPU::test_nextafter_noncontiguous_False_xpu_float64"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorNextafterXPU::test_nextafter_noncontiguous_True_xpu_float32"
    "test/inductor/test_torchinductor_opinfo.py::TestInductorNextafterXPU::test_nextafter_noncontiguous_True_xpu_float64"
    # https://github.com/intel/torch-xpu-ops/issues/5847 (closed; failure in this wheel)
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_histc_xpu_float32"
    # https://github.com/intel/torch-xpu-ops/issues/5453 (unexpected success)
    "test/xpu/test_ops_xpu.py::TestCommonXPU::test_numpy_ref_linalg_tensorinv_xpu_float64"
    # https://github.com/intel/torch-xpu-ops/issues/6012 (CPU accuracy)
    "test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcherGenericCPU::test_conv2d_binary_broadcast_shapes_cpu"
    "test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcherGenericCPU::test_conv3d_binary_broadcast_shapes_cpu"
    # https://github.com/intel/torch-xpu-ops/issues/6013 (subprocess import failure)
    "test/inductor/test_aot_inductor_package.py::TestAOTInductorPackage_cpu::test_custom_output_type_missing_pytree_registration_error"
)
