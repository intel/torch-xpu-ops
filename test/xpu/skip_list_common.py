# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

skip_dict = {
    "complex_tensor/test_complex_tensor_xpu.py": None,
    "functorch/test_ops_xpu.py": None,
    "nn/test_convolution_xpu.py": None,
    "nn/test_dropout_xpu.py": None,
    "nn/test_embedding_xpu.py": None,
    "nn/test_init_xpu.py": None,
    "nn/test_lazy_modules_xpu.py": None,
    "nn/test_load_state_dict_xpu.py": None,
    "nn/test_module_hooks_xpu.py": None,
    "nn/test_multihead_attention_xpu.py": None,
    "nn/test_packed_sequence_xpu.py": None,
    "nn/test_parametrization_xpu.py": None,
    "nn/test_pooling_xpu.py": None,
    "nn/test_pruning_xpu.py": None,
    "quantization/core/test_quantized_op_xpu.py": (
        # AssertionError: Tensor-likes are not close!
        # RuntimeError: value cannot be converted to type int without overflow
        "test_add_scalar_relu_xpu",
        # AssertionError: Tensor-likes are not close!
        "test_cat_nhwc_xpu",
    ),
    "quantization/core/test_quantized_tensor_xpu.py": None,
    "quantization/core/test_workflow_module_xpu.py": None,
    "quantization/core/test_workflow_ops_xpu.py": (
        # AssertionError:
        # Not equal to tolerance rtol=1e-06, atol=1e-06
        "test_forward_per_channel_xpu",
        # AssertionError:
        # Not equal to tolerance rtol=1e-06, atol=1e-06
        "test_forward_per_tensor_xpu",
        # AssertionError: False is not true : Expected kernel forward function to have results match the reference forward function
        "test_learnable_forward_per_channel_cpu_xpu",
    ),
    "test_autocast_xpu.py": None,
    "test_autograd_fallback_xpu.py": None,
    "test_autograd_xpu.py": (
        # skipped due to #2536, torch._C._scatter or torch._C._gather
        "test_profiler_emit_nvtx_xpu",
        "test_checkpointing_without_reentrant_dataparallel",
        "test_dataparallel_saved_tensors_hooks",
    ),
    "test_binary_ufuncs_xpu.py": ("_jiterator_",),
    "test_comparison_utils_xpu.py": None,
    "test_complex_xpu.py": None,
    "test_content_store_xpu.py": None,
    "test_dataloader_xpu.py": None,
    "test_decomp_xpu.py": (
        # Slow test case: it takes more than 10 minutes to run on XPU.
        "test_quick_core_backward_baddbmm_xpu_float64",
    ),
    "test_distributions_xpu.py": None,
    "test_dynamic_shapes_xpu.py": None,
    "test_foreach_xpu.py": (
        # RuntimeError: Tried to instantiate dummy base class CUDAGraph
        "use_cuda_graph_True",
    ),
    "test_indexing_xpu.py": None,
    "test_linalg_xpu.py": (
        # skipped due to #2309, unsupported ops: aten::_dyn_quant_pack_4bit_weight, aten::narrow_copy, aten::_histogramdd_bin_edges
        "test__dyn_quant_matmul_4bit_m_1_k_128_n_11008_xpu",
        "test__dyn_quant_matmul_4bit_m_1_k_128_n_4096_xpu",
        "test__dyn_quant_matmul_4bit_m_1_k_64_n_11008_xpu",
        "test__dyn_quant_matmul_4bit_m_1_k_64_n_4096_xpu",
        "test__dyn_quant_matmul_4bit_m_32_k_128_n_11008_xpu",
        "test__dyn_quant_matmul_4bit_m_32_k_128_n_4096_xpu",
        "test__dyn_quant_matmul_4bit_m_32_k_64_n_11008_xpu",
        "test__dyn_quant_matmul_4bit_m_32_k_64_n_4096_xpu",
        "test__dyn_quant_pack_4bit_weight_k_256_n_128_xpu",
        "test__dyn_quant_pack_4bit_weight_k_256_n_32_xpu",
        "test__dyn_quant_pack_4bit_weight_k_256_n_48_xpu",
        "test__dyn_quant_pack_4bit_weight_k_256_n_64_xpu",
        "test__dyn_quant_pack_4bit_weight_k_64_n_128_xpu",
        "test__dyn_quant_pack_4bit_weight_k_64_n_32_xpu",
        "test__dyn_quant_pack_4bit_weight_k_64_n_48_xpu",
        "test__dyn_quant_pack_4bit_weight_k_64_n_64_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_1_k_128_n_11008_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_1_k_128_n_4096_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_1_k_64_n_11008_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_1_k_64_n_4096_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_32_k_128_n_11008_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_32_k_128_n_4096_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_32_k_64_n_11008_xpu",
        "test_compile_dyn_quant_matmul_4bit_m_32_k_64_n_4096_xpu",
        "_tunableop_",
        "_tuning_tunableop_",
    ),
    "test_masked_xpu.py": None,
    "test_maskedtensor_xpu.py": None,
    "test_meta_xpu.py": (
        # skipped due to #2309, unsupported ops: aten::narrow_copy, aten::_histogramdd_bin_edges
        "narrow_copy",
        "histogramdd",
        "_jiterator_",
    ),
    "test_modules_xpu.py": None,
    "test_namedtensor_xpu.py": None,
    "test_native_functions_xpu.py": None,
    "test_native_mha_xpu.py": None,
    "test_nn_xpu.py": None,
    "test_ops_fwd_gradients_xpu.py": None,
    "test_ops_gradients_xpu.py": None,
    "test_ops_xpu.py": (
        "_jiterator_",
        # crash
        "test_dtypes__refs_nn_functional_pdist_xpu",
        # not implemented
        "histogramdd",
        "_flash_attention_",
        "_efficient_attention_",
        # Exception: The supported dtypes for linalg.multi_dot on device type xpu are incorrect!
        "test_dtypes_linalg_multi_dot_xpu",
        # For CUDA it's skipped explicitly in common_methods_invocations.py in upstream. We can skip it here
        "test_out_histc_xpu_float32",
        "test_out_mean_xpu_float32",
        # FakeTensor mismatch in outputs_alias_inputs for aten.view.default
        # Known upstream issue: https://github.com/pytorch/pytorch/issues/159150
        "test_fake_crossref_backward_amp_nn_functional_bilinear_xpu_float32",
    ),
    "test_optim_xpu.py": None,
    "test_reductions_xpu.py": None,
    "test_scatter_gather_ops_xpu.py": None,
    "test_segment_reductions_xpu.py": None,
    "test_shape_ops_xpu.py": None,
    "test_sort_and_select_xpu.py": None,
    "test_sparse_csr_xpu.py": None,
    "test_sparse_xpu.py": None,
    "test_spectral_ops_xpu.py": None,
    "test_tensor_creation_ops_xpu.py": None,
    "test_torch_xpu.py": (
        # due to #2164, CUDA specific test
        "test_no_cuda_monkeypatch",
        # skipped due to #2536, torch._C._scatter or torch._C._gather
        "test_storage_setitem_xpu_float32",
        "test_storage_error_no_attribute",
        "test_storage_setitem_xpu_uint8",
        "test_tensor_storage_type_xpu_bfloat16",
        "test_tensor_storage_type_xpu_int16",
        "test_tensor_storage_type_xpu_float16",
        "test_storage_setitem_xpu_int16",
        "test_storage_setitem_xpu_float64",
        "test_tensor_storage_type_xpu_bool",
        "test_pickle_gradscaler_xpu",
        "test_tensor_storage_type_xpu_float64",
        "test_tensor_storage_type_xpu_float32",
        "test_tensor_storage_type_xpu_int8",
        "test_storage_setitem_xpu_int32",
        "test_tensor_storage_type_xpu_uint8",
        "test_tensor_storage_type_xpu_int32",
        "test_storage_setitem_xpu_int64",
        "test_grad_scaler_pass_itself_xpu",
        "test_storage_setitem_xpu_bool",
        "test_tensor_storage_type_xpu_int64",
        "test_storage_setitem_xpu_int8",
        "test_grad_scaling_state_dict_xpu",
        "test_typed_storage_deprecation_warning",
        "test_typed_storage_internal_no_warning",
        # TypeError: map2_ is only implemented on CPU tensors
        "test_broadcast_fn_map2_xpu",
        "test_broadcast_fn_map_xpu",
        # RuntimeError: _share_fd_: only available on CPU
        "test_module_share_memory_xpu",
    ),
    "test_transformers_xpu.py": None,
    "test_type_promotion_xpu.py": None,
    "test_unary_ufuncs_xpu.py": None,
    "test_view_ops_xpu.py": None,
    "test_schema_check.py": None,
    "test_nestedtensor_xpu.py": None,
    "functorch/test_eager_transforms_xpu.py": None,
    "test_cpp_api_parity_xpu.py": None,
    "test_expanded_weights_xpu.py": None,
    "test_fake_tensor_xpu.py": (
        # https://github.com/intel/torch-xpu-ops/issues/2472
        # aten::_cudnn_rnn/aten::miopen_rnn not supported
        "test_cudnn_rnn",
    ),
    "test_matmul_cuda_xpu.py": None,
    "functorch/test_vmap_xpu.py": None,
    "dynamo/test_ctx_manager_xpu.py": None,
    "functorch/test_control_flow_xpu.py": None,
    "profiler/test_memory_profiler.py": None,
    "export/test_hop_xpu.py": None,
    "export/test_export_opinfo_xpu.py": None,
    "functorch/test_aotdispatch_xpu.py": None,
}
