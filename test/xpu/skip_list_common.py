skip_dict = {
    "test_ops_xpu.py": (
        # Jiterator is only supported on CUDA and ROCm GPUs, none are available.
        # https://github.com/intel/torch-xpu-ops/issues/584
        "_jiterator_",
        # OPs not supported
        "test_errors_dot_xpu",
        "test_errors_vdot_xpu",
        # core dump
        "test_dtypes__refs_nn_functional_pdist_xpu",
        # Reference result was farther (inf) from the precise
        # computation than the torch result was (nan)!
        "test_python_ref_executor__refs_pow_executor_aten_xpu_complex32",
        "test_python_ref_executor__refs_mul_executor_aten_xpu_complex32",
        # https://github.com/intel/torch-xpu-ops/issues/2254
        "histogramdd",
        "_vdot_",
        "_dot_",
        "_flash_attention_",
        "_efficient_attention_",
    ),
    "test_binary_ufuncs_xpu.py": (
        "test_fmod_remainder_by_zero_integral_xpu_int64",  # zero division is an undefined behavior: different handles on different backends
        "test_div_rounding_numpy_xpu_float16",  # Calculation error. XPU implementation uses opmath type.
        # AssertionError: Jiterator is only supported on CUDA and ROCm GPUs, none are available.
        "_jiterator_",
    ),
    "test_scatter_gather_ops_xpu.py": (
        # AssertionError: Tensor-likes are not equal!
        # Mismatched elements: 2 / 1870 (0.1%)
        # Greatest absolute difference: 2.220446049250313e-16 at index (14, 9, 4)
        # Greatest relative difference: 1.7039539596977877e-16 at index (15, 7, 6)
        "test_scatter_reduce_mean_xpu_float64",
    ),
    "test_autograd_fallback_xpu.py": None,
    "test_sort_and_select_xpu.py": (
        "test_sort_large_slice_xpu",
    ),  # Hard code CUDA, UT has already been rewritten to test/regressions/test_sort.py.
    "nn/test_embedding_xpu.py": (
        # CUDA implementation has no such functionality due to performance consideration.
        # skipped by CUDA for performance
        # @skipCUDAIf(True, "no out-of-bounds check on CUDA for perf.")
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_max_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_max_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_max_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_max_xpu_float64_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_mean_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_mean_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_mean_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_mean_xpu_float64_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_sum_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_sum_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_sum_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx0_mode_sum_xpu_float64_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_max_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_max_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_max_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_max_xpu_float64_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_mean_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_mean_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_mean_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_mean_xpu_float64_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_sum_xpu_float32_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_sum_xpu_float32_int64",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_sum_xpu_float64_int32",
        "test_embedding_bag_out_of_bounds_idx_padding_idx_0_mode_sum_xpu_float64_int64",
    ),
    "test_transformers_xpu.py": (
        # Efficient attention is not supported.
        "test_mem_eff_attention_large_seq_len_uniform_attention_xpu",
        # AssertionError("Torch not compiled with CUDA enabled")
        "test_mem_eff_attention_fail_with_batch_size_geq_65536",
        # https://github.com/intel/torch-xpu-ops/issues/761
        # AssertionError: False is not true
        # CPU fallback failure. To support aten::transformer_encoder_layer_forward with proper priority.
        "test_disable_fastpath_xpu",
        # NestedTensorXPU not supported
        # Could not run 'aten::_to_copy' with arguments from the 'NestedTensorXPU' backend
        "test_with_nested_tensor_input_xpu",
        # oneDNN issues
        # Double and complex datatype matmul is not supported in oneDNN
        # https://github.com/intel/torch-xpu-ops/issues/253
        "test_scaled_dot_product_attention_4D_input_dim_no_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_4D_causal_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_4D_causal_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_2D_causal_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_2D_causal_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_2D_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_4D_input_dim_2D_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_no_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_no_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_3D_causal_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_3D_causal_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_2D_causal_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_2D_causal_attn_mask_dropout_p_0_2_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_2D_attn_mask_dropout_p_0_5_xpu",
        "test_scaled_dot_product_attention_3D_input_dim_2D_attn_mask_dropout_p_0_2_xpu",
    ),
    "test_complex_xpu.py": None,
    "test_modules_xpu.py": (
        # oneDNN issues
        # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_grad_nn_MultiheadAttention_eval_mode_xpu_float64",
        "test_grad_nn_MultiheadAttention_train_mode_xpu_float64",
        "test_gradgrad_nn_MultiheadAttention_eval_mode_xpu_float64",
        "test_gradgrad_nn_MultiheadAttention_train_mode_xpu_float64",
        # Unexpected success:
        "test_cpu_gpu_parity_nn_ConvTranspose1d_xpu_complex32",
        "test_cpu_gpu_parity_nn_ConvTranspose2d_xpu_complex32",
        # CPU fallback fails
        # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # AssertionError: False is not true
        "test_to_nn_BatchNorm1d_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_BatchNorm1d_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_BatchNorm2d_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_BatchNorm2d_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_BatchNorm3d_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_BatchNorm3d_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Bilinear_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Conv1d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Conv2d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Conv3d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_ConvTranspose1d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_ConvTranspose2d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_ConvTranspose3d_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Embedding_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_GRUCell_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_GRU_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_GRU_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_GroupNorm_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_LSTMCell_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_LSTM_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_LSTM_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_LayerNorm_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Linear_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_MultiheadAttention_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_MultiheadAttention_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_PReLU_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_RMSNorm_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_RNNCell_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_RNN_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_RNN_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_TransformerDecoderLayer_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_TransformerEncoderLayer_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_TransformerEncoderLayer_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_TransformerEncoder_eval_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_TransformerEncoder_train_mode_swap_True_set_grad_True_xpu_float32",
        "test_to_nn_Transformer_swap_True_set_grad_True_xpu_float32",
        # Unexpected succuss
        "test_memory_format_nn_Conv2d_xpu_float64",
        "test_memory_format_nn_ConvTranspose2d_xpu_float64",
        "test_memory_format_nn_LazyConv2d_xpu_float64",
        "test_memory_format_nn_LazyConvTranspose2d_xpu_float64",
    ),
    "test_nn_xpu.py": (
        # AttributeError: module 'torch.xpu' has no attribute 'FloatTensor'
        "test_type",
        # rnn fallback to cpu
        "test_cudnn_weight_format",
        # oneDNN issues
        # AssertionError: MultiheadAttention does not support NestedTensor outside of its fast path. The fast path was not hit because some Tensor argument's device is neither one of cpu, cuda or privateuseone
        "test_TransformerEncoderLayer_empty_xpu",
        "test_transformerencoderlayer_xpu_float16",
        "test_transformerencoderlayer_xpu_float32",
        # oneDNN issues
        # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_transformerencoderlayer_xpu_float64",
        # Unexpected success: CUDA only test case, launch grid_y == 2**16 (larger than CUDA maximum y-dimension limit 65535) and expect fail.
        # SYCL don't have this limitation and hence can pass.
        "test_upsamplingNearest2d_launch_fail_xpu",
        # AssertionError: False is not true
        "test_ctc_loss_cudnn_xpu",  # want "xpu" in function name
        "test_ctc_loss_cudnn_tensor",  # want "xpu" in function name
        # RuntimeError: reflection_pad2d_backward_xpu does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
        "test_ReflectionPad2d_large_deterministic_xpu",
        # x_cuda = x.clone().detach().to("cuda").requires_grad_(): Torch not compiled with CUDA enabled
        "test_layer_norm_backwards_eps",
    ),
    "test_indexing_xpu.py": None,
    "nn/test_pooling_xpu.py": None,
    "nn/test_dropout_xpu.py": None,
    "test_dataloader_xpu.py": None,
    "test_tensor_creation_ops_xpu.py": None,
    "test_autocast_xpu.py": None,
    "test_autograd_xpu.py": (
        # AttributeError: module 'torch.xpu' has no attribute
        "test_profiler_emit_nvtx_xpu",
        # Double and complex datatype matmul is not supported in oneDNN
        "test_mv_grad_stride_0_xpu",
        # module 'torch._C' has no attribute '_scatter'
        "test_checkpointing_without_reentrant_dataparallel",
        "test_dataparallel_saved_tensors_hooks",
    ),
    "test_reductions_xpu.py": None,
    "test_unary_ufuncs_xpu.py": None,
    "test_masked_xpu.py": None,
    "test_view_ops_xpu.py": (
        # Need quantization support, NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_flatten_xpu",
        "test_ravel_xpu",
    ),
    "test_shape_ops_xpu.py": (
        # Need quantization support.
        # https://github.com/intel/torch-xpu-ops/issues/275
        # NotImplementedError: Could not run 'aten::empty_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_flip_xpu_float32",
    ),
    "test_content_store_xpu.py": None,
    "test_native_functions_xpu.py": None,
    "nn/test_init_xpu.py": None,
    "test_namedtensor_xpu.py": None,
    "nn/test_lazy_modules_xpu.py": None,
    "test_linalg_xpu.py": (
        # Summary:
        # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_tensordot_out_kernel_errors_with_autograd_xpu_float32",
        "test_addmm_gelu_xpu_float64",
        "test_addmm_relu_xpu_float64",
        "test_addmm_sizes_xpu_float64",
        "test_addmm_xpu_float64",
        "test_baddbmm_xpu_float64",
        "test_einsum_random_xpu_float64",
        "test_lobpcg_basic_xpu_float64",
        "test_lobpcg_ortho_xpu_float64",
        "test_pca_lowrank_xpu",
        "test_svd_lowrank_xpu_float64",
        "test_linalg_lstsq_input_checks_xpu_float32",
        "test_linalg_lstsq_input_checks_xpu_float64",
        "test_dot_invalid_args_xpu",
        "test_vdot_invalid_args_xpu",
        "test__int_mm_errors_xpu",
        # https://github.com/intel/torch-xpu-ops/issues/814
        # xpu does not have '_cuda_tunableop_is_enabled' API
        "_tunableop_",
        "test_matmul_small_brute_force_tunableop_xpu_float16",
        "test_matmul_small_brute_force_tunableop_xpu_float32",
        "test_matmul_small_brute_force_tunableop_xpu_float64",
        "test_matmul_offline_tunableop_xpu_float16",
        # XPU does not support tunable.
        "test_bmm_tunableop_rocm_xpu_float32",
        "test_numeric_check_leak_tunableop_rocm_xpu_float32",
        "test_dump_results_on_exit_tunableop_xpu_float32",
        "test_rotating_buffer_tunableop_xpu_float32",
        "test_gemm_bias_tunableop_xpu_bfloat16",
        "test_scaled_gemm_tunableop_xpu_float8_e4m3fnuz",
        "test_scaled_gemm_tunableop_xpu_float8_e5m2fnuz",
        # CUDA bias cases added in latest PyTorch
        # AttributeError: module 'torch._C' has no attribute '_cuda_tunableop_enable'
        # https://github.com/intel/torch-xpu-ops/issues/2066
        "test_matmul_check_entries_tunableop_xpu_float16",
        "test_minimum_tuning_iteration_tunableop_xpu_float16",
        "test_validator_tunableop_rocm_xpu_float32",
        "test_addmm_relu_tunableop_rocm_xpu_float32",
        "test_addmm_relu_tunableop_rocm_xpu_float64",
        "_tuning_tunableop_",
        # TODO: align input data type for convert_weight_to_int4pack with CUDA
        # XPU expects weight to be kInt, while CUDA expects kByte
        "test__int4_mm_m_32_k_32_n_48_xpu",
        "test__int4_mm_m_32_k_64_n_48_xpu",
        "test__int4_mm_m_64_k_32_n_48_xpu",
        "test__int4_mm_m_64_k_32_n_64_xpu",
        "test__int4_mm_m_64_k_64_n_48_xpu",
        "test__int4_mm_m_64_k_64_n_64_xpu",
        "test_compile_int4_mm_m_32_k_32_n_48_xpu",
        "test_compile_int4_mm_m_32_k_32_n_64_xpu",
        "test_compile_int4_mm_m_32_k_64_n_48_xpu",
        "test_compile_int4_mm_m_32_k_64_n_64_xpu",
        "test_compile_int4_mm_m_64_k_32_n_48_xpu",
        "test_compile_int4_mm_m_64_k_32_n_64_xpu",
        "test_compile_int4_mm_m_64_k_64_n_48_xpu",
        "test_compile_int4_mm_m_64_k_64_n_64_xpu",
        "test__int4_mm_m_32_k_32_n_48_xpu",
        "test__int4_mm_m_32_k_64_n_48_xpu",
        "test__int4_mm_m_64_k_32_n_48_xpu",
        "test__int4_mm_m_64_k_32_n_64_xpu",
        "test__int4_mm_m_64_k_64_n_48_xpu",
        "test__int4_mm_m_64_k_64_n_64_xpu",
        "test_compile_int4_mm_m_32_k_32_n_48_xpu",
        "test_compile_int4_mm_m_32_k_32_n_64_xpu",
        "test_compile_int4_mm_m_32_k_64_n_48_xpu",
        "test_compile_int4_mm_m_32_k_64_n_64_xpu",
        "test_compile_int4_mm_m_64_k_32_n_48_xpu",
        "test_compile_int4_mm_m_64_k_32_n_64_xpu",
        "test_compile_int4_mm_m_64_k_64_n_48_xpu",
        "test_compile_int4_mm_m_64_k_64_n_64_xpu",
        # float8 is not supported
        "test_matmul_scaled_gemm_offline_tunableop_xpu_float8_e4m3fnuz",
        "test_matmul_scaled_gemm_offline_tunableop_xpu_float8_e5m2fnuz",
        "test_scaled_gemm_offline_tunableop_xpu_float8_e4m3fnuz",
        "test_scaled_gemm_offline_tunableop_xpu_float8_e5m2fnuz",
        # case need to port for xpu
        "test_gemm_bias_offline_tunableop_xpu_bfloat16",
        # Exception is temporarily unavailable due to regression in oneMKL
        "test_inv_errors_and_warnings_xpu_float32",
        "test_inv_errors_and_warnings_xpu_float64",
        "test_inverse_errors_large_xpu_float32",
        "test_inverse_errors_large_xpu_float64",
        "test_inverse_errors_xpu_float32",
        "test_inverse_errors_xpu_float64",
        "test_inv_ex_singular_xpu_float32",
        "test_inv_ex_singular_xpu_float64",
    ),
    "test_ops_fwd_gradients_xpu.py": (
        # All of the followings are oneDNN issues
        # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_fn_fwgrad_bwgrad___rmatmul___xpu_float64",
        "test_fn_fwgrad_bwgrad_addmv_xpu_float64",
        "test_fn_fwgrad_bwgrad_addr_xpu_float64",
        "test_fn_fwgrad_bwgrad_matmul_xpu_float64",
        "test_fn_fwgrad_bwgrad_mv_xpu_float64",
        "test_forward_mode_AD___rmatmul___xpu_float64",
        "test_forward_mode_AD_addbmm_xpu_float64",
        "test_forward_mode_AD_addmm_xpu_float64",
        "test_forward_mode_AD_addmv_xpu_float64",
        "test_forward_mode_AD_baddbmm_xpu_float64",
        "test_forward_mode_AD_matmul_xpu_float64",
        "test_forward_mode_AD_mv_xpu_float64",
        "test_inplace_forward_mode_AD_addbmm_xpu_float64",
        "test_inplace_forward_mode_AD_addmm_decomposed_xpu_float64",
        "test_inplace_forward_mode_AD_addmm_xpu_float64",
        "test_inplace_forward_mode_AD_addmv_xpu_float64",
        "test_inplace_forward_mode_AD_baddbmm_xpu_float64",
        "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose2d_xpu_float64",
        "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose3d_xpu_float64",
        "test_forward_mode_AD_nn_functional_conv_transpose2d_xpu_float64",
        "test_forward_mode_AD_nn_functional_conv_transpose3d_xpu_float64",
    ),
    #    "test_matmul_cuda_xpu.py": None,
    "test_maskedtensor_xpu.py": None,
    "quantization/core/test_quantized_op_xpu.py": (
        # AssertionError: Torch not compiled with CUDA enabled
        "test_qgelu_xpu",
        "test_qrelu_xpu",
        # AttributeError: 'TestQuantizedOpsXPU' object has no attribute 'test_qsoftmax'
        "test_qsoftmax_qnnpack_xpu",
    ),
    "quantization/core/test_workflow_ops_xpu.py": None,
    "quantization/core/test_workflow_module_xpu.py": None,
    "quantization/core/test_quantized_tensor_xpu.py": (
        # Summary: Quantized OPs are not supported for XPU
        # NotImplementedError: Could not run 'aten::dequantize.self' with arguments from the 'QuantizedXPU' backend
        "test_compare_per_channel_device_numerics_xpu",
        # NotImplementedError: Could not run 'aten::dequantize.self' with arguments from the 'QuantizedXPU' backend.
        "test_compare_per_tensor_device_numerics_xpu",
        # NotImplementedError: Could not run 'aten::empty_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_cuda_quantization_does_not_pin_memory_xpu",
        # NotImplementedError: Could not run 'aten::_empty_per_channel_affine_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_per_channel_qtensor_creation_cuda_xpu",
        # NotImplementedError: Could not run 'aten::empty_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_per_channel_to_device_xpu",
        # NotImplementedError: Could not run 'aten::empty_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_per_tensor_to_device_xpu",
        # NotImplementedError: Could not run 'aten::q_scale' with arguments from the 'QuantizedXPU' backend.
        "test_qtensor_cuda_xpu",
        # NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'QuantizedXPU' backend.
        "test_qtensor_index_put_cuda_xpu",
        # NotImplementedError: Could not run 'aten::index_select' with arguments from the 'QuantizedXPU' backend.
        "test_qtensor_index_select_cuda_xpu",
        # NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend.
        "test_qtensor_masked_fill_cuda_xpu",
    ),
    "nn/test_packed_sequence_xpu.py": (
        # test case porting issue
        "test_to and not test_to_memory and not test_total",
    ),
    "test_ops_gradients_xpu.py": (
        # All are oneDNN issues
        ### Error #0 in TestBwdGradientsXPU , totally 271 , RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_fn_grad___rmatmul___xpu_float64",
        "test_fn_grad_addbmm_xpu_float64",
        "test_fn_grad_addmm_xpu_float64",
        "test_fn_grad_addmv_xpu_float64",
        "test_fn_grad_baddbmm_xpu_float64",
        "test_fn_grad_cdist_xpu_float64",
        "test_fn_grad_matmul_xpu_float64",
        "test_fn_grad_mv_xpu_float64",
        "test_fn_grad_nn_functional_multi_head_attention_forward_xpu_float64",
        "test_fn_gradgrad___rmatmul___xpu_float64",
        "test_fn_gradgrad_addmv_xpu_float64",
        "test_fn_gradgrad_addr_xpu_float64",
        "test_fn_gradgrad_matmul_xpu_float64",
        "test_fn_gradgrad_mv_xpu_float64",
        "test_inplace_grad_addbmm_xpu_float64",
        "test_inplace_grad_addmm_decomposed_xpu_float64",
        "test_inplace_grad_addmm_xpu_float64",
        "test_inplace_grad_addmv_xpu_float64",
        "test_inplace_grad_baddbmm_xpu_float64",
        "test_inplace_gradgrad_addmv_xpu_float64",
        "test_inplace_gradgrad_addr_xpu_float64",
        "test_fn_grad_nn_functional_conv_transpose2d_xpu_float64",
        "test_fn_grad_nn_functional_conv_transpose3d_xpu_float64",
        "test_fn_gradgrad_nn_functional_conv_transpose2d_xpu_float64",
        "test_fn_gradgrad_nn_functional_conv_transpose3d_xpu_float64",
        "test_fn_gradgrad_index_reduce_mean_xpu_float64",
        "test_fn_gradgrad_index_reduce_prod_xpu_float64",
        "test_inplace_gradgrad_index_reduce_mean_xpu_float64",
        "test_inplace_gradgrad_index_reduce_prod_xpu_float64",
    ),
    "test_torch_xpu.py": (
        # 'torch.xpu' has no attribute ...
        ### Error #1 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'FloatTensor'
        "test_grad_scaling_state_dict_xpu",
        ### Error #2 in TestTorchDeviceTypeXPU , totally 1 , AttributeError: 'torch.storage.TypedStorage' object has no attribute 'is_xpu'
        ### Error #3 in TestTorchDeviceTypeXPU , totally 3 , AttributeError: module 'torch.xpu' has no attribute 'ByteStorage'
        "test_storage_setitem_xpu_uint8",
        "test_tensor_storage_type_xpu_uint8",
        ### Error #4 in TestTorchDeviceTypeXPU , totally 4 , AttributeError: module 'torch.xpu' has no attribute 'FloatStorage'
        "test_storage_setitem_xpu_float32",
        "test_tensor_storage_type_xpu_float32",
        ### Error #7 in TestTorchDeviceTypeXPU , totally 1 , TypeError: map2_ is only implemented on CPU tensors
        "test_broadcast_fn_map2_xpu",
        ### Error #8 in TestTorchDeviceTypeXPU , totally 1 , TypeError: map_ is only implemented on CPU tensors
        "test_broadcast_fn_map_xpu",
        ### Error #12 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'amp'
        "test_grad_scaler_pass_itself_xpu",
        "test_pickle_gradscaler_xpu",
        ### Error #15 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: Tensor-likes are not close!
        "test_index_put_non_accumulate_deterministic_xpu",
        ### Error #17 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: False is not true
        "test_sync_warning_xpu",
        ### Error #19 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: _share_fd_: only available on CPU
        "test_module_share_memory_xpu",
        # 'torch.xpu' has no attribute ...
        ### Error #30 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'BoolStorage'
        "test_storage_setitem_xpu_bool",
        "test_tensor_storage_type_xpu_bool",
        ### Error #33 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'DoubleStorage'
        "test_storage_setitem_xpu_float64",
        "test_tensor_storage_type_xpu_float64",
        ### Error #34 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'ShortStorage'
        "test_storage_setitem_xpu_int16",
        "test_tensor_storage_type_xpu_int16",
        ### Error #35 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'IntStorage'
        "test_storage_setitem_xpu_int32",
        "test_tensor_storage_type_xpu_int32",
        ### Error #36 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'LongStorage'
        "test_storage_setitem_xpu_int64",
        "test_tensor_storage_type_xpu_int64",
        ### Error #37 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'CharStorage'
        "test_storage_setitem_xpu_int8",
        "test_tensor_storage_type_xpu_int8",
        ### Error #38 in TestTorchDeviceTypeXPU , totally 1 , AttributeError: module 'torch.xpu' has no attribute 'BFloat16Storage'
        "test_tensor_storage_type_xpu_bfloat16",
        ### Error #39 in TestTorchDeviceTypeXPU , totally 1 , AttributeError: module 'torch.xpu' has no attribute 'HalfStorage'
        "test_tensor_storage_type_xpu_float16",
        ### Module 'torch.xpu' has no attribute 'ByteStorage'
        "test_tensor_storage_type_xpu_uint8",
        # issue 302 , 8
        "test_print",
        "test_storage_error",
        "test_storage_error_no_attribute",
        # issue 302, 6
        "test_storage_error",
        "test_typed_storage_deprecation_warning",
        "test_typed_storage_internal_no_warning",
        # issue 302, 11
        "test_cuda_vitals_gpu_only_xpu",
        # torch.utils.swap_tensors AssertionError: RuntimeError not raised
        "test_swap_basic",
        # internally uses index_put deterministic implementation
        # dependent on "test_index_put_non_accumulate_deterministic"
        "test_index_copy_deterministic",
    ),
    "nn/test_multihead_attention_xpu.py": None,
    "test_native_mha_xpu.py": None,
    "test_comparison_utils_xpu.py": None,
    "test_segment_reductions_xpu.py": None,
    "nn/test_pruning_xpu.py": None,
    "test_foreach_xpu.py": (
        # RuntimeError: Tried to instantiate dummy base class CUDAGraph
        "use_cuda_graph_True",
        # randomly fails
        "test_parity__foreach_div_fastpath_inplace_xpu_complex128",
        "test_parity__foreach_div_fastpath_outplace_xpu_complex128",
        "test_parity__foreach_addcdiv_fastpath_inplace_xpu_complex128",
        "test_parity__foreach_addcdiv_fastpath_outplace_xpu_complex128",
    ),
    "nn/test_convolution_xpu.py": (
        # Summary: all of them are oneDNN related issues
        # XPU unsupport ops, skip.
        # https://github.com/intel/torch-xpu-ops/issues/348
        "test_cudnn_convolution_relu_xpu_float16",
        "test_cudnn_convolution_relu_xpu_float32",
        "test_cudnn_convolution_add_relu_xpu_float16",
        "test_cudnn_convolution_add_relu_xpu_float32",
        # accuracy issue, TODO
        "test_Conv2d_naive_groups_xpu_float16",
        # issue: https://github.com/intel/torch-xpu-ops/issues/809
        "test_thnn_conv_strided_padded_dilated",
    ),
    "test_dynamic_shapes_xpu.py": None,
    "nn/test_load_state_dict_xpu.py": None,
    "nn/test_module_hooks_xpu.py": (
        # TypeError: TestStateDictHooks.test_register_state_dict_post_hook() missing 1 required positional argument: 'private'
        # https://github.com/intel/torch-xpu-ops/issues/658
        "test_register_state_dict_post_hook",
    ),
    "nn/test_parametrization_xpu.py": None,
    "test_meta_xpu.py": (
        # https://github.com/intel/torch-xpu-ops/issues/774
        "_jiterator_",
        # RuntimeError: Short is not supported in oneDNN! Need oneDNN's support, suggest to keep skip.
        "test_dispatch_meta_outplace_nn_functional_linear_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_nn_functional_linear_xpu_int16",
        "test_meta_outplace_nn_functional_linear_xpu_int16",
        # RuntimeError: Long is not supported in oneDNN! Need oneDNN's support, suggest to keep skip.
        "test_dispatch_meta_outplace_nn_functional_linear_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_linear_xpu_int64",
        "test_meta_outplace_nn_functional_linear_xpu_int64",
        # RuntimeError: Short is not supported in oneDNN!
        "test_dispatch_meta_inplace_addbmm_xpu_int16",
        "test_dispatch_meta_inplace_addmm_decomposed_xpu_int16",
        "test_dispatch_meta_inplace_addmm_xpu_int16",
        "test_dispatch_meta_inplace_addmv_xpu_int16",
        "test_dispatch_meta_inplace_baddbmm_xpu_int16",
        "test_dispatch_meta_outplace___rmatmul___xpu_int16",
        "test_dispatch_meta_outplace_addbmm_xpu_int16",
        "test_dispatch_meta_outplace_addmm_decomposed_xpu_int16",
        "test_dispatch_meta_outplace_addmm_xpu_int16",
        "test_dispatch_meta_outplace_addmv_xpu_int16",
        "test_dispatch_meta_outplace_baddbmm_xpu_int16",
        "test_dispatch_meta_outplace_bmm_xpu_int16",
        "test_dispatch_meta_outplace_einsum_xpu_int16",
        "test_dispatch_meta_outplace_linalg_multi_dot_xpu_int16",
        "test_dispatch_meta_outplace_matmul_xpu_int16",
        "test_dispatch_meta_outplace_mm_xpu_int16",
        "test_dispatch_meta_outplace_mv_xpu_int16",
        "test_dispatch_meta_outplace_nn_functional_bilinear_xpu_int16",
        "test_dispatch_meta_outplace_tensordot_xpu_int16",
        "test_dispatch_symbolic_meta_inplace_addbmm_xpu_int16",
        "test_dispatch_symbolic_meta_inplace_addmm_decomposed_xpu_int16",
        "test_dispatch_symbolic_meta_inplace_addmm_xpu_int16",
        "test_dispatch_symbolic_meta_inplace_addmv_xpu_int16",
        "test_dispatch_symbolic_meta_inplace_baddbmm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace___rmatmul___xpu_int16",
        "test_dispatch_symbolic_meta_outplace_addbmm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_addmm_decomposed_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_addmm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_addmv_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_baddbmm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_bmm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_einsum_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_linalg_multi_dot_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_matmul_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_mm_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_mv_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_nn_functional_bilinear_xpu_int16",
        "test_dispatch_symbolic_meta_outplace_tensordot_xpu_int16",
        "test_meta_inplace_addbmm_xpu_int16",
        "test_meta_inplace_addmm_decomposed_xpu_int16",
        "test_meta_inplace_addmm_xpu_int16",
        "test_meta_inplace_addmv_xpu_int16",
        "test_meta_inplace_baddbmm_xpu_int16",
        "test_meta_outplace___rmatmul___xpu_int16",
        "test_meta_outplace_addbmm_xpu_int16",
        "test_meta_outplace_addmm_decomposed_xpu_int16",
        "test_meta_outplace_addmm_xpu_int16",
        "test_meta_outplace_addmv_xpu_int16",
        "test_meta_outplace_baddbmm_xpu_int16",
        "test_meta_outplace_bmm_xpu_int16",
        "test_meta_outplace_einsum_xpu_int16",
        "test_meta_outplace_linalg_multi_dot_xpu_int16",
        "test_meta_outplace_matmul_xpu_int16",
        "test_meta_outplace_mm_xpu_int16",
        "test_meta_outplace_mv_xpu_int16",
        "test_meta_outplace_nn_functional_bilinear_xpu_int16",
        "test_meta_outplace_tensordot_xpu_int16",
        # RuntimeError: could not create a primitive descriptor for a matmul primitive
        "test_dispatch_meta_inplace_addbmm_xpu_int32",
        "test_dispatch_meta_inplace_addmm_decomposed_xpu_int32",
        "test_dispatch_meta_inplace_addmm_xpu_int32",
        "test_dispatch_meta_inplace_addmv_xpu_int32",
        "test_dispatch_meta_inplace_baddbmm_xpu_int32",
        "test_dispatch_meta_outplace___rmatmul___xpu_int32",
        "test_dispatch_meta_outplace_addbmm_xpu_int32",
        "test_dispatch_meta_outplace_addmm_decomposed_xpu_int32",
        "test_dispatch_meta_outplace_addmm_xpu_int32",
        "test_dispatch_meta_outplace_addmv_xpu_int32",
        "test_dispatch_meta_outplace_baddbmm_xpu_int32",
        "test_dispatch_meta_outplace_bmm_xpu_int32",
        "test_dispatch_meta_outplace_einsum_xpu_int32",
        "test_dispatch_meta_outplace_linalg_multi_dot_xpu_int32",
        "test_dispatch_meta_outplace_matmul_xpu_int32",
        "test_dispatch_meta_outplace_mm_xpu_int32",
        "test_dispatch_meta_outplace_mv_xpu_int32",
        "test_dispatch_meta_outplace_nn_functional_bilinear_xpu_int32",
        "test_dispatch_meta_outplace_nn_functional_linear_xpu_int32",
        "test_dispatch_meta_outplace_tensordot_xpu_int32",
        "test_dispatch_symbolic_meta_inplace_addbmm_xpu_int32",
        "test_dispatch_symbolic_meta_inplace_addmm_decomposed_xpu_int32",
        "test_dispatch_symbolic_meta_inplace_addmm_xpu_int32",
        "test_dispatch_symbolic_meta_inplace_addmv_xpu_int32",
        "test_dispatch_symbolic_meta_inplace_baddbmm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace___rmatmul___xpu_int32",
        "test_dispatch_symbolic_meta_outplace_addbmm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_addmm_decomposed_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_addmm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_addmv_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_baddbmm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_bmm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_einsum_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_linalg_multi_dot_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_matmul_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_mm_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_mv_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_nn_functional_bilinear_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_nn_functional_linear_xpu_int32",
        "test_dispatch_symbolic_meta_outplace_tensordot_xpu_int32",
        "test_meta_inplace_addbmm_xpu_int32",
        "test_meta_inplace_addmm_decomposed_xpu_int32",
        "test_meta_inplace_addmm_xpu_int32",
        "test_meta_inplace_addmv_xpu_int32",
        "test_meta_inplace_baddbmm_xpu_int32",
        "test_meta_outplace___rmatmul___xpu_int32",
        "test_meta_outplace_addbmm_xpu_int32",
        "test_meta_outplace_addmm_decomposed_xpu_int32",
        "test_meta_outplace_addmm_xpu_int32",
        "test_meta_outplace_addmv_xpu_int32",
        "test_meta_outplace_baddbmm_xpu_int32",
        "test_meta_outplace_bmm_xpu_int32",
        "test_meta_outplace_einsum_xpu_int32",
        "test_meta_outplace_linalg_multi_dot_xpu_int32",
        "test_meta_outplace_matmul_xpu_int32",
        "test_meta_outplace_mm_xpu_int32",
        "test_meta_outplace_mv_xpu_int32",
        "test_meta_outplace_nn_functional_bilinear_xpu_int32",
        "test_meta_outplace_nn_functional_linear_xpu_int32",
        "test_meta_outplace_tensordot_xpu_int32",
        # RuntimeError: Long is not supported in oneDNN!
        "test_dispatch_meta_inplace_addbmm_xpu_int64",
        "test_dispatch_meta_inplace_addmm_decomposed_xpu_int64",
        "test_dispatch_meta_inplace_addmm_xpu_int64",
        "test_dispatch_meta_inplace_addmv_xpu_int64",
        "test_dispatch_meta_inplace_baddbmm_xpu_int64",
        "test_dispatch_meta_outplace___rmatmul___xpu_int64",
        "test_dispatch_meta_outplace_addbmm_xpu_int64",
        "test_dispatch_meta_outplace_addmm_decomposed_xpu_int64",
        "test_dispatch_meta_outplace_addmm_xpu_int64",
        "test_dispatch_meta_outplace_addmv_xpu_int64",
        "test_dispatch_meta_outplace_baddbmm_xpu_int64",
        "test_dispatch_meta_outplace_bmm_xpu_int64",
        "test_dispatch_meta_outplace_einsum_xpu_int64",
        "test_dispatch_meta_outplace_linalg_multi_dot_xpu_int64",
        "test_dispatch_meta_outplace_matmul_xpu_int64",
        "test_dispatch_meta_outplace_mm_xpu_int64",
        "test_dispatch_meta_outplace_mv_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_bilinear_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv1d_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv2d_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv3d_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose1d_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose2d_xpu_int64",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose3d_xpu_int64",
        "test_dispatch_meta_outplace_tensordot_xpu_int64",
        "test_dispatch_symbolic_meta_inplace_addbmm_xpu_int64",
        "test_dispatch_symbolic_meta_inplace_addmm_decomposed_xpu_int64",
        "test_dispatch_symbolic_meta_inplace_addmm_xpu_int64",
        "test_dispatch_symbolic_meta_inplace_addmv_xpu_int64",
        "test_dispatch_symbolic_meta_inplace_baddbmm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace___rmatmul___xpu_int64",
        "test_dispatch_symbolic_meta_outplace_addbmm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_addmm_decomposed_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_addmm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_addmv_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_baddbmm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_bmm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_einsum_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_linalg_multi_dot_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_matmul_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_mm_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_mv_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_bilinear_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv1d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv2d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv3d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose1d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose2d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose3d_xpu_int64",
        "test_dispatch_symbolic_meta_outplace_tensordot_xpu_int64",
        "test_meta_inplace_addbmm_xpu_int64",
        "test_meta_inplace_addmm_decomposed_xpu_int64",
        "test_meta_inplace_addmm_xpu_int64",
        "test_meta_inplace_addmv_xpu_int64",
        "test_meta_inplace_baddbmm_xpu_int64",
        "test_meta_outplace___rmatmul___xpu_int64",
        "test_meta_outplace_addbmm_xpu_int64",
        "test_meta_outplace_addmm_decomposed_xpu_int64",
        "test_meta_outplace_addmm_xpu_int64",
        "test_meta_outplace_addmv_xpu_int64",
        "test_meta_outplace_baddbmm_xpu_int64",
        "test_meta_outplace_bmm_xpu_int64",
        "test_meta_outplace_einsum_xpu_int64",
        "test_meta_outplace_linalg_multi_dot_xpu_int64",
        "test_meta_outplace_matmul_xpu_int64",
        "test_meta_outplace_mm_xpu_int64",
        "test_meta_outplace_mv_xpu_int64",
        "test_meta_outplace_nn_functional_bilinear_xpu_int64",
        "test_meta_outplace_nn_functional_conv1d_xpu_int64",
        "test_meta_outplace_nn_functional_conv2d_xpu_int64",
        "test_meta_outplace_nn_functional_conv3d_xpu_int64",
        "test_meta_outplace_nn_functional_conv_transpose1d_xpu_int64",
        "test_meta_outplace_nn_functional_conv_transpose2d_xpu_int64",
        "test_meta_outplace_nn_functional_conv_transpose3d_xpu_int64",
        "test_meta_outplace_tensordot_xpu_int64",
        # RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive
        "test_dispatch_meta_outplace_nn_functional_conv_transpose2d_xpu_bfloat16",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose2d_xpu_complex",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose2d_xpu_float",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose3d_xpu_bfloat16",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose3d_xpu_complex",
        "test_dispatch_meta_outplace_nn_functional_conv_transpose3d_xpu_float",
        "test_dispatch_symbolic_meta_outplace_all_strides_nn_functional_conv_transpose2d_xpu_float32",
        "test_dispatch_symbolic_meta_outplace_all_strides_nn_functional_conv_transpose3d_xpu_float32",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose2d_xpu_bfloat16",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose2d_xpu_complex",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose2d_xpu_float",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose3d_xpu_bfloat16",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose3d_xpu_complex",
        "test_dispatch_symbolic_meta_outplace_nn_functional_conv_transpose3d_xpu_float",
        "test_meta_outplace_nn_functional_conv_transpose2d_xpu_bfloat16",
        "test_meta_outplace_nn_functional_conv_transpose2d_xpu_complex",
        "test_meta_outplace_nn_functional_conv_transpose2d_xpu_float",
        "test_meta_outplace_nn_functional_conv_transpose3d_xpu_bfloat16",
        "test_meta_outplace_nn_functional_conv_transpose3d_xpu_complex",
        "test_meta_outplace_nn_functional_conv_transpose3d_xpu_float",
        # Not implemented, try these cases after implementing vdot
        "test_dispatch_meta_outplace_vdot_xpu_complex",
        "test_dispatch_symbolic_meta_outplace_vdot_xpu_complex",
        "test_meta_outplace_vdot_xpu_complex",
        # Unexpected success:
        "test_dispatch_symbolic_meta_outplace_all_strides_narrow_copy_xpu_float32",
    ),
    "test_type_promotion_xpu.py": None,
    "test_distributions_xpu.py": None,
    "test_optim_xpu.py": None,
    "test_spectral_ops_xpu.py": (
        # CUDA specific case
        "test_cufft_plan_cache_xpu_float64",
    ),
    "test_sparse_xpu.py": (
        "test_bmm_deterministic_xpu_float64",  # - AssertionError: Torch not compiled with CUDA enabled
        "test_bmm_oob_xpu",  # - NotImplementedError: Could not run 'aten::bmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was ...
        "test_bmm_xpu_float64",  # - NotImplementedError: Could not run 'aten::bmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was ...
        "test_dsmm_xpu_float64",  # - NotImplementedError: Could not run 'aten::mm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was o...
        "test_empty_like_xpu_float64",  # - AssertionError: "Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend" does not match "Could not run 'aten::empty_strided' with argu...
        "test_factory_device_type_inference_xpu",  # - RuntimeError: PyTorch is not linked with support for cuda devices
        "test_hsmm_xpu_float64",  # - NotImplementedError: Could not run 'aten::hspmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or wa...
        "test_mv_xpu_float64",  # - NotImplementedError: Could not run 'aten::mm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was o...
        "test_new_device_single_gpu_xpu",  # - RuntimeError: PyTorch was compiled without CUDA support
        "test_print_coalesced_xpu_float64",  # - RuntimeError: I got this output for TestSparseXPU.test_print_coalesced_xpu_float64:
        "test_print_uncoalesced_xpu_float64",  # - RuntimeError: I got this output for TestSparseXPU.test_print_uncoalesced_xpu_float64
        "test_sparse_addmm_xpu_bfloat16",  # - NotImplementedError: Could not run 'aten::addmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or wa...
        "test_sparse_addmm_xpu_float16",  # - NotImplementedError: Could not run 'aten::addmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or wa...
        "test_sparse_addmm_xpu_float64",  # - NotImplementedError: Could not run 'aten::addmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or wa...
        "test_sparse_matmul_xpu_float32",  # - NotImplementedError: Could not run 'aten::_sparse_sparse_matmul' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for thi...
        "test_sparse_matmul_xpu_float64",  # - RuntimeError: Double and complex datatype matmul is not supported in oneDNN
        "test_sparse_mm_xpu_float64",  # - NotImplementedError: Could not run 'aten::addmm' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or wa...
    ),
    "functorch/test_ops_functorch_xpu.py": None,
}
