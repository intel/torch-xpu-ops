skip_dict = {
    "test_ops_xpu.py": (
        # Calculation error between XPU implementation and CPU implementation,
    # 1. Compiler optimization causes failing to promote data type to higher precision.
    # 2. Accumulate error is amplified by some operations in some extreme cases. (std::exp(extreme_large_num))
    # 3. Accumulate error is amplified by a large number of accumalate operations.
    # 4. Accumulate error is different on different implementations due to different accumulation order.
    #     a. Different kernel implementations.
    #     b. Different std functions. (std::log, std::tanh, std::exp)
    # 5. The result of division between two same float values is not 1.
    # 6. std functions get different results when input is nan or inf between GCC and SYCL.
    "test_compare_cpu_cumsum_xpu_bfloat16",
    "test_compare_cpu_cumsum_xpu_float16",
    "test_compare_cpu_logcumsumexp_xpu_bfloat16",
    "test_compare_cpu_logcumsumexp_xpu_complex128",
    "test_compare_cpu_log_xpu_complex64",
    "test_compare_cpu_log10_xpu_complex64",
    "test_compare_cpu_log1p_xpu_complex64",
    "test_compare_cpu_log2_xpu_complex64",
    "test_compare_cpu_log2_xpu_complex128",
    "test_compare_cpu_mul_xpu_complex64",
    "test_compare_cpu_pow_xpu_complex128",
    "test_compare_cpu_pow_xpu_complex64",
    "test_compare_cpu_tan_xpu_complex128",
    "test_compare_cpu_tan_xpu_complex64",
    "test_compare_cpu_tanh_xpu_complex128",
    "test_compare_cpu_tanh_xpu_complex64",
    "test_compare_cpu_rsqrt_xpu_bfloat16",
    "test_compare_cpu_pow_xpu_bfloat16",
    # cuda has the same issue on this case
    "test_compare_cpu__refs_rsub_xpu_bfloat16",
    "test_compare_cpu_add_xpu_bfloat16",
    "test_compare_cpu_sub_xpu_bfloat16",
    "test_compare_cpu_acos_xpu_complex128",
    "test_compare_cpu_acos_xpu_complex64",
    "test_compare_cpu_acosh_xpu_complex64",
    "test_compare_cpu_cross_xpu_float16",
    "test_compare_cpu_floor_divide_xpu_bfloat16",
    "test_compare_cpu_floor_divide_xpu_float16",
    "test_compare_cpu_polygamma_polygamma_n_0_xpu_bfloat16",
    "test_compare_cpu_polygamma_polygamma_n_0_xpu_float16",
    "test_compare_cpu_exp_xpu_bfloat16",
    "test_compare_cpu_exp_xpu_complex128",
    "test_compare_cpu_exp_xpu_complex64",
    "test_compare_cpu_acosh_xpu_complex64",
    "test_compare_cpu_asin_xpu_complex128",
    "test_compare_cpu_asin_xpu_complex64",
    "test_compare_cpu_asinh_xpu_complex128",
    "test_compare_cpu_asinh_xpu_complex64",
    "test_compare_cpu_atan_xpu_complex128",
    "test_compare_cpu_atan_xpu_complex64",
    "test_compare_cpu_exp2_xpu_complex128",
    "test_compare_cpu_exp2_xpu_complex64",
    "test_compare_cpu_nextafter_xpu_bfloat16",
    # CUDA does not support the data type either
    "test_non_standard_bool_values_native_dropout_backward_xpu_bool",
    # Need FP64 golden ref for more accurate comparison
    "test_compare_cpu_log_softmax_xpu_bfloat16",
    "test_compare_cpu__softmax_backward_data_xpu_bfloat16",
    "test_compare_cpu__softmax_backward_data_xpu_float16",
    # TestCompositeCompliance::test_cow_input
    # XPU Tensor fails in copy-on-write cases
    # AssertionError: False is not true : Keyword argument 'output grad 0' during backward call unexpectedly materializes. Either set `supports_cow_input_no_materialize_backward=False` in this operation's OpInfo, add the arg to the OpInfo's `allow_cow_input_materialize_backward` list, or change the implementation to avoid materialization.
    # https://github.com/intel/torch-xpu-ops/issues/281
    "test_cow_input_addr_xpu_float32",
    "test_cow_input_cdist_xpu_float32",
    "test_cow_input_nn_functional_multi_head_attention_forward_xpu_float32",
    # XPU implementation is correct.
    # std::exp{-inf, nan}, the result is (±0,±0) (signs are unspecified)
    # std::exp{-inf, inf}, the result is (±0,±0) (signs are unspecified)
    # CPU implementation gets NaN in the cases.
    # https://en.cppreference.com/w/cpp/numeric/complex/exp
    "test_compare_cpu_sigmoid_xpu_complex64",
    "test_compare_cpu_sigmoid_xpu_complex128",
    # Special handle (different calculation order) in CPU reference impl.
    # https://github.com/pytorch/pytorch/blob/c97e3ebb96d7457075b019b94411e8c2d058e68b/aten/src/ATen/native/EmbeddingBag.cpp#L300
    "test_compare_cpu_nn_functional_embedding_bag_xpu_bfloat16",
    # Double and complex datatype matmul is not supported in oneDNN
    "test_compare_cpu_cdist_xpu_float64",
    "test_compare_cpu_nn_functional_grid_sample_xpu_float64",
    # CPU change: https://github.com/pytorch/pytorch/pull/134812
    # Issue link: https://github.com/intel/torch-xpu-ops/issues/1061
    "test_compare_cpu_grid_sampler_2d_xpu_bfloat16",
    "test_compare_cpu_grid_sampler_2d_xpu_float16",
    "test_compare_cpu_nn_functional_grid_sample_xpu_bfloat16",
    "test_compare_cpu_nn_functional_grid_sample_xpu_float16",
    # bilinear interpolate includes large calculation steps, accuracy reduces in half-precision
    # Not in CUDA test scope too
    "test_compare_cpu_nn_functional_upsample_bilinear_xpu_bfloat16",
    "test_compare_cpu_nn_functional_upsample_bilinear_xpu_float16",
    # Same as cuda, AssertionError: Tensor-likes are not close!Should skip.
    # Not in CUDA test scope too
    # https://github.com/intel/torch-xpu-ops/issues/845
    "test_compare_cpu_nn_functional_adaptive_avg_pool3d_xpu_bfloat16",
    "test_compare_cpu_nn_functional_adaptive_avg_pool3d_xpu_float16",
    # Same as cuda, AssertionError: Tensor-likes are not close!Should skip.
    # Not in CUDA test scope too
    "test_compare_cpu_nn_functional_interpolate_trilinear_xpu_bfloat16",
    "test_compare_cpu_nn_functional_interpolate_trilinear_xpu_float16",
    # CPU result is not golden reference
    "test_compare_cpu_nn_functional_group_norm_xpu_bfloat16",
    "test_compare_cpu_nn_functional_group_norm_xpu_float16",
    "test_compare_cpu_nn_functional_nll_loss_xpu_bfloat16",
    "test_compare_cpu_nn_functional_nll_loss_xpu_float16",
    "test_compare_cpu_nn_functional_batch_norm_xpu_bfloat16",
    "test_compare_cpu__batch_norm_with_update_xpu_bfloat16",
    "test_compare_cpu__batch_norm_with_update_xpu_float16",
    "test_compare_cpu_nn_functional_huber_loss_xpu_bfloat16",
    "test_compare_cpu_nansum_xpu_bfloat16",
    "test_compare_cpu_nanmean_xpu_bfloat16",
    # Align with CUDA impl by using accumulate type. But CPU doesn't use.
    # When XPU uses original data type, the case passes.
    "test_compare_cpu_logit_xpu_bfloat16",
    # precison error
    #     Mismatched elements: 1 / 24 (4.2%)
    # Greatest absolute difference: 0.03125 at index (0, 1, 0, 1) (up to 0.001 allowed)
    # Greatest relative difference: 0.0048828125 at index (0, 1, 0, 1) (up to 0.001 allowed)
    "test_compare_cpu_nn_functional_interpolate_bilinear_xpu_bfloat16",
    # RuntimeError: "compute_index_ranges_weights" not implemented for 'Half'
    "test_compare_cpu_nn_functional_interpolate_bilinear_xpu_float16",
    #The results of XPU and CUDA are consistent, but the results of CPU and CUDA are inconsistent
    "test_compare_cpu_nn_functional_interpolate_linear_xpu_bfloat16",
    "test_compare_cpu_nn_functional_interpolate_linear_xpu_float16",
    # bicubic interpolate includes large calculation steps, accuracy reduces in half-precision
    # Not in CUDA test scope too
    "test_compare_cpu_nn_functional_interpolate_bicubic_xpu_bfloat16",
    "test_compare_cpu_nn_functional_interpolate_bicubic_xpu_float16",
    #AssertionError: Tensor-likes are not close! CUDA fails too.
    "test_compare_cpu_nn_functional_interpolate_area_xpu_bfloat16",
    "test_compare_cpu_nn_functional_interpolate_area_xpu_float16",
    # Not all operators are implemented for XPU tested in the case.
    # Retrieve it once the operator is implemented.

    # Precision error.
    # Mismatched elements: 1 / 812 (0.1%)
    # Greatest absolute difference: 0.03125 at index (610,) (up to 0.001 allowed)
    # Greatest relative difference: 0.00396728515625 at index (610,) (up to 0.001 allowed)
    "test_compare_cpu_hypot_xpu_bfloat16",
    # RuntimeError: Expected both inputs to be Half, Float or Double tensors but got BFloat16 and BFloat16.
    # Polar's backward is calculated using complex(), which does not support bfloat16. CUDA fails with same error.
    "test_compare_cpu_polar_xpu_bfloat16",
    # Precision error.
    # Mismatched elements: 1 / 25 (4.0%)
    # Greatest absolute difference: 0.00146484375 at index (0, 0) (up to 0.001 allowed)
    # Greatest relative difference: 0.0163116455078125 at index (0, 0) (up to 0.001 allowed)
    "test_compare_cpu_sub_xpu_float16",
    # different results for value index due to unstable sort.
    # XPU and CUDA have the same result.
    "test_compare_cpu_kthvalue_xpu_bfloat16",
    "test_compare_cpu_kthvalue_xpu_int16",
    "test_compare_cpu_kthvalue_xpu_int32",
    "test_compare_cpu_kthvalue_xpu_int64",
    "test_compare_cpu_kthvalue_xpu_int8",
    "test_compare_cpu_kthvalue_xpu_uint8",
    "test_compare_cpu_median_xpu_int16",
    "test_compare_cpu_median_xpu_int32",
    "test_compare_cpu_median_xpu_int64",
    "test_compare_cpu_median_xpu_int8",
    "test_compare_cpu_median_xpu_uint8",
    "test_compare_cpu_nanmedian_xpu_int16",
    "test_compare_cpu_nanmedian_xpu_int32",
    "test_compare_cpu_nanmedian_xpu_int64",
    "test_compare_cpu_nanmedian_xpu_int8",
    "test_compare_cpu_nanmedian_xpu_uint8",

    # sort algorithm is different to cpu
    "_compare_cpu_argsort_xpu_",
    "test_non_standard_bool_values_argsort_xpu_bool", # stock pytorch commit: e7cf7d0

    # AssertionError: The values for attribute 'dtype' do not match: torch.float32 != torch.bfloat16
    # https://github.com/intel/torch-xpu-ops/issues/780
    "test_compare_cpu_native_layer_norm_xpu_bfloat16",
    "test_compare_cpu_native_layer_norm_xpu_float16",

    # AssertionError: Tensor-likes are not close!
    # https://github.com/intel/torch-xpu-ops/issues/781
    "test_compare_cpu_square_xpu_complex64",


    # The operator 'aten::_assert_async.msg' is not currently implemented for the XPU device.
    "test_operator_multinomial_xpu_float32",
    "test_view_replay_multinomial_xpu_float32"

    # returned index is dependent on input data and implementation detail, and no
    # specification is given to uniquely identify the correct index 
    # (e.g. index with maximal / minimal value)
    "test_compare_cpu_mode",

    # nextafter: Numeric error due to `std::nextafter` difference between CPU (GCC) and XPU (SYCL)
    # https://github.com/intel/torch-xpu-ops/issues/623
    "test_compare_cpu_nextafter_xpu_float16",

    # AssertionError: Tensor-likes are not close!
    # Greatest absolute difference: 0.0625 at index (1,) (up to 0.001 allowed)
    #  Greatest relative difference: 0.00640869140625 at index (1,) (up to 0.001 allowed)
    "test_compare_cpu_xlogy_xpu_bfloat16",
    ),
}
