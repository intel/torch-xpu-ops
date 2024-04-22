import os
import sys

skip_list = (
    # Skip list of base line
    "test_compare_cpu_nn_functional_conv1d_xpu_float32",
    "test_compare_cpu_nn_functional_conv2d_xpu_float32",
    "test_compare_cpu_sparse_sampled_addmm_xpu_float32",
    "test_compare_cpu_to_sparse_xpu_float32",
    "test_dtypes___rdiv___xpu",
    "test_dtypes___rmod___xpu",
    "test_dtypes_abs_xpu",
    "test_dtypes_jiterator_2inputs_2outputs_xpu",
    "test_dtypes_jiterator_4inputs_with_extra_args_xpu",
    "test_dtypes_jiterator_binary_return_by_ref_xpu",
    "test_dtypes_jiterator_binary_xpu",
    "test_dtypes_jiterator_unary_xpu",
    "test_dtypes_nn_functional_batch_norm_without_cudnn_xpu",
    "test_dtypes_nn_functional_conv1d_xpu",
    "test_dtypes_nn_functional_conv2d_xpu",
    "test_dtypes_nn_functional_conv3d_xpu",
    "test_dtypes_nn_functional_conv_transpose1d_xpu",
    "test_dtypes_nn_functional_conv_transpose2d_xpu",
    "test_dtypes_nn_functional_conv_transpose3d_xpu",
    "test_dtypes_nn_functional_max_pool1d_xpu",
    "test_dtypes_nn_functional_softsign_xpu",
    "test_dtypes_reciprocal_xpu",
    "test_dtypes_sgn_xpu",
    "test_dtypes_sparse_sampled_addmm_xpu",
    "test_dtypes_square_xpu",
    "test_errors_cat_xpu",
    "test_errors_dot_xpu",
    "test_errors_gather_xpu",
    "test_errors_index_select_xpu",
    "test_errors_kthvalue_xpu",
    "test_errors_masked_select_xpu",
    "test_errors_sparse_mul_layout0_xpu",
    "test_errors_sparse_mul_layout1_xpu",
    "test_errors_sparse_mul_layout2_xpu",
    "test_errors_sparse_mul_layout3_xpu",
    "test_errors_sparse_mul_layout4_xpu",
    "test_errors_take_xpu",
    "test_errors_vdot_xpu",
    "test_non_standard_bool_values___rdiv___xpu_bool",
    "test_non_standard_bool_values_jiterator_2inputs_2outputs_xpu_bool",
    "test_non_standard_bool_values_jiterator_4inputs_with_extra_args_xpu_bool",
    "test_non_standard_bool_values_jiterator_binary_return_by_ref_xpu_bool",
    "test_non_standard_bool_values_jiterator_binary_xpu_bool",
    "test_non_standard_bool_values_jiterator_unary_xpu_bool",
    "test_non_standard_bool_values_reciprocal_xpu_bool",
    "test_non_standard_bool_values_square_xpu_bool",
    "test_non_standard_bool_values_to_sparse_xpu_bool",
    "test_noncontiguous_samples___rdiv___xpu_int64",
    "test_noncontiguous_samples_jiterator_2inputs_2outputs_xpu_complex64",
    "test_noncontiguous_samples_jiterator_2inputs_2outputs_xpu_float32",
    "test_noncontiguous_samples_jiterator_2inputs_2outputs_xpu_int64",
    "test_noncontiguous_samples_jiterator_4inputs_with_extra_args_xpu_complex64",
    "test_noncontiguous_samples_jiterator_4inputs_with_extra_args_xpu_float32",
    "test_noncontiguous_samples_jiterator_4inputs_with_extra_args_xpu_int64",
    "test_noncontiguous_samples_jiterator_binary_return_by_ref_xpu_complex64",
    "test_noncontiguous_samples_jiterator_binary_return_by_ref_xpu_float32",
    "test_noncontiguous_samples_jiterator_binary_return_by_ref_xpu_int64",
    "test_noncontiguous_samples_jiterator_binary_xpu_complex64",
    "test_noncontiguous_samples_jiterator_binary_xpu_float32",
    "test_noncontiguous_samples_jiterator_binary_xpu_int64",
    "test_noncontiguous_samples_jiterator_unary_xpu_complex64",
    "test_noncontiguous_samples_jiterator_unary_xpu_float32",
    "test_noncontiguous_samples_jiterator_unary_xpu_int64",
    "test_noncontiguous_samples_linalg_det_xpu_float32",
    "test_noncontiguous_samples_linalg_slogdet_xpu_float32",
    "test_noncontiguous_samples_linalg_solve_ex_xpu_float32",
    "test_noncontiguous_samples_linalg_solve_xpu_float32",
    "test_noncontiguous_samples_linalg_tensorsolve_xpu_float32",
    "test_noncontiguous_samples_logdet_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv1d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv1d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv1d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_conv2d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv2d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv2d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_conv3d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv3d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv3d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_conv_transpose1d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv_transpose1d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv_transpose1d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_conv_transpose2d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv_transpose2d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv_transpose2d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_conv_transpose3d_xpu_complex64",
    "test_noncontiguous_samples_nn_functional_conv_transpose3d_xpu_float32",
    "test_noncontiguous_samples_nn_functional_conv_transpose3d_xpu_int64",
    "test_noncontiguous_samples_nn_functional_group_norm_xpu_float32",
    "test_noncontiguous_samples_nn_functional_rrelu_xpu_float32",
    "test_noncontiguous_samples_reciprocal_xpu_int64",
    "test_numpy_ref_jiterator_2inputs_2outputs_xpu_complex128",
    "test_numpy_ref_jiterator_2inputs_2outputs_xpu_float64",
    "test_numpy_ref_jiterator_2inputs_2outputs_xpu_int64",
    "test_numpy_ref_jiterator_4inputs_with_extra_args_xpu_complex128",
    "test_numpy_ref_jiterator_4inputs_with_extra_args_xpu_float64",
    "test_numpy_ref_jiterator_4inputs_with_extra_args_xpu_int64",
    "test_numpy_ref_linalg_tensorinv_xpu_float64",
    "test_numpy_ref_linalg_tensorsolve_xpu_float64",
    "test_numpy_ref_nn_functional_conv_transpose1d_xpu_complex128",
    "test_numpy_ref_nn_functional_conv_transpose1d_xpu_float64",
    "test_numpy_ref_nn_functional_group_norm_xpu_float64",
    "test_numpy_ref_nn_functional_pdist_xpu_float64",
    "test_out_addr_xpu_float32",
    "test_out_jiterator_2inputs_2outputs_xpu_float32",
    "test_out_jiterator_4inputs_with_extra_args_xpu_float32",
    "test_out_jiterator_binary_return_by_ref_xpu_float32",
    "test_out_jiterator_binary_xpu_float32",
    "test_out_jiterator_unary_xpu_float32",
    "test_out_mode_xpu_float32",
    "test_out_nanmean_xpu_float32",
    "test_out_nn_functional_conv1d_xpu_float32",
    "test_out_nn_functional_conv2d_xpu_float32",
    "test_out_nn_functional_conv3d_xpu_float32",
    "test_out_nn_functional_conv_transpose1d_xpu_float32",
    "test_out_nn_functional_conv_transpose2d_xpu_float32",
    "test_out_nn_functional_conv_transpose3d_xpu_float32",
    "test_out_requires_grad_error_sparse_sampled_addmm_xpu_complex64",
    "test_out_requires_grad_error_sparse_sampled_addmm_xpu_float32",
    "test_out_to_sparse_xpu_float32",
    "test_out_warning__native_batch_norm_legit_xpu",
    "test_out_warning_jiterator_2inputs_2outputs_xpu",
    "test_out_warning_jiterator_4inputs_with_extra_args_xpu",
    "test_out_warning_jiterator_binary_return_by_ref_xpu",
    "test_out_warning_jiterator_binary_xpu",
    "test_out_warning_jiterator_unary_xpu",
    "test_out_warning_nanmean_xpu",
    "test_out_warning_native_batch_norm_xpu",
    "test_out_warning_nn_functional_conv1d_xpu",
    "test_out_warning_nn_functional_conv2d_xpu",
    "test_out_warning_nn_functional_conv3d_xpu",
    "test_out_warning_nn_functional_conv_transpose1d_xpu",
    "test_out_warning_nn_functional_conv_transpose2d_xpu",
    "test_out_warning_nn_functional_conv_transpose3d_xpu",
    "test_out_warning_nn_functional_logsigmoid_xpu",
    "test_out_warning_to_sparse_xpu",
    "test_promotes_int_to_float___rdiv___xpu_bool",
    "test_promotes_int_to_float___rdiv___xpu_int16",
    "test_promotes_int_to_float___rdiv___xpu_int32",
    "test_promotes_int_to_float___rdiv___xpu_int64",
    "test_promotes_int_to_float___rdiv___xpu_int8",
    "test_promotes_int_to_float___rdiv___xpu_uint8",
    "test_promotes_int_to_float_reciprocal_xpu_bool",
    "test_promotes_int_to_float_reciprocal_xpu_int16",
    "test_promotes_int_to_float_reciprocal_xpu_int32",
    "test_promotes_int_to_float_reciprocal_xpu_int64",
    "test_promotes_int_to_float_reciprocal_xpu_int8",
    "test_promotes_int_to_float_reciprocal_xpu_uint8",
    "test_python_ref__refs_div_trunc_rounding_xpu_bfloat16",
    "test_python_ref__refs_div_trunc_rounding_xpu_float16",
    "test_python_ref__refs_floor_divide_xpu_float16",
    "test_python_ref__refs_floor_divide_xpu_float32",
    "test_python_ref__refs_floor_divide_xpu_float64",
    "test_python_ref__refs_floor_divide_xpu_int16",
    "test_python_ref__refs_floor_divide_xpu_int32",
    "test_python_ref__refs_floor_divide_xpu_int64",
    "test_python_ref__refs_floor_divide_xpu_int8",
    "test_python_ref__refs_floor_divide_xpu_uint8",
    "test_python_ref__refs_linspace_tensor_overload_xpu_int16",
    "test_python_ref__refs_linspace_tensor_overload_xpu_int32",
    "test_python_ref__refs_linspace_tensor_overload_xpu_int64",
    "test_python_ref__refs_linspace_tensor_overload_xpu_int8",
    "test_python_ref__refs_linspace_tensor_overload_xpu_uint8",
    "test_python_ref__refs_linspace_xpu_int16",
    "test_python_ref__refs_linspace_xpu_int32",
    "test_python_ref__refs_linspace_xpu_int64",
    "test_python_ref__refs_linspace_xpu_int8",
    "test_python_ref__refs_linspace_xpu_uint8",
    "test_python_ref__refs_logaddexp_xpu_complex128",
    "test_python_ref__refs_logaddexp_xpu_complex64",
    "test_python_ref__refs_native_layer_norm_xpu_bfloat16",
    "test_python_ref__refs_native_layer_norm_xpu_float16",
    "test_python_ref__refs_native_layer_norm_xpu_float32",
    "test_python_ref__refs_nn_functional_group_norm_xpu_bfloat16",
    "test_python_ref__refs_nn_functional_group_norm_xpu_float16",
    "test_python_ref__refs_nn_functional_group_norm_xpu_float32",
    "test_python_ref__refs_nn_functional_group_norm_xpu_float64",
    "test_python_ref__refs_nn_functional_hinge_embedding_loss_xpu_bfloat16",
    "test_python_ref__refs_nn_functional_hinge_embedding_loss_xpu_float16", "test_python_ref__refs_nn_functional_margin_ranking_loss_xpu_bfloat16",
    "test_python_ref__refs_nn_functional_margin_ranking_loss_xpu_float16",
    "test_python_ref__refs_nn_functional_triplet_margin_loss_xpu_uint8",
    "test_python_ref__refs_reciprocal_xpu_bool",
    "test_python_ref__refs_reciprocal_xpu_int16",
    "test_python_ref__refs_reciprocal_xpu_int32",
    "test_python_ref__refs_reciprocal_xpu_int64",
    "test_python_ref__refs_reciprocal_xpu_int8",
    "test_python_ref__refs_reciprocal_xpu_uint8",
    "test_python_ref__refs_square_xpu_bool",
    "test_python_ref__refs_trunc_xpu_float64",
    "test_python_ref_executor__refs_div_trunc_rounding_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_div_trunc_rounding_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_int16",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_int32",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_int8",
    "test_python_ref_executor__refs_floor_divide_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_int16",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_int32",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_int8",
    "test_python_ref_executor__refs_geometric_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_linspace_executor_aten_xpu_int16",
    "test_python_ref_executor__refs_linspace_executor_aten_xpu_int32",
    "test_python_ref_executor__refs_linspace_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_linspace_executor_aten_xpu_int8",
    "test_python_ref_executor__refs_linspace_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_xpu_int16",
    "test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_xpu_int32",
    "test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_xpu_int8",
    "test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_log_normal_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_log_normal_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_log_normal_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_log_normal_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_logaddexp_executor_aten_xpu_complex128",
    "test_python_ref_executor__refs_logaddexp_executor_aten_xpu_complex64",
    "test_python_ref_executor__refs_native_layer_norm_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_native_layer_norm_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_native_layer_norm_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_nn_functional_alpha_dropout_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_nn_functional_alpha_dropout_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_nn_functional_alpha_dropout_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_nn_functional_alpha_dropout_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_nn_functional_group_norm_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_nn_functional_group_norm_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_nn_functional_group_norm_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_nn_functional_group_norm_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_nn_functional_hinge_embedding_loss_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_nn_functional_hinge_embedding_loss_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_nn_functional_margin_ranking_loss_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_nn_functional_margin_ranking_loss_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_nn_functional_nll_loss_executor_aten_xpu_bfloat16",
    "test_python_ref_executor__refs_nn_functional_nll_loss_executor_aten_xpu_float32",
    "test_python_ref_executor__refs_nn_functional_nll_loss_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_nn_functional_triplet_margin_loss_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_bool",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_int16",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_int32",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_int8",
    "test_python_ref_executor__refs_reciprocal_executor_aten_xpu_uint8",
    "test_python_ref_executor__refs_square_executor_aten_xpu_bool",
    "test_python_ref_executor__refs_vdot_executor_aten_xpu_complex128",
    "test_python_ref_executor__refs_vdot_executor_aten_xpu_complex64",
    "test_python_ref_meta__refs_nn_functional_group_norm_xpu_bfloat16",
    "test_python_ref_meta__refs_nn_functional_group_norm_xpu_float16",
    "test_python_ref_meta__refs_nn_functional_group_norm_xpu_float32",
    "test_python_ref_meta__refs_nn_functional_group_norm_xpu_float64",
    "test_python_ref_torch_fallback__refs_div_trunc_rounding_xpu_bfloat16",
    "test_python_ref_torch_fallback__refs_div_trunc_rounding_xpu_float16",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_float16",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_float32",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_float64",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_int16",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_int32",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_int64",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_int8",
    "test_python_ref_torch_fallback__refs_floor_divide_xpu_uint8",
    "test_python_ref_torch_fallback__refs_linspace_tensor_overload_xpu_int16",
    "test_python_ref_torch_fallback__refs_linspace_tensor_overload_xpu_int32",
    "test_python_ref_torch_fallback__refs_linspace_tensor_overload_xpu_int64",
    "test_python_ref_torch_fallback__refs_linspace_tensor_overload_xpu_int8",
    "test_python_ref_torch_fallback__refs_linspace_tensor_overload_xpu_uint8",
    "test_python_ref_torch_fallback__refs_linspace_xpu_int16",
    "test_python_ref_torch_fallback__refs_linspace_xpu_int32",
    "test_python_ref_torch_fallback__refs_linspace_xpu_int64",
    "test_python_ref_torch_fallback__refs_linspace_xpu_int8",
    "test_python_ref_torch_fallback__refs_linspace_xpu_uint8",
    "test_python_ref_torch_fallback__refs_logaddexp_xpu_complex128",
    "test_python_ref_torch_fallback__refs_logaddexp_xpu_complex64",
    "test_python_ref_torch_fallback__refs_native_layer_norm_xpu_bfloat16",
    "test_python_ref_torch_fallback__refs_native_layer_norm_xpu_float16",
    "test_python_ref_torch_fallback__refs_native_layer_norm_xpu_float32",
    "test_python_ref_torch_fallback__refs_nn_functional_group_norm_xpu_bfloat16",
    "test_python_ref_torch_fallback__refs_nn_functional_group_norm_xpu_float16",
    "test_python_ref_torch_fallback__refs_nn_functional_group_norm_xpu_float32",
    "test_python_ref_torch_fallback__refs_nn_functional_group_norm_xpu_float64",
    "test_python_ref_torch_fallback__refs_nn_functional_hinge_embedding_loss_xpu_bfloat16",
    "test_python_ref_torch_fallback__refs_nn_functional_hinge_embedding_loss_xpu_float16",
    "test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_xpu_bfloat16",
    "test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_xpu_float16",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_bool",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_int16",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_int32",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_int64",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_int8",
    "test_python_ref_torch_fallback__refs_reciprocal_xpu_uint8",
    "test_python_ref_torch_fallback__refs_sinh_xpu_complex128",
    "test_python_ref_torch_fallback__refs_special_multigammaln_mvlgamma_p_5_xpu_int32",
    "test_python_ref_torch_fallback__refs_square_xpu_bool",
    "test_python_ref_torch_fallback__refs_vdot_xpu_complex128",
    "test_python_ref_torch_fallback__refs_vdot_xpu_complex64",
    "test_variant_consistency_eager_chalf_xpu_complex64",
    "test_variant_consistency_eager_chalf_xpu_float32",
    "test_variant_consistency_eager_conj_physical_xpu_complex64",
    "test_variant_consistency_eager_jiterator_2inputs_2outputs_xpu_complex64",
    "test_variant_consistency_eager_jiterator_2inputs_2outputs_xpu_float32",
    "test_variant_consistency_eager_jiterator_4inputs_with_extra_args_xpu_complex64",
    "test_variant_consistency_eager_jiterator_4inputs_with_extra_args_xpu_float32",
    "test_variant_consistency_eager_jiterator_binary_return_by_ref_xpu_complex64",
    "test_variant_consistency_eager_jiterator_binary_return_by_ref_xpu_float32",
    "test_variant_consistency_eager_jiterator_binary_xpu_complex64",
    "test_variant_consistency_eager_jiterator_binary_xpu_float32",
    "test_variant_consistency_eager_jiterator_unary_xpu_complex64",
    "test_variant_consistency_eager_jiterator_unary_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv1d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv1d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv2d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv2d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv3d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv3d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv_transpose1d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv_transpose1d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv_transpose2d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv_transpose2d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_conv_transpose3d_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_conv_transpose3d_xpu_float32",
    "test_variant_consistency_eager_nn_functional_group_norm_xpu_float32",
    "test_variant_consistency_eager_nn_functional_rrelu_xpu_float32",
    "test_variant_consistency_eager_to_sparse_xpu_complex64",
    "test_variant_consistency_eager_to_sparse_xpu_float32",
    "test_compare_cpu__native_batch_norm_legit_xpu_float32",
    "test_compare_cpu__refs_special_zeta_xpu_float32",
    "test_compare_cpu_linalg_lu_factor_ex_xpu_float32",
    "test_compare_cpu_linalg_lu_factor_xpu_float32",
    "test_compare_cpu_linalg_lu_xpu_float32",
    "test_compare_cpu_native_batch_norm_xpu_float32",
    "test_compare_cpu_special_hermite_polynomial_h_xpu_float32",
    "test_compare_cpu_special_zeta_xpu_float32",
    "test_out_cholesky_inverse_xpu_float32",
    "test_out_geqrf_xpu_float32",
    "test_out_narrow_copy_xpu_float32",
    "test_out_ormqr_xpu_float32",
    "test_out_triangular_solve_xpu_float32",
    "test_python_ref__refs_heaviside_xpu_int64",
    "test_python_ref__refs_special_bessel_j0_xpu_int64",
    "test_python_ref_errors__refs_dstack_xpu",
    "test_python_ref_errors__refs_hstack_xpu",
    "test_python_ref_errors__refs_linalg_cross_xpu",
    "test_python_ref_errors__refs_masked_fill_xpu",
    "test_python_ref_errors__refs_vstack_xpu",
    "test_python_ref_executor__refs_mul_executor_aten_xpu_complex32",
    "test_python_ref__refs_special_multigammaln_mvlgamma_p_5_xpu_float64",
    "test_python_ref_executor__refs_minimum_executor_aten_xpu_int64",
    "test_python_ref_executor__refs_special_multigammaln_mvlgamma_p_3_executor_aten_xpu_float64",
    "test_numpy_ref_nn_functional_rms_norm_xpu_complex128",
    "test_python_ref__refs_square_xpu_complex128",
    "test_python_ref__refs_square_xpu_complex64",
    "test_python_ref_executor__refs_istft_executor_aten_xpu_complex128",
    "test_python_ref_executor__refs_square_executor_aten_xpu_complex128",
    "test_python_ref_torch_fallback__refs_square_xpu_complex128",
    "test_python_ref_torch_fallback__refs_square_xpu_complex64",

    # Skip list of new added when porting XPU operators.
    # See: https://github.com/intel/torch-xpu-ops/issues/128
    "test_noncontiguous_samples_native_dropout_backward_xpu_int64", # The implementation aligns with CUDA, RuntimeError: "masked_scale" not implemented for 'Long'.
    "test_non_standard_bool_values_native_dropout_backward_xpu_bool", # The implementation aligns with CUDA, RuntimeError: "masked_scale" not implemented for 'Bool'.
    "test_compare_cpu_nn_functional_alpha_dropout_xpu_float32", # CUDA xfail.
    "test_dtypes_native_dropout_backward_xpu", # Test architecture issue. Cannot get correct claimed supported data type for "masked_scale".
    "test_dtypes_nn_functional_linear_xpu", # https://github.com/intel/torch-xpu-ops/issues/157
)


skip_options = " -k 'not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += "'"

test_command = "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v _test_ops.py"
test_command += skip_options

res = os.system(test_command)
exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
