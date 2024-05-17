import os
import sys

def launch_test(test_case, skip_list=None, exe_list=None):
    if skip_list != None:
        skip_options = " -k 'not " + skip_list[0]
        for skip_case in skip_list[1:]:
            skip_option = " and not " + skip_case
            skip_options += skip_option
        skip_options += "'"
        test_command = "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v " + test_case
        test_command += skip_options
        return os.system(test_command)
    elif exe_list != None:
        exe_options = " -k '" + exe_list[0]
        for exe_case in exe_list[1:]:
            exe_option = " and " + exe_case
            exe_options += exe_option
        exe_options += "'"
        test_command = "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v " + test_case
        test_command += exe_options
        return os.system(test_command)
    else:
        test_command = "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v " + test_case
        return os.system(test_command)

res = 0

# test_ops
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
    "test_non_standard_bool_values_scatter_reduce_amax_xpu_bool", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_non_standard_bool_values_scatter_reduce_amin_xpu_bool", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_non_standard_bool_values_scatter_reduce_prod_xpu_bool", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_dtypes_scatter_reduce_amax_xpu", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_dtypes_scatter_reduce_amin_xpu", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_dtypes_scatter_reduce_prod_xpu", # Align with CUDA dtypes - "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_non_standard_bool_values_argsort_xpu_bool", # The implementation aligns with CUDA, RuntimeError: "argsort" not implemented for 'Bool'.
    "test_non_standard_bool_values_msort_xpu_bool", # The implementation aligns with CUDA, RuntimeError: "msort" not implemented for 'Bool'.
    "test_non_standard_bool_values_sort_xpu_bool", # The implementation aligns with CUDA, RuntimeError: "sort" not implemented for 'Bool'.

    # https://github.com/intel/torch-xpu-ops/issues/157
    # Segfault:
    "test_dtypes_nn_functional_linear_xpu", # https://github.com/intel/torch-xpu-ops/issues/157
    "test_dtypes_nn_functional_multi_head_attention_forward_xpu", # https://github.com/intel/torch-xpu-ops/issues/157
    "test_dtypes_pca_lowrank_xpu", # https://github.com/intel/torch-xpu-ops/issues/157
    "test_dtypes_svd_lowrank_xpu", # https://github.com/intel/torch-xpu-ops/issues/157
    "test_noncontiguous_samples_nn_functional_linear_xpu_int64", # https://github.com/intel/torch-xpu-ops/issues/157
    "test_dtypes__refs_nn_functional_pdist_xpu", # https://github.com/intel/torch-xpu-ops/issues/157

    # https://github.com/intel/torch-xpu-ops/issues/157
    # Failures:
    "test_compare_cpu_addmm_xpu_float32",
    "test_compare_cpu_addmv_xpu_float32",
    "test_dtypes__refs_linalg_svd_xpu",
    "test_dtypes_addbmm_xpu",
    "test_dtypes_addmm_decomposed_xpu",
    "test_dtypes_addmm_xpu",
    "test_dtypes_addmv_xpu",
    "test_dtypes_addr_xpu",
    "test_dtypes_baddbmm_xpu",
    "test_dtypes_bmm_xpu",
    "test_dtypes_cholesky_inverse_xpu",
    "test_dtypes_cholesky_solve_xpu",
    "test_dtypes_cholesky_xpu",
    "test_dtypes_corrcoef_xpu",
    "test_dtypes_cov_xpu",
    "test_dtypes_linalg_cholesky_ex_xpu",
    "test_dtypes_linalg_cholesky_xpu",
    "test_dtypes_linalg_cond_xpu",
    "test_dtypes_linalg_det_singular_xpu",
    "test_dtypes_linalg_det_xpu",
    "test_dtypes_linalg_eig_xpu",
    "test_dtypes_linalg_eigh_xpu",
    "test_dtypes_linalg_eigvals_xpu",
    "test_dtypes_linalg_eigvalsh_xpu",
    "test_dtypes_linalg_inv_ex_xpu",
    "test_dtypes_linalg_inv_xpu",
    "test_dtypes_linalg_ldl_factor_ex_xpu",
    "test_dtypes_linalg_ldl_factor_xpu",
    "test_dtypes_linalg_ldl_solve_xpu",
    "test_dtypes_linalg_lstsq_grad_oriented_xpu",
    "test_dtypes_linalg_lstsq_xpu",
    "test_dtypes_linalg_lu_factor_ex_xpu",
    "test_dtypes_linalg_lu_factor_xpu",
    "test_dtypes_linalg_lu_solve_xpu",
    "test_dtypes_linalg_lu_xpu",
    "test_dtypes_linalg_matrix_power_xpu",
    "test_dtypes_linalg_matrix_rank_hermitian_xpu",
    "test_dtypes_linalg_matrix_rank_xpu",
    "test_dtypes_linalg_pinv_hermitian_xpu",
    "test_dtypes_linalg_pinv_xpu",
    "test_dtypes_linalg_qr_xpu",
    "test_dtypes_linalg_slogdet_xpu",
    "test_dtypes_linalg_solve_ex_xpu",
    "test_dtypes_linalg_solve_xpu",
    "test_dtypes_linalg_svd_xpu",
    "test_dtypes_linalg_tensorinv_xpu",
    "test_dtypes_linalg_tensorsolve_xpu",
    "test_dtypes_logdet_xpu",
    "test_dtypes_lu_solve_xpu",
    "test_dtypes_lu_xpu",
    "test_dtypes_mv_xpu",
    "test_dtypes_nn_functional_scaled_dot_product_attention_xpu",
    "test_dtypes_norm_nuc_xpu",
    "test_dtypes_pinverse_xpu",
    "test_dtypes_qr_xpu",
    "test_dtypes_svd_xpu",
    "test_dtypes_tensordot_xpu",
    "test_dtypes_triangular_solve_xpu",
    "test_noncontiguous_samples___rmatmul___xpu_complex64",
    "test_noncontiguous_samples___rmatmul___xpu_int64",
    "test_noncontiguous_samples_addbmm_xpu_complex64",
    "test_noncontiguous_samples_addbmm_xpu_float32",
    "test_noncontiguous_samples_addbmm_xpu_int64",
    "test_noncontiguous_samples_addmm_decomposed_xpu_complex64",
    "test_noncontiguous_samples_addmm_decomposed_xpu_int64",
    "test_noncontiguous_samples_addmm_xpu_complex64",
    "test_noncontiguous_samples_addmm_xpu_float32",
    "test_noncontiguous_samples_addmm_xpu_int64",
    "test_noncontiguous_samples_addmv_xpu_complex64",
    "test_noncontiguous_samples_addmv_xpu_float32",
    "test_noncontiguous_samples_addmv_xpu_int64",
    "test_noncontiguous_samples_addr_xpu_complex64",
    "test_noncontiguous_samples_baddbmm_xpu_complex64",
    "test_noncontiguous_samples_baddbmm_xpu_int64",
    "test_noncontiguous_samples_bmm_xpu_complex64",
    "test_noncontiguous_samples_bmm_xpu_int64",
    "test_noncontiguous_samples_cholesky_inverse_xpu_complex64",
    "test_noncontiguous_samples_cholesky_solve_xpu_complex64",
    "test_noncontiguous_samples_cholesky_xpu_complex64",
    "test_noncontiguous_samples_corrcoef_xpu_complex64",
    "test_noncontiguous_samples_cov_xpu_complex64",
    "test_noncontiguous_samples_einsum_xpu_complex64",
    "test_noncontiguous_samples_einsum_xpu_int64",
    "test_noncontiguous_samples_geqrf_xpu_complex64",
    "test_noncontiguous_samples_inner_xpu_complex64",
    "test_noncontiguous_samples_inner_xpu_int64",
    "test_noncontiguous_samples_linalg_cholesky_ex_xpu_complex64",
    "test_noncontiguous_samples_linalg_cholesky_xpu_complex64",
    "test_noncontiguous_samples_linalg_cond_xpu_complex64",
    "test_noncontiguous_samples_linalg_det_xpu_complex64",
    "test_noncontiguous_samples_linalg_eig_xpu_complex64",
    "test_noncontiguous_samples_linalg_eig_xpu_float32",
    "test_noncontiguous_samples_linalg_eigh_xpu_complex64",
    "test_noncontiguous_samples_linalg_eigvals_xpu_complex64",
    "test_noncontiguous_samples_linalg_eigvalsh_xpu_complex64",
    "test_noncontiguous_samples_linalg_householder_product_xpu_complex64",
    "test_noncontiguous_samples_linalg_inv_ex_xpu_complex64",
    "test_noncontiguous_samples_linalg_inv_xpu_complex64",
    "test_noncontiguous_samples_linalg_ldl_factor_ex_xpu_complex64",
    "test_noncontiguous_samples_linalg_ldl_factor_xpu_complex64",
    "test_noncontiguous_samples_linalg_ldl_solve_xpu_complex64",
    "test_noncontiguous_samples_linalg_lstsq_grad_oriented_xpu_complex64",
    "test_noncontiguous_samples_linalg_lstsq_xpu_complex64",
    "test_noncontiguous_samples_linalg_lu_factor_ex_xpu_complex64",
    "test_noncontiguous_samples_linalg_lu_factor_xpu_complex64",
    "test_noncontiguous_samples_linalg_lu_solve_xpu_complex64",
    "test_noncontiguous_samples_linalg_lu_xpu_complex64",
    "test_noncontiguous_samples_linalg_matrix_norm_xpu_complex64",
    "test_noncontiguous_samples_linalg_matrix_power_xpu_complex64",
    "test_noncontiguous_samples_linalg_matrix_rank_hermitian_xpu_complex64",
    "test_noncontiguous_samples_linalg_matrix_rank_xpu_complex64",
    "test_noncontiguous_samples_linalg_norm_subgradients_at_zero_xpu_complex64",
    "test_noncontiguous_samples_linalg_norm_xpu_complex64",
    "test_noncontiguous_samples_linalg_pinv_hermitian_xpu_complex64",
    "test_noncontiguous_samples_linalg_pinv_singular_xpu_complex64",
    "test_noncontiguous_samples_linalg_pinv_xpu_complex64",
    "test_noncontiguous_samples_linalg_qr_xpu_complex64",
    "test_noncontiguous_samples_linalg_slogdet_xpu_complex64",
    "test_noncontiguous_samples_linalg_solve_ex_xpu_complex64",
    "test_noncontiguous_samples_linalg_solve_triangular_xpu_complex64",
    "test_noncontiguous_samples_linalg_solve_xpu_complex64",
    "test_noncontiguous_samples_linalg_svd_xpu_complex64",
    "test_noncontiguous_samples_linalg_svdvals_xpu_complex64",
    "test_noncontiguous_samples_linalg_tensorinv_xpu_complex64",
    "test_noncontiguous_samples_linalg_tensorsolve_xpu_complex64",
    "test_noncontiguous_samples_logdet_xpu_complex64",
    "test_noncontiguous_samples_lu_solve_xpu_complex64",
    "test_noncontiguous_samples_lu_xpu_complex64",
    "test_noncontiguous_samples_matmul_xpu_complex64",
    "test_noncontiguous_samples_matmul_xpu_int64",
    "test_noncontiguous_samples_mm_xpu_complex64",
    "test_noncontiguous_samples_mm_xpu_int64",
    "test_noncontiguous_samples_mv_xpu_complex64",
    "test_noncontiguous_samples_mv_xpu_int64",
    "test_noncontiguous_samples_nn_functional_bilinear_xpu_int64",
    "test_noncontiguous_samples_nn_functional_linear_xpu_complex64",
    "test_noncontiguous_samples_norm_nuc_xpu_complex64",
    "test_noncontiguous_samples_ormqr_xpu_complex64",
    "test_noncontiguous_samples_pinverse_xpu_complex64",
    "test_noncontiguous_samples_qr_xpu_complex64",
    "test_noncontiguous_samples_svd_xpu_complex64",
    "test_noncontiguous_samples_tensordot_xpu_complex64",
    "test_noncontiguous_samples_tensordot_xpu_int64",
    "test_noncontiguous_samples_triangular_solve_xpu_complex64",
    "test_numpy_ref_addbmm_xpu_complex128",
    "test_numpy_ref_addbmm_xpu_float64",
    "test_numpy_ref_addbmm_xpu_int64",
    "test_numpy_ref_linalg_tensorinv_xpu_complex128",
    "test_out_addbmm_xpu_float32",
    "test_out_addmm_xpu_float32",
    "test_out_addmv_xpu_float32",
    "test_out_baddbmm_xpu_float32",
    "test_out_mm_xpu_float32",
    "test_out_mv_xpu_float32",
    "test_out_requires_grad_error_addbmm_xpu_complex64",
    "test_out_requires_grad_error_addmm_decomposed_xpu_complex64",
    "test_out_requires_grad_error_addmm_xpu_complex64",
    "test_out_requires_grad_error_addmv_xpu_complex64",
    "test_out_requires_grad_error_baddbmm_xpu_complex64",
    "test_out_requires_grad_error_bmm_xpu_complex64",
    "test_out_requires_grad_error_cholesky_inverse_xpu_complex64",
    "test_out_requires_grad_error_cholesky_solve_xpu_complex64",
    "test_out_requires_grad_error_cholesky_xpu_complex64",
    "test_out_requires_grad_error_inner_xpu_complex64",
    "test_out_requires_grad_error_linalg_cholesky_ex_xpu_complex64",
    "test_out_requires_grad_error_linalg_cholesky_xpu_complex64",
    "test_out_requires_grad_error_linalg_det_singular_xpu_complex64",
    "test_out_requires_grad_error_linalg_eig_xpu_complex64",
    "test_out_requires_grad_error_linalg_eigh_xpu_complex64",
    "test_out_requires_grad_error_linalg_eigvals_xpu_complex64",
    "test_out_requires_grad_error_linalg_eigvalsh_xpu_complex64",
    "test_out_requires_grad_error_linalg_inv_ex_xpu_complex64",
    "test_out_requires_grad_error_linalg_inv_xpu_complex64",
    "test_out_requires_grad_error_linalg_lstsq_xpu_complex64",
    "test_out_requires_grad_error_linalg_lu_factor_xpu_complex64",
    "test_out_requires_grad_error_linalg_lu_solve_xpu_complex64",
    "test_out_requires_grad_error_linalg_multi_dot_xpu_complex64",
    "test_out_requires_grad_error_linalg_pinv_hermitian_xpu_complex64",
    "test_out_requires_grad_error_linalg_pinv_xpu_complex64",
    "test_out_requires_grad_error_linalg_qr_xpu_complex64",
    "test_out_requires_grad_error_linalg_solve_ex_xpu_complex64",
    "test_out_requires_grad_error_linalg_solve_xpu_complex64",
    "test_out_requires_grad_error_linalg_tensorinv_xpu_complex64",
    "test_out_requires_grad_error_lu_solve_xpu_complex64",
    "test_out_requires_grad_error_lu_xpu_complex64",
    "test_out_requires_grad_error_mm_xpu_complex64",
    "test_out_requires_grad_error_mv_xpu_complex64",
    "test_out_requires_grad_error_nn_functional_linear_xpu_complex64",
    "test_out_requires_grad_error_qr_xpu_complex64",
    "test_out_requires_grad_error_tensordot_xpu_complex64",
    "test_out_requires_grad_error_triangular_solve_xpu_complex64",
    "test_out_warning_addmm_decomposed_xpu",
    "test_out_warning_addmm_xpu",
    "test_out_warning_addmv_xpu",
    "test_out_warning_baddbmm_xpu",
    "test_out_warning_bmm_xpu",
    "test_out_warning_matmul_xpu",
    "test_out_warning_mm_xpu",
    "test_out_warning_mv_xpu",
    "test_out_warning_nn_functional_linear_xpu",
    "test_python_ref__refs_linalg_svd_xpu_complex128",
    "test_python_ref__refs_linalg_svd_xpu_complex64",
    "test_python_ref__refs_linalg_svd_xpu_float64",
    "test_python_ref_executor__refs_linalg_svd_executor_aten_xpu_complex128",
    "test_python_ref_executor__refs_linalg_svd_executor_aten_xpu_complex64",
    "test_python_ref_executor__refs_linalg_svd_executor_aten_xpu_float64",
    "test_python_ref_executor__refs_nn_functional_nll_loss_executor_aten_xpu_float16",
    "test_python_ref_executor__refs_nn_functional_pdist_executor_aten_xpu_float64",
    "test_python_ref_meta__refs_linalg_svd_xpu_complex128",
    "test_python_ref_meta__refs_linalg_svd_xpu_complex64",
    "test_python_ref_meta__refs_linalg_svd_xpu_float64",
    "test_python_ref_meta__refs_nn_functional_pdist_xpu_float64",
    "test_python_ref_torch_fallback__refs_linalg_svd_xpu_complex128",
    "test_python_ref_torch_fallback__refs_linalg_svd_xpu_complex64",
    "test_python_ref_torch_fallback__refs_linalg_svd_xpu_float64",
    "test_python_ref_torch_fallback__refs_nn_functional_pdist_xpu_float64",
    "test_variant_consistency_eager___rmatmul___xpu_complex64",
    "test_variant_consistency_eager_addmm_decomposed_xpu_complex64",
    "test_variant_consistency_eager_addmm_xpu_complex64",
    "test_variant_consistency_eager_addmm_xpu_float32",
    "test_variant_consistency_eager_addmv_xpu_complex64",
    "test_variant_consistency_eager_addmv_xpu_float32",
    "test_variant_consistency_eager_baddbmm_xpu_complex64",
    "test_variant_consistency_eager_baddbmm_xpu_float32",
    "test_variant_consistency_eager_bmm_xpu_complex64",
    "test_variant_consistency_eager_cholesky_inverse_xpu_complex64",
    "test_variant_consistency_eager_cholesky_solve_xpu_complex64",
    "test_variant_consistency_eager_cholesky_xpu_complex64",
    "test_variant_consistency_eager_corrcoef_xpu_complex64",
    "test_variant_consistency_eager_cov_xpu_complex64",
    "test_variant_consistency_eager_einsum_xpu_complex64",
    "test_variant_consistency_eager_geqrf_xpu_complex64",
    "test_variant_consistency_eager_inner_xpu_complex64",
    "test_variant_consistency_eager_linalg_cholesky_ex_xpu_complex64",
    "test_variant_consistency_eager_linalg_cholesky_xpu_complex64",
    "test_variant_consistency_eager_linalg_cond_xpu_complex64",
    "test_variant_consistency_eager_linalg_det_singular_xpu_complex64",
    "test_variant_consistency_eager_linalg_det_xpu_complex64",
    "test_variant_consistency_eager_linalg_eig_xpu_complex64",
    "test_variant_consistency_eager_linalg_eigh_xpu_complex64",
    "test_variant_consistency_eager_linalg_eigvals_xpu_complex64",
    "test_variant_consistency_eager_linalg_eigvalsh_xpu_complex64",
    "test_variant_consistency_eager_linalg_householder_product_xpu_complex64",
    "test_variant_consistency_eager_linalg_inv_ex_xpu_complex64",
    "test_variant_consistency_eager_linalg_inv_xpu_complex64",
    "test_variant_consistency_eager_linalg_ldl_factor_ex_xpu_complex64",
    "test_variant_consistency_eager_linalg_ldl_factor_xpu_complex64",
    "test_variant_consistency_eager_linalg_ldl_solve_xpu_complex64",
    "test_variant_consistency_eager_linalg_lstsq_grad_oriented_xpu_complex64",
    "test_variant_consistency_eager_linalg_lstsq_xpu_complex64",
    "test_variant_consistency_eager_linalg_lu_factor_xpu_complex64",
    "test_variant_consistency_eager_linalg_lu_solve_xpu_complex64",
    "test_variant_consistency_eager_linalg_matrix_norm_xpu_complex64",
    "test_variant_consistency_eager_linalg_matrix_power_xpu_complex64",
    "test_variant_consistency_eager_linalg_matrix_rank_hermitian_xpu_complex64",
    "test_variant_consistency_eager_linalg_matrix_rank_xpu_complex64",
    "test_variant_consistency_eager_linalg_multi_dot_xpu_complex64",
    "test_variant_consistency_eager_linalg_norm_subgradients_at_zero_xpu_complex64",
    "test_variant_consistency_eager_linalg_norm_xpu_complex64",
    "test_variant_consistency_eager_linalg_pinv_hermitian_xpu_complex64",
    "test_variant_consistency_eager_linalg_pinv_singular_xpu_complex64",
    "test_variant_consistency_eager_linalg_pinv_xpu_complex64",
    "test_variant_consistency_eager_linalg_qr_xpu_complex64",
    "test_variant_consistency_eager_linalg_slogdet_xpu_complex64",
    "test_variant_consistency_eager_linalg_solve_ex_xpu_complex64",
    "test_variant_consistency_eager_linalg_solve_triangular_xpu_complex64",
    "test_variant_consistency_eager_linalg_solve_xpu_complex64",
    "test_variant_consistency_eager_linalg_svd_xpu_complex64",
    "test_variant_consistency_eager_linalg_svdvals_xpu_complex64",
    "test_variant_consistency_eager_linalg_tensorinv_xpu_complex64",
    "test_variant_consistency_eager_linalg_tensorsolve_xpu_complex64",
    "test_variant_consistency_eager_logdet_xpu_complex64",
    "test_variant_consistency_eager_lu_solve_xpu_complex64",
    "test_variant_consistency_eager_lu_xpu_complex64",
    "test_variant_consistency_eager_matmul_xpu_complex64",
    "test_variant_consistency_eager_mm_xpu_complex64",
    "test_variant_consistency_eager_mv_xpu_complex64",
    "test_variant_consistency_eager_nn_functional_linear_xpu_complex64",
    "test_variant_consistency_eager_norm_nuc_xpu_complex64",
    "test_variant_consistency_eager_ormqr_xpu_complex64",
    "test_variant_consistency_eager_pinverse_xpu_complex64",
    "test_variant_consistency_eager_qr_xpu_complex64",
    "test_variant_consistency_eager_svd_xpu_complex64",
    "test_variant_consistency_eager_tensordot_xpu_complex64",
    "test_variant_consistency_eager_triangular_solve_xpu_complex64",
)
res += launch_test("test_ops_xpu.py", skip_list)

# test_binary_ufuncs
skip_list = (
    "jiterator", # Jiterator is only supported by CUDA
    "cuda", # Skip cuda hard-coded case
    "test_fmod_remainder_by_zero_integral_xpu_int64", # zero division is an undefined behavior: different handles on different backends
    "test_div_rounding_numpy_xpu_float16", # CPU fail
    "test_cpu_tensor_pow_cuda_scalar_tensor_xpu", # CUDA hard-coded
    "test_type_promotion_bitwise_and_xpu", # align CUDA dtype
    "test_type_promotion_bitwise_or_xpu", # align CUDA dtype
    "test_type_promotion_bitwise_xor_xpu", # align CUDA dtype
    "test_type_promotion_max_binary_xpu", # align CUDA dtype
    "test_type_promotion_maximum_xpu", # align CUDA dtype
    "test_type_promotion_min_binary_xpu", # align CUDA dtype
    "test_type_promotion_minimum_xpu", # align CUDA dtype
    "test_pow_xpu_int16", # align CUDA dtype
    "test_pow_xpu_int32", # align CUDA dtype
    "test_pow_xpu_int64", # align CUDA dtype
    "test_pow_xpu_int8", # align CUDA dtype
    "test_pow_xpu_uint8", # align CUDA dtype
    "test_logaddexp_xpu_complex128", # CPU fail
    "test_logaddexp_xpu_complex64", # CPU fail
    "test_type_promotion_clamp_max_xpu", # align CUDA dtype, CUDA XFAIL
    "test_type_promotion_clamp_min_xpu", # align CUDA dtype, CUDA XFAIL
)
res += launch_test("test_binary_ufuncs_xpu.py", skip_list)

# test_reductions
skip_list = (
    "test_accreal_type_xpu",  # Skip CPU device case
    "test_cumprod_integer_upcast_xpu",  # Skip CPU device case
    "test_cumsum_integer_upcast_xpu",  # Skip CPU device case
    "test_dim_default_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_default_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_dim_default_keepdim_mean_xpu",  # CUDA skip
    "test_dim_default_keepdim_prod_xpu",  # CUDA skip
    "test_dim_default_keepdim_std_xpu",  # CUDA skip
    "test_dim_default_keepdim_sum_xpu",  # CUDA skip
    "test_dim_default_keepdim_var_xpu",  # CUDA skip
    "test_dim_empty__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_empty__refs_sum_xpu",  # CUDA skip
    "test_dim_empty_count_nonzero_xpu",  # CUDA skip
    "test_dim_empty_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_empty_keepdim__refs_sum_xpu",  # CUDA skip
    "test_dim_empty_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_dim_empty_keepdim_masked_logsumexp_xpu",  # CUDA skip
    "test_dim_empty_keepdim_mean_xpu",  # CUDA skip
    "test_dim_empty_keepdim_nanmean_xpu",  # CUDA skip
    "test_dim_empty_keepdim_std_unbiased_xpu",  # CUDA skip
    "test_dim_empty_keepdim_std_xpu",  # CUDA skip
    "test_dim_empty_keepdim_sum_xpu",  # CUDA skip
    "test_dim_empty_keepdim_var_unbiased_xpu",  # CUDA skip
    "test_dim_empty_keepdim_var_xpu",  # CUDA skip
    "test_dim_empty_masked_logsumexp_xpu",  # CUDA skip
    "test_dim_empty_mean_xpu",  # CUDA skip
    "test_dim_empty_nanmean_xpu",  # CUDA skip
    "test_dim_empty_std_unbiased_xpu",  # CUDA skip
    "test_dim_empty_std_xpu",  # CUDA skip
    "test_dim_empty_sum_xpu",  # CUDA skip
    "test_dim_empty_var_unbiased_xpu",  # CUDA skip
    "test_dim_empty_var_xpu",  # CUDA skip
    "test_dim_multi_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_multi_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_dim_multi_unsorted_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_multi_unsorted_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_dim_none_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_none_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_dim_none_keepdim_prod_xpu",  # CUDA skip
    "test_dim_none_prod_xpu",  # CUDA skip
    "test_dim_reduction_lastdim_xpu_bfloat16",  # CUDA skip, CPU available
    "test_dim_reduction_lastdim_xpu_float32",  # CUDA skip, CPU available
    "test_dim_single_keepdim__refs_count_nonzero_xpu",  # CUDA skip
    "test_dim_single_keepdim_count_nonzero_xpu",  # CUDA skip
    "test_empty_tensor_empty_slice_masked_logsumexp_xpu",  # CUDA skip
    "test_histc_lowp_xpu_bfloat16",  # Skip CPU devive cases
    "test_histc_lowp_xpu_float16",  # Skip CPU devive cases
    "test_histogram_error_handling_xpu_float32",  # CUDA skip, CPU available
    "test_histogram_xpu_float32",  # CUDA skip, CPU available
    "test_histogramdd_xpu_float32",  # CUDA skip, CPU available
    "test_logcumsumexp_complex_xpu_complex128", # test require SciPy, but SciPy not found
    "test_logcumsumexp_complex_xpu_complex64",  # test require SciPy, but SciPy not found
    "test_logsumexp_dim_xpu",  # CUDA skip
    "test_logsumexp_xpu",  # CUDA skip
    "test_max_elementwise_xpu",  # Skip CPU devive cases
    "test_max_mixed_devices_xpu",  # Skip CPU devive cases
    "test_mean_dim_xpu",  # Skip CPU devive cases
    "test_mean_int_with_optdtype_xpu",  # CUDA skip
    "test_min_elementwise_xpu",  # Skip CPU devive cases
    "test_min_mixed_devices_xpu",  # Skip CPU devive cases
    "test_nansum_complex_xpu_complex128",  # Skip CPU devive cases
    "test_nansum_complex_xpu_complex64",  # Skip CPU devive cases
    "test_prod_integer_upcast_xpu",  # Skip CPU devive cases
    "test_prod_lowp_xpu_bfloat16",  # Skip CPU devive cases
    "test_prod_lowp_xpu_float16",  # Skip CPU devive cases
    "test_prod_xpu_float32",  # Skip CPU devive cases
    "test_ref_duplicate_values__refs_std_xpu_float16",  # CUDA skip
    "test_ref_duplicate_values__refs_sum_xpu_float16",  # CUDA skip
    "test_ref_duplicate_values_prod_xpu_complex64",  # CUDA skip
    "test_ref_duplicate_values_prod_xpu_float16",  # CUDA skip
    "test_ref_duplicate_values_prod_xpu_uint8",  # CUDA skip
    "test_ref_duplicate_values_std_xpu_float16",  # CUDA skip
    "test_ref_duplicate_values_var_xpu_complex128",  # CUDA skip
    "test_ref_duplicate_values_var_xpu_complex64",  # CUDA skip
    "test_ref_duplicate_values_var_xpu_float16",  # CUDA skip
    "test_ref_duplicate_values_var_xpu_float32",  # CUDA skip
    "test_ref_duplicate_values_var_xpu_float64",  # CUDA skip
    "test_ref_large_input_64bit_indexing__refs_all_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_amax_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_amin_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_any_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_count_nonzero_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_mean_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_prod_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_std_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_sum_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing__refs_var_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_all_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_amax_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_amin_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_any_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_argmax_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_argmin_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_count_nonzero_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_amax_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_amin_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_argmax_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_argmin_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_mean_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_prod_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_std_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_sum_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_masked_var_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_mean_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_nanmean_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_nansum_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_prod_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_std_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_sum_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_large_input_64bit_indexing_var_xpu_float64",  # common_device_type.py::_has_sufficient_memory has no case for device xpu
    "test_ref_small_input__refs_prod_xpu_complex64",  # CUDA skip, CPU available
    "test_ref_small_input__refs_prod_xpu_float16",  # CUDA skip, CPU available
    "test_ref_small_input__refs_std_xpu_float16",  # CUDA skip
    "test_ref_small_input__refs_sum_xpu_float16",  # CUDA skip
    "test_ref_small_input__refs_var_xpu_complex128",  # CUDA skip
    "test_ref_small_input__refs_var_xpu_complex64",  # CUDA skip
    "test_ref_small_input__refs_var_xpu_float16",  # CUDA skip
    "test_ref_small_input__refs_var_xpu_float32",  # CUDA skip
    "test_ref_small_input__refs_var_xpu_float64",  # CUDA skip
    "test_ref_small_input_mean_xpu_float16",  # CUDA skip
    "test_ref_small_input_nanmean_xpu_float16",  # CUDA skip
    "test_ref_small_input_nansum_xpu_float16",  # CUDA skip
    "test_ref_small_input_prod_xpu_complex64",  # CUDA skip
    "test_ref_small_input_prod_xpu_float16",  # CUDA skip
    "test_ref_small_input_std_xpu_float16",  # CUDA skip
    "test_ref_small_input_sum_xpu_float16",  # CUDA skip
    "test_ref_small_input_var_xpu_complex128",  # CUDA skip
    "test_ref_small_input_var_xpu_complex64",  # CUDA skip
    "test_ref_small_input_var_xpu_float16",  # CUDA skip
    "test_ref_small_input_var_xpu_float32",  # CUDA skip
    "test_ref_small_input_var_xpu_float64",  # CUDA skip
    "test_reference_masked_masked_prod_xpu_bool",  # CUDA skip
    "test_reference_masked_masked_prod_xpu_int16",  # CUDA skip
    "test_reference_masked_masked_prod_xpu_int32",  # CUDA skip
    "test_reference_masked_masked_prod_xpu_int8",  # CUDA skip
    "test_reference_masked_masked_sum_xpu_bool",  # CUDA skip
    "test_reference_masked_masked_sum_xpu_int16",  # CUDA skip
    "test_reference_masked_masked_sum_xpu_int32",  # CUDA skip
    "test_reference_masked_masked_sum_xpu_int8",  # CUDA skip
    "test_std_dim_xpu",  # Skip CPU devive cases
    "test_sum_all_xpu_bool",  # Skip CPU devive cases
    "test_sum_all_xpu_float64",  # Skip CPU devive cases
    "test_sum_dim_xpu",  # Skip CPU devive cases
    "test_sum_integer_upcast_xpu",  # Skip CPU devive cases
    "test_sum_noncontig_lowp_xpu_bfloat16",  # Skip CPU devive cases
    "test_sum_noncontig_lowp_xpu_float16",  # Skip CPU devive cases
    "test_sum_out_xpu_float64",  # Skip CPU devive cases
    "test_sum_parallel_xpu",  # Skip CPU devive cases
    "test_tensor_reduce_ops_empty_xpu",  # Skip cases for non-Scipy
    "test_var_dim_xpu",  # Skip CPU devive cases
    "test_dim_reduction_xpu_float16",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_bfloat16",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_float32",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_float64",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_int16",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_int32",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_int8",  # Failed, mode not supports XPU
    "test_dim_reduction_xpu_int64", # Failed, mode not supports XPU
    "test_mode_xpu_float32",  # Failed, mode not supports XPU
    "test_mode_xpu_float64",  # Failed, mode not supports XPU
    "test_mode_xpu_int16",  # Failed, mode not supports XPU
    "test_mode_xpu_int32",  # Failed, mode not supports XPU
    "test_mode_xpu_int64",  # Failed, mode not supports XPU
    "test_mode_xpu_int8",  # Failed, mode not supports XPU
    "test_mode_xpu_uint8",  # Failed, mode not supports XPU
    "test_ref_extremal_values_mean_xpu_complex64",  # CUDA skip
    "test_ref_small_input_masked_prod_xpu_float16",  # Tensor-likes are not close
    "test_dim_reduction_fns_fn_name_mode_xpu_int8", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_int64", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_int32", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_int16", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_float64", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_float32", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_float16", # Failed, mode not supports xpu
    "test_dim_reduction_fns_fn_name_mode_xpu_bfloat16", # Failed, mode not supports xpu
)
res += launch_test("test_reductions_xpu.py", skip_list=skip_list)

# test_scatter_gather_ops
skip_list = (
    "test_gather_backward_with_empty_index_tensor_sparse_grad_True_xpu_float32", # Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend.
    "test_gather_backward_with_empty_index_tensor_sparse_grad_True_xpu_float64", # Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend.
    "test_scatter__reductions_xpu_complex64", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'ComplexFloat'
    "test_scatter_reduce_amax_xpu_bool", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_scatter_reduce_amin_xpu_bool", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_scatter_reduce_mean_xpu_complex128", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'ComplexDouble'
    "test_scatter_reduce_mean_xpu_complex64", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'ComplexFloat'
    "test_scatter_reduce_prod_xpu_bool", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'Bool'
    "test_scatter_reduce_prod_xpu_complex128", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'ComplexDouble'
    "test_scatter_reduce_prod_xpu_complex64", # align CUDA dtype - RuntimeError: "scatter_gather_base_kernel_func" not implemented for 'ComplexFloat'
)
res += launch_test("test_scatter_gather_ops_xpu.py", skip_list)

res += launch_test("test_autograd_fallback.py")

# test_sort_and_select
skip_list = (
    # The following isin case fails on CPU fallback, as it could be backend-specific.
    "test_isin_different_devices_xpu_float32", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_float64", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_int16", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_int32", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_int64", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_int8", # AssertionError: RuntimeError not raised
    "test_isin_different_devices_xpu_uint8", # AssertionError: RuntimeError not raised

    "test_isin_different_dtypes_xpu", # RuntimeError: "isin_default_cpu" not implemented for 'Half'"

    "test_sort_large_slice_xpu", # Hard code CUDA
)
res += launch_test("test_sort_and_select_xpu.py", skip_list)

nn_test_embedding_skip_list = (
    # Skip list of base line
    # Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors'
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int32_float64",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int64_float32",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int64_float64",
    "test_embedding_backward_xpu_float64",
    "test_embedding_bag_1D_padding_idx_xpu_float32",
    "test_embedding_bag_1D_padding_idx_xpu_float64",
    "test_embedding_bag_2D_padding_idx_xpu_float32",
    "test_embedding_bag_2D_padding_idx_xpu_float64",
    "test_embedding_bag_bfloat16_xpu_int32_int64",
    "test_embedding_bag_bfloat16_xpu_int64_int32",
    "test_embedding_bag_bfloat16_xpu_int64_int64",
    "test_embedding_bag_device_xpu_int32_int32_float64",
    "test_embedding_bag_device_xpu_int32_int64_float64",
    "test_embedding_bag_device_xpu_int64_int32_bfloat16",
    "test_embedding_bag_device_xpu_int64_int32_float16",
    "test_embedding_bag_device_xpu_int64_int32_float32",
    "test_embedding_bag_device_xpu_int64_int32_float64",
    "test_embedding_bag_device_xpu_int64_int64_bfloat16",
    "test_embedding_bag_device_xpu_int64_int64_float16",
    "test_embedding_bag_device_xpu_int64_int64_float32",
    "test_embedding_bag_device_xpu_int64_int64_float64",
    "test_embedding_bag_half_xpu_int32_int64",
    "test_embedding_bag_half_xpu_int64_int32",
    "test_embedding_bag_half_xpu_int64_int64",

    # CPU fallback error: RuntimeError: expected scalar type Long but found Int
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int32_int32_bfloat16",
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int32_int32_float16",
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int32_int32_float32",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int32_float32",
    "test_EmbeddingBag_per_sample_weights_and_offsets_xpu_int32_int32_bfloat16",
    "test_EmbeddingBag_per_sample_weights_and_offsets_xpu_int32_int32_float16",
    "test_EmbeddingBag_per_sample_weights_and_offsets_xpu_int32_int32_float32",
    "test_embedding_bag_bfloat16_xpu_int32_int32",
    "test_embedding_bag_device_xpu_int32_int32_bfloat16",
    "test_embedding_bag_device_xpu_int32_int32_float16",
    "test_embedding_bag_device_xpu_int32_int32_float32",
    "test_embedding_bag_device_xpu_int32_int64_bfloat16",
    "test_embedding_bag_device_xpu_int32_int64_float16",
    "test_embedding_bag_device_xpu_int32_int64_float32",
    "test_embedding_bag_half_xpu_int32_int32",

    # CPU fallback error: AssertionError: Tensor-likes are not close!
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int32_int64_bfloat16",
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int64_int32_bfloat16",
    "test_EmbeddingBag_per_sample_weights_and_new_offsets_xpu_int64_int64_bfloat16",
)
res += launch_test("nn/test_embedding_xpu.py", nn_test_embedding_skip_list)

# test_transformers
skip_list = (
    # AssertionError: False is not true
    # CPU fallback failure. To support aten::transformer_encoder_layer_forward with proper priority.
    "test_disable_fastpath_xpu",
    # We have no mechanism to handle SDPBackend::ERROR so far. Will give a fully support when we support all SDPBackends.
    "test_dispatch_fails_no_backend_xpu",
    # Could not run 'aten::_to_copy' with arguments from the 'NestedTensorXPU' backend
    "test_with_nested_tensor_input_xpu",
    # Double and complex datatype matmul is not supported in oneDNN
    "test_sdp_math_gradcheck_contiguous_inputs_False_xpu",
    "test_sdp_math_gradcheck_contiguous_inputs_True_xpu",
    "test_transformerencoder_batch_first_True_training_True_enable_nested_tensor_True_xpu",
    "test_transformerencoder_batch_first_True_training_True_enable_nested_tensor_False_xpu",
    "test_transformerencoder_batch_first_True_training_False_enable_nested_tensor_True_xpu",
    "test_transformerencoder_batch_first_True_training_False_enable_nested_tensor_False_xpu",
    "test_transformerencoder_batch_first_False_training_True_enable_nested_tensor_True_xpu",
    "test_transformerencoder_batch_first_False_training_True_enable_nested_tensor_False_xpu",
    "test_transformerencoder_batch_first_False_training_False_enable_nested_tensor_True_xpu",
    "test_transformerencoder_batch_first_False_training_False_enable_nested_tensor_False_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_no_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_no_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_no_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_causal_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_causal_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_causal_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_4D_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_causal_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_causal_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_causal_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_4D_input_dim_2D_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_no_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_no_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_no_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_causal_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_causal_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_causal_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_3D_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_causal_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_causal_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_causal_attn_mask_dropout_p_0_0_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_attn_mask_dropout_p_0_5_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_attn_mask_dropout_p_0_2_xpu",
    "test_scaled_dot_product_attention_3D_input_dim_2D_attn_mask_dropout_p_0_0_xpu",
    # AssertionError: Torch not compiled with CUDA enabled
    "test_mha_native_args_nb_heads_8_bias_True_xpu",
    "test_mha_native_args_nb_heads_8_bias_False_xpu",
    "test_mha_native_args_nb_heads_1_bias_True_xpu",
    "test_mha_native_args_nb_heads_1_bias_False_xpu",
)
res += launch_test("test_transformers_xpu.py", skip_list)

# test_complex
skip_list = (
    #Skip CPU case
    "test_eq_xpu_complex128",
    "test_eq_xpu_complex64",
    "test_ne_xpu_complex128",
    "test_ne_xpu_complex64",
)
res += launch_test("test_complex_xpu.py", skip_list)

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
