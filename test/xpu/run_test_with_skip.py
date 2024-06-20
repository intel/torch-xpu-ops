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
            exe_option = " or " + exe_case
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
    "test_dtypes_sgn_xpu", # Skip this case due to mis-alignment on test case: "The following dtypes did not work in forward but are listed by the OpInfo: {Complex32}""
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
    "test_conj_view_conj_physical_xpu_complex64",
    "test_neg_conj_view_conj_physical_xpu_complex128",

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
    "test_complex_half_reference_testing_sigmoid_xpu_complex32", # Didn't align with CUDA, RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_dtypes_sigmoid_xpu", # Didn't align with CUDA, RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_python_ref__refs_sigmoid_xpu_complex32", # Didn't align with CUDA, RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_python_ref_errors__refs_where_xpu", # align with CUDA, AssertionError: "Expected all tensors to be on the same device" does not match "Tensor on device xpu:0 is not on the expected device cpu!"
    "test_python_ref_executor__refs_sigmoid_executor_aten_xpu_complex32", # Didn't align with CUDA, RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_python_ref_torch_fallback__refs_sigmoid_xpu_complex32", # Didn't align with CUDA, RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_dtypes_view_as_complex_xpu", # Didn't align with CUDA, The following dtypes did not work in backward but are listed by the OpInfo: {torch.bfloat16}
    "test_dtypes_view_as_real_xpu", # Didn't align with CUDA, The following dtypes did not work in backward but are listed by the OpInfo: {torch.bfloat16}
    "test_python_ref_executor__refs_pow_executor_aten_xpu_complex32", # Didn't align with CUDA, Unexpected success

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
    # oneDNN issues
    # RuntimeError: value cannot be converted to type float without overflow
    "test_conj_view_addbmm_xpu_complex64",
    "test_neg_conj_view_addbmm_xpu_complex128",

    # CPU fallback error: AssertionError: Tensor-likes are not close!
    "test_neg_view_nn_functional_rrelu_xpu_float64",

    # CPU fallback fails
    # RuntimeError: input tensor must have at least one element, but got input_sizes = [1, 0, 1]
    "test_neg_view__refs_nn_functional_group_norm_xpu_float64",
    "test_neg_view_nn_functional_group_norm_xpu_float64",

    # CUDA skip,reproduce the UT in CUDA,CUDA fail
    "test_neg_view_nn_functional_dropout_xpu_float64",

    ### Error #0 in TestMathBitsXPU , RuntimeError: Double and complex datatype matmul is not supported in oneDNN
    # https://github.com/intel/torch-xpu-ops/issues/254
    "test_conj_view___rmatmul___xpu_complex64",
    "test_conj_view__refs_linalg_svd_xpu_complex64",
    "test_conj_view_addmm_decomposed_xpu_complex64",
    "test_conj_view_addmm_xpu_complex64",
    "test_conj_view_addmv_xpu_complex64",
    "test_conj_view_addr_xpu_complex64",
    "test_conj_view_baddbmm_xpu_complex64",
    "test_conj_view_bmm_xpu_complex64",
    "test_conj_view_cholesky_inverse_xpu_complex64",
    "test_conj_view_cholesky_solve_xpu_complex64",
    "test_conj_view_cholesky_xpu_complex64",
    "test_conj_view_corrcoef_xpu_complex64",
    "test_conj_view_cov_xpu_complex64",
    "test_conj_view_einsum_xpu_complex64",
    "test_conj_view_geqrf_xpu_complex64",
    "test_conj_view_inner_xpu_complex64",
    "test_conj_view_linalg_cholesky_ex_xpu_complex64",
    "test_conj_view_linalg_cholesky_xpu_complex64",
    "test_conj_view_linalg_cond_xpu_complex64",
    "test_conj_view_linalg_det_singular_xpu_complex64",
    "test_conj_view_linalg_det_xpu_complex64",
    "test_conj_view_linalg_eig_xpu_complex64",
    "test_conj_view_linalg_eigh_xpu_complex64",
    "test_conj_view_linalg_eigvals_xpu_complex64",
    "test_conj_view_linalg_eigvalsh_xpu_complex64",
    "test_conj_view_linalg_householder_product_xpu_complex64",
    "test_conj_view_linalg_inv_ex_xpu_complex64",
    "test_conj_view_linalg_inv_xpu_complex64",
    "test_conj_view_linalg_ldl_factor_ex_xpu_complex64",
    "test_conj_view_linalg_ldl_factor_xpu_complex64",
    "test_conj_view_linalg_ldl_solve_xpu_complex64",
    "test_conj_view_linalg_lstsq_grad_oriented_xpu_complex64",
    "test_conj_view_linalg_lstsq_xpu_complex64",
    "test_conj_view_linalg_lu_factor_xpu_complex64",
    "test_conj_view_linalg_lu_solve_xpu_complex64",
    "test_conj_view_linalg_matrix_norm_xpu_complex64",
    "test_conj_view_linalg_matrix_power_xpu_complex64",
    "test_conj_view_linalg_matrix_rank_hermitian_xpu_complex64",
    "test_conj_view_linalg_matrix_rank_xpu_complex64",
    "test_conj_view_linalg_multi_dot_xpu_complex64",
    "test_conj_view_linalg_norm_subgradients_at_zero_xpu_complex64",
    "test_conj_view_linalg_norm_xpu_complex64",
    "test_conj_view_linalg_pinv_hermitian_xpu_complex64",
    "test_conj_view_linalg_pinv_singular_xpu_complex64",
    "test_conj_view_linalg_pinv_xpu_complex64",
    "test_conj_view_linalg_qr_xpu_complex64",
    "test_conj_view_linalg_slogdet_xpu_complex64",
    "test_conj_view_linalg_solve_ex_xpu_complex64",
    "test_conj_view_linalg_solve_triangular_xpu_complex64",
    "test_conj_view_linalg_solve_xpu_complex64",
    "test_conj_view_linalg_svd_xpu_complex64",
    "test_conj_view_linalg_svdvals_xpu_complex64",
    "test_conj_view_linalg_tensorinv_xpu_complex64",
    "test_conj_view_linalg_tensorsolve_xpu_complex64",
    "test_conj_view_logdet_xpu_complex64",
    "test_conj_view_lu_solve_xpu_complex64",
    "test_conj_view_lu_xpu_complex64",
    "test_conj_view_matmul_xpu_complex64",
    "test_conj_view_mm_xpu_complex64",
    "test_conj_view_mv_xpu_complex64",
    "test_conj_view_nn_functional_linear_xpu_complex64",
    "test_conj_view_norm_nuc_xpu_complex64",
    "test_conj_view_ormqr_xpu_complex64",
    "test_conj_view_pinverse_xpu_complex64",
    "test_conj_view_qr_xpu_complex64",
    "test_conj_view_svd_xpu_complex64",
    "test_conj_view_tensordot_xpu_complex64",
    "test_conj_view_triangular_solve_xpu_complex64",
    "test_neg_conj_view_addmm_decomposed_xpu_complex128",
    "test_neg_conj_view_addmm_xpu_complex128",
    "test_neg_conj_view_addmv_xpu_complex128",
    "test_neg_conj_view_addr_xpu_complex128",
    "test_neg_conj_view_baddbmm_xpu_complex128",
    "test_neg_conj_view_bmm_xpu_complex128",
    "test_neg_conj_view_cholesky_inverse_xpu_complex128",
    "test_neg_conj_view_cholesky_solve_xpu_complex128",
    "test_neg_conj_view_cholesky_xpu_complex128",
    "test_neg_conj_view_corrcoef_xpu_complex128",
    "test_neg_conj_view_cov_xpu_complex128",
    "test_neg_conj_view_geqrf_xpu_complex128",
    "test_neg_conj_view_inner_xpu_complex128",
    "test_neg_conj_view_linalg_cholesky_ex_xpu_complex128",
    "test_neg_conj_view_linalg_cholesky_xpu_complex128",
    "test_neg_conj_view_linalg_cond_xpu_complex128",
    "test_neg_conj_view_linalg_det_singular_xpu_complex128",
    "test_neg_conj_view_linalg_eig_xpu_complex128",
    "test_neg_conj_view_linalg_eigh_xpu_complex128",
    "test_neg_conj_view_linalg_eigvals_xpu_complex128",
    "test_neg_conj_view_linalg_eigvalsh_xpu_complex128",
    "test_neg_conj_view_linalg_householder_product_xpu_complex128",
    "test_neg_conj_view_linalg_inv_ex_xpu_complex128",
    "test_neg_conj_view_linalg_inv_xpu_complex128",
    "test_neg_conj_view_linalg_ldl_factor_ex_xpu_complex128",
    "test_neg_conj_view_linalg_ldl_factor_xpu_complex128",
    "test_neg_conj_view_linalg_ldl_solve_xpu_complex128",
    "test_neg_conj_view_linalg_lstsq_grad_oriented_xpu_complex128",
    "test_neg_conj_view_linalg_lstsq_xpu_complex128",
    "test_neg_conj_view_linalg_lu_factor_xpu_complex128",
    "test_neg_conj_view_linalg_lu_solve_xpu_complex128",
    "test_neg_conj_view_linalg_matrix_rank_hermitian_xpu_complex128",
    "test_neg_conj_view_linalg_matrix_rank_xpu_complex128",
    "test_neg_conj_view_linalg_multi_dot_xpu_complex128",
    "test_neg_conj_view_linalg_pinv_hermitian_xpu_complex128",
    "test_neg_conj_view_linalg_pinv_singular_xpu_complex128",
    "test_neg_conj_view_linalg_pinv_xpu_complex128",
    "test_neg_conj_view_linalg_qr_xpu_complex128",
    "test_neg_conj_view_linalg_solve_ex_xpu_complex128",
    "test_neg_conj_view_linalg_solve_triangular_xpu_complex128",
    "test_neg_conj_view_linalg_solve_xpu_complex128",
    "test_neg_conj_view_linalg_svdvals_xpu_complex128",
    "test_neg_conj_view_linalg_tensorinv_xpu_complex128",
    "test_neg_conj_view_linalg_tensorsolve_xpu_complex128",
    "test_neg_conj_view_lu_solve_xpu_complex128",
    "test_neg_conj_view_lu_xpu_complex128",
    "test_neg_conj_view_mm_xpu_complex128",
    "test_neg_conj_view_mv_xpu_complex128",
    "test_neg_conj_view_nn_functional_linear_xpu_complex128",
    "test_neg_conj_view_norm_nuc_xpu_complex128",
    "test_neg_conj_view_ormqr_xpu_complex128",
    "test_neg_conj_view_pinverse_xpu_complex128",
    "test_neg_conj_view_qr_xpu_complex128",
    "test_neg_conj_view_tensordot_xpu_complex128",
    "test_neg_conj_view_triangular_solve_xpu_complex128",
    "test_neg_view___rmatmul___xpu_float64",
    "test_neg_view__refs_linalg_svd_xpu_float64",
    "test_neg_view__refs_nn_functional_pdist_xpu_float64",
    "test_neg_view_addbmm_xpu_float64",
    "test_neg_view_addmm_decomposed_xpu_float64",
    "test_neg_view_addmm_xpu_float64",
    "test_neg_view_addmv_xpu_float64",
    "test_neg_view_addr_xpu_float64",
    "test_neg_view_baddbmm_xpu_float64",
    "test_neg_view_bmm_xpu_float64",
    "test_neg_view_cdist_xpu_float64",
    "test_neg_view_cholesky_inverse_xpu_float64",
    "test_neg_view_cholesky_solve_xpu_float64",
    "test_neg_view_cholesky_xpu_float64",
    "test_neg_view_corrcoef_xpu_float64",
    "test_neg_view_cov_xpu_float64",
    "test_neg_view_einsum_xpu_float64",
    "test_neg_view_geqrf_xpu_float64",
    "test_neg_view_inner_xpu_float64",
    "test_neg_view_linalg_cholesky_ex_xpu_float64",
    "test_neg_view_linalg_cholesky_xpu_float64",
    "test_neg_view_linalg_cond_xpu_float64",
    "test_neg_view_linalg_det_singular_xpu_float64",
    "test_neg_view_linalg_det_xpu_float64",
    "test_neg_view_linalg_eig_xpu_float64",
    "test_neg_view_linalg_eigh_xpu_float64",
    "test_neg_view_linalg_eigvals_xpu_float64",
    "test_neg_view_linalg_eigvalsh_xpu_float64",
    "test_neg_view_linalg_householder_product_xpu_float64",
    "test_neg_view_linalg_inv_ex_xpu_float64",
    "test_neg_view_linalg_inv_xpu_float64",
    "test_neg_view_linalg_ldl_factor_ex_xpu_float64",
    "test_neg_view_linalg_ldl_factor_xpu_float64",
    "test_neg_view_linalg_ldl_solve_xpu_float64",
    "test_neg_view_linalg_lstsq_grad_oriented_xpu_float64",
    "test_neg_view_linalg_lstsq_xpu_float64",
    "test_neg_view_linalg_lu_factor_xpu_float64",
    "test_neg_view_linalg_lu_solve_xpu_float64",
    "test_neg_view_linalg_matrix_norm_xpu_float64",
    "test_neg_view_linalg_matrix_power_xpu_float64",
    "test_neg_view_linalg_matrix_rank_hermitian_xpu_float64",
    "test_neg_view_linalg_matrix_rank_xpu_float64",
    "test_neg_view_linalg_multi_dot_xpu_float64",
    "test_neg_view_linalg_norm_subgradients_at_zero_xpu_float64",
    "test_neg_view_linalg_norm_xpu_float64",
    "test_neg_view_linalg_pinv_hermitian_xpu_float64",
    "test_neg_view_linalg_pinv_singular_xpu_float64",
    "test_neg_view_linalg_pinv_xpu_float64",
    "test_neg_view_linalg_qr_xpu_float64",
    "test_neg_view_linalg_slogdet_xpu_float64",
    "test_neg_view_linalg_solve_ex_xpu_float64",
    "test_neg_view_linalg_solve_triangular_xpu_float64",
    "test_neg_view_linalg_solve_xpu_float64",
    "test_neg_view_linalg_svd_xpu_float64",
    "test_neg_view_linalg_svdvals_xpu_float64",
    "test_neg_view_linalg_tensorinv_xpu_float64",
    "test_neg_view_linalg_tensorsolve_xpu_float64",
    "test_neg_view_logdet_xpu_float64",
    "test_neg_view_lu_solve_xpu_float64",
    "test_neg_view_lu_xpu_float64",
    "test_neg_view_matmul_xpu_float64",
    "test_neg_view_mm_xpu_float64",
    "test_neg_view_mv_xpu_float64",
    "test_neg_view_nn_functional_bilinear_xpu_float64",
    "test_neg_view_nn_functional_linear_xpu_float64",
    "test_neg_view_nn_functional_multi_head_attention_forward_xpu_float64",
    "test_neg_view_nn_functional_scaled_dot_product_attention_xpu_float64",
    "test_neg_view_norm_nuc_xpu_float64",
    "test_neg_view_ormqr_xpu_float64",
    "test_neg_view_pca_lowrank_xpu_float64",
    "test_neg_view_pinverse_xpu_float64",
    "test_neg_view_qr_xpu_float64",
    "test_neg_view_svd_lowrank_xpu_float64",
    "test_neg_view_svd_xpu_float64",
    "test_neg_view_tensordot_xpu_float64",
    "test_neg_view_triangular_solve_xpu_float64",
    "test_noncontiguous_samples_pca_lowrank_xpu_complex64",
    "test_noncontiguous_samples_svd_lowrank_xpu_complex64",
    "test_variant_consistency_eager_pca_lowrank_xpu_complex64",
    "test_variant_consistency_eager_svd_lowrank_xpu_complex64",
    "test_conj_view_pca_lowrank_xpu_complex64",
    "test_conj_view_svd_lowrank_xpu_complex64",

    ### Error #1 in TestMathBitsXPU , RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive
    # https://github.com/intel/torch-xpu-ops/issues/253
    "test_conj_view_nn_functional_conv_transpose2d_xpu_complex64",
    "test_conj_view_nn_functional_conv_transpose3d_xpu_complex64",
    "test_neg_view_nn_functional_conv_transpose2d_xpu_float64",
    "test_neg_view_nn_functional_conv_transpose3d_xpu_float64",

    ### Error #2 in TestMathBitsXPU , NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend. 
    # https://github.com/intel/torch-xpu-ops/issues/242
    "test_conj_view_to_sparse_xpu_complex64",
    "test_neg_conj_view_to_sparse_xpu_complex128",
    "test_neg_view_to_sparse_xpu_float64",
)
res += launch_test("test_ops_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_binary_ufuncs
skip_list = (
    "jiterator", # Jiterator is only supported by CUDA
    "cuda", # Skip cuda hard-coded case
    "test_fmod_remainder_by_zero_integral_xpu_int64", # zero division is an undefined behavior: different handles on different backends
    "test_div_rounding_numpy_xpu_float16", # Calculation error. XPU implementation uses opmath type.
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
    "test_div_rounding_nonfinite_xpu_bfloat16", # CPU result is not golden reference
    "test_div_rounding_nonfinite_xpu_float16", # CPU result is not golden reference
)
res += launch_test("test_binary_ufuncs_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

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
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("test_autograd_fallback.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_sort_and_select
skip_list = (
    # The following isin case fails on CPU fallback, as it could be backend-specific.
    "test_isin_xpu_float16", # RuntimeError: "isin_default_cpu" not implemented for 'Half'
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
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

nn_test_embedding_skip_list = (
    # NotImplementedError: Could not run 'aten::_indices' with arguments from the 'SparseXPU' backend.
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int32_float16",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int32_float32",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int32_float64",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int64_float16",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int64_float32",
    "test_EmbeddingBag_per_sample_weights_and_no_offsets_xpu_int64_float64",
    "test_embedding_backward_xpu_float16",
    "test_embedding_backward_xpu_float64",
    "test_embedding_bag_1D_padding_idx_xpu_bfloat16",
    "test_embedding_bag_1D_padding_idx_xpu_float16",
    "test_embedding_bag_2D_padding_idx_xpu_bfloat16",
    "test_embedding_bag_2D_padding_idx_xpu_float16",
    "test_embedding_bag_bfloat16_xpu_int32_int32",
    "test_embedding_bag_bfloat16_xpu_int32_int64",
    "test_embedding_bag_bfloat16_xpu_int64_int32",
    "test_embedding_bag_bfloat16_xpu_int64_int64",
    "test_embedding_bag_device_xpu_int32_int32_float16",
    "test_embedding_bag_device_xpu_int32_int32_float32",
    "test_embedding_bag_device_xpu_int32_int32_float64",
    "test_embedding_bag_device_xpu_int32_int64_float16",
    "test_embedding_bag_device_xpu_int32_int64_float32",
    "test_embedding_bag_device_xpu_int32_int64_float64",
    "test_embedding_bag_device_xpu_int64_int32_float16",
    "test_embedding_bag_device_xpu_int64_int32_float32",
    "test_embedding_bag_device_xpu_int64_int32_float64",
    "test_embedding_bag_device_xpu_int64_int64_float16",
    "test_embedding_bag_device_xpu_int64_int64_float32",
    "test_embedding_bag_device_xpu_int64_int64_float64",
    "test_embedding_bag_half_xpu_int32_int32",
    "test_embedding_bag_half_xpu_int32_int64",
    "test_embedding_bag_half_xpu_int64_int32",
    "test_embedding_bag_half_xpu_int64_int64",

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
)
res += launch_test("nn/test_embedding_xpu.py", nn_test_embedding_skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

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
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_complex
skip_list = (
    #Skip CPU case
    "test_eq_xpu_complex128",
    "test_eq_xpu_complex64",
    "test_ne_xpu_complex128",
    "test_ne_xpu_complex64",
)
res += launch_test("test_complex_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_modules
skip_list = (
    # XPU tensor compatible issue
    # RuntimeError: don't know how to determine data location of torch.storage.UntypedStorage
    "test_save_load_nn_",

    # oneDNN issues
    # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
    "test_cpu_gpu_parity_nn_Bilinear_xpu_float64",
    "test_cpu_gpu_parity_nn_GRUCell_xpu_float64",
    "test_cpu_gpu_parity_nn_GRU_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_GRU_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_LSTMCell_xpu_float64",
    "test_cpu_gpu_parity_nn_LSTM_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_LSTM_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_Linear_xpu_float64",
    "test_cpu_gpu_parity_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_RNNCell_xpu_float64",
    "test_cpu_gpu_parity_nn_RNN_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_RNN_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_TransformerDecoderLayer_xpu_float64",
    "test_cpu_gpu_parity_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_cpu_gpu_parity_nn_Transformer_xpu_float64",
    "test_forward_nn_Bilinear_xpu_float64",
    "test_forward_nn_GRUCell_xpu_float64",
    "test_forward_nn_GRU_eval_mode_xpu_float64",
    "test_forward_nn_GRU_train_mode_xpu_float64",
    "test_forward_nn_LSTMCell_xpu_float64",
    "test_forward_nn_LSTM_eval_mode_xpu_float64",
    "test_forward_nn_LSTM_train_mode_xpu_float64",
    "test_forward_nn_Linear_xpu_float64",
    "test_forward_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_forward_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_forward_nn_RNNCell_xpu_float64",
    "test_forward_nn_RNN_eval_mode_xpu_float64",
    "test_forward_nn_RNN_train_mode_xpu_float64",
    "test_forward_nn_TransformerDecoderLayer_xpu_float64",
    "test_forward_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_forward_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_forward_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_forward_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_forward_nn_Transformer_xpu_float64",
    "test_grad_nn_Bilinear_xpu_float64",
    "test_grad_nn_GRUCell_xpu_float64",
    "test_grad_nn_GRU_eval_mode_xpu_float64",
    "test_grad_nn_GRU_train_mode_xpu_float64",
    "test_grad_nn_LSTMCell_xpu_float64",
    "test_grad_nn_LSTM_eval_mode_xpu_float64",
    "test_grad_nn_LSTM_train_mode_xpu_float64",
    "test_grad_nn_Linear_xpu_float64",
    "test_grad_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_grad_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_grad_nn_RNNCell_xpu_float64",
    "test_grad_nn_RNN_eval_mode_xpu_float64",
    "test_grad_nn_RNN_train_mode_xpu_float64",
    "test_grad_nn_TransformerDecoderLayer_xpu_float64",
    "test_grad_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_grad_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_grad_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_grad_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_grad_nn_Transformer_xpu_float64",
    "test_gradgrad_nn_Bilinear_xpu_float64",
    "test_gradgrad_nn_GRUCell_xpu_float64",
    "test_gradgrad_nn_GRU_eval_mode_xpu_float64",
    "test_gradgrad_nn_GRU_train_mode_xpu_float64",
    "test_gradgrad_nn_LSTMCell_xpu_float64",
    "test_gradgrad_nn_LSTM_eval_mode_xpu_float64",
    "test_gradgrad_nn_LSTM_train_mode_xpu_float64",
    "test_gradgrad_nn_Linear_xpu_float64",
    "test_gradgrad_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_gradgrad_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_gradgrad_nn_RNNCell_xpu_float64",
    "test_gradgrad_nn_RNN_eval_mode_xpu_float64",
    "test_gradgrad_nn_RNN_train_mode_xpu_float64",
    "test_gradgrad_nn_TransformerDecoderLayer_xpu_float64",
    "test_gradgrad_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_gradgrad_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_gradgrad_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_gradgrad_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_gradgrad_nn_Transformer_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_Bilinear_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_GRUCell_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_LSTMCell_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_Linear_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_RNNCell_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_TransformerDecoderLayer_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_TransformerEncoderLayer_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_TransformerEncoder_xpu_float64",
    "test_if_train_and_eval_modes_differ_nn_Transformer_xpu_float64",
    "test_memory_format_nn_GRUCell_xpu_float64",
    "test_memory_format_nn_GRU_eval_mode_xpu_float64",
    "test_memory_format_nn_GRU_train_mode_xpu_float64",
    "test_memory_format_nn_LSTMCell_xpu_float64",
    "test_memory_format_nn_LSTM_eval_mode_xpu_float64",
    "test_memory_format_nn_LSTM_train_mode_xpu_float64",
    "test_memory_format_nn_RNNCell_xpu_float64",
    "test_memory_format_nn_RNN_eval_mode_xpu_float64",
    "test_memory_format_nn_RNN_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_Bilinear_xpu_float64",
    "test_multiple_device_transfer_nn_GRUCell_xpu_float64",
    "test_multiple_device_transfer_nn_GRU_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_GRU_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_LSTMCell_xpu_float64",
    "test_multiple_device_transfer_nn_LSTM_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_LSTM_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_Linear_xpu_float64",
    "test_multiple_device_transfer_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_RNNCell_xpu_float64",
    "test_multiple_device_transfer_nn_RNN_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_RNN_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_TransformerDecoderLayer_xpu_float64",
    "test_multiple_device_transfer_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_Transformer_xpu_float64",
    "test_non_contiguous_tensors_nn_Bilinear_xpu_float64",
    "test_non_contiguous_tensors_nn_GRUCell_xpu_float64",
    "test_non_contiguous_tensors_nn_GRU_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_GRU_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_LSTMCell_xpu_float64",
    "test_non_contiguous_tensors_nn_LSTM_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_LSTM_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_Linear_xpu_float64",
    "test_non_contiguous_tensors_nn_MultiheadAttention_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_MultiheadAttention_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_RNNCell_xpu_float64",
    "test_non_contiguous_tensors_nn_RNN_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_RNN_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_TransformerDecoderLayer_xpu_float64",
    "test_non_contiguous_tensors_nn_TransformerEncoderLayer_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_TransformerEncoderLayer_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_TransformerEncoder_eval_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_TransformerEncoder_train_mode_xpu_float64",
    "test_non_contiguous_tensors_nn_Transformer_xpu_float64",
    # AssertionError: Tensor-likes are not close!
    "test_cpu_gpu_parity_nn_ConvTranspose1d_xpu_complex32",
    "test_cpu_gpu_parity_nn_ConvTranspose2d_xpu_complex32",
    "test_cpu_gpu_parity_nn_ConvTranspose3d_xpu_complex32",
    # torch.autograd.gradcheck.GradcheckError: Backward is not reentrant, i.e., running backward with same input and grad_output multiple times gives different values, although analytical gradient matches numerical gradient.The ...
    "test_grad_nn_Conv3d_xpu_float64",
    "test_grad_nn_ConvTranspose3d_xpu_float64",
    "test_grad_nn_LazyConv3d_xpu_float64",
    "test_grad_nn_LazyConvTranspose3d_xpu_float64",
    # AssertionError: False is not true
    "test_memory_format_nn_Conv2d_xpu_float64",
    "test_memory_format_nn_ConvTranspose2d_xpu_float64",
    "test_memory_format_nn_LazyConv2d_xpu_float64",
    "test_memory_format_nn_LazyConvTranspose2d_xpu_float64",

    # CPU fallback fails
    # AssertionError: Tensor-likes are not close!
    "test_cpu_gpu_parity_nn_CrossEntropyLoss_xpu_float16",

    # CPU fallback could not cover these
    # NotImplementedError: Could not run 'aten::_thnn_fused_gru_cell' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build pro...
    "test_cpu_gpu_parity_nn_GRUCell_xpu_float32",
    "test_cpu_gpu_parity_nn_GRU_eval_mode_xpu_float32",
    "test_cpu_gpu_parity_nn_GRU_train_mode_xpu_float32",
    "test_forward_nn_GRUCell_xpu_float32",
    "test_forward_nn_GRU_eval_mode_xpu_float32",
    "test_forward_nn_GRU_train_mode_xpu_float32",
    "test_if_train_and_eval_modes_differ_nn_GRUCell_xpu_float32",
    "test_memory_format_nn_GRUCell_xpu_float32",
    "test_memory_format_nn_GRU_eval_mode_xpu_float32",
    "test_memory_format_nn_GRU_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_GRUCell_xpu_float32",
    "test_multiple_device_transfer_nn_GRU_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_GRU_train_mode_xpu_float32",
    "test_non_contiguous_tensors_nn_GRUCell_xpu_float32",
    "test_non_contiguous_tensors_nn_GRU_eval_mode_xpu_float32",
    "test_non_contiguous_tensors_nn_GRU_train_mode_xpu_float32",

    # CUDA xfails
    # Failed: Unexpected success
    "test_memory_format_nn_AdaptiveAvgPool2d_xpu_float32",
    "test_memory_format_nn_AdaptiveAvgPool2d_xpu_float64",

    # CPU fallback fails
    # AssertionError: False is not true
    "test_memory_format_nn_ReflectionPad3d_xpu_float32",
    "test_memory_format_nn_ReflectionPad3d_xpu_float64",
    "test_memory_format_nn_ReplicationPad2d_xpu_float32",
    "test_memory_format_nn_ReplicationPad2d_xpu_float64",
    "test_memory_format_nn_ReplicationPad3d_xpu_float32",
    "test_memory_format_nn_ReplicationPad3d_xpu_float64",

    # CPU fallback fails
    # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    "test_memory_format_nn_GroupNorm_xpu_bfloat16",
    "test_memory_format_nn_GroupNorm_xpu_float16",
    "test_memory_format_nn_GroupNorm_xpu_float32",
    "test_memory_format_nn_GroupNorm_xpu_float64",

    # CPU fallback fails
    # Could not run 'aten::_thnn_fused_gru_cell' with arguments from the 'CPU' backend.
    "test_to_nn_GRUCell_swap_True_set_grad_False_xpu_float32",
    "test_to_nn_GRU_eval_mode_swap_True_set_grad_False_xpu_float32",
    "test_to_nn_GRU_train_mode_swap_True_set_grad_False_xpu_float32 ",

    # CUDA bias cases
    # AssertionError: Torch not compiled with CUDA enabled
    "test_multiple_device_transfer_nn_BCELoss_xpu_float32",
    "test_multiple_device_transfer_nn_BCELoss_xpu_float64",
    "test_multiple_device_transfer_nn_BCEWithLogitsLoss_xpu_float32",
    "test_multiple_device_transfer_nn_BCEWithLogitsLoss_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm1d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm1d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm1d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm1d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm2d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm2d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm2d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm2d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm3d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm3d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_BatchNorm3d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_BatchNorm3d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_Bilinear_xpu_float32",
    "test_multiple_device_transfer_nn_Conv1d_xpu_float32",
    "test_multiple_device_transfer_nn_Conv1d_xpu_float64",
    "test_multiple_device_transfer_nn_Conv2d_xpu_float32",
    "test_multiple_device_transfer_nn_Conv2d_xpu_float64",
    "test_multiple_device_transfer_nn_Conv3d_xpu_float32",
    "test_multiple_device_transfer_nn_Conv3d_xpu_float64",
    "test_multiple_device_transfer_nn_ConvTranspose1d_xpu_complex128",
    "test_multiple_device_transfer_nn_ConvTranspose1d_xpu_complex32",
    "test_multiple_device_transfer_nn_ConvTranspose1d_xpu_complex64",
    "test_multiple_device_transfer_nn_ConvTranspose1d_xpu_float32",
    "test_multiple_device_transfer_nn_ConvTranspose1d_xpu_float64",
    "test_multiple_device_transfer_nn_ConvTranspose2d_xpu_complex128",
    "test_multiple_device_transfer_nn_ConvTranspose2d_xpu_complex32",
    "test_multiple_device_transfer_nn_ConvTranspose2d_xpu_complex64",
    "test_multiple_device_transfer_nn_ConvTranspose2d_xpu_float32",
    "test_multiple_device_transfer_nn_ConvTranspose2d_xpu_float64",
    "test_multiple_device_transfer_nn_ConvTranspose3d_xpu_complex128",
    "test_multiple_device_transfer_nn_ConvTranspose3d_xpu_complex32",
    "test_multiple_device_transfer_nn_ConvTranspose3d_xpu_complex64",
    "test_multiple_device_transfer_nn_ConvTranspose3d_xpu_float32",
    "test_multiple_device_transfer_nn_ConvTranspose3d_xpu_float64",
    "test_multiple_device_transfer_nn_CrossEntropyLoss_xpu_float16",
    "test_multiple_device_transfer_nn_CrossEntropyLoss_xpu_float32",
    "test_multiple_device_transfer_nn_CrossEntropyLoss_xpu_float64",
    "test_multiple_device_transfer_nn_Embedding_xpu_float32",
    "test_multiple_device_transfer_nn_Embedding_xpu_float64",
    "test_multiple_device_transfer_nn_FractionalMaxPool2d_xpu_float32",
    "test_multiple_device_transfer_nn_FractionalMaxPool2d_xpu_float64",
    "test_multiple_device_transfer_nn_FractionalMaxPool3d_xpu_float32",
    "test_multiple_device_transfer_nn_FractionalMaxPool3d_xpu_float64",
    "test_multiple_device_transfer_nn_GroupNorm_xpu_bfloat16",
    "test_multiple_device_transfer_nn_GroupNorm_xpu_float16",
    "test_multiple_device_transfer_nn_GroupNorm_xpu_float32",
    "test_multiple_device_transfer_nn_GroupNorm_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm1d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm1d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm1d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm1d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm2d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm2d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm2d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm2d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm3d_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm3d_eval_mode_xpu_float64",
    "test_multiple_device_transfer_nn_InstanceNorm3d_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_InstanceNorm3d_train_mode_xpu_float64",
    "test_multiple_device_transfer_nn_LSTMCell_xpu_float32",
    "test_multiple_device_transfer_nn_LSTM_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_LSTM_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_LayerNorm_xpu_float32",
    "test_multiple_device_transfer_nn_LayerNorm_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConv1d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConv1d_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConv2d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConv2d_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConv3d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConv3d_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConvTranspose1d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConvTranspose1d_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConvTranspose2d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConvTranspose2d_xpu_float64",
    "test_multiple_device_transfer_nn_LazyConvTranspose3d_xpu_float32",
    "test_multiple_device_transfer_nn_LazyConvTranspose3d_xpu_float64",
    "test_multiple_device_transfer_nn_Linear_xpu_float32",
    "test_multiple_device_transfer_nn_MultiLabelSoftMarginLoss_xpu_float32",
    "test_multiple_device_transfer_nn_MultiLabelSoftMarginLoss_xpu_float64",
    "test_multiple_device_transfer_nn_MultiMarginLoss_xpu_float32",
    "test_multiple_device_transfer_nn_MultiMarginLoss_xpu_float64",
    "test_multiple_device_transfer_nn_MultiheadAttention_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_MultiheadAttention_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_NLLLoss_xpu_float32",
    "test_multiple_device_transfer_nn_NLLLoss_xpu_float64",
    "test_multiple_device_transfer_nn_PReLU_xpu_float32",
    "test_multiple_device_transfer_nn_PReLU_xpu_float64",
    "test_multiple_device_transfer_nn_RMSNorm_xpu_float32",
    "test_multiple_device_transfer_nn_RMSNorm_xpu_float64",
    "test_multiple_device_transfer_nn_RNNCell_xpu_float32",
    "test_multiple_device_transfer_nn_RNN_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_RNN_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_TransformerDecoderLayer_xpu_float32",
    "test_multiple_device_transfer_nn_TransformerEncoderLayer_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_TransformerEncoderLayer_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_TransformerEncoder_eval_mode_xpu_float32",
    "test_multiple_device_transfer_nn_TransformerEncoder_train_mode_xpu_float32",
    "test_multiple_device_transfer_nn_Transformer_xpu_float32",
)
res += launch_test("test_modules_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_nn
skip_list = (
    # CUDA bias cases
    # AssertionError: Torch not compiled with CUDA enabled
    "test_CTCLoss_cudnn_xpu",
    "test_ctc_loss_cudnn_xpu",
    "test_grid_sample_bfloat16_precision_xpu",
    "test_grid_sample_half_precision_xpu",
    "test_grid_sample_large_xpu",
    "test_layernorm_half_precision_xpu",
    "test_layernorm_weight_bias_xpu",
    "test_masked_softmax_devices_parity_xpu",
    # AssertionError: 'CUDA error: device-side assert triggered' not found in 'PYTORCH_API_USAGE torch.python.import\nPYTORCH_API_USAGE c10d.python.import\nPYTORCH_API_USAGE aten.init.xpu\nPYTORCH_API_USAGE tensor.create\n/home/...
    "test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu_float16",
    "test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu_float32",
    # AssertionError: MultiheadAttention does not support NestedTensor outside of its fast path. The fast path was not hit because some Tensor argument's device is neither one of cpu, cuda or privateuseone
    "test_TransformerEncoderLayer_empty_xpu",

    # oneDNN issues
    # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
    "test_GRU_grad_and_gradgrad_xpu_float64",
    "test_LSTM_grad_and_gradgrad_xpu_float64",
    "test_lstmcell_backward_only_one_output_grad_xpu_float64",
    "test_module_to_empty_xpu_float64",
    "test_rnn_fused_xpu_float64",
    "test_rnn_retain_variables_xpu_float64",
    "test_transformerencoderlayer_xpu_float64",
    "test_variable_sequence_xpu_float64",

    # CPU fallback fails
    # RuntimeError: input tensor must have at least one element, but got input_sizes = [1, 0, 1]
    "test_GroupNorm_empty_xpu",
    # AssertionError: Tensor-likes are not close!
    "test_GroupNorm_memory_format_xpu",
    "test_transformerencoderlayer_gelu_xpu_float16",
    "test_transformerencoderlayer_xpu_float16",
    # AssertionError: Scalars are not close!
    "test_InstanceNorm1d_general_xpu",
    "test_InstanceNorm2d_general_xpu",
    "test_InstanceNorm3d_general_xpu",
    # AssertionError: AssertionError not raised
    "test_batchnorm_simple_average_mixed_xpu_bfloat16",
    "test_batchnorm_simple_average_mixed_xpu_float16",
    "test_batchnorm_simple_average_xpu_float32",
    "test_batchnorm_update_stats_xpu",
    "test_batchnorm_simple_average_xpu_bfloat16",
    # AssertionError: False is not true
    "test_device_mask_xpu",
    "test_overwrite_module_params_on_conversion_cpu_device_xpu",
    # AssertionError: RuntimeError not raised
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_False_num_channels_3_mode_bicubic_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_False_num_channels_3_mode_bilinear_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_False_num_channels_5_mode_bicubic_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_False_num_channels_5_mode_bilinear_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_True_num_channels_3_mode_bicubic_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_True_num_channels_3_mode_bilinear_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_True_num_channels_5_mode_bicubic_uint8_xpu_uint8",
    "test_upsamplingBiMode2d_nonsupported_dtypes_antialias_True_num_channels_5_mode_bilinear_uint8_xpu_uint8",
    # Failed: Unexpected success
    "test_upsamplingNearest2d_launch_fail_xpu",

    # CPU fallback could not cover
    # NotImplementedError: Could not run 'aten::_thnn_fused_gru_cell' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build pro...
    "test_rnn_fused_xpu_float32",
    "test_rnn_retain_variables_xpu_float16",
    "test_rnn_retain_variables_xpu_float32",
)
res += launch_test("test_nn_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_indexing
skip_list = (
    # CPU bias cases
    # It is kernel assert on XPU implementation not exception on host.
    # We are same as CUDA implementation. And CUDA skips these cases.
    "test_trivial_fancy_out_of_bounds_xpu",
    "test_advancedindex",

    # CUDA bias case
    "test_index_put_accumulate_with_optional_tensors_xpu",
)
res += launch_test("test_indexing_xpu.py",skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_pooling
skip_list = (
    # CUDA bias case
    "test_max_pool2d_indices_xpu", # AssertionError: Torch not compiled with CUDA enabled
    "test_max_pool2d_xpu", # AssertionError: Torch not compiled with CUDA enabled

    # CPU fallback fails
    "test_pooling_bfloat16_xpu", # RuntimeError: "avg_pool3d_out_frame" not implemented for 'BFloat16'
    "test_AdaptiveMaxPool3d_indices_xpu_float16", # "adaptive_max_pool3d_cpu" not implemented for 'Half'
    "test_max_pool_nan_inf_xpu_float16", # "adaptive_max_pool3d_cpu" not implemented for 'Half'
    "test_maxpool_indices_no_batch_dim_xpu_float16", # "adaptive_max_pool3d_cpu" not implemented for 'Half'
    "test_pool_large_size_xpu_bfloat16", # "avg_pool3d_out_frame" not implemented for 'BFloat16'
    "test_pool_large_size_xpu_float16", # "avg_pool3d_out_frame" not implemented for 'Half'
    "test_adaptive_pooling_empty_output_size_xpu_float16", # "adaptive_max_pool3d_cpu" not implemented for 'Half'
)
res += launch_test("nn/test_pooling_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# nn/test_dropout
skip_list = (
    # Cannot freeze rng state. Need enhance test infrastructure to make XPU
    # compatible in freeze_rng_state.
    # https://github.com/intel/torch-xpu-ops/issues/259
    "test_Dropout1d_xpu",
    "test_Dropout3d_xpu",
)
res += launch_test("nn/test_dropout_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_tensor_creation_ops
skip_list = (
    # CPU only (vs Numpy). CUDA skips these cases since non-deterministic results are outputed for inf and nan.
    "test_float_to_int_conversion_finite_xpu_int8",
    "test_float_to_int_conversion_finite_xpu_int16",

    # sparse
    "test_tensor_ctor_device_inference_xpu",

    # Dispatch issue. It is a composite operator. But it is implemented by
    # DispatchStub. XPU doesn't support DispatchStub.
    "test_kaiser_window_xpu",

    # CUDA bias case
    "test_randperm_xpu",
)
res += launch_test("test_tensor_creation_ops_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_autocast
skip_list = (
    # Frontend API support
    # Unsupported XPU runtime functionality, '_set_cached_tensors_enabled'
    # https://github.com/intel/torch-xpu-ops/issues/223
    "test_cache_disabled",
)
res += launch_test("test_autocast_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_autograd
skip_list = (
    # Segment fault
    "test_resize_version_bump_xpu",
    # c10::NotImplementedError
    "test_autograd_composite_implicit_and_dispatch_registration_xpu",
    "test_autograd_multiple_dispatch_registrations_xpu",
    # NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend。
    "test_sparse_mask_autograd_xpu",
    "test_sparse_ctor_getter_backward_xpu_float64",
    "test_sparse_ctor_getter_backward_xpu_complex128",
    "test_sparse_backward_xpu_float64",
    "test_sparse_backward_xpu_complex128",
    # AttributeError: module 'torch.xpu' has no attribute
    "test_graph_save_on_cpu_cuda",
    "test_checkpointing_without_reentrant_memory_savings",
    # CUDA hard-code
    "test_profiler_emit_nvtx_xpu",
    # Double and complex datatype matmul is not supported in oneDNN
    "test_mv_grad_stride_0_xpu",
    # module 'torch._C' has no attribute '_scatter'
    "test_checkpointing_without_reentrant_dataparallel",
    "test_dataparallel_saved_tensors_hooks",
    # AssertionError: "none of output has requires_grad=True" does not match "PyTorch was compiled without CUDA support"
    "test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True",
    # PyTorch was compiled without CUDA support
    "test_checkpointing_non_reentrant_autocast_gpu",
    # Skip if without LAPACK
    "test_lobpcg",
    # Skip device count < 2
    "test_backward_device_xpu",
    "test_inputbuffer_add_multidevice_xpu",
    "test_unused_output_device_xpu",
    # Skip CPU case
    "test_copy__xpu",
    "test_checkpointing_non_reentrant_autocast_cpu",
    "test_per_dispatch_key_input_saving_xpu",
)
res += launch_test("test_autograd_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

# test_reductions
skip_list = (
    # CPU/CUDA bias code in aten::mode_out
    # https://github.com/intel/torch-xpu-ops/issues/327
    # RuntimeError: mode only supports CPU AND CUDA device type, got: xpu
    "test_dim_reduction",
    "test_mode",
    "test_dim_reduction_fns_fn_name_mode",

    # CUDA skips the case in opdb.
    # https://github.com/intel/torch-xpu-ops/issues/222
    "test_ref_extremal_values_mean_xpu_complex64",

    # CPU fallback fails (CPU vs Numpy).
    "test_ref_small_input_masked_prod_xpu_float16",
)
res += launch_test("test_reductions_xpu.py", skip_list=skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list=(
    # AssertionError: Jiterator is only supported on CUDA and ROCm GPUs, none are available.
    "_jiterator_",

    # CPU Fallback fails: Tensor-likes are not close!
    "test_reference_numerics_extremal__refs_acos_xpu_complex128",
    "test_reference_numerics_extremal__refs_asin_xpu_complex128",
    "test_reference_numerics_extremal__refs_asin_xpu_complex64",
    "test_reference_numerics_extremal__refs_atan_xpu_complex128",
    "test_reference_numerics_extremal__refs_atan_xpu_complex64",
    "test_reference_numerics_extremal__refs_exp2_xpu_complex128",
    "test_reference_numerics_extremal__refs_exp2_xpu_complex64",
    "test_reference_numerics_extremal__refs_nn_functional_tanhshrink_xpu_complex64",
    "test_reference_numerics_extremal_acos_xpu_complex128",
    "test_reference_numerics_extremal_asin_xpu_complex128",
    "test_reference_numerics_extremal_asin_xpu_complex64",
    "test_reference_numerics_extremal_atan_xpu_complex128",
    "test_reference_numerics_extremal_atan_xpu_complex64",
    "test_reference_numerics_extremal_exp2_xpu_complex128",
    "test_reference_numerics_extremal_exp2_xpu_complex64",
    "test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex64",
    "test_reference_numerics_large__refs_atan_xpu_complex128",
    "test_reference_numerics_large__refs_atan_xpu_complex64",
    "test_reference_numerics_large_atan_xpu_complex128",
    "test_reference_numerics_large_atan_xpu_complex64",
    "test_reference_numerics_normal__refs_nn_functional_tanhshrink_xpu_complex64",
    "test_reference_numerics_normal_nn_functional_tanhshrink_xpu_complex64",
    "test_reference_numerics_small__refs_atan_xpu_complex128",
    "test_reference_numerics_small__refs_atan_xpu_complex64",
    "test_reference_numerics_small_atan_xpu_complex128",
    "test_reference_numerics_small_atan_xpu_complex64",
    "test_reference_numerics_large__refs_atan_xpu_complex32",
    "test_reference_numerics_large__refs_tanh_xpu_complex32",
    "test_reference_numerics_large_tanh_xpu_complex32",
    "test_reference_numerics_small__refs_atan_xpu_complex32",

    # For extreme value processing, Numpy and XPU results are inconsistent
    "test_reference_numerics_extremal__refs_log_xpu_complex64",
    "test_reference_numerics_extremal_log_xpu_complex64",
    "test_reference_numerics_extremal__refs_tanh_xpu_complex128",
    "test_reference_numerics_extremal__refs_tanh_xpu_complex64",
    "test_reference_numerics_extremal_tanh_xpu_complex128",
    "test_reference_numerics_extremal_tanh_xpu_complex64",

    # CPU Fallback fails
    # New ATen operators fails on CPU Fallback.
    # E.g. aten::special_spherical_bessel_j0, aten::special_airy_ai.
    "_special_",

    # Failed: Unexpected success
    "test_reference_numerics_large__refs_rsqrt_xpu_complex32",
    "test_reference_numerics_large_rsqrt_xpu_complex32",

    # RuntimeError: "sigmoid_xpu" not implemented for 'ComplexHalf'
    "test_batch_vs_slicing_sigmoid_xpu_complex32",
    "test_contig_size1_large_dim_sigmoid_xpu_complex32",
    "test_contig_size1_sigmoid_xpu_complex32",
    "test_contig_vs_every_other_sigmoid_xpu_complex32",
    "test_contig_vs_transposed_sigmoid_xpu_complex32",
    "test_non_contig_expand_sigmoid_xpu_complex32",
    "test_non_contig_index_sigmoid_xpu_complex32",
    "test_non_contig_sigmoid_xpu_complex32",
    "test_reference_numerics_normal_sigmoid_xpu_complex32",
    "test_reference_numerics_small_sigmoid_xpu_complex32",
)
res += launch_test("test_unary_ufuncs_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list=(
    # RuntimeError: is_coalesced expected sparse coordinate tensor layout but got Sparse.
    "test_mask_layout_sparse_coo_masked_amax_xpu_bfloat16",
    "test_mask_layout_sparse_coo_masked_amax_xpu_float16",
    "test_mask_layout_sparse_coo_masked_amax_xpu_float32",
    "test_mask_layout_sparse_coo_masked_amax_xpu_float64",
    "test_mask_layout_sparse_coo_masked_amin_xpu_bfloat16",
    "test_mask_layout_sparse_coo_masked_amin_xpu_float16",
    "test_mask_layout_sparse_coo_masked_amin_xpu_float32",
    "test_mask_layout_sparse_coo_masked_amin_xpu_float64",
    "test_mask_layout_sparse_coo_masked_prod_xpu_bfloat16",
    "test_mask_layout_sparse_coo_masked_prod_xpu_bool",
    "test_mask_layout_sparse_coo_masked_prod_xpu_complex128",
    "test_mask_layout_sparse_coo_masked_prod_xpu_complex64",
    "test_mask_layout_sparse_coo_masked_prod_xpu_float16",
    "test_mask_layout_sparse_coo_masked_prod_xpu_float32",
    "test_mask_layout_sparse_coo_masked_prod_xpu_float64",
    "test_mask_layout_sparse_coo_masked_prod_xpu_int16",
    "test_mask_layout_sparse_coo_masked_prod_xpu_int32",
    "test_mask_layout_sparse_coo_masked_prod_xpu_int64",
    "test_mask_layout_sparse_coo_masked_prod_xpu_int8",
    "test_mask_layout_sparse_coo_masked_prod_xpu_uint8",
    # NotImplementedError: Could not run 'aten::_values' with arguments from the 'SparseXPU' backend. 
    "test_mask_layout_sparse_coo_masked_sum_xpu_bfloat16",
    "test_mask_layout_sparse_coo_masked_sum_xpu_bool",
    "test_mask_layout_sparse_coo_masked_sum_xpu_complex128",
    "test_mask_layout_sparse_coo_masked_sum_xpu_complex64",
    "test_mask_layout_sparse_coo_masked_sum_xpu_float16",
    "test_mask_layout_sparse_coo_masked_sum_xpu_float32",
    "test_mask_layout_sparse_coo_masked_sum_xpu_float64",
    "test_mask_layout_sparse_coo_masked_sum_xpu_int16",
    "test_mask_layout_sparse_coo_masked_sum_xpu_int32",
    "test_mask_layout_sparse_coo_masked_sum_xpu_int64",
    "test_mask_layout_sparse_coo_masked_sum_xpu_int8",
    "test_mask_layout_sparse_coo_masked_sum_xpu_uint8",
    # CPU and CUDA bias code in SparseCsrTensor.cpp.
    # RuntimeError: device type of values (xpu) must be CPU or CUDA or Meta : 
    "test_mask_layout_sparse_csr_masked_amax_xpu_bfloat16",
    "test_mask_layout_sparse_csr_masked_amax_xpu_float16",
    "test_mask_layout_sparse_csr_masked_amax_xpu_float32",
    "test_mask_layout_sparse_csr_masked_amax_xpu_float64",
    "test_mask_layout_sparse_csr_masked_amin_xpu_bfloat16",
    "test_mask_layout_sparse_csr_masked_amin_xpu_float16",
    "test_mask_layout_sparse_csr_masked_amin_xpu_float32",
    "test_mask_layout_sparse_csr_masked_amin_xpu_float64",
    "test_mask_layout_sparse_csr_masked_mean_xpu_bfloat16",
    "test_mask_layout_sparse_csr_masked_mean_xpu_float16",
    "test_mask_layout_sparse_csr_masked_mean_xpu_float32",
    "test_mask_layout_sparse_csr_masked_mean_xpu_float64",
    "test_mask_layout_sparse_csr_masked_prod_xpu_bfloat16",
    "test_mask_layout_sparse_csr_masked_prod_xpu_bool",
    "test_mask_layout_sparse_csr_masked_prod_xpu_complex128",
    "test_mask_layout_sparse_csr_masked_prod_xpu_complex64",
    "test_mask_layout_sparse_csr_masked_prod_xpu_float16",
    "test_mask_layout_sparse_csr_masked_prod_xpu_float32",
    "test_mask_layout_sparse_csr_masked_prod_xpu_float64",
    "test_mask_layout_sparse_csr_masked_prod_xpu_int16",
    "test_mask_layout_sparse_csr_masked_prod_xpu_int32",
    "test_mask_layout_sparse_csr_masked_prod_xpu_int64",
    "test_mask_layout_sparse_csr_masked_prod_xpu_int8",
    "test_mask_layout_sparse_csr_masked_prod_xpu_uint8",
    "test_mask_layout_sparse_csr_masked_sum_xpu_bfloat16",
    "test_mask_layout_sparse_csr_masked_sum_xpu_bool",
    "test_mask_layout_sparse_csr_masked_sum_xpu_complex128",
    "test_mask_layout_sparse_csr_masked_sum_xpu_complex64",
    "test_mask_layout_sparse_csr_masked_sum_xpu_float16",
    "test_mask_layout_sparse_csr_masked_sum_xpu_float32",
    "test_mask_layout_sparse_csr_masked_sum_xpu_float64",
    "test_mask_layout_sparse_csr_masked_sum_xpu_int16",
    "test_mask_layout_sparse_csr_masked_sum_xpu_int32",
    "test_mask_layout_sparse_csr_masked_sum_xpu_int64",
    "test_mask_layout_sparse_csr_masked_sum_xpu_int8",
    "test_mask_layout_sparse_csr_masked_sum_xpu_uint8",
    "test_mask_layout_strided_masked_mean_xpu_bfloat16",
    "test_mask_layout_strided_masked_mean_xpu_float16",
    "test_mask_layout_strided_masked_mean_xpu_float32",
    "test_mask_layout_strided_masked_mean_xpu_float64",
    # NotImplementedError: Could not run 'aten::_to_dense' with arguments from the 'SparseXPU' backend. 
    "test_mask_layout_strided_masked_amax_xpu_bfloat16",
    "test_mask_layout_strided_masked_amax_xpu_float16",
    "test_mask_layout_strided_masked_amax_xpu_float32",
    "test_mask_layout_strided_masked_amax_xpu_float64",
    "test_mask_layout_strided_masked_amin_xpu_bfloat16",
    "test_mask_layout_strided_masked_amin_xpu_float16",
    "test_mask_layout_strided_masked_amin_xpu_float32",
    "test_mask_layout_strided_masked_amin_xpu_float64",
    "test_mask_layout_strided_masked_prod_xpu_bfloat16",
    "test_mask_layout_strided_masked_prod_xpu_bool",
    "test_mask_layout_strided_masked_prod_xpu_complex128",
    "test_mask_layout_strided_masked_prod_xpu_complex64",
    "test_mask_layout_strided_masked_prod_xpu_float16",
    "test_mask_layout_strided_masked_prod_xpu_float32",
    "test_mask_layout_strided_masked_prod_xpu_float64",
    "test_mask_layout_strided_masked_prod_xpu_int16",
    "test_mask_layout_strided_masked_prod_xpu_int32",
    "test_mask_layout_strided_masked_prod_xpu_int64",
    "test_mask_layout_strided_masked_prod_xpu_int8",
    "test_mask_layout_strided_masked_prod_xpu_uint8",
    "test_mask_layout_strided_masked_sum_xpu_bfloat16",
    "test_mask_layout_strided_masked_sum_xpu_bool",
    "test_mask_layout_strided_masked_sum_xpu_complex128",
    "test_mask_layout_strided_masked_sum_xpu_complex64",
    "test_mask_layout_strided_masked_sum_xpu_float16",
    "test_mask_layout_strided_masked_sum_xpu_float32",
    "test_mask_layout_strided_masked_sum_xpu_float64",
    "test_mask_layout_strided_masked_sum_xpu_int16",
    "test_mask_layout_strided_masked_sum_xpu_int32",
    "test_mask_layout_strided_masked_sum_xpu_int64",
    "test_mask_layout_strided_masked_sum_xpu_int8",
    "test_mask_layout_strided_masked_sum_xpu_uint8",
)
res += launch_test("test_masked_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = ( 
    # Need quantization support, NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend. 
    "test_flatten_xpu",
    "test_ravel_xpu",
)
res += launch_test("./test_view_ops_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = ( 
        # Need quantization support.
        # https://github.com/intel/torch-xpu-ops/issues/275
        # NotImplementedError: Could not run 'aten::empty_quantized' with arguments from the 'QuantizedXPU' backend. 
        "test_flip_xpu_float32",
        # RuntimeError: "trace" not implemented for 'Half'
        "test_trace_xpu_float16",
)
res += launch_test("test_shape_ops_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("test_content_store_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("test_native_functions_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("nn/test_init_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("test_namedtensor_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("nn/test_lazy_modules_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list=(
# RuntimeError: Double and complex datatype matmul is not supported in oneDNN 
    "test_1_sized_with_0_strided_xpu_float64",
    "test_addbmm_xpu_complex128",
    "test_addbmm_xpu_complex64",
    "test_addbmm_xpu_float64",
    "test_addmm_gelu_xpu_float64",
    "test_addmm_relu_xpu_float64",
    "test_addmm_sizes_xpu_float64",
    "test_addmm_xpu_complex128",
    "test_addmm_xpu_complex64",
    "test_addmm_xpu_float64",
    "test_addmv_rowmajor_colmajor_incx_incy_lda_xpu_float64",
    "test_addmv_xpu_complex128",
    "test_addmv_xpu_complex64",
    "test_addmv_xpu_float64",
    "test_baddbmm_xpu_complex128",
    "test_baddbmm_xpu_complex64",
    "test_baddbmm_xpu_float64",
    "test_bmm_xpu_complex128",
    "test_bmm_xpu_complex64",
    "test_bmm_xpu_float64",
    "test_cholesky_errors_and_warnings_xpu_complex128",
    "test_cholesky_errors_and_warnings_xpu_complex64",
    "test_cholesky_errors_and_warnings_xpu_float64",
    "test_cholesky_ex_xpu_complex128",
    "test_cholesky_ex_xpu_complex64",
    "test_cholesky_ex_xpu_float64",
    "test_cholesky_inverse_xpu_complex128",
    "test_cholesky_inverse_xpu_complex64",
    "test_cholesky_inverse_xpu_float64",
    "test_cholesky_solve_backward_xpu_float64",
    "test_cholesky_solve_batched_many_batches_xpu_complex128",
    "test_cholesky_solve_batched_many_batches_xpu_complex64",
    "test_cholesky_solve_batched_many_batches_xpu_float64",
    "test_cholesky_solve_batched_xpu_complex128",
    "test_cholesky_solve_batched_xpu_complex64",
    "test_cholesky_solve_batched_xpu_float64",
    "test_cholesky_solve_xpu_complex128",
    "test_cholesky_solve_xpu_complex64",
    "test_cholesky_solve_xpu_float64",
    "test_cholesky_xpu_complex128",
    "test_cholesky_xpu_complex64",
    "test_cholesky_xpu_float64",
    "test_corner_cases_of_cublasltmatmul_xpu_complex128",
    "test_corner_cases_of_cublasltmatmul_xpu_complex64",
    "test_corner_cases_of_cublasltmatmul_xpu_float64",
    "test_det_logdet_slogdet_batched_xpu_float64",
    "test_det_logdet_slogdet_xpu_float64",
    "test_eig_check_magma_xpu_float32",
    "test_einsum_random_xpu_complex128",
    "test_einsum_random_xpu_float64",
    "test_einsum_sublist_format_xpu_complex128",
    "test_einsum_sublist_format_xpu_float64",
    "test_einsum_xpu_complex128",
    "test_einsum_xpu_float64",
    "test_inner_xpu_complex64",
    "test_inverse_many_batches_xpu_complex128",
    "test_inverse_many_batches_xpu_complex64",
    "test_inverse_many_batches_xpu_float64",
    "test_inverse_xpu_complex128",
    "test_inverse_xpu_complex64",
    "test_inverse_xpu_float64",
    "test_ldl_factor_xpu_complex128",
    "test_ldl_factor_xpu_complex64",
    "test_ldl_factor_xpu_float64",
    "test_ldl_solve_xpu_complex128",
    "test_ldl_solve_xpu_complex64",
    "test_ldl_solve_xpu_float64",
    "test_linalg_lstsq_batch_broadcasting_xpu_complex128",
    "test_linalg_lstsq_batch_broadcasting_xpu_complex64",
    "test_linalg_lstsq_batch_broadcasting_xpu_float64",
    "test_linalg_lstsq_xpu_complex128",
    "test_linalg_lstsq_xpu_complex64",
    "test_linalg_lstsq_xpu_float64",
    "test_linalg_lu_family_xpu_complex128",
    "test_linalg_lu_family_xpu_complex64",
    "test_linalg_lu_family_xpu_float64",
    "test_linalg_lu_solve_xpu_complex128",
    "test_linalg_lu_solve_xpu_complex64",
    "test_linalg_lu_solve_xpu_float64",
    "test_linalg_solve_triangular_broadcasting_xpu_complex128",
    "test_linalg_solve_triangular_broadcasting_xpu_complex64",
    "test_linalg_solve_triangular_broadcasting_xpu_float64",
    "test_linalg_solve_triangular_large_xpu_complex128",
    "test_linalg_solve_triangular_large_xpu_complex64",
    "test_linalg_solve_triangular_large_xpu_float64",
    "test_linalg_solve_triangular_xpu_complex128",
    "test_linalg_solve_triangular_xpu_complex64",
    "test_linalg_solve_triangular_xpu_float64",
    "test_lobpcg_basic_xpu_float64",
    "test_lobpcg_ortho_xpu_float64",
    "test_lu_solve_batched_broadcasting_xpu_complex128",
    "test_lu_solve_batched_broadcasting_xpu_complex64",
    "test_lu_solve_batched_broadcasting_xpu_float64",
    "test_lu_solve_batched_many_batches_xpu_complex128",
    "test_lu_solve_batched_many_batches_xpu_complex64",
    "test_lu_solve_batched_many_batches_xpu_float64",
    "test_lu_solve_batched_xpu_complex128",
    "test_lu_solve_batched_xpu_complex64",
    "test_lu_solve_batched_xpu_float64",
    "test_lu_solve_large_matrices_xpu_complex128",
    "test_lu_solve_large_matrices_xpu_complex64",
    "test_lu_solve_large_matrices_xpu_float64",
    "test_lu_solve_xpu_complex128",
    "test_lu_solve_xpu_complex64",
    "test_lu_solve_xpu_float64",
    "test_matmul_out_kernel_errors_with_autograd_xpu_complex64",
    "test_matmul_small_brute_force_1d_Nd_xpu_complex64",
    "test_matmul_small_brute_force_2d_Nd_xpu_complex64",
    "test_matmul_small_brute_force_3d_Nd_xpu_complex64",
    "test_matrix_power_negative_xpu_complex128",
    "test_matrix_power_negative_xpu_float64",
    "test_matrix_power_non_negative_xpu_complex128",
    "test_matrix_power_non_negative_xpu_float64",
    "test_matrix_rank_atol_rtol_xpu_float64",
    "test_matrix_rank_xpu_complex128",
    "test_matrix_rank_xpu_complex64",
    "test_matrix_rank_xpu_float64",
    "test_mm_bmm_non_memory_dense_xpu",
    "test_mm_conjtranspose_xpu",
    "test_mm_xpu_complex128",
    "test_mm_xpu_complex64",
    "test_mm_xpu_float64",
    "test_multi_dot_xpu_complex128",
    "test_multi_dot_xpu_float64",
    "test_old_cholesky_batched_many_batches_xpu_float64",
    "test_old_cholesky_batched_upper_xpu_complex128",
    "test_old_cholesky_batched_upper_xpu_complex64",
    "test_old_cholesky_batched_upper_xpu_float64",
    "test_old_cholesky_batched_xpu_complex128",
    "test_old_cholesky_batched_xpu_complex64",
    "test_old_cholesky_batched_xpu_float64",
    "test_old_cholesky_xpu_complex128",
    "test_old_cholesky_xpu_complex64",
    "test_old_cholesky_xpu_float64",
    "test_ormqr_xpu_complex128",
    "test_ormqr_xpu_complex64",
    "test_ormqr_xpu_float64",
    "test_pca_lowrank_xpu",
    "test_pinv_errors_and_warnings_xpu_complex128",
    "test_pinv_errors_and_warnings_xpu_complex64",
    "test_pinv_errors_and_warnings_xpu_float64",
    "test_pinv_xpu_complex128",
    "test_pinv_xpu_complex64",
    "test_pinv_xpu_float64",
    "test_pinverse_xpu_complex128",
    "test_pinverse_xpu_complex64",
    "test_pinverse_xpu_float64",
    "test_slogdet_xpu_complex128",
    "test_slogdet_xpu_complex64",
    "test_slogdet_xpu_float64",
    "test_solve_batched_broadcasting_xpu_complex128",
    "test_solve_batched_broadcasting_xpu_complex64",
    "test_solve_batched_broadcasting_xpu_float64",
    "test_solve_xpu_complex128",
    "test_solve_xpu_complex64",
    "test_solve_xpu_float64",
    "test_strided_mm_bmm_xpu_float64",
    "test_svd_lowrank_xpu_complex128",
    "test_svd_lowrank_xpu_float64",
    "test_svd_xpu_complex128",
    "test_svd_xpu_complex64",
    "test_svd_xpu_float64",
    "test_triangular_solve_batched_broadcasting_xpu_complex128",
    "test_triangular_solve_batched_broadcasting_xpu_complex64",
    "test_triangular_solve_batched_broadcasting_xpu_float64",
    "test_triangular_solve_batched_many_batches_xpu_complex128",
    "test_triangular_solve_batched_many_batches_xpu_complex64",
    "test_triangular_solve_batched_many_batches_xpu_float64",
    "test_triangular_solve_batched_xpu_complex128",
    "test_triangular_solve_batched_xpu_complex64",
    "test_triangular_solve_batched_xpu_float64",
    "test_triangular_solve_xpu_complex128",
    "test_triangular_solve_xpu_complex64",
    "test_triangular_solve_xpu_float64",
# https://github.com/intel/torch-xpu-ops/issues/317
# addmm.out, addmv.out, addr, linalg_lstsq, linalg_vector_norm.out, norm.out, vdot&dot lack XPU support and fallback to CPU 
    "test_addmm_sizes_xpu_complex128",
    "test_addmm_sizes_xpu_complex64",
    "test_blas_alpha_beta_empty_xpu_complex128",
    "test_blas_alpha_beta_empty_xpu_complex64",
    "test_addr_float_and_complex_xpu_bfloat16",
    "test_addr_float_and_complex_xpu_complex128",
    "test_addr_float_and_complex_xpu_complex64",
    "test_addr_float_and_complex_xpu_float16",
    "test_addr_float_and_complex_xpu_float32",
    "test_addr_float_and_complex_xpu_float64",
    "test_addr_integral_xpu_int16",
    "test_addr_integral_xpu_int32",
    "test_addr_integral_xpu_int64",
    "test_addr_integral_xpu_int8",
    "test_addr_integral_xpu_uint8",
    "test_linalg_lstsq_input_checks_xpu_complex128",
    "test_linalg_lstsq_input_checks_xpu_complex64",
    "test_linalg_lstsq_input_checks_xpu_float32",
    "test_linalg_lstsq_input_checks_xpu_float64",
    "test_norm_fused_type_promotion_xpu_bfloat16",
    "test_norm_fused_type_promotion_xpu_float16",
    "test_dot_invalid_args_xpu",
    "test_vdot_invalid_args_xpu",
# xpu does not have '_cuda_tunableop_is_enabled' API
    "test_matmul_small_brute_force_tunableop_xpu_float16",
    "test_matmul_small_brute_force_tunableop_xpu_float32",
    "test_matmul_small_brute_force_tunableop_xpu_float64",
# these case passed in a env with triton, but triton did not install in pre-ci. 
    "test_compile_int4_mm_m_32_k_32_n_48_xpu",
    "test_compile_int4_mm_m_32_k_32_n_64_xpu",
    "test_compile_int4_mm_m_32_k_64_n_48_xpu",
    "test_compile_int4_mm_m_32_k_64_n_64_xpu",
    "test_compile_int4_mm_m_64_k_32_n_48_xpu",
    "test_compile_int4_mm_m_64_k_32_n_64_xpu",
    "test_compile_int4_mm_m_64_k_64_n_48_xpu",
    "test_compile_int4_mm_m_64_k_64_n_64_xpu",
    )
res += launch_test("test_linalg_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list=(
#RuntimeError: Double and complex datatype matmul is not supported in oneDNN 
    "test_fn_fwgrad_bwgrad___rmatmul___xpu_complex128",
    "test_fn_fwgrad_bwgrad___rmatmul___xpu_float64",
    "test_fn_fwgrad_bwgrad_addbmm_xpu_float64",
    "test_fn_fwgrad_bwgrad_addmm_decomposed_xpu_complex128",
    "test_fn_fwgrad_bwgrad_addmm_decomposed_xpu_float64",
    "test_fn_fwgrad_bwgrad_addmm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_addmm_xpu_float64",
    "test_fn_fwgrad_bwgrad_addmv_xpu_complex128",
    "test_fn_fwgrad_bwgrad_addmv_xpu_float64",
    "test_fn_fwgrad_bwgrad_addr_xpu_complex128",
    "test_fn_fwgrad_bwgrad_addr_xpu_float64",
    "test_fn_fwgrad_bwgrad_baddbmm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_baddbmm_xpu_float64",
    "test_fn_fwgrad_bwgrad_bmm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_bmm_xpu_float64",
    "test_fn_fwgrad_bwgrad_cholesky_inverse_xpu_complex128",
    "test_fn_fwgrad_bwgrad_cholesky_inverse_xpu_float64",
    "test_fn_fwgrad_bwgrad_cholesky_solve_xpu_complex128",
    "test_fn_fwgrad_bwgrad_cholesky_solve_xpu_float64",
    "test_fn_fwgrad_bwgrad_cholesky_xpu_complex128",
    "test_fn_fwgrad_bwgrad_cholesky_xpu_float64",
    "test_fn_fwgrad_bwgrad_corrcoef_xpu_complex128",
    "test_fn_fwgrad_bwgrad_corrcoef_xpu_float64",
    "test_fn_fwgrad_bwgrad_einsum_xpu_complex128",
    "test_fn_fwgrad_bwgrad_einsum_xpu_float64",
    "test_fn_fwgrad_bwgrad_inner_xpu_complex128",
    "test_fn_fwgrad_bwgrad_inner_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_cholesky_ex_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_cholesky_ex_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_cholesky_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_cholesky_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_cond_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_cond_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_det_singular_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_det_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_det_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_eig_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_eig_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_eigh_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_eigh_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_eigvals_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_eigvals_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_eigvalsh_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_eigvalsh_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_householder_product_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_householder_product_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_inv_ex_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_inv_ex_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_inv_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_inv_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_lstsq_grad_oriented_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_lstsq_grad_oriented_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_lu_factor_ex_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_lu_factor_ex_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_lu_factor_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_lu_factor_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_lu_solve_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_lu_solve_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_lu_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_lu_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_matrix_norm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_matrix_norm_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_matrix_power_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_matrix_power_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_multi_dot_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_multi_dot_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_norm_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_pinv_hermitian_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_pinv_hermitian_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_pinv_singular_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_pinv_singular_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_pinv_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_pinv_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_qr_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_qr_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_slogdet_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_slogdet_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_solve_ex_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_solve_ex_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_solve_triangular_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_solve_triangular_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_solve_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_solve_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_svd_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_svd_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_svdvals_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_svdvals_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_tensorinv_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_tensorinv_xpu_float64",
    "test_fn_fwgrad_bwgrad_linalg_tensorsolve_xpu_complex128",
    "test_fn_fwgrad_bwgrad_linalg_tensorsolve_xpu_float64",
    "test_fn_fwgrad_bwgrad_logdet_xpu_complex128",
    "test_fn_fwgrad_bwgrad_logdet_xpu_float64",
    "test_fn_fwgrad_bwgrad_lu_solve_xpu_complex128",
    "test_fn_fwgrad_bwgrad_lu_solve_xpu_float64",
    "test_fn_fwgrad_bwgrad_lu_xpu_complex128",
    "test_fn_fwgrad_bwgrad_lu_xpu_float64",
    "test_fn_fwgrad_bwgrad_matmul_xpu_complex128",
    "test_fn_fwgrad_bwgrad_matmul_xpu_float64",
    "test_fn_fwgrad_bwgrad_mm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_mm_xpu_float64",
    "test_fn_fwgrad_bwgrad_mv_xpu_complex128",
    "test_fn_fwgrad_bwgrad_mv_xpu_float64",
    "test_fn_fwgrad_bwgrad_nn_functional_bilinear_xpu_float64",
    "test_fn_fwgrad_bwgrad_nn_functional_linear_xpu_complex128",
    "test_fn_fwgrad_bwgrad_nn_functional_linear_xpu_float64",
    "test_fn_fwgrad_bwgrad_nn_functional_multi_head_attention_forward_xpu_float64",
    "test_fn_fwgrad_bwgrad_nn_functional_scaled_dot_product_attention_xpu_float64",
    "test_fn_fwgrad_bwgrad_norm_nuc_xpu_complex128",
    "test_fn_fwgrad_bwgrad_norm_nuc_xpu_float64",
    "test_fn_fwgrad_bwgrad_ormqr_xpu_complex128",
    "test_fn_fwgrad_bwgrad_ormqr_xpu_float64",
    "test_fn_fwgrad_bwgrad_pca_lowrank_xpu_float64",
    "test_fn_fwgrad_bwgrad_pinverse_xpu_complex128",
    "test_fn_fwgrad_bwgrad_pinverse_xpu_float64",
    "test_fn_fwgrad_bwgrad_qr_xpu_complex128",
    "test_fn_fwgrad_bwgrad_qr_xpu_float64",
    "test_fn_fwgrad_bwgrad_svd_lowrank_xpu_float64",
    "test_fn_fwgrad_bwgrad_svd_xpu_complex128",
    "test_fn_fwgrad_bwgrad_svd_xpu_float64",
    "test_fn_fwgrad_bwgrad_tensordot_xpu_complex128",
    "test_fn_fwgrad_bwgrad_tensordot_xpu_float64",
    "test_forward_mode_AD___rmatmul___xpu_complex128",
    "test_forward_mode_AD___rmatmul___xpu_float64",
    "test_forward_mode_AD_addbmm_xpu_float64",
    "test_forward_mode_AD_addmm_decomposed_xpu_complex128",
    "test_forward_mode_AD_addmm_decomposed_xpu_float64",
    "test_forward_mode_AD_addmm_xpu_complex128",
    "test_forward_mode_AD_addmm_xpu_float64",
    "test_forward_mode_AD_addmv_xpu_complex128",
    "test_forward_mode_AD_addmv_xpu_float64",
    "test_forward_mode_AD_baddbmm_xpu_complex128",
    "test_forward_mode_AD_baddbmm_xpu_float64",
    "test_forward_mode_AD_bmm_xpu_complex128",
    "test_forward_mode_AD_bmm_xpu_float64",
    "test_forward_mode_AD_cholesky_inverse_xpu_complex128",
    "test_forward_mode_AD_cholesky_inverse_xpu_float64",
    "test_forward_mode_AD_cholesky_solve_xpu_complex128",
    "test_forward_mode_AD_cholesky_solve_xpu_float64",
    "test_forward_mode_AD_cholesky_xpu_complex128",
    "test_forward_mode_AD_cholesky_xpu_float64",
    "test_forward_mode_AD_corrcoef_xpu_complex128",
    "test_forward_mode_AD_corrcoef_xpu_float64",
    "test_forward_mode_AD_dot_xpu_complex128",
    "test_forward_mode_AD_dot_xpu_float64",
    "test_forward_mode_AD_einsum_xpu_complex128",
    "test_forward_mode_AD_einsum_xpu_float64",
    "test_forward_mode_AD_inner_xpu_complex128",
    "test_forward_mode_AD_inner_xpu_float64",
    "test_forward_mode_AD_linalg_cholesky_ex_xpu_complex128",
    "test_forward_mode_AD_linalg_cholesky_ex_xpu_float64",
    "test_forward_mode_AD_linalg_cholesky_xpu_complex128",
    "test_forward_mode_AD_linalg_cholesky_xpu_float64",
    "test_forward_mode_AD_linalg_cond_xpu_complex128",
    "test_forward_mode_AD_linalg_cond_xpu_float64",
    "test_forward_mode_AD_linalg_det_singular_xpu_complex128",
    "test_forward_mode_AD_linalg_det_singular_xpu_float64",
    "test_forward_mode_AD_linalg_det_xpu_complex128",
    "test_forward_mode_AD_linalg_det_xpu_float64",
    "test_forward_mode_AD_linalg_eig_xpu_complex128",
    "test_forward_mode_AD_linalg_eig_xpu_float64",
    "test_forward_mode_AD_linalg_eigh_xpu_complex128",
    "test_forward_mode_AD_linalg_eigh_xpu_float64",
    "test_forward_mode_AD_linalg_eigvals_xpu_complex128",
    "test_forward_mode_AD_linalg_eigvals_xpu_float64",
    "test_forward_mode_AD_linalg_eigvalsh_xpu_complex128",
    "test_forward_mode_AD_linalg_eigvalsh_xpu_float64",
    "test_forward_mode_AD_linalg_householder_product_xpu_complex128",
    "test_forward_mode_AD_linalg_householder_product_xpu_float64",
    "test_forward_mode_AD_linalg_inv_ex_xpu_complex128",
    "test_forward_mode_AD_linalg_inv_ex_xpu_float64",
    "test_forward_mode_AD_linalg_inv_xpu_complex128",
    "test_forward_mode_AD_linalg_inv_xpu_float64",
    "test_forward_mode_AD_linalg_lstsq_grad_oriented_xpu_complex128",
    "test_forward_mode_AD_linalg_lstsq_grad_oriented_xpu_float64",
    "test_forward_mode_AD_linalg_lu_factor_ex_xpu_complex128",
    "test_forward_mode_AD_linalg_lu_factor_ex_xpu_float64",
    "test_forward_mode_AD_linalg_lu_factor_xpu_complex128",
    "test_forward_mode_AD_linalg_lu_factor_xpu_float64",
    "test_forward_mode_AD_linalg_lu_solve_xpu_complex128",
    "test_forward_mode_AD_linalg_lu_solve_xpu_float64",
    "test_forward_mode_AD_linalg_lu_xpu_complex128",
    "test_forward_mode_AD_linalg_lu_xpu_float64",
    "test_forward_mode_AD_linalg_matrix_norm_xpu_complex128",
    "test_forward_mode_AD_linalg_matrix_norm_xpu_float64",
    "test_forward_mode_AD_linalg_matrix_power_xpu_complex128",
    "test_forward_mode_AD_linalg_matrix_power_xpu_float64",
    "test_forward_mode_AD_linalg_multi_dot_xpu_complex128",
    "test_forward_mode_AD_linalg_multi_dot_xpu_float64",
    "test_forward_mode_AD_linalg_norm_xpu_float64",
    "test_forward_mode_AD_linalg_pinv_hermitian_xpu_complex128",
    "test_forward_mode_AD_linalg_pinv_hermitian_xpu_float64",
    "test_forward_mode_AD_linalg_pinv_singular_xpu_complex128",
    "test_forward_mode_AD_linalg_pinv_singular_xpu_float64",
    "test_forward_mode_AD_linalg_pinv_xpu_complex128",
    "test_forward_mode_AD_linalg_pinv_xpu_float64",
    "test_forward_mode_AD_linalg_qr_xpu_complex128",
    "test_forward_mode_AD_linalg_qr_xpu_float64",
    "test_forward_mode_AD_linalg_slogdet_xpu_complex128",
    "test_forward_mode_AD_linalg_slogdet_xpu_float64",
    "test_forward_mode_AD_linalg_solve_ex_xpu_complex128",
    "test_forward_mode_AD_linalg_solve_ex_xpu_float64",
    "test_forward_mode_AD_linalg_solve_triangular_xpu_complex128",
    "test_forward_mode_AD_linalg_solve_triangular_xpu_float64",
    "test_forward_mode_AD_linalg_solve_xpu_complex128",
    "test_forward_mode_AD_linalg_solve_xpu_float64",
    "test_forward_mode_AD_linalg_svd_xpu_complex128",
    "test_forward_mode_AD_linalg_svd_xpu_float64",
    "test_forward_mode_AD_linalg_svdvals_xpu_complex128",
    "test_forward_mode_AD_linalg_svdvals_xpu_float64",
    "test_forward_mode_AD_linalg_tensorinv_xpu_complex128",
    "test_forward_mode_AD_linalg_tensorinv_xpu_float64",
    "test_forward_mode_AD_linalg_tensorsolve_xpu_complex128",
    "test_forward_mode_AD_linalg_tensorsolve_xpu_float64",
    "test_forward_mode_AD_logdet_xpu_complex128",
    "test_forward_mode_AD_logdet_xpu_float64",
    "test_forward_mode_AD_lu_solve_xpu_complex128",
    "test_forward_mode_AD_lu_solve_xpu_float64",
    "test_forward_mode_AD_lu_xpu_complex128",
    "test_forward_mode_AD_lu_xpu_float64",
    "test_forward_mode_AD_matmul_xpu_complex128",
    "test_forward_mode_AD_matmul_xpu_float64",
    "test_forward_mode_AD_mm_xpu_complex128",
    "test_forward_mode_AD_mm_xpu_float64",
    "test_forward_mode_AD_mv_xpu_complex128",
    "test_forward_mode_AD_mv_xpu_float64",
    "test_forward_mode_AD_nn_functional_bilinear_xpu_float64",
    "test_forward_mode_AD_nn_functional_linear_xpu_complex128",
    "test_forward_mode_AD_nn_functional_linear_xpu_float64",
    "test_forward_mode_AD_norm_nuc_xpu_complex128",
    "test_forward_mode_AD_norm_nuc_xpu_float64",
    "test_forward_mode_AD_pca_lowrank_xpu_float64",
    "test_forward_mode_AD_pinverse_xpu_complex128",
    "test_forward_mode_AD_pinverse_xpu_float64",
    "test_forward_mode_AD_qr_xpu_complex128",
    "test_forward_mode_AD_qr_xpu_float64",
    "test_forward_mode_AD_svd_lowrank_xpu_float64",
    "test_forward_mode_AD_svd_xpu_complex128",
    "test_forward_mode_AD_svd_xpu_float64",
    "test_forward_mode_AD_tensordot_xpu_complex128",
    "test_forward_mode_AD_tensordot_xpu_float64",
    "test_forward_mode_AD_triangular_solve_xpu_complex128",
    "test_forward_mode_AD_triangular_solve_xpu_float64",
    "test_inplace_forward_mode_AD_addbmm_xpu_float64",
    "test_inplace_forward_mode_AD_addmm_decomposed_xpu_complex128",
    "test_inplace_forward_mode_AD_addmm_decomposed_xpu_float64",
    "test_inplace_forward_mode_AD_addmm_xpu_complex128",
    "test_inplace_forward_mode_AD_addmm_xpu_float64",
    "test_inplace_forward_mode_AD_addmv_xpu_complex128",
    "test_inplace_forward_mode_AD_addmv_xpu_float64",
    "test_inplace_forward_mode_AD_baddbmm_xpu_complex128",
    "test_inplace_forward_mode_AD_baddbmm_xpu_float64",
    "test_forward_mode_AD_pca_lowrank_xpu_complex128",
    "test_forward_mode_AD_svd_lowrank_xpu_complex128",
#RuntimeError: value cannot be converted to type float without overflow 
    "test_fn_fwgrad_bwgrad_addbmm_xpu_complex128",
    "test_forward_mode_AD_addbmm_xpu_complex128",
    "test_inplace_forward_mode_AD_addbmm_xpu_complex128",
#torch.autograd.gradcheck.GradcheckError: While considering the real part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 0, 
    "test_fn_fwgrad_bwgrad_linalg_norm_xpu_complex128",
#torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 0, 
    "test_fn_fwgrad_bwgrad_linalg_vector_norm_xpu_complex128",
    "test_fn_fwgrad_bwgrad_masked_normalize_xpu_complex128",
    "test_fn_fwgrad_bwgrad_norm_inf_xpu_complex128",
    "test_fn_fwgrad_bwgrad_renorm_xpu_complex128",
    "test_forward_mode_AD_linalg_norm_xpu_complex128",
    "test_forward_mode_AD_linalg_vector_norm_xpu_complex128",
    "test_forward_mode_AD_masked_normalize_xpu_complex128",
    "test_forward_mode_AD_norm_inf_xpu_complex128",
    "test_forward_mode_AD_renorm_xpu_complex128",
    "test_inplace_forward_mode_AD_renorm_xpu_complex128",
#RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive 
    "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose2d_xpu_complex128",
    "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose2d_xpu_float64",
    "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose3d_xpu_complex128",
    "test_fn_fwgrad_bwgrad_nn_functional_conv_transpose3d_xpu_float64",
    "test_forward_mode_AD_nn_functional_conv_transpose2d_xpu_complex128",
    "test_forward_mode_AD_nn_functional_conv_transpose2d_xpu_float64",
    "test_forward_mode_AD_nn_functional_conv_transpose3d_xpu_complex128",
    "test_forward_mode_AD_nn_functional_conv_transpose3d_xpu_float64",
#RuntimeError: input tensor must have at least one element, but got input_sizes = [1, 0, 1] 
    "test_fn_fwgrad_bwgrad_nn_functional_group_norm_xpu_float64",
    "test_forward_mode_AD_nn_functional_group_norm_xpu_float64",
#torch.autograd.gradcheck.GradcheckError: Jacobian computed with forward mode mismatch for output 0 with respect to input 0, 
    "test_fn_fwgrad_bwgrad_nn_functional_rrelu_xpu_float64",
    "test_forward_mode_AD_nn_functional_rrelu_xpu_float64",
#RuntimeError: DispatchStub: unsupported device typexpu 
    "test_inplace_forward_mode_AD_conj_physical_xpu_complex128",
)
res += launch_test("test_ops_fwd_gradients_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = (
    # eye fallbacks to CPU and does not support Float8_e4m3fn
    "test_cache_disabled",
)
res += launch_test("test_matmul_cuda_xpu.py",skip_list=skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)


skip_list = (
    #RuntimeError: is_coalesced expected sparse coordinate tensor layout but got Sparse  
    "test_contiguous_xpu",
    "test_invalid_sparse_coo_values_xpu",
    "test_to_dense_and_sparse_coo_xpu",
    "test_to_dense_xpu",
    "test_to_sparse_xpu",
    "test_binary_core_add_layout1_xpu_float16",
    "test_binary_core_add_layout1_xpu_float32",
    "test_binary_core_add_layout1_xpu_float64",
    "test_binary_core_atan2_layout1_xpu_float16",
    "test_binary_core_atan2_layout1_xpu_float32",
    "test_binary_core_atan2_layout1_xpu_float64",
    "test_binary_core_div_floor_rounding_layout1_xpu_float16",
    "test_binary_core_div_floor_rounding_layout1_xpu_float32",
    "test_binary_core_div_floor_rounding_layout1_xpu_float64",
    "test_binary_core_div_no_rounding_mode_layout1_xpu_float16",
    "test_binary_core_div_no_rounding_mode_layout1_xpu_float32",
    "test_binary_core_div_no_rounding_mode_layout1_xpu_float64",
    "test_binary_core_div_trunc_rounding_layout1_xpu_float16",
    "test_binary_core_div_trunc_rounding_layout1_xpu_float32",
    "test_binary_core_div_trunc_rounding_layout1_xpu_float64",
    "test_binary_core_eq_layout1_xpu_float16",
    "test_binary_core_eq_layout1_xpu_float32",
    "test_binary_core_eq_layout1_xpu_float64",
    "test_binary_core_floor_divide_layout1_xpu_float16",
    "test_binary_core_floor_divide_layout1_xpu_float32",
    "test_binary_core_floor_divide_layout1_xpu_float64",
    "test_binary_core_fmax_layout1_xpu_float16",
    "test_binary_core_fmax_layout1_xpu_float32",
    "test_binary_core_fmax_layout1_xpu_float64",
    "test_binary_core_fmin_layout1_xpu_float16",
    "test_binary_core_fmin_layout1_xpu_float32",
    "test_binary_core_fmin_layout1_xpu_float64",
    "test_binary_core_fmod_layout1_xpu_float16",
    "test_binary_core_fmod_layout1_xpu_float32",
    "test_binary_core_fmod_layout1_xpu_float64",
    "test_binary_core_ge_layout1_xpu_float16",
    "test_binary_core_ge_layout1_xpu_float32",
    "test_binary_core_ge_layout1_xpu_float64",
    "test_binary_core_gt_layout1_xpu_float16",
    "test_binary_core_gt_layout1_xpu_float32",
    "test_binary_core_gt_layout1_xpu_float64",
    "test_binary_core_le_layout1_xpu_float16",
    "test_binary_core_le_layout1_xpu_float32",
    "test_binary_core_le_layout1_xpu_float64",
    "test_binary_core_logaddexp_layout1_xpu_float16",
    "test_binary_core_logaddexp_layout1_xpu_float32",
    "test_binary_core_logaddexp_layout1_xpu_float64",
    "test_binary_core_lt_layout1_xpu_float16",
    "test_binary_core_lt_layout1_xpu_float32",
    "test_binary_core_lt_layout1_xpu_float64",
    "test_binary_core_maximum_layout1_xpu_float16",
    "test_binary_core_maximum_layout1_xpu_float32",
    "test_binary_core_maximum_layout1_xpu_float64",
    "test_binary_core_minimum_layout1_xpu_float16",
    "test_binary_core_minimum_layout1_xpu_float32",
    "test_binary_core_minimum_layout1_xpu_float64",
    "test_binary_core_mul_layout1_xpu_float16",
    "test_binary_core_mul_layout1_xpu_float32",
    "test_binary_core_mul_layout1_xpu_float64",
    "test_binary_core_ne_layout1_xpu_float16",
    "test_binary_core_ne_layout1_xpu_float32",
    "test_binary_core_ne_layout1_xpu_float64",
    "test_binary_core_nextafter_layout1_xpu_float16",
    "test_binary_core_nextafter_layout1_xpu_float32",
    "test_binary_core_nextafter_layout1_xpu_float64",
    "test_binary_core_remainder_layout1_xpu_float16",
    "test_binary_core_remainder_layout1_xpu_float32",
    "test_binary_core_remainder_layout1_xpu_float64",
    "test_binary_core_sub_layout1_xpu_float16",
    "test_binary_core_sub_layout1_xpu_float32",
    "test_binary_core_sub_layout1_xpu_float64",
    "test_binary_core_true_divide_layout1_xpu_float16",
    "test_binary_core_true_divide_layout1_xpu_float32",
    "test_binary_core_true_divide_layout1_xpu_float64",
    "test_reduction_all_amax_layout1_xpu_float16",
    "test_reduction_all_amax_layout1_xpu_float32",
    "test_reduction_all_amax_layout1_xpu_float64",
    "test_reduction_all_amin_layout1_xpu_float16",
    "test_reduction_all_amin_layout1_xpu_float32",
    "test_reduction_all_amin_layout1_xpu_float64",
    "test_reduction_all_argmax_layout1_xpu_float16",
    "test_reduction_all_argmax_layout1_xpu_float32",
    "test_reduction_all_argmax_layout1_xpu_float64",
    "test_reduction_all_argmin_layout1_xpu_float16",
    "test_reduction_all_argmin_layout1_xpu_float32",
    "test_reduction_all_argmin_layout1_xpu_float64",
    "test_reduction_all_prod_layout1_xpu_float32",
    "test_reduction_all_prod_layout1_xpu_float64",
    "test_reduction_all_sum_layout1_xpu_float16",
    "test_reduction_all_sum_layout1_xpu_float64",
    #RuntimeError: device type of values (xpu) must be CPU or CUDA or Meta
    "test_invalid_sparse_layout_xpu",
    "test_to_dense_and_sparse_csr_xpu",
    "test_binary_core_add_layout2_xpu_float16",
    "test_binary_core_add_layout2_xpu_float32",
    "test_binary_core_add_layout2_xpu_float64",
    "test_binary_core_atan2_layout2_xpu_float16",
    "test_binary_core_atan2_layout2_xpu_float32",
    "test_binary_core_atan2_layout2_xpu_float64",
    "test_binary_core_div_floor_rounding_layout2_xpu_float16",
    "test_binary_core_div_floor_rounding_layout2_xpu_float32",
    "test_binary_core_div_floor_rounding_layout2_xpu_float64",
    "test_binary_core_div_no_rounding_mode_layout2_xpu_float16",
    "test_binary_core_div_no_rounding_mode_layout2_xpu_float32",
    "test_binary_core_div_no_rounding_mode_layout2_xpu_float64",
    "test_binary_core_div_trunc_rounding_layout2_xpu_float16",
    "test_binary_core_div_trunc_rounding_layout2_xpu_float32",
    "test_binary_core_div_trunc_rounding_layout2_xpu_float64",
    "test_binary_core_eq_layout2_xpu_float16",
    "test_binary_core_eq_layout2_xpu_float32",
    "test_binary_core_eq_layout2_xpu_float64",
    "test_binary_core_floor_divide_layout2_xpu_float16",
    "test_binary_core_floor_divide_layout2_xpu_float32",
    "test_binary_core_floor_divide_layout2_xpu_float64",
    "test_binary_core_fmax_layout2_xpu_float16",
    "test_binary_core_fmax_layout2_xpu_float32",
    "test_binary_core_fmax_layout2_xpu_float64",
    "test_binary_core_fmin_layout2_xpu_float16",
    "test_binary_core_fmin_layout2_xpu_float32",
    "test_binary_core_fmin_layout2_xpu_float64",
    "test_binary_core_fmod_layout2_xpu_float16",
    "test_binary_core_fmod_layout2_xpu_float32",
    "test_binary_core_fmod_layout2_xpu_float64",
    "test_binary_core_ge_layout2_xpu_float16",
    "test_binary_core_ge_layout2_xpu_float32",
    "test_binary_core_ge_layout2_xpu_float64",
    "test_binary_core_gt_layout2_xpu_float16",
    "test_binary_core_gt_layout2_xpu_float32",
    "test_binary_core_gt_layout2_xpu_float64",
    "test_binary_core_le_layout2_xpu_float16",
    "test_binary_core_le_layout2_xpu_float32",
    "test_binary_core_le_layout2_xpu_float64",
    "test_binary_core_logaddexp_layout2_xpu_float16",
    "test_binary_core_logaddexp_layout2_xpu_float32",
    "test_binary_core_logaddexp_layout2_xpu_float64",
    "test_binary_core_lt_layout2_xpu_float16",
    "test_binary_core_lt_layout2_xpu_float32",
    "test_binary_core_lt_layout2_xpu_float64",
    "test_binary_core_maximum_layout2_xpu_float16",
    "test_binary_core_maximum_layout2_xpu_float32",
    "test_binary_core_maximum_layout2_xpu_float64",
    "test_binary_core_minimum_layout2_xpu_float16",
    "test_binary_core_minimum_layout2_xpu_float32",
    "test_binary_core_minimum_layout2_xpu_float64",
    "test_binary_core_mul_layout2_xpu_float16",
    "test_binary_core_mul_layout2_xpu_float32",
    "test_binary_core_mul_layout2_xpu_float64",
    "test_binary_core_ne_layout2_xpu_float16",
    "test_binary_core_ne_layout2_xpu_float32",
    "test_binary_core_ne_layout2_xpu_float64",
    "test_binary_core_nextafter_layout2_xpu_float16",
    "test_binary_core_nextafter_layout2_xpu_float32",
    "test_binary_core_nextafter_layout2_xpu_float64",
    "test_binary_core_remainder_layout2_xpu_float16",
    "test_binary_core_remainder_layout2_xpu_float32",
    "test_binary_core_remainder_layout2_xpu_float64",
    "test_binary_core_sub_layout2_xpu_float16",
    "test_binary_core_sub_layout2_xpu_float32",
    "test_binary_core_sub_layout2_xpu_float64",
    "test_binary_core_true_divide_layout2_xpu_float16",
    "test_binary_core_true_divide_layout2_xpu_float32",
    "test_binary_core_true_divide_layout2_xpu_float64",
    "test_reduction_all_amax_layout2_xpu_float16",
    "test_reduction_all_amax_layout2_xpu_float32",
    "test_reduction_all_amax_layout2_xpu_float64",
    "test_reduction_all_amin_layout2_xpu_float16",
    "test_reduction_all_amin_layout2_xpu_float32",
    "test_reduction_all_amin_layout2_xpu_float64",
    "test_reduction_all_prod_layout2_xpu_float32",
    "test_reduction_all_prod_layout2_xpu_float64",
    "test_reduction_all_sum_layout2_xpu_float16",
    "test_reduction_all_sum_layout2_xpu_float64",
)
res += launch_test("test_maskedtensor_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

res += launch_test("nn/test_packed_sequence_xpu.py")
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = (
    ### Error #0 in TestBwdGradientsXPU , totally 271 , RuntimeError: Double and complex datatype matmul is not supported in oneDNN
 
    "test_fn_grad___rmatmul___xpu_complex128",
    "test_fn_grad___rmatmul___xpu_float64",
    "test_fn_grad_addbmm_xpu_float64",
    "test_fn_grad_addmm_decomposed_xpu_complex128",
    "test_fn_grad_addmm_decomposed_xpu_float64",
    "test_fn_grad_addmm_xpu_complex128",
    "test_fn_grad_addmm_xpu_float64",
    "test_fn_grad_addmv_xpu_complex128",
    "test_fn_grad_addmv_xpu_float64",
    "test_fn_grad_addr_xpu_complex128",
    "test_fn_grad_addr_xpu_float64",
    "test_fn_grad_baddbmm_xpu_complex128",
    "test_fn_grad_baddbmm_xpu_float64",
    "test_fn_grad_bmm_xpu_complex128",
    "test_fn_grad_bmm_xpu_float64",
    "test_fn_grad_cdist_xpu_float64",
    "test_fn_grad_cholesky_inverse_xpu_complex128",
    "test_fn_grad_cholesky_inverse_xpu_float64",
    "test_fn_grad_cholesky_solve_xpu_complex128",
    "test_fn_grad_cholesky_solve_xpu_float64",
    "test_fn_grad_cholesky_xpu_complex128",
    "test_fn_grad_cholesky_xpu_float64",
    "test_fn_grad_corrcoef_xpu_complex128",
    "test_fn_grad_corrcoef_xpu_float64",
    "test_fn_grad_einsum_xpu_complex128",
    "test_fn_grad_einsum_xpu_float64",
    "test_fn_grad_inner_xpu_complex128",
    "test_fn_grad_inner_xpu_float64",
    "test_fn_grad_linalg_cholesky_ex_xpu_complex128",
    "test_fn_grad_linalg_cholesky_ex_xpu_float64",
    "test_fn_grad_linalg_cholesky_xpu_complex128",
    "test_fn_grad_linalg_cholesky_xpu_float64",
    "test_fn_grad_linalg_cond_xpu_complex128",
    "test_fn_grad_linalg_cond_xpu_float64",
    "test_fn_grad_linalg_det_singular_xpu_complex128",
    "test_fn_grad_linalg_det_singular_xpu_float64",
    "test_fn_grad_linalg_det_xpu_complex128",
    "test_fn_grad_linalg_det_xpu_float64",
    "test_fn_grad_linalg_eig_xpu_complex128",
    "test_fn_grad_linalg_eig_xpu_float64",
    "test_fn_grad_linalg_eigh_xpu_complex128",
    "test_fn_grad_linalg_eigh_xpu_float64",
    "test_fn_grad_linalg_eigvals_xpu_complex128",
    "test_fn_grad_linalg_eigvals_xpu_float64",
    "test_fn_grad_linalg_eigvalsh_xpu_complex128",
    "test_fn_grad_linalg_eigvalsh_xpu_float64",
    "test_fn_grad_linalg_householder_product_xpu_complex128",
    "test_fn_grad_linalg_householder_product_xpu_float64",
    "test_fn_grad_linalg_inv_ex_xpu_complex128",
    "test_fn_grad_linalg_inv_ex_xpu_float64",
    "test_fn_grad_linalg_inv_xpu_complex128",
    "test_fn_grad_linalg_inv_xpu_float64",
    "test_fn_grad_linalg_lstsq_grad_oriented_xpu_complex128",
    "test_fn_grad_linalg_lstsq_grad_oriented_xpu_float64",
    "test_fn_grad_linalg_lu_factor_ex_xpu_complex128",
    "test_fn_grad_linalg_lu_factor_ex_xpu_float64",
    "test_fn_grad_linalg_lu_factor_xpu_complex128",
    "test_fn_grad_linalg_lu_factor_xpu_float64",
    "test_fn_grad_linalg_lu_solve_xpu_complex128",
    "test_fn_grad_linalg_lu_solve_xpu_float64",
    "test_fn_grad_linalg_lu_xpu_complex128",
    "test_fn_grad_linalg_lu_xpu_float64",
    "test_fn_grad_linalg_matrix_norm_xpu_complex128",
    "test_fn_grad_linalg_matrix_norm_xpu_float64",
    "test_fn_grad_linalg_matrix_power_xpu_complex128",
    "test_fn_grad_linalg_matrix_power_xpu_float64",
    "test_fn_grad_linalg_multi_dot_xpu_complex128",
    "test_fn_grad_linalg_multi_dot_xpu_float64",
    "test_fn_grad_linalg_norm_xpu_float64",
    "test_fn_grad_linalg_pinv_hermitian_xpu_complex128",
    "test_fn_grad_linalg_pinv_hermitian_xpu_float64",
    "test_fn_grad_linalg_pinv_singular_xpu_complex128",
    "test_fn_grad_linalg_pinv_singular_xpu_float64",
    "test_fn_grad_linalg_pinv_xpu_complex128",
    "test_fn_grad_linalg_pinv_xpu_float64",
    "test_fn_grad_linalg_qr_xpu_complex128",
    "test_fn_grad_linalg_qr_xpu_float64",
    "test_fn_grad_linalg_slogdet_xpu_complex128",
    "test_fn_grad_linalg_slogdet_xpu_float64",
    "test_fn_grad_linalg_solve_ex_xpu_complex128",
    "test_fn_grad_linalg_solve_ex_xpu_float64",
    "test_fn_grad_linalg_solve_triangular_xpu_complex128",
    "test_fn_grad_linalg_solve_triangular_xpu_float64",
    "test_fn_grad_linalg_solve_xpu_complex128",
    "test_fn_grad_linalg_solve_xpu_float64",
    "test_fn_grad_linalg_svd_xpu_complex128",
    "test_fn_grad_linalg_svd_xpu_float64",
    "test_fn_grad_linalg_svdvals_xpu_complex128",
    "test_fn_grad_linalg_svdvals_xpu_float64",
    "test_fn_grad_linalg_tensorinv_xpu_complex128",
    "test_fn_grad_linalg_tensorinv_xpu_float64",
    "test_fn_grad_linalg_tensorsolve_xpu_complex128",
    "test_fn_grad_linalg_tensorsolve_xpu_float64",
    "test_fn_grad_logdet_xpu_complex128",
    "test_fn_grad_logdet_xpu_float64",
    "test_fn_grad_lu_solve_xpu_complex128",
    "test_fn_grad_lu_solve_xpu_float64",
    "test_fn_grad_lu_xpu_complex128",
    "test_fn_grad_lu_xpu_float64",
    "test_fn_grad_matmul_xpu_complex128",
    "test_fn_grad_matmul_xpu_float64",
    "test_fn_grad_mm_xpu_complex128",
    "test_fn_grad_mm_xpu_float64",
    "test_fn_grad_mv_xpu_complex128",
    "test_fn_grad_mv_xpu_float64",
    "test_fn_grad_nn_functional_bilinear_xpu_float64",
    "test_fn_grad_nn_functional_linear_xpu_complex128",
    "test_fn_grad_nn_functional_linear_xpu_float64",
    "test_fn_grad_nn_functional_multi_head_attention_forward_xpu_float64",
    "test_fn_grad_nn_functional_scaled_dot_product_attention_xpu_float64",
    "test_fn_grad_norm_nuc_xpu_complex128",
    "test_fn_grad_norm_nuc_xpu_float64",
    "test_fn_grad_ormqr_xpu_complex128",
    "test_fn_grad_ormqr_xpu_float64",
    "test_fn_grad_pca_lowrank_xpu_float64",
    "test_fn_grad_pinverse_xpu_complex128",
    "test_fn_grad_pinverse_xpu_float64",
    "test_fn_grad_qr_xpu_complex128",
    "test_fn_grad_qr_xpu_float64",
    "test_fn_grad_svd_lowrank_xpu_float64",
    "test_fn_grad_svd_xpu_complex128",
    "test_fn_grad_svd_xpu_float64",
    "test_fn_grad_tensordot_xpu_complex128",
    "test_fn_grad_tensordot_xpu_float64",
    "test_fn_grad_triangular_solve_xpu_complex128",
    "test_fn_grad_triangular_solve_xpu_float64",
    "test_fn_gradgrad___rmatmul___xpu_complex128",
    "test_fn_gradgrad___rmatmul___xpu_float64",
    "test_fn_gradgrad_addbmm_xpu_float64",
    "test_fn_gradgrad_addmm_decomposed_xpu_complex128",
    "test_fn_gradgrad_addmm_decomposed_xpu_float64",
    "test_fn_gradgrad_addmm_xpu_complex128",
    "test_fn_gradgrad_addmm_xpu_float64",
    "test_fn_gradgrad_addmv_xpu_complex128",
    "test_fn_gradgrad_addmv_xpu_float64",
    "test_fn_gradgrad_addr_xpu_complex128",
    "test_fn_gradgrad_addr_xpu_float64",
    "test_fn_gradgrad_baddbmm_xpu_complex128",
    "test_fn_gradgrad_baddbmm_xpu_float64",
    "test_fn_gradgrad_bmm_xpu_complex128",
    "test_fn_gradgrad_bmm_xpu_float64",
    "test_fn_gradgrad_cholesky_inverse_xpu_complex128",
    "test_fn_gradgrad_cholesky_inverse_xpu_float64",
    "test_fn_gradgrad_cholesky_solve_xpu_complex128",
    "test_fn_gradgrad_cholesky_solve_xpu_float64",
    "test_fn_gradgrad_cholesky_xpu_complex128",
    "test_fn_gradgrad_cholesky_xpu_float64",
    "test_fn_gradgrad_corrcoef_xpu_complex128",
    "test_fn_gradgrad_corrcoef_xpu_float64",
    "test_fn_gradgrad_einsum_xpu_complex128",
    "test_fn_gradgrad_einsum_xpu_float64",
    "test_fn_gradgrad_inner_xpu_complex128",
    "test_fn_gradgrad_inner_xpu_float64",
    "test_fn_gradgrad_linalg_cholesky_ex_xpu_complex128",
    "test_fn_gradgrad_linalg_cholesky_ex_xpu_float64",
    "test_fn_gradgrad_linalg_cholesky_xpu_complex128",
    "test_fn_gradgrad_linalg_cholesky_xpu_float64",
    "test_fn_gradgrad_linalg_cond_xpu_complex128",
    "test_fn_gradgrad_linalg_cond_xpu_float64",
    "test_fn_gradgrad_linalg_det_singular_xpu_float64",
    "test_fn_gradgrad_linalg_det_xpu_complex128",
    "test_fn_gradgrad_linalg_det_xpu_float64",
    "test_fn_gradgrad_linalg_eig_xpu_complex128",
    "test_fn_gradgrad_linalg_eig_xpu_float64",
    "test_fn_gradgrad_linalg_eigh_xpu_complex128",
    "test_fn_gradgrad_linalg_eigh_xpu_float64",
    "test_fn_gradgrad_linalg_eigvals_xpu_complex128",
    "test_fn_gradgrad_linalg_eigvals_xpu_float64",
    "test_fn_gradgrad_linalg_eigvalsh_xpu_complex128",
    "test_fn_gradgrad_linalg_eigvalsh_xpu_float64",
    "test_fn_gradgrad_linalg_householder_product_xpu_complex128",
    "test_fn_gradgrad_linalg_householder_product_xpu_float64",
    "test_fn_gradgrad_linalg_inv_ex_xpu_complex128",
    "test_fn_gradgrad_linalg_inv_ex_xpu_float64",
    "test_fn_gradgrad_linalg_inv_xpu_complex128",
    "test_fn_gradgrad_linalg_inv_xpu_float64",
    "test_fn_gradgrad_linalg_lstsq_grad_oriented_xpu_complex128",
    "test_fn_gradgrad_linalg_lstsq_grad_oriented_xpu_float64",
    "test_fn_gradgrad_linalg_lu_factor_ex_xpu_complex128",
    "test_fn_gradgrad_linalg_lu_factor_ex_xpu_float64",
    "test_fn_gradgrad_linalg_lu_factor_xpu_complex128",
    "test_fn_gradgrad_linalg_lu_factor_xpu_float64",
    "test_fn_gradgrad_linalg_lu_solve_xpu_complex128",
    "test_fn_gradgrad_linalg_lu_solve_xpu_float64",
    "test_fn_gradgrad_linalg_lu_xpu_complex128",
    "test_fn_gradgrad_linalg_lu_xpu_float64",
    "test_fn_gradgrad_linalg_matrix_norm_xpu_complex128",
    "test_fn_gradgrad_linalg_matrix_norm_xpu_float64",
    "test_fn_gradgrad_linalg_matrix_power_xpu_complex128",
    "test_fn_gradgrad_linalg_matrix_power_xpu_float64",
    "test_fn_gradgrad_linalg_multi_dot_xpu_complex128",
    "test_fn_gradgrad_linalg_multi_dot_xpu_float64",
    "test_fn_gradgrad_linalg_pinv_hermitian_xpu_complex128",
    "test_fn_gradgrad_linalg_pinv_hermitian_xpu_float64",
    "test_fn_gradgrad_linalg_pinv_singular_xpu_complex128",
    "test_fn_gradgrad_linalg_pinv_singular_xpu_float64",
    "test_fn_gradgrad_linalg_pinv_xpu_complex128",
    "test_fn_gradgrad_linalg_pinv_xpu_float64",
    "test_fn_gradgrad_linalg_qr_xpu_complex128",
    "test_fn_gradgrad_linalg_qr_xpu_float64",
    "test_fn_gradgrad_linalg_slogdet_xpu_complex128",
    "test_fn_gradgrad_linalg_slogdet_xpu_float64",
    "test_fn_gradgrad_linalg_solve_ex_xpu_complex128",
    "test_fn_gradgrad_linalg_solve_ex_xpu_float64",
    "test_fn_gradgrad_linalg_solve_triangular_xpu_complex128",
    "test_fn_gradgrad_linalg_solve_triangular_xpu_float64",
    "test_fn_gradgrad_linalg_solve_xpu_complex128",
    "test_fn_gradgrad_linalg_solve_xpu_float64",
    "test_fn_gradgrad_linalg_svd_xpu_complex128",
    "test_fn_gradgrad_linalg_svd_xpu_float64",
    "test_fn_gradgrad_linalg_svdvals_xpu_complex128",
    "test_fn_gradgrad_linalg_svdvals_xpu_float64",
    "test_fn_gradgrad_linalg_tensorinv_xpu_complex128",
    "test_fn_gradgrad_linalg_tensorinv_xpu_float64",
    "test_fn_gradgrad_linalg_tensorsolve_xpu_complex128",
    "test_fn_gradgrad_linalg_tensorsolve_xpu_float64",
    "test_fn_gradgrad_logdet_xpu_complex128",
    "test_fn_gradgrad_logdet_xpu_float64",
    "test_fn_gradgrad_lu_solve_xpu_complex128",
    "test_fn_gradgrad_lu_solve_xpu_float64",
    "test_fn_gradgrad_lu_xpu_complex128",
    "test_fn_gradgrad_lu_xpu_float64",
    "test_fn_gradgrad_matmul_xpu_complex128",
    "test_fn_gradgrad_matmul_xpu_float64",
    "test_fn_gradgrad_mm_xpu_complex128",
    "test_fn_gradgrad_mm_xpu_float64",
    "test_fn_gradgrad_mv_xpu_complex128",
    "test_fn_gradgrad_mv_xpu_float64",
    "test_fn_gradgrad_nn_functional_bilinear_xpu_float64",
    "test_fn_gradgrad_nn_functional_linear_xpu_complex128",
    "test_fn_gradgrad_nn_functional_linear_xpu_float64",
    "test_fn_gradgrad_nn_functional_multi_head_attention_forward_xpu_float64",
    "test_fn_gradgrad_nn_functional_scaled_dot_product_attention_xpu_float64",
    "test_fn_gradgrad_norm_nuc_xpu_complex128",
    "test_fn_gradgrad_norm_nuc_xpu_float64",
    "test_fn_gradgrad_ormqr_xpu_complex128",
    "test_fn_gradgrad_ormqr_xpu_float64",
    "test_fn_gradgrad_pca_lowrank_xpu_float64",
    "test_fn_gradgrad_pinverse_xpu_complex128",
    "test_fn_gradgrad_pinverse_xpu_float64",
    "test_fn_gradgrad_qr_xpu_complex128",
    "test_fn_gradgrad_qr_xpu_float64",
    "test_fn_gradgrad_svd_lowrank_xpu_float64",
    "test_fn_gradgrad_svd_xpu_complex128",
    "test_fn_gradgrad_svd_xpu_float64",
    "test_fn_gradgrad_tensordot_xpu_complex128",
    "test_fn_gradgrad_tensordot_xpu_float64",
    "test_fn_gradgrad_triangular_solve_xpu_complex128",
    "test_fn_gradgrad_triangular_solve_xpu_float64",
    "test_inplace_grad_addbmm_xpu_float64",
    "test_inplace_grad_addmm_decomposed_xpu_complex128",
    "test_inplace_grad_addmm_decomposed_xpu_float64",
    "test_inplace_grad_addmm_xpu_complex128",
    "test_inplace_grad_addmm_xpu_float64",
    "test_inplace_grad_addmv_xpu_complex128",
    "test_inplace_grad_addmv_xpu_float64",
    "test_inplace_grad_addr_xpu_complex128",
    "test_inplace_grad_addr_xpu_float64",
    "test_inplace_grad_baddbmm_xpu_complex128",
    "test_inplace_grad_baddbmm_xpu_float64",
    "test_inplace_gradgrad_addbmm_xpu_float64",
    "test_inplace_gradgrad_addmm_decomposed_xpu_complex128",
    "test_inplace_gradgrad_addmm_decomposed_xpu_float64",
    "test_inplace_gradgrad_addmm_xpu_complex128",
    "test_inplace_gradgrad_addmm_xpu_float64",
    "test_inplace_gradgrad_addmv_xpu_complex128",
    "test_inplace_gradgrad_addmv_xpu_float64",
    "test_inplace_gradgrad_addr_xpu_complex128",
    "test_inplace_gradgrad_addr_xpu_float64",
    "test_inplace_gradgrad_baddbmm_xpu_complex128",
    "test_inplace_gradgrad_baddbmm_xpu_float64",
    "test_fn_grad_pca_lowrank_xpu_complex128",
    "test_fn_grad_svd_lowrank_xpu_complex128",
    "test_fn_gradgrad_pca_lowrank_xpu_complex128",
    "test_fn_gradgrad_svd_lowrank_xpu_complex128",

    ### Error #1 in TestBwdGradientsXPU , totally 4 , RuntimeError: value cannot be converted to type float without overflow
    
    "test_fn_grad_addbmm_xpu_complex128",
    "test_fn_gradgrad_addbmm_xpu_complex128",
    "test_inplace_grad_addbmm_xpu_complex128",
    "test_inplace_gradgrad_addbmm_xpu_complex128",

    ### Error #2 in TestBwdGradientsXPU , totally 8 , torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
    
    "test_fn_grad_bernoulli_xpu_float64",
    "test_fn_grad_linalg_norm_xpu_complex128",
    "test_fn_grad_linalg_vector_norm_xpu_complex128",
    "test_fn_grad_nn_functional_rrelu_xpu_float64",
    "test_fn_grad_norm_inf_xpu_complex128",
    "test_fn_gradgrad_nn_functional_rrelu_xpu_float64",
    "test_inplace_grad_nn_functional_rrelu_xpu_float64",
    "test_inplace_gradgrad_nn_functional_rrelu_xpu_float64",

    ### Error #3 in TestBwdGradientsXPU , totally 8 , torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex outputs only, Jacobian mismatch for output 0 with respect to input 0,
    
    "test_fn_grad_masked_normalize_xpu_complex128",
    "test_fn_grad_renorm_xpu_complex128",
    "test_fn_gradgrad_linalg_vector_norm_xpu_complex128",
    "test_fn_gradgrad_masked_normalize_xpu_complex128",
    "test_fn_gradgrad_norm_inf_xpu_complex128",
    "test_fn_gradgrad_renorm_xpu_complex128",
    "test_inplace_grad_renorm_xpu_complex128",
    "test_inplace_gradgrad_renorm_xpu_complex128",

    ### Error #4 in TestBwdGradientsXPU , totally 8 , RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive
    
    "test_fn_grad_nn_functional_conv_transpose2d_xpu_complex128",
    "test_fn_grad_nn_functional_conv_transpose2d_xpu_float64",
    "test_fn_grad_nn_functional_conv_transpose3d_xpu_complex128",
    "test_fn_grad_nn_functional_conv_transpose3d_xpu_float64",
    "test_fn_gradgrad_nn_functional_conv_transpose2d_xpu_complex128",
    "test_fn_gradgrad_nn_functional_conv_transpose2d_xpu_float64",
    "test_fn_gradgrad_nn_functional_conv_transpose3d_xpu_complex128",
    "test_fn_gradgrad_nn_functional_conv_transpose3d_xpu_float64",

    ### Error #5 in TestBwdGradientsXPU , totally 2 , RuntimeError: input tensor must have at least one element, but got input_sizes = [1, 0, 1]
    
    "test_fn_grad_nn_functional_group_norm_xpu_float64",
    "test_fn_gradgrad_nn_functional_group_norm_xpu_float64",

    ### Error #6 in TestBwdGradientsXPU , totally 5 , torch.autograd.gradcheck.GradcheckError: Backward is not reentrant, i.e., running backward with same input and grad_output multiple times gives different values, although analytical gradient matches numerical gradient.The tolerance for nondeterminism was 0.0.
    
    "test_fn_grad_nn_functional_max_pool2d_xpu_float64",
    "test_fn_gradgrad_index_reduce_mean_xpu_float64",
    "test_fn_gradgrad_index_reduce_prod_xpu_float64",
    "test_inplace_gradgrad_index_reduce_mean_xpu_float64",
    "test_inplace_gradgrad_index_reduce_prod_xpu_float64",

    ### Error #7 in TestBwdGradientsXPU , totally 2 , NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::_sparse_coo_tensor_with_dims_and_tensors' is only available for these backends: [XPU, Meta, SparseCPU, SparseMeta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastXPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    
    "test_fn_grad_to_sparse_xpu_float64",
    "test_fn_gradgrad_to_sparse_xpu_float64",

    ### Error #8 in TestBwdGradientsXPU , totally 2 , RuntimeError: DispatchStub: unsupported device typexpu
    
    "test_inplace_grad_conj_physical_xpu_complex128",
    "test_inplace_gradgrad_conj_physical_xpu_complex128",
)

res += launch_test("test_ops_gradients_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = (
    # issue 302
    ### Error #0 in TestTorchDeviceTypeXPU , totally 11 , RuntimeError: expected scalar type Long but found Int
 
    "test_index_reduce_reduce_mean_xpu_bfloat16",
    "test_index_reduce_reduce_mean_xpu_float16",
    "test_index_reduce_reduce_mean_xpu_float32",
    "test_index_reduce_reduce_mean_xpu_float64",
    "test_index_reduce_reduce_mean_xpu_int16",
    "test_index_reduce_reduce_mean_xpu_int32",
    "test_index_reduce_reduce_mean_xpu_int64",
    "test_index_reduce_reduce_mean_xpu_int8",
    "test_index_reduce_reduce_mean_xpu_uint8",

    ### Error #1 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'FloatTensor'
    

    "test_grad_scaling_state_dict_xpu",

    ### Error #2 in TestTorchDeviceTypeXPU , totally 1 , AttributeError: 'torch.storage.TypedStorage' object has no attribute 'is_xpu'

    ### Error #3 in TestTorchDeviceTypeXPU , totally 3 , AttributeError: module 'torch.xpu' has no attribute 'ByteStorage'
    

    "test_storage_setitem_xpu_uint8",
    "test_tensor_storage_type_xpu_uint8",

    ### Error #4 in TestTorchDeviceTypeXPU , totally 4 , AttributeError: module 'torch.xpu' has no attribute 'FloatStorage'
    


    "test_storage_setitem_xpu_float32",
    "test_tensor_storage_type_xpu_float32",

    ### Error #5 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: Scalars are not equal!
    
    "test_bernoulli_edge_cases_xpu_float16",
    "test_strides_propagation_xpu",


    ### Error #7 in TestTorchDeviceTypeXPU , totally 1 , TypeError: map2_ is only implemented on CPU tensors
    
    "test_broadcast_fn_map2_xpu",

    ### Error #8 in TestTorchDeviceTypeXPU , totally 1 , TypeError: map_ is only implemented on CPU tensors
    
    "test_broadcast_fn_map_xpu",

    ### Error #9 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: Double and complex datatype matmul is not supported in oneDNN
    
    "test_corrcoef_xpu_complex64",

    ### Error #10 in TestTorchDeviceTypeXPU , totally 1 , AssertionError: True is not false
    
    "test_discontiguous_out_cumsum_xpu",

    ### Error #11 in TestTorchDeviceTypeXPU , totally 1 , AssertionError: tensor(False, device='xpu:0') is not true
    
    "test_exponential_no_zero_xpu_float16",

    ### Error #12 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'amp'
    
    "test_grad_scaler_pass_itself_xpu",
    "test_pickle_gradscaler_xpu",

    ### Error #13 in TestTorchDeviceTypeXPU , totally 3 , NotImplementedError: Could not run 'aten::_copy_from_and_resize' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::_copy_from_and_resize' is only available for these backends: [XPU, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastXPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    
    "test_grad_scaling_autocast_foreach2_fused_True_AdamW_xpu_float32",
    "test_grad_scaling_autocast_foreach2_fused_True_Adam_xpu_float32",
    "test_grad_scaling_autocast_foreach2_fused_True_SGD_xpu_float32",

    ### Error #14 in TestTorchDeviceTypeXPU , totally 2 , NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseXPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::_sparse_coo_tensor_with_dims_and_tensors' is only available for these backends: [XPU, Meta, SparseCPU, SparseMeta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastXPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    
    "test_grad_scaling_unscale_sparse_xpu_float32",
    "test_memory_format_empty_like_xpu",

    ### Error #15 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: Tensor-likes are not close!
    
    "test_gradient_all_xpu_float32",
    "test_index_put_non_accumulate_deterministic_xpu",

    ### Error #16 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation.
    
    "test_index_fill_mem_overlap_xpu",

    ### Error #17 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: False is not true
    
    "test_is_set_to_xpu",
    "test_pin_memory_from_constructor_xpu",

    ### Error #18 in TestTorchDeviceTypeXPU , totally 2 , AssertionError: Torch not compiled with CUDA enabled
    
    "test_memory_format_cpu_and_cuda_ops_xpu",
    "test_sync_warning_xpu",

    ### Error #19 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: _share_fd_: only available on CPU
    
    "test_module_share_memory_xpu",

    ### Error #20 in TestTorchDeviceTypeXPU , totally 3 , RuntimeError: Expected a 'cpu' device type for generator but found 'xpu'
    
    "test_multinomial_deterministic_xpu_float16",
    "test_multinomial_deterministic_xpu_float32",
    "test_multinomial_deterministic_xpu_float64",

    ### Error #21 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: multinomial expects Long tensor out, got: Float
    
    "test_multinomial_device_constrain_xpu",

    ### Error #22 in TestTorchDeviceTypeXPU , totally 1 , AssertionError: "Expected all tensors to be on the same device" does not match "multinomial expects Long tensor out, got: Float"
    
    "test_multinomial_device_constrain_xpu",

    ### Error #23 in TestTorchDeviceTypeXPU , totally 26 , AssertionError: RuntimeError not raised : expected a non-deterministic error, but it was not raised
    
    "test_nondeterministic_alert_AdaptiveAvgPool2d_xpu",
    "test_nondeterministic_alert_AdaptiveAvgPool3d_xpu",
    "test_nondeterministic_alert_AdaptiveMaxPool2d_xpu",
    "test_nondeterministic_alert_CTCLoss_xpu",
    "test_nondeterministic_alert_EmbeddingBag_max_xpu",
    "test_nondeterministic_alert_FractionalMaxPool2d_xpu",
    "test_nondeterministic_alert_FractionalMaxPool3d_xpu",
    "test_nondeterministic_alert_MaxPool3d_xpu",
    "test_nondeterministic_alert_NLLLoss_xpu",
    "test_nondeterministic_alert_ReflectionPad1d_xpu",
    "test_nondeterministic_alert_ReflectionPad2d_xpu",
    "test_nondeterministic_alert_ReflectionPad3d_xpu",
    "test_nondeterministic_alert_ReplicationPad1d_xpu",
    "test_nondeterministic_alert_ReplicationPad2d_xpu",
    "test_nondeterministic_alert_ReplicationPad3d_xpu",
    "test_nondeterministic_alert_bincount_xpu",
    "test_nondeterministic_alert_grid_sample_2d_xpu",
    "test_nondeterministic_alert_grid_sample_3d_xpu",
    "test_nondeterministic_alert_histc_xpu",
    "test_nondeterministic_alert_interpolate_bicubic_xpu",
    "test_nondeterministic_alert_interpolate_bilinear_xpu",
    "test_nondeterministic_alert_interpolate_linear_xpu",
    "test_nondeterministic_alert_interpolate_trilinear_xpu",
    "test_nondeterministic_alert_kthvalue_xpu_float64",
    "test_nondeterministic_alert_median_xpu_float64",
    "test_nondeterministic_alert_put_accumulate_xpu",

    ### Error #24 in TestTorchDeviceTypeXPU , totally 1 , AttributeError: 'TestTorchDeviceTypeXPU' object has no attribute 'check_device_nondeterministic_alert'
    
    "test_nondeterministic_alert_AvgPool3d_xpu",

    ### Error #25 in TestTorchDeviceTypeXPU , totally 2 , RuntimeError: "max_unpool2d" not implemented for 'Half'
    
    "test_nondeterministic_alert_MaxUnpool1d_xpu_float16",
    "test_nondeterministic_alert_MaxUnpool2d_xpu_float16",

    ### Error #26 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: "max_unpool3d" not implemented for 'Half'
    
    "test_nondeterministic_alert_MaxUnpool3d_xpu_float16",

    ### Error #27 in TestTorchDeviceTypeXPU , totally 1 , AssertionError: RuntimeError not raised
    
    "test_put_mem_overlap_xpu",

    ### Error #28 in TestTorchDeviceTypeXPU , totally 1 , RuntimeError: "lshift_cpu" not implemented for 'Float'
    
    "test_shift_mem_overlap_xpu",

    ### Error #29 in TestTorchDeviceTypeXPU , totally 1 , AssertionError: "unsupported operation" does not match ""lshift_cpu" not implemented for 'Float'"
    
    "test_shift_mem_overlap_xpu",

    ### Error #30 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'BoolStorage'
    
    "test_storage_setitem_xpu_bool",
    "test_tensor_storage_type_xpu_bool",

    ### Error #31 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'ComplexDoubleStorage'
    
    "test_storage_setitem_xpu_complex128",
    "test_tensor_storage_type_xpu_complex128",

    ### Error #32 in TestTorchDeviceTypeXPU , totally 2 , AttributeError: module 'torch.xpu' has no attribute 'ComplexFloatStorage'
    
    "test_storage_setitem_xpu_complex64",
    "test_tensor_storage_type_xpu_complex64",

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

    ### Error #40 in TestTorchDeviceTypeXPU , totally 1 , FAILED test_torch_xpu.py::TestTorch::test_index_add - RuntimeError: expected ...
    
    "test_tensor_storage_type_xpu_uint8",

    ### Error #41 in TestTorchDeviceTypeXPU , totally 1 , FAILED test_torch_xpu.py::TestTorch::test_print - AttributeError: module 'tor...
    
    "test_tensor_storage_type_xpu_uint8",

    ### Error #42 in TestTorchDeviceTypeXPU , totally 1 , FAILED test_torch_xpu.py::TestTorch::test_storage_error - AttributeError: 'to...
    
    "test_tensor_storage_type_xpu_uint8",

    # issue 302, 12
    "test_index_add",
    "test_index_add_all_dtypes", 

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
)
res += launch_test("test_torch_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)

skip_list = (
    # known oneDNN issue
    # RuntimeError: Double and complex datatype matmul is not supported in oneDNN
    "test_multihead_attention_dtype_batch_first_xpu_float64",
    "test_multihead_attention_dtype_xpu_float64",
    "test_multihead_attn_fast_path_query_and_bias_have_different_dtypes_xpu_float64",
    "test_multihead_attn_fast_path_small_test_xpu_float64",
    "test_multihead_attn_in_proj_bias_none_xpu_float64",
    "test_multihead_attn_in_proj_weight_none_xpu_float64",
    # issue 342
    "test_multihead_self_attn_two_masks_fast_path_mock_xpu",
)

res += launch_test("nn/test_multihead_attention_xpu.py", skip_list)
if res != 0:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
