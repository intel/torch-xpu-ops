import os
import sys

skip_list = (
    # Calculation error between XPU implementation and CPU implementation,
    # 1. Compiler optimization causes failing to promote data type to higher precision.
    # 2. Accumulate error is amplified by some operations in some extreme cases. (std::exp(extreme_large_num))
    # 3. Accumulate error is amplified by a large number of accumalate operations.
    # 4. Accumulate error is different on different implementations due to different accumulation order.
    #     a. Different kernel implementations.
    #     b. Different std functions. (std::log, std::tanh, std::exp)
    # 5. The result of division between two same float values is not 1.
    "test_compare_cpu_cumsum_xpu_bfloat16",
    "test_compare_cpu_cumsum_xpu_float16",
    "test_compare_cpu_log_xpu_complex64",
    "test_compare_cpu_mul_xpu_complex64",
    "test_compare_cpu_pow_xpu_complex128",
    "test_compare_cpu_pow_xpu_complex64",
    "test_compare_cpu_tanh_xpu_complex128",
    "test_compare_cpu_tanh_xpu_complex64",
    "test_compare_cpu_rsqrt_xpu_bfloat16",
    "test_compare_cpu__refs_rsub_xpu_bfloat16",
    "test_compare_cpu_add_xpu_bfloat16",
    "test_compare_cpu_sub_xpu_bfloat16",
    "test_compare_cpu_acos_xpu_complex128",
    "test_compare_cpu_acos_xpu_complex64",
    "test_compare_cpu_acosh_xpu_complex64",

    # CPU result is not golden reference
    "test_compare_cpu_div_floor_rounding_xpu_bfloat16",
    "test_compare_cpu_div_trunc_rounding_xpu_float16",
    "test_compare_cpu_div_trunc_rounding_xpu_bfloat16",
    "test_compare_cpu_addr_xpu_float16",

    # CUDA does not support the data type either
    "test_compare_cpu_native_dropout_backward_xpu_bool",
    "test_compare_cpu_native_dropout_backward_xpu_int16",
    "test_compare_cpu_native_dropout_backward_xpu_int32",
    "test_compare_cpu_native_dropout_backward_xpu_int64",
    "test_compare_cpu_native_dropout_backward_xpu_int8",
    "test_compare_cpu_native_dropout_backward_xpu_uint8",
    "test_non_standard_bool_values_native_dropout_backward_xpu_bool",

    # Need FP64 golden ref for more accurate comparison
    "test_compare_cpu_log_softmax_xpu_bfloat16",

    # TestCompositeCompliance
    # CPU fallback fails
    # Require implementing aten::embedding_renorm_
    "test_forward_ad_nn_functional_embedding_xpu_float32",
    "test_backward_nn_functional_embedding_xpu_float32",
    "test_cow_input_nn_functional_embedding_xpu_float32",
    "test_forward_ad_nn_functional_embedding_xpu_float32",
    "test_view_replay_nn_functional_embedding_xpu_float32",

    # TestCompositeCompliance::test_cow_input
    # XPU Tensor fails in copy-on-write cases
    # AssertionError: False is not true : Keyword argument 'output grad 0' during backward call unexpectedly materializes. Either set `supports_cow_input_no_materialize_backward=False` in this operation's OpInfo, add the arg to the OpInfo's `allow_cow_input_materialize_backward` list, or change the implementation to avoid materialization.
    # https://github.com/intel/torch-xpu-ops/issues/281
    "test_cow_input",

    # XPU implementation is correct.
    # std::exp{-inf, nan}, the result is (±0,±0) (signs are unspecified)
    # std::exp{-inf, inf}, the result is (±0,±0) (signs are unspecified)
    # CPU implementation gets NaN in the cases.
    # https://en.cppreference.com/w/cpp/numeric/complex/exp
    "test_compare_cpu_sigmoid_xpu_complex64",
    "test_compare_cpu_sigmoid_xpu_complex128",

    # Align with CUDA dtypes - RuntimeError: "avg_pool2d_out_xpu" not implemented for 'Long'
    "test_compare_cpu_nn_functional_avg_pool2d_xpu_int64",

    # Special handle (different calculation order) in CPU reference impl.
    # https://github.com/pytorch/pytorch/blob/c97e3ebb96d7457075b019b94411e8c2d058e68b/aten/src/ATen/native/EmbeddingBag.cpp#L300
    "test_compare_cpu_nn_functional_embedding_bag_xpu_bfloat16",
    "test_compare_cpu_nn_functional_embedding_bag_xpu_float16",

    # Not implemented operators, aten::embedding_renorm_.
    # To retrieve cases when the operators are supported.
    # https://github.com/intel/torch-xpu-ops/issues/380
    "test_compare_cpu_nn_functional_embedding_bag_xpu_float32",
    "test_compare_cpu_nn_functional_embedding_bag_xpu_float64",
    "test_view_replay_nn_functional_embedding_bag_xpu_float32",

    # Not implemented operators, aten::_embedding_bag_backward.
    # To retrieve cases when the operators are supported.
    # https://github.com/intel/torch-xpu-ops/issues/536
    "test_backward_nn_functional_embedding_bag_xpu_float32",

    #Double and complex datatype matmul is not supported in oneDNN
    "test_compare_cpu_cdist_xpu_float64",

    # CPU reference fail. `abs_cpu` does not support bool.
    # The case should be skipped by PyTorch test infrastructure, but not be
    # skipped correctly after https://github.com/pytorch/pytorch/pull/124147
    # https://github.com/intel/torch-xpu-ops/issues/412
    "test_compare_cpu_abs_xpu_bool",

    # bilinear interpolate includes large calculation steps, accuracy reduces in half-precision
    # Not in CUDA test scope too
    "test_compare_cpu_nn_functional_upsample_bilinear_xpu_bfloat16",
    "test_compare_cpu_nn_functional_upsample_bilinear_xpu_float16",

    # CPU result is not golden reference
    "test_compare_cpu_nn_functional_group_norm_xpu_bfloat16",
    "test_compare_cpu_nn_functional_group_norm_xpu_float16",
    "test_compare_cpu_nn_functional_batch_norm_xpu_bfloat16",
    "test_compare_cpu__batch_norm_with_update_xpu_bfloat16",
    "test_compare_cpu__batch_norm_with_update_xpu_float16",

    # Not implemented operators, aten::upsample_linear1d, aten::upsample_bilinear2d,
    # aten::upsample_trilinear3d
    "nn_functional_interpolate_linear",
    "nn_functional_interpolate_bilinear",
    "nn_functional_interpolate_trilinear",

    # bicubic interpolate includes large calculation steps, accuracy reduces in half-precision
    # Not in CUDA test scope too
    "test_compare_cpu_nn_functional_interpolate_bicubic_xpu_bfloat16",
    "test_compare_cpu_nn_functional_interpolate_bicubic_xpu_float16",
    # Not all operators are implemented for XPU tested in the case.
    # Retrieve it once the operator is implemented.
    # Error: The operator 'aten::glu_jvp' is not currently implemented for the XPU device.
    "test_forward_ad_nn_functional_glu_xpu_float32",

    # Precision error.
    # Mismatched elements: 1 / 812 (0.1%)
    # Greatest absolute difference: 0.03125 at index (610,) (up to 0.001 allowed)
    # Greatest relative difference: 0.00396728515625 at index (610,) (up to 0.001 allowed)
    "test_compare_cpu_hypot_xpu_bfloat16",

    # Regressions due to PyTorch uplift (Numeric difference in float and bfloat)
    # https://github.com/intel/torch-xpu-ops/issues/549
    # Example fail log
    # FAILED test_ops_xpu.py::TestCommonXPU::test_compare_cpu_nn_functional_batch_norm_xpu_float16 - AssertionError: Tensor-likes are not close!
    # Mismatched elements: 3 / 72 (4.2%)
    # Greatest absolute difference: 0.0029296875 at index (0, 1, 1, 0) (up to 0.001 allowed)
    # Greatest relative difference: 0.0032501220703125 at index (2, 1, 2, 1) (up to 0.001 allowed)
    "test_compare_cpu_nn_functional_batch_norm_xpu_float16",
    "test_compare_cpu_std_mean_xpu_bfloat16",
    "test_compare_cpu_sub_xpu_float16",
    "test_compare_cpu_var_mean_xpu_bfloat16",
)


skip_options = " -k 'not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += "'"

test_command = "PYTORCH_TEST_WITH_SLOW=1 pytest -v test_ops_xpu.py"
test_command += skip_options

res = os.system(test_command)
exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
