import os
import sys

skip_list = (
    # Known issue
    "test_compare_cpu__refs_rsub_xpu_bfloat16", # CPU result is not golden reference
    "test_compare_cpu_add_xpu_bfloat16",# CPU result is not golden reference
    "test_compare_cpu_cumsum_xpu_bfloat16",# Caused by different accumulation order
    "test_compare_cpu_cumsum_xpu_float16", # Caused by different accumulation order
    "test_compare_cpu_log_softmax_xpu_bfloat16", # Need FP64 golden ref for more accurate comparison
    "test_compare_cpu_div_floor_rounding_xpu_bfloat16",# CPU result is not golden reference
    "test_compare_cpu_div_trunc_rounding_xpu_float16",# CPU result is not golden reference
    "test_compare_cpu_log_xpu_complex64", # Calculation error between XPU implementation and CPU implementation
    "test_compare_cpu_mul_xpu_complex64", # Calculation error between XPU implementation and CPU implementation
    "test_compare_cpu_native_dropout_backward_xpu_bool", # CUDA does not support either
    "test_compare_cpu_native_dropout_backward_xpu_int16", # CUDA does not support either
    "test_compare_cpu_native_dropout_backward_xpu_int32", # CUDA does not support either
    "test_compare_cpu_native_dropout_backward_xpu_int64", # CUDA does not support either
    "test_compare_cpu_native_dropout_backward_xpu_int8", # CUDA does not support either
    "test_compare_cpu_native_dropout_backward_xpu_uint8", # CUDA does not support either
    "test_compare_cpu_pow_xpu_complex128", # Calculation error between XPU implementation and CPU implementation
    "test_compare_cpu_pow_xpu_complex64", # Calculation error between XPU implementation and CPU implementation
    "test_compare_cpu_rsqrt_xpu_bfloat16", # CPU result is not golden reference
    "test_compare_cpu_sub_xpu_bfloat16", # CPU result is not golden reference
    "test_compare_cpu_tanh_xpu_complex128", # Calculation error between XPU implementation and CPU implementation
    "test_compare_cpu_tanh_xpu_complex64",  # Calculation error between XPU implementation and CPU implementation
    "test_non_standard_bool_values_native_dropout_backward_xpu_bool", # CUDA does not support either

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
