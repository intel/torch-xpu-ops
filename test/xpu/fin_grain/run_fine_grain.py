import os
import sys

skip_list = (
    # Known issue
    "test_compare_cpu__refs_rsub_xpu_bfloat16",
    "test_compare_cpu_add_xpu_bfloat16",
    "test_compare_cpu_cumsum_xpu_bfloat16",
    "test_compare_cpu_cumsum_xpu_float16",
    "test_compare_cpu_div_floor_rounding_xpu_bfloat16",
    "test_compare_cpu_div_trunc_rounding_xpu_float16",
    "test_compare_cpu_index_add_xpu_bool",
    "test_compare_cpu_index_put_xpu_bool",
    "test_compare_cpu_log_xpu_complex64",
    "test_compare_cpu_mul_xpu_complex64",
    "test_compare_cpu_native_dropout_backward_xpu_bool",
    "test_compare_cpu_native_dropout_backward_xpu_int16",
    "test_compare_cpu_native_dropout_backward_xpu_int32",
    "test_compare_cpu_native_dropout_backward_xpu_int64",
    "test_compare_cpu_native_dropout_backward_xpu_int8",
    "test_compare_cpu_native_dropout_backward_xpu_uint8",
    "test_compare_cpu_pow_xpu_complex128",
    "test_compare_cpu_pow_xpu_complex64",
    "test_compare_cpu_rsqrt_xpu_bfloat16",
    "test_compare_cpu_sub_xpu_bfloat16",
    "test_compare_cpu_tanh_xpu_complex128",
    "test_compare_cpu_tanh_xpu_complex64",
    "test_non_standard_bool_values_native_dropout_backward_xpu_bool",
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
