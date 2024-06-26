import os
import sys


# Cases in the file is too slow to run all suites on CPU. So add white list.


def launch_test(test_case, skip_list=None, exe_list=None):
    if skip_list != None:
        skip_options = " -k 'not " + skip_list[0]
        for skip_case in skip_list[1:]:
            skip_option = " and not " + skip_case
            skip_options += skip_option
        skip_options += "'"
        test_command = (
            "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v "
            + test_case
        )
        test_command += skip_options
        return os.system(test_command)
    elif exe_list != None:
        exe_options = " -k '" + exe_list[0]
        for exe_case in exe_list[1:]:
            exe_option = " or " + exe_case
            exe_options += exe_option
        exe_options += "'"
        test_command = (
            "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v "
            + test_case
        )
        test_command += exe_options
        return os.system(test_command)
    else:
        test_command = (
            "PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -v "
            + test_case
        )
        return os.system(test_command)


res = 0

# test_foreach
# full skip_list is in Issue #469
execute_list = (
    "_foreach_add_ and not slowpath",
    "_foreach_mul_ and not slowpath",
    "_foreach_div_ and not slowpath",
    "_foreach_addcmul_ and not slowpath",
    # Compiler optimization on data type conversion brings the precision error.
    "_foreach_addcdiv_ and not slowpath and not test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_xpu_float16",

    "_foreach_sqrt_ and not slowpath",
    "_foreach_lerp_ and not slowpath",
    # CPU Fallback fail
    # RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Char
    # test_foreach_reduce_large_input__foreach_norm_xpu_uint8
    "_foreach_norm_ and not slow and not cuda and not test_foreach_reduce_large_input__foreach_norm_xpu_bool and not test_foreach_reduce_large_input__foreach_norm_xpu_int16 and not test_foreach_reduce_large_input__foreach_norm_xpu_int32 and not test_foreach_reduce_large_input__foreach_norm_xpu_int64 and not test_foreach_reduce_large_input__foreach_norm_xpu_int8 and not test_foreach_reduce_large_input__foreach_norm_xpu_uint8"
)
res += launch_test("test_foreach_xpu.py", exe_list=execute_list)

# test_decomp
# full skip_list is in Issue #470
execute_list = (
    "test_comprehensive_nn_functional_cross_entropy_xpu",
    "test_comprehensive_nn_functional_nll_loss_xpu_bfloat16",
    "test_comprehensive_nn_functional_nll_loss_xpu_float32",
    "test_comprehensive_nn_functional_nll_loss_xpu_float64",
    "bincount",
)
res += launch_test("test_decomp_xpu.py", exe_list=execute_list)

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
