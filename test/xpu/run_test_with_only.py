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

# test_foreach
execute_list = (
    "_foreach_add_ and not slowpath",
    "_foreach_mul_ and not slowpath",
    "_foreach_div_ and not slowpath",
    "_foreach_addcmul_ and not slowpath",
    # Compiler optimization on data type conversion brings the precision error.
    "_foreach_addcdiv_ and not slowpath and not test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_xpu_float16",
    "_foreach_sqrt_ and not slowpath",
)
res += launch_test("test_foreach_xpu.py", exe_list=execute_list)

# test_decomp
execute_list = (
    "test_comprehensive_nn_functional_cross_entropy_xpu",
    "test_comprehensive_nn_functional_nll_loss_xpu_bfloat16",
    "test_comprehensive_nn_functional_nll_loss_xpu_float32",
    "test_comprehensive_nn_functional_nll_loss_xpu_float64"
)
res += launch_test("test_decomp_xpu.py", exe_list=execute_list)

# test_torch
execute_list = (
    "test_pin_memory",
)
res += launch_test("test_torch_xpu.py", exe_list=execute_list)

# test_comparison_utils
res += launch_test("test_comparison_utils_xpu.py")

# test_pruning
res += launch_test("nn/test_pruning_xpu.py")

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
