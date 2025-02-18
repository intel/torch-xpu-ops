import os
import sys

# Cases in the file is too slow to run all suites on CPU. So add white list.


def launch_test(test_case, skip_list=None, exe_list=None):
    os.environ["PYTORCH_ENABLE_XPU_FALLBACK"] = "1"
    os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
    if skip_list is not None:
        skip_options = ' -k "not ' + skip_list[0]
        for skip_case in skip_list[1:]:
            skip_option = " and not " + skip_case
            skip_options += skip_option
        skip_options += '"'
        test_command = (
            "pytest -v "
            + "--junit-xml=./op_ut_with_only.xml "
            + test_case
            + skip_options
        )
        return os.system(test_command)
    elif exe_list is not None:
        exe_options = ' -k "' + exe_list[0]
        for exe_case in exe_list[1:]:
            exe_option = " or " + exe_case
            exe_options += exe_option
        exe_options += '"'
        test_command = (
            "pytest -v "
            + "--junit-xml=./op_ut_with_only.xml "
            + test_case
            + exe_options
        )
        return os.system(test_command)
    else:
        test_command = "pytest -v --junit-xml=./op_ut_with_only.xml " + test_case
        return os.system(test_command)


res = 0

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

if os.name == "nt":
    sys.exit(res)
else:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
