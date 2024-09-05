import os
import sys
from xpu_test_utils import launch_test


# Cases in the file is too slow to run all suites on CPU. So add white list.

# test_decomp
# full skip_list is in Issue #470
execute_list = (
    "test_comprehensive_nn_functional_cross_entropy_xpu",
    "test_comprehensive_nn_functional_nll_loss_xpu_bfloat16",
    "test_comprehensive_nn_functional_nll_loss_xpu_float32",
    "test_comprehensive_nn_functional_nll_loss_xpu_float64",
    "bincount",
)
return_code, count_buf, fails = launch_test("test_decomp_xpu.py", exe_list=execute_list)

if os.name == "nt":
    sys.exit(return_code)
else:    
    exit_code = os.WEXITSTATUS(return_code)
    sys.exit(exit_code)
