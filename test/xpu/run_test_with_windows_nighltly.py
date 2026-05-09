import os
import sys

from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl

res = 0
IS_WINDOWS = sys.platform == "win32"


def launch_test(test_case, skip_list=None, exe_list=None):
    os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
    module_name = test_case.replace(".py", "").replace("/", ".").replace("\\", ".")
    if skip_list is not None:
        if skip_list:
            skip_options = ' -k "not ' + skip_list[0]
            for skip_case in skip_list[1:]:
                skip_option = " and not " + skip_case
                skip_options += skip_option
            skip_options += '"'
            test_command = (
                f"pytest --junit-xml=./op_ut_with_skip.{module_name}.xml " + test_case
            )
            test_command += skip_options
        else:
            test_command = (
                f"pytest --junit-xml=./op_ut_with_all.{module_name}.xml " + test_case
            )
    elif exe_list is not None:
        if exe_list:
            exe_options = ' -k "' + exe_list[0]
            for exe_case in exe_list[1:]:
                exe_option = " or " + exe_case
                exe_options += exe_option
            exe_options += '"'
            test_command = (
                f"pytest --junit-xml=./op_ut_with_exe.{module_name}.xml " + test_case
            )
            test_command += exe_options
        else:
            test_command = (
                f"pytest --junit-xml=./op_ut_with_all.{module_name}.xml " + test_case
            )
    else:
        test_command = (
            f"pytest --junit-xml=./op_ut_with_all.{module_name}.xml " + test_case
        )
    return os.system(test_command)


skip_files_list = [
    "test_autocast_xpu.py",
    "test_autograd_fallback_xpu.py",
    "test_autograd_xpu.py",
    # "test_binary_ufuncs_xpu.py",
    "test_comparison_utils_xpu.py",
    "test_complex_xpu.py",
    "test_content_store_xpu.py",
    "test_dataloader_xpu.py",
    "test_decomp.py",
    "test_decomp_xpu.py",
    "test_distributions_xpu.py",
    "test_dynamic_shapes_xpu.py",
    "test_foreach_xpu.py",
    # "test_indexing_xpu.py",
    "test_linalg_xpu.py",
    "test_maskedtensor_xpu.py",
    # "test_masked_xpu.py",
    "test_matmul_cuda_xpu.py",
    "test_meta_xpu.py",
    # "test_modules_xpu.py",
    "test_namedtensor_xpu.py",
    "test_native_functions_xpu.py",
    "test_native_mha_xpu.py",
    "test_nestedtensor_xpu.py",
    "test_nn_xpu.py",
    "test_ops_fwd_gradients_xpu.py",
    # "test_ops_gradients_xpu.py",
    # "test_ops_xpu.py",
    "test_optim_xpu.py",
    # "test_reductions_xpu.py",
    # "test_scatter_gather_ops_xpu.py",
    "test_segment_reductions_xpu.py",
    "test_shape_ops_xpu.py",
    "test_sort_and_select_xpu.py",
    "test_sparse_csr_xpu.py",
    "test_sparse_xpu.py",
    "test_spectral_ops_xpu.py",
    "test_tensor_creation_ops_xpu.py",
    # "test_torch_xpu.py",
    # "test_transformers_xpu.py",
    "test_type_promotion_xpu.py",
    # "test_unary_ufuncs_xpu.py",
    # "test_view_ops_xpu.py",
    "functorch/test_ops_xpu.py",
]

print("Current working directory:", os.getcwd())
print("Files in directory:")
for file in os.listdir("."):
    if file.endswith(".py"):
        print(f"  {file}")

for key in skip_dict:
    # Check if key is in skip list
    if key in skip_files_list:
        print(f"\n=== Skipping test file: {key} ===")
        continue

    skip_list = skip_dict.get(key)
    if skip_list is None:
        skip_list = []

    if IS_WINDOWS and key in skip_dict_win:
        win_skip_list = skip_dict_win[key]
        if isinstance(win_skip_list, tuple):
            skip_list.extend(list(win_skip_list))
        elif win_skip_list is not None:
            skip_list.extend(win_skip_list)
    if IS_WINDOWS and key in skip_dict_win_lnl:
        win_lnl_skip_list = skip_dict_win_lnl[key]
        if isinstance(win_lnl_skip_list, tuple):
            skip_list.extend(list(win_lnl_skip_list))
        elif win_lnl_skip_list is not None:
            skip_list.extend(win_lnl_skip_list)

    print(f"\n=== Processing test case: {key} ===")
    res += launch_test(key, skip_list=skip_list)

if os.name == "nt":
    sys.exit(res)
else:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
