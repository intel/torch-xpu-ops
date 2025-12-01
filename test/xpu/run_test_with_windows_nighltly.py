import os
import sys

from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl
from xpu_test_utils import launch_test

res = 0
IS_WINDOWS = sys.platform == "win32"

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

    skip_list = skip_dict[key]
    if IS_WINDOWS and key in skip_dict_win:
        skip_list += skip_dict_win[key]
    if IS_WINDOWS and key in skip_dict_win_lnl:
        skip_list += skip_dict_win_lnl[key]

    print(f"\n=== Processing test case: {key} ===")
    res += launch_test(key, skip_list=skip_list)

if os.name == "nt":
    sys.exit(res)
else:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
