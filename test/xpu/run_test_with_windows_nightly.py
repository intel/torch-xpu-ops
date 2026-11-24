import os
import sys

from skip_list_common import PYTORCH_TEST_DIR, skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl
from xpu_test_utils import launch_test

res = 0
IS_WINDOWS = sys.platform == "win32"


skip_files_list = [
    f"{PYTORCH_TEST_DIR}/test_autocast.py",
    f"{PYTORCH_TEST_DIR}/test_autograd_fallback.py",
    "test_autograd_xpu.py",
    # "test_binary_ufuncs_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_comparison_utils.py",
    "test_complex_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_content_store.py",
    "test_dataloader_xpu.py",
    "test_decomp.py",
    "test_decomp_xpu.py",
    "test_distributions_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_dynamic_shapes.py",
    f"{PYTORCH_TEST_DIR}/test_foreach.py",
    "test_linalg_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_maskedtensor.py",
    # f"{PYTORCH_TEST_DIR}/test_masked.py",
    "test_matmul_cuda_xpu.py",
    "test_meta_xpu.py",
    "test_native_functions_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_native_mha.py",
    "test_nestedtensor_xpu.py",
    "test_nn_xpu.py",
    f"{PYTORCH_TEST_DIR}/test_ops_fwd_gradients.py",
    # f"{PYTORCH_TEST_DIR}/test_ops_gradients.py",
    # "test_ops_xpu.py",
    "test_optim_xpu.py",
    # f"{PYTORCH_TEST_DIR}/"test_reductions.py",
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
    # f"{PYTORCH_TEST_DIR}/test_view_ops.py",
    f"{PYTORCH_TEST_DIR}/functorch/test_ops.py",
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

    skip_list: list[str] = list(skip_dict.get(key) or ())
    if IS_WINDOWS:
        skip_list.extend(skip_dict_win.get(key) or ())
        skip_list.extend(skip_dict_win_lnl.get(key) or ())

    print(f"\n=== Processing test case: {key} ===")
    if launch_test(key, skip_list=skip_list):
        res = 1

sys.exit(res)
