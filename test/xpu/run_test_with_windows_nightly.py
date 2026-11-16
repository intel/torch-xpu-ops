"""Run the torch-xpu-ops unit tests of the Windows nightly, one pytest process per file."""

import sys

from skip_list_common import PYTORCH_TEST_DIR, skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl
from ut_launcher import main, merge_skip_dicts

# Files the Windows nightly does not run
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
    "test_maskedtensor_xpu.py",
    # "test_masked_xpu.py",
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
    "functorch/test_ops_xpu.py",
]

if __name__ == "__main__":
    overlays = [skip_dict_win, skip_dict_win_lnl] if sys.platform == "win32" else []
    sys.exit(
        main(
            merge_skip_dicts(skip_dict, *overlays),
            __doc__,
            skip_files=skip_files_list,
        )
    )
