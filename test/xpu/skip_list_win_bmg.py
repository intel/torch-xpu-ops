skip_dict = {
    # tensor(0.-0.j, device='xpu:0', dtype=torch.complex32) tensor(nan+nanj, device='xpu:0', dtype=torch.complex32) (1.5707964+0j)
    "test_unary_ufuncs_xpu.py": None,
    # https://github.com/intel/torch-xpu-ops/issues/1171
    # AssertionError: 'Assertion maxind >= 0 && maxind < outputImageSize failed' not found in '\nAssertHandler::printMessage\n' : The expected error was not found
    "nn\test_pooling_xpu.py": (
        "test_MaxUnpool_index_errors_case1_xpu",
        "test_MaxUnpool_index_errors_case2_xpu",
        "test_MaxUnpool_index_errors_case4_xpu",
        "test_MaxUnpool_index_errors_case6_xpu",
        "test_MaxUnpool_index_errors_case7_xpu",
        "test_MaxUnpool_index_errors_case9_xpu",
    ),
    "functorch/test_ops_functorch_xpu.py": None,
}
