skip_dict = {
    # tensor(0.-0.j, device='xpu:0', dtype=torch.complex32) tensor(nan+nanj, device='xpu:0', dtype=torch.complex32) (1.5707964+0j)
    "test_unary_ufuncs_xpu.pyy": (
        "test_reference_numerics_small_acos_xpu_complex32",
        "test_reference_numerics_small_asin_xpu_complex32",
        "test_reference_numerics_small_asinh_xpu_complex32",
        "test_reference_numerics_small_atan_xpu_complex32",
        "test_reference_numerics_small_atanh_xpu_complex32",
        # Need to check compiler std::sin() on inf+infj
        "test_reference_numerics_extremal__refs_sin_xpu_complex128",
        "test_reference_numerics_extremal__refs_sin_xpu_complex64",
        "test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex128",
        "test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex64",
        "test_reference_numerics_extremal_sin_xpu_complex128",
        "test_reference_numerics_extremal_sin_xpu_complex64",
        "test_reference_numerics_extremal_sinh_xpu_complex128",
        "test_reference_numerics_extremal_sinh_xpu_complex64",
        "test_reference_numerics_large__refs_sin_xpu_complex32",
        "test_reference_numerics_large_sin_xpu_complex32",
        # Known issue of exp accuracy
        # tensor(13437.7000-501.j, device='xpu:0', dtype=torch.complex128) tensor(inf+infj, device='xpu:0', dtype=torch.complex128) (-inf+infj)
        "test_reference_numerics_large__refs_exp_xpu_complex128",
        "test_reference_numerics_large_exp_xpu_complex128",
        "test_reference_numerics_small_exp_xpu_complex32",
        ":test_reference_numerics_normal_special_i1_xpu_float32",
        "test_reference_numerics_normal_sigmoid_xpu_complex32",
    ),
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
}
