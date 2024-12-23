skip_dict = {
    "test_ops_xpu.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1173
        # Fatal Python error: Illegal instruction
        "test_compare_cpu_grid_sampler_2d_xpu_float64",
        "test_compare_cpu_cosh_xpu_complex64",
        "test_compare_cpu_nn_functional_softshrink_xpu_bfloat16",
        "test_compare_cpu_nn_functional_softshrink_xpu_float16",
        "test_compare_cpu_nn_functional_softshrink_xpu_float32",
        "test_compare_cpu_nn_functional_softshrink_xpu_float64",
        "test_compare_cpu_square_xpu_complex128",
    ),
}
