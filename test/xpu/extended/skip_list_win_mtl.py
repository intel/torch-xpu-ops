skip_dict = {
    # failed on MTL windows, skip first for Preci
    "test_ops_xpu.py": (
        "test_compare_cpu_sqrt_xpu_complex64",
        "test_backward_nn_functional_adaptive_avg_pool2d_xpu_float32",

        "test_compare_cpu_cosh_xpu_complex128",
        "test_compare_cpu_frexp_xpu_bfloat16",
        "test_compare_cpu_frexp_xpu_float16",
        "test_compare_cpu_frexp_xpu_float32",
        "test_compare_cpu_frexp_xpu_float64",
        "test_compare_cpu_max_pool2d_with_indices_backward_xpu_bfloat16",
        "test_compare_cpu_max_pool2d_with_indices_backward_xpu_float16",
        "test_compare_cpu_max_pool2d_with_indices_backward_xpu_float32",
        "test_compare_cpu_max_pool2d_with_indices_backward_xpu_float64",
        "test_compare_cpu_nn_functional_avg_pool2d_xpu_bfloat16",
        "test_compare_cpu_nn_functional_avg_pool2d_xpu_float32",
        "test_compare_cpu_nn_functional_avg_pool3d_xpu_float32",
    ),
}
