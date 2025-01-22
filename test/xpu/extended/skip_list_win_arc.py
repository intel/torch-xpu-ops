skip_dict = {
    # SYCL Compiler on Windows removed the following operations when '-cl-poison-unsupported-fp64-kernels' is on
    # Hence, skip the following windows specific errors
    "test_ops_xpu.py": (
        "test_compare_cpu_sqrt_xpu_complex64",
        "test_backward_nn_functional_adaptive_avg_pool2d_xpu_float32",
    ),
}
