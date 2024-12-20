skip_dict = {
    "test_ops_xpu.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1173
        # Fatal Python error: Illegal instruction
        "test_compare_cpu_grid_sampler_2d_xpu_float64",
    ),
}
