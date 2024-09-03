skip_dict = {
    "test_ops_xpu.py": (
        # RuntimeError: Required aspect fp64 is not supported on the device
        # https://github.com/intel/torch-xpu-ops/issues/628
        "test_compare_cpu_bincount_xpu_int16",
        "test_compare_cpu_bincount_xpu_int32",
        "test_compare_cpu_bincount_xpu_int64",
        "test_compare_cpu_bincount_xpu_int8",
        "test_compare_cpu_bincount_xpu_uint8",
    ),
}
