skip_dict = {
    "test_ops_xpu.py": (
        # RuntimeError: Required aspect fp64 is not supported on the device
        # https://github.com/intel/torch-xpu-ops/issues/628
        "test_compare_cpu_bincount_xpu_int16",
        "test_compare_cpu_bincount_xpu_int32",
        "test_compare_cpu_bincount_xpu_int64",
        "test_compare_cpu_bincount_xpu_int8",
        "test_compare_cpu_bincount_xpu_uint8",
        # RuntimeError: Kernel is incompatible with all devices in devs
        # https://github.com/intel/torch-xpu-ops/issues/1150
        "test_compare_cpu_logcumsumexp_xpu_float16",
        "test_compare_cpu_logcumsumexp_xpu_float32",
        "test_compare_cpu_nn_functional_pdist_xpu_float32",
        "test_compare_cpu_tril_indices_xpu_int32",
        "test_compare_cpu_tril_indices_xpu_int64",
        "test_compare_cpu_triu_indices_xpu_int32",
        "test_compare_cpu_triu_indices_xpu_int64",
        "test_backward_logcumsumexp_xpu_float32",
        "test_backward_nn_functional_pdist_xpu_float32",
        "test_forward_ad_logcumsumexp_xpu_float32",
        "test_operator_logcumsumexp_xpu_float32",
        "test_operator_nn_functional_pdist_xpu_float32",
        "test_view_replay_logcumsumexp_xpu_float32",
        "test_view_replay_nn_functional_pdist_xpu_float32",
    ),
}
