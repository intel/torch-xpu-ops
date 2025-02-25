skip_dict = {
    "test_ops_xpu.py": (
        # RuntimeError: Required aspect fp64 is not supported on the device
        # https://github.com/intel/torch-xpu-ops/issues/628
        "test_compare_cpu_bincount_xpu_int16",
        "test_compare_cpu_bincount_xpu_int32",
        "test_compare_cpu_bincount_xpu_int64",
        "test_compare_cpu_bincount_xpu_int8",
        "test_compare_cpu_bincount_xpu_uint8",
        "test_compare_cpu_igamma_xpu_float32",
        "test_compare_cpu_igammac_xpu_float32",
        "test_compare_cpu_logcumsumexp_xpu_complex64",
        "test_compare_cpu_logspace_tensor_overload_xpu_complex64",
        "test_compare_cpu_logspace_xpu_complex64",
        "test_compare_cpu_rsqrt_xpu_complex64",
        "test_backward_nn_functional_rrelu_xpu_float32",
        "test_cow_input_igamma_xpu_float32",
        "test_cow_input_igammac_xpu_float32",
        "test_cow_input_nn_functional_rrelu_xpu_float32",
        "test_forward_ad_nn_functional_rrelu_xpu_float32",
        "test_operator_igamma_xpu_float32",
        "test_operator_igammac_xpu_float32",
        "test_operator_nn_functional_rrelu_xpu_float32",
        "test_view_replay_igamma_xpu_float32",
        "test_view_replay_igammac_xpu_float32",
        "test_view_replay_nn_functional_rrelu_xpu_float32",
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
        "test_cow_input_nn_functional_pdist_xpu_float32",
        # NotImplementedError: The operator 'aten::_upsample_bilinear2d_aa_backward.grad_input'
        # is not currently implemented for the XPU device
        "test_backward__upsample_bilinear2d_aa_xpu_float32",
        "test_cow_input__upsample_bilinear2d_aa_xpu_float32",
    ),
}
