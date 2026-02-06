# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

skip_dict = {
    "complex_tensor/test_complex_tensor_xpu.py": None,
    "functorch/test_ops_xpu.py": None,
    "nn/test_convolution_xpu.py": None,
    "nn/test_dropout_xpu.py": None,
    "nn/test_embedding_xpu.py": None,
    "nn/test_init_xpu.py": None,
    "nn/test_lazy_modules_xpu.py": None,
    "nn/test_load_state_dict_xpu.py": None,
    "nn/test_module_hooks_xpu.py": None,
    "nn/test_multihead_attention_xpu.py": None,
    "nn/test_packed_sequence_xpu.py": None,
    "nn/test_parametrization_xpu.py": None,
    "nn/test_pooling_xpu.py": None,
    "nn/test_pruning_xpu.py": None,
    "quantization/core/test_quantized_op_xpu.py": None,
    "quantization/core/test_quantized_tensor_xpu.py": None,
    "quantization/core/test_workflow_module_xpu.py": None,
    "quantization/core/test_workflow_ops_xpu.py": None,
    "test_autocast_xpu.py": None,
    "test_autograd_fallback_xpu.py": None,
    "test_autograd_xpu.py": None,
    "test_binary_ufuncs_xpu.py": ("_jiterator_",),
    "test_comparison_utils_xpu.py": None,
    "test_complex_xpu.py": None,
    "test_content_store_xpu.py": None,
    "test_dataloader_xpu.py": None,
    "test_decomp_xpu.py": None,
    "test_distributions_xpu.py": None,
    "test_dynamic_shapes_xpu.py": None,
    "test_foreach_xpu.py": (
        # RuntimeError: Tried to instantiate dummy base class CUDAGraph
        "use_cuda_graph_True",
    ),
    "test_indexing_xpu.py": None,
    "test_linalg_xpu.py": (
        "_tunableop_",
        "_tuning_tunableop_",
    ),
    "test_masked_xpu.py": None,
    "test_maskedtensor_xpu.py": None,
    "test_meta_xpu.py": ("_jiterator_",),
    "test_modules_xpu.py": None,
    "test_namedtensor_xpu.py": None,
    "test_native_functions_xpu.py": None,
    "test_native_mha_xpu.py": None,
    "test_nn_xpu.py": None,
    "test_ops_fwd_gradients_xpu.py": None,
    "test_ops_gradients_xpu.py": None,
    "test_ops_xpu.py": (
        "_jiterator_",
        # crash
        "test_dtypes__refs_nn_functional_pdist_xpu",
        # not implemented
        "histogramdd",
        "_flash_attention_",
        "_efficient_attention_",
        # Exception: The supported dtypes for linalg.multi_dot on device type xpu are incorrect!
        "test_dtypes_linalg_multi_dot_xpu",
    ),
    "test_optim_xpu.py": None,
    "test_reductions_xpu.py": None,
    "test_scatter_gather_ops_xpu.py": None,
    "test_segment_reductions_xpu.py": None,
    "test_shape_ops_xpu.py": None,
    "test_sort_and_select_xpu.py": None,
    "test_sparse_csr_xpu.py": None,
    "test_sparse_xpu.py": None,
    "test_spectral_ops_xpu.py": None,
    "test_tensor_creation_ops_xpu.py": None,
    "test_torch_xpu.py": (
        # TypeError: map2_ is only implemented on CPU tensors
        "test_broadcast_fn_map2_xpu",
        "test_broadcast_fn_map_xpu",
        # RuntimeError: _share_fd_: only available on CPU
        "test_module_share_memory_xpu",
    ),
    "test_transformers_xpu.py": None,
    "test_type_promotion_xpu.py": None,
    "test_unary_ufuncs_xpu.py": ("_jiterator_",),
    "test_view_ops_xpu.py": None,
    "test_schema_check.py": None,
    "test_nestedtensor_xpu.py": None,
    "functorch/test_eager_transforms_xpu.py": None,
    "test_cpp_api_parity_xpu.py": None,
    "test_expanded_weights_xpu.py": None,
    "test_fake_tensor_xpu.py": None,
    "test_matmul_cuda_xpu.py": None,
    "functorch/test_vmap_xpu.py": None,
    "test/xpu/dynamo/test_ctx_manager_xpu.py": None,
    "functorch/test_control_flow_xpu.py": None,
}
