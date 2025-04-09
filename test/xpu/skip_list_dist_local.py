skip_dict = {
    "../../../../test/distributed/fsdp/test_checkpoint_wrapper.py": None,
    # https://github.com/intel/torch-xpu-ops/issues/1536
    # "../../../../test/distributed/fsdp/test_distributed_checkpoint.py": (
    #    "test_distributed_checkpoint_state_dict_type0_xpu",
    #    "test_distributed_checkpoint_state_dict_type1_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_fsdp_apply.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_backward_prefetch.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_checkpoint.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_clip_grad_norm.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_comm.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_comm_hooks.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_core.py": (
        "test_delayed_optim_step_offload_true_no_shard_xpu",
        "test_transformer_no_grad_mixed_precision_True_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_dtensor_state_dict.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_exec_order.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_fine_tune.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_flatten_params.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_freezing_weights.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_fx.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_grad_acc.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_hybrid_shard.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_ignored_modules.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_input.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_memory.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_meta.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_misc.py": (
        "test_fsdp_zero2_eval_with_prefetch",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_mixed_precision.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_forward.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_wrapping.py": None,
    # https://github.com/intel/torch-xpu-ops/issues/1537
    "../../../../test/distributed/fsdp/test_fsdp_optim_state.py": (
        "test_use_orig_params",
    ),
    # Performance check, skip
    # "../../../../test/distributed/fsdp/test_fsdp_overlap.py": (
    #    "test_forward_overlap",
    #    "test_forward_overlap_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_fsdp_pure_fp16.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_state_dict.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_tp_integration.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_traversal.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_uneven.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_unshard_params.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_use_orig_params.py": None,
    "../../../../test/distributed/fsdp/test_hsdp_dtensor_state_dict.py": None,
    "../../../../test/distributed/fsdp/test_shard_utils.py": None,
    "../../../../test/distributed/fsdp/test_utils.py": None,
    "../../../../test/distributed/fsdp/test_wrap.py": None,
    "../../../../test/distributed/test_backends.py": None,
    "../../../../test/distributed/test_c10d_common.py": None,
    "../../../../test/distributed/test_c10d_functional_native.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1508
        # RuntimeError: oneCCL: coll_param.cpp:455 validate: EXCEPTION: average operation is not supported for the scheduler path
        "test_reduce_scatter_tensor_coalesced",
        "test_reduce_scatter_tensor_single",
        # https://github.com/intel/torch-xpu-ops/issues/1525
        # ValueError: trying to initialize the default process group twice!
        "test_inductor_all_gather_into_tensor_coalesced",
        "test_inductor_all_gather_into_tensor_single",
        "test_inductor_all_reduce_coalesced",
        "test_inductor_all_reduce_non_contig_input",
        "test_inductor_all_reduce_single",
        "test_inductor_all_to_all_single",
        "test_inductor_broadcast",
        "test_inductor_inplace_op_on_view",
        "test_inductor_reduce_scatter_tensor_coalesced",
        "test_inductor_reduce_scatter_tensor_single",
        "test_inductor_reuse_buffer_after_inplace_collective",
        "test_ranks_and_tag",
        "test_wait_tensor",
    ),
    "../../../../test/distributed/test_c10d_logger.py": None,
    "../../../../test/distributed/test_c10d_object_collectives.py": (
        # RuntimeError: Process 0 terminated or timed out after 300.09047198295593 seconds
        # https://github.com/intel/torch-xpu-ops/issues/1535
        "test_gather_object_cpu",
        "test_gather_object_xpu",
        "test_gather_object_list_cpu",
        "test_gather_object_list_xpu",
    ),
    "../../../../test/distributed/test_compute_comm_reordering.py": None,
    "../../../../test/distributed/test_control_collectives.py": None,
    "../../../../test/distributed/test_device_mesh.py": None,
    "../../../../test/distributed/test_dynamo_distributed.py": (
        # AttributeError:'torch._C._distributed_c10d.ProcessGroupXCCL' object has no attribute '_set_default_timeout'
        "test_asymmetric_compilation",
        "test_asymmetric_compilation_with_fx_cache",
        # ValueError: FlexAttention is only supported on CUDA or CPU devices. Found input tensors on xpu device.
        "test_compiled_flex_attention_full_model_ddp",
        "test_compiled_flex_attention_local_ddp",
        # torch._dynamo.exc.InternalTorchDynamoError: AttributeError: __enter__
        # https://github.com/intel/torch-xpu-ops/issues/1527
        "test_compiler_collectives_automatic_dynamic_scalar",
        "test_compiler_collectives_automatic_dynamic_speculation_divergence",
        "test_compiler_collectives_automatic_dynamic_tensor",
        "test_compiler_collectives_dim_mismatch",
        "test_compiler_collectives_graph_break_empty_graph_still_collective",
        "test_compiler_collectives_missing_source",
        "test_compiler_collectives_scalar_missing_source",
        "test_compiler_collectives_type_mismatch",
        "test_ddp_activation_checkpointing",
        "test_ddp_baseline_aot_eager_multiprocess",
        "test_fsdp_activation_checkpointing",
        "test_fsdp_aot_eager",
        "test_fsdp_inductor",
        "test_fsdp_setattr",
        "test_fsdp_unspecialized_forced_getattr_inline",
        "test_fsdp_unspecialized_forced_getattr_no_inline",
        # RuntimeError: UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
        # https://github.com/intel/torch-xpu-ops/issues/1526
        "test_get_pg_attr",
    ),
    "../../../../test/distributed/test_fake_pg.py": None,
    "../../../../test/distributed/test_functional_api.py": (
        # RuntimeError: UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
        # https://github.com/intel/torch-xpu-ops/issues/1526
        "test_tracing_xpu",
        "test_tracing and test_tracing_with_fakepg and test_tracing_with_fakepg_xpu and test_tracing_with_dce_code and test_tracing_with_dce_code_xpu",
    ),
    "../../../../test/distributed/test_multi_threaded_pg.py": (
        # oneccl not support multi-threaded well, so skip it first.
        "test_bwd_sees_fwd_pg",
    ),
    "../../../../test/distributed/test_store.py": None,
    "../../../../test/distributed/pipelining/test_backward.py": None,
    "../../../../test/distributed/pipelining/test_microbatch.py": None,
    "../../../../test/distributed/pipelining/test_pipe.py": None,
    "../../../../test/distributed/pipelining/test_schedule.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
    "../../../../test/distributed/pipelining/test_unflatten.py": None,
    "../../../../test/distributed/tensor/parallel/test_micro_pipeline_tp.py": (
        # NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter'
        # is not currently implemented for the XPU device
        # https://github.com/intel/torch-xpu-ops/issues/1547
        "test_dtensor_seq_par_shard_dim_0",
        "test_dtensor_seq_par_shard_dim_1",
        "test_fuse_matmul_reduce_scatter_A_dims_2_scatter_dim_0",
        "test_fuse_matmul_reduce_scatter_A_dims_2_scatter_dim_1",
        "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_0",
        "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_1",
        "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_2",
        # AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: ......'
        # https://github.com/intel/torch-xpu-ops/issues/1548
        "test_fuse_all_gather_matmul_A_dims_2_gather_dim_0_return_A_False",
        "test_fuse_all_gather_matmul_A_dims_2_gather_dim_0_return_A_True",
        "test_fuse_all_gather_matmul_A_dims_3_gather_dim_0_return_A_False",
        "test_fuse_all_gather_matmul_A_dims_3_gather_dim_0_return_A_True",
        "test_fuse_all_gather_matmul_A_dims_3_gather_dim_1_return_A_False",
        "test_fuse_all_gather_matmul_A_dims_3_gather_dim_1_return_A_True",
        # AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......'
        # https://github.com/intel/torch-xpu-ops/issues/1549
        "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_0_return_A_False",
        "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_0_return_A_True",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_0_return_A_False",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_0_return_A_True",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_1_return_A_False",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_1_return_A_True",
        # NotImplementedError: The operator 'aten::_scaled_mm.out' is not currently implemented for the XPU device.
        # https://github.com/intel/torch-xpu-ops/issues/1550
        "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_1_return_A_False",
        "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_1_return_A_True",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_2_return_A_False",
        "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_2_return_A_True",
        # NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter'
        # is not currently implemented for the XPU device.
        # https://github.com/intel/torch-xpu-ops/issues/1551
        "test_fuse_scaled_matmul_reduce_scatter_A_dims_2_scatter_dim_0",
        "test_fuse_scaled_matmul_reduce_scatter_A_dims_2_scatter_dim_1",
        "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_0",
        "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_1",
        "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_2",
        "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_0",
        "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_1",
        "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_2",
    ),
    "../../../../test/distributed/tensor/parallel/test_tp_examples.py": (
        # RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!
        # https://github.com/intel/torch-xpu-ops/issues/1555
        "test/distributed/tensor/parallel/test_tp_examples.py::DistTensorParallelExampleTest::test_transformer_req_grad_seq_parallel_float32_thaw_all",
        "test_transformer_req_grad_seq_parallel_float32_thaw_layers_0_attention_wv__layers_0_feed_forward_w1__layers_1_feed_forward_w2__layers_1_ffn_norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_layers_1_ffn_norm__norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_output__tok_embeddings",
        "test_transformer_training_is_seq_parallel_False_float32",
        "test_transformer_training_is_seq_parallel_True_float32",
        # NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered.
        # https://github.com/intel/torch-xpu-ops/issues/1556
        "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output",
    ),
    "../../../../test/distributed/tensor/parallel/test_tp_random_state.py": None,
    "../../../../test/distributed/tensor/parallel/test_parallelize_api.py": None,
    "../../../../test/distributed/tensor/parallel/test_tp_style.py": None,
    "../../../../test/distributed/tensor/test_api.py": None,
    "../../../../test/distributed/tensor/test_attention.py": None,
    "../../../../test/distributed/tensor/test_common_rules.py": None,
    "../../../../test/distributed/tensor/test_dtensor.py": None,
    "../../../../test/distributed/tensor/test_dtensor_compile.py": None,
    "../../../../test/distributed/tensor/test_experimental_ops.py": None,
    "../../../../test/distributed/tensor/test_init.py": None,
    "../../../../test/distributed/tensor/test_math_ops.py": (
        # RuntimeError: oneCCL: coll_param.cpp:455 validate: EXCEPTION: average operation is not supported for the scheduler path
        # https://github.com/intel/torch-xpu-ops/issues/1508
        "test_mean",
        "test_nll_loss_and_cross_entropy",
    ),
    "../../../../test/distributed/tensor/test_random_ops.py": None,
    "../../../../test/distributed/tensor/test_redistribute.py": None,
    "../../../../test/distributed/tensor/test_tensor_ops.py": None,
    "../../../../test/distributed/tensor/experimental/test_register_sharding.py": None,
}

skip_dict_python = {
    "distributed/test_c10d_ops_xccl.py": None,
    "distributed/test_c10d_xccl.py": None,
    "../../../../test/distributed/pipelining/test_schedule_multiproc.py": None,  # Hang error.
    "../../../../test/distributed/pipelining/test_stage.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
}
