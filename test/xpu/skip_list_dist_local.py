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
    "../../../../test/distributed/fsdp/test_fsdp_core.py": None,
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
        # fsdp accuracy gaps
        # https://github.com/intel/torch-xpu-ops/issues/1504, Performance test, should skip
        "test_fsdp_optimizer_overlap",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_mixed_precision.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_forward.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_wrapping.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_optim_state.py": None,
    # Performance check, skip
    # "../../../../test/distributed/fsdp/test_fsdp_overlap.py": (
    #    # fsdp accuracy gaps
        # https://github.com/intel/torch-xpu-ops/issues/1504
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
    "../../../../test/distributed/test_c10d_logger.py": None,
    "../../../../test/distributed/test_c10d_object_collectives.py": (
        # RuntimeError: Process 2 exited with error code 10 and exception: ; AssertionError: Scalars are not equal!
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_scatter_object_list_cpu",
        "test_scatter_object_list_xpu",
    ),
    "../../../../test/distributed/test_compute_comm_reordering.py": None,
    "../../../../test/distributed/test_control_collectives.py": None,
    "../../../../test/distributed/test_device_mesh.py": (
        # RuntimeError: Process 1 exited with error code 10 and exception:
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_scatter_1d",
        "test_scatter_uneven",
    ),
    "../../../../test/distributed/test_dynamo_distributed.py": None,
    "../../../../test/distributed/test_fake_pg.py": None,
    "../../../../test/distributed/test_functional_api.py": None,
    "../../../../test/distributed/test_inductor_collectives.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1581, 2.8 skipped
        # Fatal Python error: Segmentation fault
        "test_dynamo_rewrite_dist_all_gather",
        "test_dynamo_rewrite_dist_all_gather_list",
        "test_dynamo_rewrite_dist_all_gather_args_match",
        "test_dynamo_rewrite_dist_reduce_scatter",
        "test_dynamo_support_collective_op_with_async_op_False",
        "test_dynamo_trace_reduce_scatter_tensor",
        "test_dynamo_trace_all_gather_tensor",
        "test_dynamo_trace_allgather_coalesced",
        "test_inductor_reduce_scatter_coalesced",
        "test_inductor_all_gather_coalesced",
        "test_reorder_peak_memory",
        "test_reorder_respects_wait_dep",
    ),
    "../../../../test/distributed/test_multi_threaded_pg.py": None,
    "../../../../test/distributed/test_store.py": None,
    "../../../../test/distributed/pipelining/test_backward.py": None,
    # (
    #     # fsdp accuracy gaps
          # https://github.com/intel/torch-xpu-ops/issues/1504
    #     "test_stage_backward_weight_multiple_iters_xpu",
    #     "test_stage_backward_weight_xpu",
    #     "test_stage_backward_xpu",
    # ),
    "../../../../test/distributed/pipelining/test_microbatch.py": None,
    # (
    #     # fsdp accuracy gaps
          # https://github.com/intel/torch-xpu-ops/issues/1504, need retest with oneccl fix
    #     "test_chunk_spec_xpu",
    # ),
    "../../../../test/distributed/pipelining/test_pipe.py": None,
    "../../../../test/distributed/pipelining/test_schedule.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
    "../../../../test/distributed/pipelining/test_unflatten.py": None,
    "../../../../test/distributed/tensor/parallel/test_micro_pipeline_tp.py": None,
        # NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter'
        # is not currently implemented for the XPU device
        # https://github.com/intel/torch-xpu-ops/issues/1547, 2.8 skipped
        # "test_dtensor_seq_par_shard_dim_0",
        # "test_dtensor_seq_par_shard_dim_1",
        # "test_fuse_matmul_reduce_scatter_A_dims_2_scatter_dim_0",
        # "test_fuse_matmul_reduce_scatter_A_dims_2_scatter_dim_1",
        # "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_0",
        # "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_1",
        # "test_fuse_matmul_reduce_scatter_A_dims_3_scatter_dim_2",
        # AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: ......'
        # https://github.com/intel/torch-xpu-ops/issues/1548, 2.8 skipped
        # "test_fuse_all_gather_matmul_A_dims_2_gather_dim_0_return_A_False",
        # "test_fuse_all_gather_matmul_A_dims_2_gather_dim_0_return_A_True",
        # "test_fuse_all_gather_matmul_A_dims_3_gather_dim_0_return_A_False",
        # "test_fuse_all_gather_matmul_A_dims_3_gather_dim_0_return_A_True",
        # "test_fuse_all_gather_matmul_A_dims_3_gather_dim_1_return_A_False",
        # "test_fuse_all_gather_matmul_A_dims_3_gather_dim_1_return_A_True",
        # AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......'
        # https://github.com/intel/torch-xpu-ops/issues/1549, 2.8 skipped
        # "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_0_return_A_False",
        # "test_fuse_all_gather_scaled_matmul_A_dims_2_gather_dim_0_return_A_True",
        # "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_0_return_A_False",
        # "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_0_return_A_True",
        # "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_1_return_A_False",
        # "test_fuse_all_gather_scaled_matmul_A_dims_3_gather_dim_1_return_A_True",
        # NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter'
        # is not currently implemented for the XPU device.
        # https://github.com/intel/torch-xpu-ops/issues/1551, 2.8 skipped
        # "test_fuse_scaled_matmul_reduce_scatter_A_dims_2_scatter_dim_0",
        # "test_fuse_scaled_matmul_reduce_scatter_A_dims_2_scatter_dim_1",
        # "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_0",
        # "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_1",
        # "test_fuse_scaled_matmul_reduce_scatter_A_dims_3_scatter_dim_2",
        # "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_0",
        # "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_1",
        # "test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape_scatter_dim_2",
    # ),
    "../../../../test/distributed/tensor/parallel/test_tp_examples.py": (
        # RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!
        # https://github.com/intel/torch-xpu-ops/issues/1555, 2.8 skipped
        # "test_transformer_req_grad_seq_parallel_float32_thaw_all",
        # "test_transformer_req_grad_seq_parallel_float32_thaw_layers_0_attention_wv__layers_0_feed_forward_w1__layers_1_feed_forward_w2__layers_1_ffn_norm__output__tok_embeddings",
        # "test_transformer_req_grad_seq_parallel_float32_thaw_layers_1_ffn_norm__norm__output__tok_embeddings",
        # "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output__tok_embeddings",
        # "test_transformer_req_grad_seq_parallel_float32_thaw_output__tok_embeddings",
        # "test_transformer_training_is_seq_parallel_False_float32",
        # "test_transformer_training_is_seq_parallel_True_float32",
        # NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered.
        # https://github.com/intel/torch-xpu-ops/issues/1556, 2.8 skipped
        # "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output",
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_loss_parallel",
        "test_mlp_training_is_seq_parallel_False_recompute_activation_False",
        "test_mlp_training_is_seq_parallel_True_recompute_activation_False",
        "test_transformer_req_grad_float64_thaw_all",
        "test_transformer_training_is_seq_parallel_False_float64",
        "test_transformer_training_is_seq_parallel_True_float64",
    ),
    "../../../../test/distributed/tensor/parallel/test_tp_random_state.py": None,
    "../../../../test/distributed/tensor/parallel/test_parallelize_api.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_linear_col_wise_parallel",
        "test_parallelize_mlp_with_module_api",
        "test_parallelize_mlp_with_module_api_nested",
        "test_parallelize_module_multi_wildcard",
        "test_parallelize_module_src_data_rank",
        "test_parallelize_module_with_digit",
        "test_parallelize_module_with_question",
        "test_parallelize_module_with_star",
        "test_under_devicemesh_context",
        "test_linear_row_wise_parallel",
        "test_parallelize_module_with_no_match",
        "test_parallelize_module_with_root_module",
    ),
    "../../../../test/distributed/tensor/parallel/test_tp_style.py": None,
    "../../../../test/distributed/tensor/test_api.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_distribute_tensor_rank",
        "test_distribute_tensor_uneven_sharding",
    ),
    "../../../../test/distributed/tensor/test_attention.py": None,
    "../../../../test/distributed/tensor/test_common_rules.py": None,
    "../../../../test/distributed/tensor/test_dtensor.py": None,
    "../../../../test/distributed/tensor/test_dtensor_compile.py": None,
    "../../../../test/distributed/tensor/test_experimental_ops.py": None,
    "../../../../test/distributed/tensor/test_init.py": None,
    "../../../../test/distributed/tensor/test_math_ops.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_cumsum",
        "test_layer_norm_bwd",
        "test_layer_norm_bwd_req_grad",
        "test_layer_norm_fwd",
        "test_linear_op_reductions",
        "test_shard0_svd",
        "test_softmax_fwd",
        "test_topk",
    ),
    "../../../../test/distributed/tensor/test_random_ops.py": None,
    "../../../../test/distributed/tensor/test_redistribute.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_redistribute_shard_dim_change",
        "test_redistribute_uneven_sharding",
        "test_shard_to_replicate_forward_backward",
        "test_shard_to_replicate_forward_backward_datatype_conversion",
        "test_multi_dim_mesh",
    ),
    "../../../../test/distributed/tensor/test_tensor_ops.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_aten_contiguous",
        "test_gather",
        "test_index",
        "test_slice",
        "test_stack",
        "test_where_type_promotion",
    ),
    "../../../../test/distributed/tensor/experimental/test_register_sharding.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_argmax",
        "test_softmax_fwd",
    ),
    # FSDP2
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_autograd.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_nontensor_activations",
        "test_unused_forward_module",
        "test_unused_forward_output",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_comm.py": (
        # ValueError: Cannot use ReduceOp.PREMUL_SUM with XCCL 
        # https://github.com/intel/torch-xpu-ops/issues/1571, 2.8 skipped
        "test_set_reduce_scatter_divide_factor",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_compile.py": None,
    #     # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised
    #     # https://github.com/intel/torch-xpu-ops/issues/1665, 2.8 skipped
    #     "test_transformer_backend_inductor_fullgraph_True",
    # ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_extensions.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_frozen.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_train_mixed_requires_grad_per_group",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_ignore_params.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_init.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_logging.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_memory.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_compute_dtype",
        "test_grad_acc_with_reduce_dtype",
        "test_reduce_dtype",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_overlap.py": (
        # Performance test, should skip
        "test_fully_shard_training_overlap",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_state_dict.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_state.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_training.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_1f1b_microbatching",
        "test_gradient_accumulation",
    ),
    "../../../../test/distributed/_composable/test_replicate_with_compiler.py": None,
    #     # AssertionError: Tensor-likes are not close!
    #     # https://github.com/intel/torch-xpu-ops/issues/1668, 2.8 skipped
    #     "test_compile_backward_only",
    #     "test_compile_bf16",
    #     "test_compile_fp16",
    #     "test_compile_gpu",
    #     "test_compile_gpu_ac",
    # ),
    "../../../../test/distributed/_shard/test_sharder.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_custom_sharder",
    ),
    "../../../../test/distributed/_shard/sharded_tensor/test_logger.py": None,
    "../../../../test/distributed/_shard/sharded_tensor/test_sharded_tensor.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_shard_parameter",
        "test_shard_tensor",
        "test_shard_tensor_with_empty_shard",
    ),
    "../../../../test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_sharded_tensor_reshard",
    ),
    "../../../../test/distributed/_shard/sharding_plan/test_sharding_plan.py": None,
    "../../../../test/distributed/_shard/sharding_spec/test_sharding_spec.py": None,
    "../../../../test/distributed/_tools/test_fsdp2_mem_tracker.py": None,
    # (
    #     # RuntimeError: oneCCL: coll_param.cpp:455 validate: EXCEPTION: average operation is not supported for the scheduler path
    #     # https://github.com/intel/torch-xpu-ops/issues/1508, 2.8 skipped
    #     "test_tracker_with_activation_checkpointing",
    # ),
    "../../../../test/distributed/_tools/test_mem_tracker.py": None,
    "../../../../test/distributed/_tools/test_memory_tracker.py": None,
    "../../../../test/distributed/_tools/test_mod_tracker.py": None,
    "../../../../test/distributed/checkpoint/e2e/test_e2e_save_and_load.py": None,
    "../../../../test/distributed/checkpoint/e2e/test_fine_tuning.py": None,
    "../../../../test/distributed/checkpoint/e2e/test_fsdp_ep.py": None,
    "../../../../test/distributed/checkpoint/fsdp/test_fsdp_dsd.py": None,
    "../../../../test/distributed/checkpoint/test_checkpoint.py": None,
    "../../../../test/distributed/checkpoint/test_compatibility.py": None,
    "../../../../test/distributed/checkpoint/test_dedup_tensors.py": None,
    "../../../../test/distributed/checkpoint/test_dtensor_checkpoint.py": None,
    "../../../../test/distributed/checkpoint/test_dtensor_resharding.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_1d_to_1d_reshard_placement_change_extensions0",
        "test_1d_to_1d_reshard_placement_change_extensions1",
        "test_2d_to_2d_reshard_placement_change",
        "test_1d_to_2d_reshard_mesh_change",
        "test_2d_to_1d_reshard_mesh_change",
    ),
    "../../../../test/distributed/checkpoint/test_file_system_checkpoint.py": None,
    "../../../../test/distributed/checkpoint/test_file_system_checkpoint_cpu.py": None,
    "../../../../test/distributed/checkpoint/test_format_utils.py": None,
    "../../../../test/distributed/checkpoint/test_fsdp_model_state.py": None,
    "../../../../test/distributed/checkpoint/test_fsdp_optim_state.py": None,
    "../../../../test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_fsdp_to_tp"
    ),
    "../../../../test/distributed/checkpoint/test_fsspec.py": None,
    "../../../../test/distributed/checkpoint/test_hsdp_checkpoint.py": None,
    "../../../../test/distributed/checkpoint/test_nested_dict.py": None,
    "../../../../test/distributed/checkpoint/test_planner.py": None,
    "../../../../test/distributed/checkpoint/test_save_load_api.py": None,
    "../../../../test/distributed/checkpoint/test_state_dict.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_setting_meta_device_model",
        "test_multi_param_groups",
    ),
    "../../../../test/distributed/checkpoint/test_state_dict_utils.py": None,
    "../../../../test/distributed/checkpoint/test_tp_checkpoint.py": None,
    "../../../../test/distributed/checkpoint/test_traverse.py": None,
    "../../../../test/distributed/checkpoint/test_utils.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_barriers.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_builder.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_checkpoint_process.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_checkpoint_reader.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_checkpoint_writer.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_checkpointer.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_staging.py": None,
    "../../../../test/distributed/checkpoint/_experimental/test_types.py": None,
    "../../../../test/distributed/elastic/events/lib_test.py": None,
    "../../../../test/distributed/elastic/metrics/api_test.py": None,
    "../../../../test/distributed/elastic/multiprocessing/api_test.py": None,
    "../../../../test/distributed/elastic/test_control_plane.py": None,
    "../../../../test/distributed/elastic/timer/api_test.py": None,
    "../../../../test/distributed/elastic/timer/local_timer_example.py": None,
    "../../../../test/distributed/elastic/timer/local_timer_test.py": None,
    "../../../../test/distributed/elastic/utils/distributed_test.py": None,
    "../../../../test/distributed/elastic/utils/logging_test.py": None,
    "../../../../test/distributed/elastic/utils/util_test.py": None,
    "../../../../test/distributed/optim/test_apply_optimizer_in_backward.py": None,
    "../../../../test/distributed/optim/test_named_optimizer.py": None,
    "../../../../test/distributed/optim/test_zero_redundancy_optimizer.py": None,
}

skip_dict_python = {
    "distributed/test_c10d_ops_xccl.py": None,
    "distributed/test_c10d_xccl.py": None,
    "../../../../test/distributed/test_c10d_functional_native.py": None,
    # "../../../../test/distributed/pipelining/test_schedule_multiproc.py": None,  # Hang error.
    "../../../../test/distributed/pipelining/test_stage.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
}
