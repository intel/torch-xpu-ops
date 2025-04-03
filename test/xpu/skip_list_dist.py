skip_dict = {
    "../../../../test/distributed/fsdp/test_fsdp_checkpoint.py": (
        "test_checkpoint_fsdp_wrapping_cpu_offload0_offload_activations_False_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload1_offload_activations_False_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload1_offload_activations_True_use_orig_params_False",
        "test_checkpoint_submodule_use_reentrant_False_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_apply.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_clip_grad_norm.py": (
        "test_ddp_parity_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_comm.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_core.py": (
        "test_delayed_optim_step_offload_false_no_shard_xpu",
        "test_delayed_optim_step_offload_false_none_xpu",
        "test_delayed_optim_step_offload_false_shard_grad_op_xpu",
        "test_delayed_optim_step_offload_true_none_xpu",
        "test_delayed_optim_step_offload_true_shard_grad_op_xpu",
        "test_delayed_reduce_scatter_offload_false_no_shard_xpu",
        "test_delayed_reduce_scatter_offload_false_none_xpu",
        "test_delayed_reduce_scatter_offload_false_shard_grad_op_xpu",
        "test_delayed_reduce_scatter_offload_true_none_xpu",
        "test_delayed_reduce_scatter_offload_true_shard_grad_op_xpu",
        "test_mixture_of_experts_offload_false_no_shard_xpu",
        "test_mixture_of_experts_offload_false_none_xpu",
        "test_mixture_of_experts_offload_false_shard_grad_op_xpu",
        "test_mixture_of_experts_offload_true_none_xpu",
        "test_mixture_of_experts_offload_true_shard_grad_op_xpu",
        "test_mixture_of_experts_with_delay_before_free_offload_false_no_shard_xpu",
        "test_mixture_of_experts_with_delay_before_free_offload_false_none_xpu",
        "test_mixture_of_experts_with_delay_before_free_offload_false_shard_grad_op_xpu",
        "test_mixture_of_experts_with_delay_before_free_offload_true_none_xpu",
        "test_mixture_of_experts_with_delay_before_free_offload_true_shard_grad_op_xpu",
        "test_nested_always_wrap_model_offload_false_no_shard_xpu",
        "test_nested_always_wrap_model_offload_false_none_xpu",
        "test_nested_always_wrap_model_offload_false_shard_grad_op_xpu",
        "test_nested_always_wrap_model_offload_true_none_xpu",
        "test_nested_always_wrap_model_offload_true_shard_grad_op_xpu",
        "test_nested_wrapped_model_offload_false_no_shard_xpu",
        "test_nested_wrapped_model_offload_false_none_xpu",
        "test_nested_wrapped_model_offload_false_shard_grad_op_xpu",
        "test_nested_wrapped_model_offload_true_none_xpu",
        "test_nested_wrapped_model_offload_true_shard_grad_op_xpu",
        "test_transformer_offload_false_no_shard_xpu",
        "test_transformer_offload_false_none_xpu",
        "test_transformer_offload_false_shard_grad_op_xpu",
        "test_transformer_offload_true_none_xpu",
        "test_transformer_offload_true_shard_grad_op_xpu",
        # https://github.com/intel/torch-xpu-ops/issues/1475
        "test_transformer_no_grad_mixed_precision_True_xpu",
        "test_transformer_no_grad_mixed_precision_False_xpu",
    ),
    # Will add them back after debugging
    # "../../../../test/distributed/fsdp/test_fsdp_dtensor_state_dict.py": (
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_is_even_sharded_model_True_xpu",
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_is_even_sharded_model_True_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_is_even_sharded_model_True_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_is_even_sharded_model_True_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_is_even_sharded_model_True_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_is_even_sharded_model_False_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_is_even_sharded_model_True_xpu",
    #    "test_fsdp_init_with_device_mesh_is_even_sharded_model_False_xpu",
    #    "test_fsdp_init_with_device_mesh_is_even_sharded_model_True_xpu",
    #    "test_raises_warning_or_errors_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_fsdp_exec_order.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_fine_tune.py": (
        "test_parity_with_non_frozen_fsdp_xpu",
        "test_parity_with_ddp_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_fx.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_input.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_forward.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_wrapping.py": (
        "test_transformer_no_grad_mixed_precision_True_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_uneven.py": None,
    # Will add them back after debugging
    # "../../../../test/distributed/fsdp/test_hsdp_dtensor_state_dict.py": (
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_xpu",
    #    "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_xpu",
    #    "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_xpu",
    #    "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_xpu",
    #    "test_hsdp_init_with_device_mesh_xpu",
    #    "test_root_module_is_not_FSDP_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_utils.py": None,
    "../../../../test/distributed/test_backends.py": None,
    "../../../../test/distributed/test_c10d_common.py": None,
    "../../../../test/distributed/test_c10d_functional_native.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1508
        #RuntimeError: oneCCL: coll_param.cpp:455 validate: EXCEPTION: average operation is not supported for the scheduler path
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
        "test_tracing and test_tracing_with_fakepg and test_tracing_with_fakepg_xpu and test_tracing_with_dce_code and test_tracing_with_dce_code_xpu"
    ),
    "../../../../test/distributed/test_multi_threaded_pg.py": (
        # oneccl not support multi-threaded well, so skip it first.
        "test_bwd_sees_fwd_pg",
    ),
    "../../../../test/distributed/test_store.py": None,
    "../../../../test/distributed/pipelining/test_backward.py": None,
    "../../../../test/distributed/tensor/parallel/test_tp_random_state.py": None,
    "../../../../test/distributed/pipelining/test_backward.py": None,
    "../../../../test/distributed/pipelining/test_microbatch.py": None,
    "../../../../test/distributed/pipelining/test_pipe.py": None,
    "../../../../test/distributed/pipelining/test_schedule.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
    "../../../../test/distributed/pipelining/test_unflatten.py": None,
}

skip_dict_python = {
    "distributed/test_c10d_ops_xccl.py": None,
    "distributed/test_c10d_xccl.py": None,
    "../../../../test/distributed/pipelining/test_schedule_multiproc.py": None, # Hang error.
    "../../../../test/distributed/pipelining/test_stage.py": None,
}
