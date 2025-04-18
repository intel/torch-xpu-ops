skip_dict = {
    "../../../../test/distributed/fsdp/test_checkpoint_wrapper.py": None,
    # https://github.com/intel/torch-xpu-ops/issues/1536
    # "../../../../test/distributed/fsdp/test_distributed_checkpoint.py": (
    #    "test_distributed_checkpoint_state_dict_type0_xpu",
    #    "test_distributed_checkpoint_state_dict_type1_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_fsdp_apply.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_backward_prefetch.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_checkpoint.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_basic_checkpoint_end_to_end_cpu_offload1_offload_activations_False_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload0_offload_activations_False_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload0_offload_activations_True_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload1_offload_activations_False_use_orig_params_False",
        "test_checkpoint_fsdp_wrapping_cpu_offload1_offload_activations_True_use_orig_params_False",
        "test_checkpoint_submodule_use_reentrant_False_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_clip_grad_norm.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_ddp_parity_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_comm.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_comm_hooks.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_bf16_hook_has_wrapping_False_sharding_strategy0",
        "test_bf16_hook_has_wrapping_False_sharding_strategy1",
        "test_bf16_hook_has_wrapping_False_sharding_strategy2",
        "test_bf16_hook_has_wrapping_True_sharding_strategy0",
        "test_bf16_hook_has_wrapping_True_sharding_strategy1",
        "test_bf16_hook_has_wrapping_True_sharding_strategy2",
        "test_fp16_hook_has_wrapping_False_sharding_strategy1",
        "test_fp16_hook_has_wrapping_False_sharding_strategy2",
        "test_fp16_hook_has_wrapping_True_sharding_strategy0",
        "test_fp16_hook_has_wrapping_True_sharding_strategy1",
        "test_fp16_hook_has_wrapping_True_sharding_strategy2",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_core.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_delayed_optim_step_offload_true_no_shard_xpu",
        "test_transformer_no_grad_mixed_precision_True_xpu",
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
        "test_transformer_offload_false_none_xpu",
        "test_transformer_offload_false_shard_grad_op_xpu",
        "test_transformer_offload_true_none_xpu",
        "test_transformer_offload_true_shard_grad_op_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_dtensor_state_dict.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        " test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_is_even_sharded_model_False_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_exec_order.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_fine_tune.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_hooks_multi_traversal_xpu",
        "test_parity_with_ddp_xpu",
        "test_parity_with_non_frozen_fsdp_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_flatten_params.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_freezing_weights.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_False_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_False_disable_autograd_True_forward_prefetch_True ",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_False_forward_prefetch_True",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_False",
        "test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_RequiresGrad_freeze_after_wrap_fsdp_True_disable_autograd_True_forward_prefetch_True",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_fx.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_grad_acc.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_hybrid_shard.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_ignored_modules.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_input.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_memory.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_meta.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_misc.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1535
        "test_fsdp_zero2_eval_with_prefetch",
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_fsdp_optimizer_overlap",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_mixed_precision.py": (
        "test_buffer_dtype_no_root_handle",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_multiple_forward.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_multi_forward_cpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_multiple_wrapping.py": None,
    # https://github.com/intel/torch-xpu-ops/issues/1537
    "../../../../test/distributed/fsdp/test_fsdp_optim_state.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_flatten_sharded_optim_state_dict_nested",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_True",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_True_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_True_use_diff_optim_inputs_True",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_True_rank0_only_False_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_True_rank0_only_False_use_diff_optim_inputs_True",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_True_rank0_only_True_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_True_rank0_only_True_use_diff_optim_inputs_True",
        "test_optim_state_dict_nested_state_dict_type1_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type1_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_True",
        "test_optim_state_dict_nested_state_dict_type1_use_multiple_param_groups_True_rank0_only_False_use_diff_optim_inputs_False",
        "test_optim_state_dict_nested_state_dict_type1_use_multiple_param_groups_True_rank0_only_False_use_diff_optim_inputs_True",
        "test_rekey_optim_state_dict_to_ids_state_dict_type0_use_multiple_param_groups_False",
        "test_rekey_optim_state_dict_to_ids_state_dict_type0_use_multiple_param_groups_True",
        "test_rekey_optim_state_dict_to_ids_state_dict_type1_use_multiple_param_groups_False",
        "test_rekey_optim_state_dict_to_ids_state_dict_type1_use_multiple_param_groups_True",
        "test_rekey_optim_state_dict_to_names",
        "test_scatter_full_optim_state_dict_nested_halve_world_size",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_False_use_diff_optim_inputs_False",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_False_use_diff_optim_inputs_True",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_True_use_diff_optim_inputs_False",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_True_use_diff_optim_inputs_True",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_False_use_diff_optim_inputs_False",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_False_use_diff_optim_inputs_True",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_True_use_diff_optim_inputs_False",
        "test_scatter_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_True_use_diff_optim_inputs_True",
        "test_shard_full_optim_state_dict_nested_halve_world_size",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_False_use_diff_optim_inputs_False",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_False_use_diff_optim_inputs_True",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_True_use_diff_optim_inputs_False",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_True_use_diff_optim_inputs_True",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_False_use_diff_optim_inputs_False",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_False_use_diff_optim_inputs_True",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_True_use_diff_optim_inputs_False",
        "test_shard_full_optim_state_dict_nested_use_multiple_param_groups_True_wrap_alt_True_use_diff_optim_inputs_True",
        "test_use_orig_params",
    ),
    # Performance check, skip
    # "../../../../test/distributed/fsdp/test_fsdp_overlap.py": (
    #    # https://github.com/intel/torch-xpu-ops/issues/1504
    #    "test_forward_overlap",
    #    "test_forward_overlap_xpu",
    # ),
    "../../../../test/distributed/fsdp/test_fsdp_pure_fp16.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_fsdp_ddp_parity_with_grad_scaler_offload_false_none_none_none",
        "test_fsdp_ddp_parity_with_grad_scaler_offload_false_shard_grad_op_none_none",
        "test_fsdp_ddp_parity_with_grad_scaler_offload_true_none_none_none",
        "test_fsdp_ddp_parity_with_grad_scaler_offload_true_shard_grad_op_none_none",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_state_dict.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_state_dict_save_load_flow_state_dict_type_local_state_dict",
        "test_state_dict_save_load_flow_state_dict_type_sharded_state_dict",
        "test_state_dict_save_load_flow_state_dict_type_state_dict",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_tp_integration.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_traversal.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_uneven.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_unshard_params.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_use_orig_params.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_diff_hyperparams_sharding_strategy_str_full_shard",
        "test_diff_hyperparams_sharding_strategy_str_no_shard",
        "test_diff_hyperparams_sharding_strategy_str_shard_grad_op",
        "test_no_sync_correctness",
    ),
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
        # RuntimeError: Process 2 exited with error code 10 and exception: ; AssertionError: Scalars are not equal!
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_scatter_object_list_cpu",
        "test_scatter_object_list_xpu",
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
        # https://github.com/intel/torch-xpu-ops/issues/1509
        "test_bwd_sees_fwd_pg",
    ),
    "../../../../test/distributed/test_store.py": None,
    "../../../../test/distributed/pipelining/test_backward.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_stage_backward_weight_multiple_iters_xpu",
        "test_stage_backward_weight_xpu",
        "test_stage_backward_xpu",
    ),
    "../../../../test/distributed/pipelining/test_microbatch.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_chunk_spec_xpu",
    ),
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
        "test_transformer_req_grad_seq_parallel_float32_thaw_all",
        "test_transformer_req_grad_seq_parallel_float32_thaw_layers_0_attention_wv__layers_0_feed_forward_w1__layers_1_feed_forward_w2__layers_1_ffn_norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_layers_1_ffn_norm__norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output__tok_embeddings",
        "test_transformer_req_grad_seq_parallel_float32_thaw_output__tok_embeddings",
        "test_transformer_training_is_seq_parallel_False_float32",
        "test_transformer_training_is_seq_parallel_True_float32",
        # NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered.
        # https://github.com/intel/torch-xpu-ops/issues/1556
        "test_transformer_req_grad_seq_parallel_float32_thaw_norm__output",
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_loss_parallel",
        "test_mlp_training_is_seq_parallel_False_recompute_activation_False",
        "test_mlp_training_is_seq_parallel_True_recompute_activation_False",
        "test_transformer_req_grad_float64_thaw_all",
        "test_transformer_training_is_seq_parallel_False_float64",
        "test_transformer_training_is_seq_parallel_True_float64",
        "test_sequence_parallel_style",
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
    ),
    "../../../../test/distributed/tensor/parallel/test_tp_style.py": None,
    "../../../../test/distributed/tensor/test_api.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_distribute_tensor_rank",
        "test_distribute_tensor_uneven_sharding",
    ),
    "../../../../test/distributed/tensor/test_attention.py": None,
    "../../../../test/distributed/tensor/test_common_rules.py": None,
    "../../../../test/distributed/tensor/test_dtensor.py": (
        # Passed with updated test code for world_size 8
        "test_auto_implicit_replication",
        "test_default_value_sub_mesh",
        "test_device_mesh_nd",
        "test_dtensor_2d_mesh",
        "test_dtensor_api_device_mesh_context_manager",
        "test_dtensor_device_mesh_device_conversion",
        "test_dtensor_spec_local_shard_offset",
        "test_from_local_sub_mesh",
        "test_implicit_replication",
        "test_metadata_consistency_check",
        "test_redistribute_sub_mesh",
        "test_split_tensor_1D",
    ),
    "../../../../test/distributed/tensor/test_dtensor_compile.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_2d_fsdp_tp_compile",
    ),
    "../../../../test/distributed/tensor/test_experimental_ops.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1535
        "test_bernoulli",
    ),
    "../../../../test/distributed/tensor/test_init.py": None,
    "../../../../test/distributed/tensor/test_math_ops.py": (
        # RuntimeError: oneCCL: coll_param.cpp:455 validate: EXCEPTION: average operation is not supported for the scheduler path
        # https://github.com/intel/torch-xpu-ops/issues/1508
        "test_mean",
        "test_nll_loss_and_cross_entropy",
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
    "../../../../test/distributed/tensor/test_random_ops.py": (
        # Need to update world size
        "test_hsdp_tp_model_meta_init",
    ),
    "../../../../test/distributed/tensor/test_redistribute.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_redistribute_shard_dim_multi_dim_mesh",
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
        "test_op_out_variant",
        "test_slice",
        "test_stack",
        "test_where_type_promotion",
    ),
    "../../../../test/distributed/tensor/experimental/test_register_sharding.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_argmax",
        "test_softmax_fwd",
    ),
    "../../../../test/distributed/_shard/test_sharder.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_custom_sharder",
    ),
    # FSDP2
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_autograd.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_nontensor_activations",
        "test_unused_forward_module",
        "test_unused_forward_output",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_clip_grad_norm_2d",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_comm.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1571
        "test_set_reduce_scatter_divide_factor",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_compile.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_extensions.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_frozen.py": (
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_train_mixed_requires_grad_per_group",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1508
        "test_gradient_scaler",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_ignore_params.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_init.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_logging.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_memory.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1535
        "test_fully_shard_training_memory",
    ),
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
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_state_dict.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1572
        "test_dp_state_dict_cpu_offload",
    ),
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_state.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_training.py": (
        # https://github.com/intel/torch-xpu-ops/issues/1508
        "test_post_optim_event",
        # https://github.com/intel/torch-xpu-ops/issues/1504
        "test_train_parity_multi_group_unshard_async_op",
        "test_train_parity_with_activation_checkpointing",
        # https://jira.devtools.intel.com/browse/MLSL-3625
        "test_1f1b_microbatching",
        "test_gradient_accumulation",
    ),
}

skip_dict_python = {
    "distributed/test_c10d_ops_xccl.py": None,
    "distributed/test_c10d_xccl.py": None,
    # "../../../../test/distributed/pipelining/test_schedule_multiproc.py": None,  # Hang error.
    "../../../../test/distributed/pipelining/test_stage.py": None,
    "../../../../test/distributed/pipelining/test_transformer.py": None,
}
