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
    "../../../../test/distributed/fsdp/test_fsdp_dtensor_state_dict.py": (
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_is_even_sharded_model_True_xpu",
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_is_even_sharded_model_True_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_is_even_sharded_model_True_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_is_even_sharded_model_True_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_is_even_sharded_model_True_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_is_even_sharded_model_False_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_is_even_sharded_model_True_xpu",
        "test_fsdp_init_with_device_mesh_is_even_sharded_model_False_xpu",
        "test_fsdp_init_with_device_mesh_is_even_sharded_model_True_xpu",
        "test_raises_warning_or_errors_xpu",
    ),
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
    "../../../../test/distributed/fsdp/test_hsdp_dtensor_state_dict.py": (
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_False_xpu",
        "test_dtensor_sharded_model_load_state_dict_offload_to_cpu_True_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_False_xpu",
        "test_dtensor_sharded_optim_load_state_dict_offload_to_cpu_True_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_False_xpu",
        "test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_xpu",
        "test_hsdp_init_with_device_mesh_xpu",
        "test_root_module_is_not_FSDP_xpu",
    ),
    "../../../../test/distributed/fsdp/test_utils.py": None,
}
