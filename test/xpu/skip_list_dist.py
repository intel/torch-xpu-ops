skip_dict = {
    "../../../../test/distributed/fsdp/test_fsdp_checkpoint.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_apply.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_clip_grad_norm.py": (
        "test_ddp_parity_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_comm.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_comm_hooks.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_core.py": None,
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
    "../../../../test/distributed/fsdp/test_utils.py": None,
    "distributed/test_c10d_xccl.py": (
        # https://github.com/intel/torch-xpu-ops/issues/2046
        "test_unwaited",
    ),
    "distributed/test_c10d_ops_xccl.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_misc.py": None,
    "../../../../test/distributed/test_functional_api.py": (
        # depends on https://github.com/pytorch/pytorch/pull/159473
        "test_tracing_with_fakepg_xpu",
    ),
    "../../../../test/distributed/_tools/test_fsdp2_mem_tracker.py": None,
    "../../../../test/distributed/_tools/test_mem_tracker.py": None,
    "../../../../test/distributed/_tools/test_memory_tracker.py": None,
}
