# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

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
    "../../../../test/distributed/fsdp/test_fsdp_flatten_params.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_unshard_params.py": None,
    "../../../../test/distributed/fsdp/test_wrap.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_fx.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_input.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_forward.py": None,
    "../../../../test/distributed/fsdp/test_fsdp_multiple_wrapping.py": (
        "test_transformer_no_grad_mixed_precision_True_xpu",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_uneven.py": None,
    "../../../../test/distributed/fsdp/test_utils.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_comm.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_state_dict.py": (
        "test_cached_state_dict",
        "test_dp_state_dict_cpu_offload",
    ),
    "../../../../test/distributed/fsdp/test_fsdp_misc.py": None,
    "../../../../test/distributed/test_functional_api.py": None,
    "../../../../test/distributed/_composable/fsdp/test_fully_shard_frozen.py": None,
    "../../../../test/distributed/_composable/test_checkpoint.py": None,
    "../../../../test/distributed/_composable/test_contract.py": None,
    "distributed/test_c10d_xccl.py": None,
    "distributed/test_c10d_ops_xccl.py": None,
    "../../../../test/distributed/test_c10d_functional_native.py": None,
    "../../../../test/distributed/_tools/test_fsdp2_mem_tracker.py": None,
    "../../../../test/distributed/_tools/test_mem_tracker.py": None,
    "../../../../test/distributed/_tools/test_memory_tracker.py": None,
    "../../../../test/distributed/tensor/test_random_ops.py": None,
    "../../../../test/distributed/tensor/test_math_ops.py": None,
}
