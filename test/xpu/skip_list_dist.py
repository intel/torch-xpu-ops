# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from skip_list_common import PYTORCH_TEST_DIR

skip_dict = {
    f"{PYTORCH_TEST_DIR}/distributed/_composable/fsdp/test_fully_shard_comm.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_composable/fsdp/test_fully_shard_frozen.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_composable/fsdp/test_fully_shard_state_dict.py": (
        "test_cached_state_dict",
        "test_dp_state_dict_cpu_offload",
    ),
    f"{PYTORCH_TEST_DIR}/distributed/_composable/test_checkpoint.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_composable/test_contract.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_tools/test_fsdp2_mem_tracker.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_tools/test_mem_tracker.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/_tools/test_memory_tracker.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_apply.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_checkpoint.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_clip_grad_norm.py": (
        "test_ddp_parity_xpu",
    ),
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_comm.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_comm_hooks.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_core.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_exec_order.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_fine_tune.py": (
        "test_parity_with_non_frozen_fsdp_xpu",
        "test_parity_with_ddp_xpu",
    ),
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_flatten_params.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_fx.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_input.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_misc.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_multiple_forward.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_multiple_wrapping.py": (
        "test_transformer_no_grad_mixed_precision_True_xpu",
    ),
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_sharded_grad_scaler.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_uneven.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_fsdp_unshard_params.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_utils.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/fsdp/test_wrap.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/pipelining/test_backward.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/pipelining/test_microbatch.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/tensor/test_math_ops.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/tensor/test_random_ops.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/test_c10d_functional_native.py": None,
    "distributed/test_c10d_ops_xccl.py": None,
    "distributed/test_c10d_xccl.py": None,
    f"{PYTORCH_TEST_DIR}/distributed/test_functional_api.py": None,
}
