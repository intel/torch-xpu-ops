# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Reproducer for:
#   [upstream_ut] test/dynamo/test_higher_order_ops.py
#   ActivationCheckpointingTests::test_dropout
#   failed with RuntimeError: CUDA not available
#
# Root cause:
#   CUDARngStateHelper.get_torch_state_as_tuple raised
#   "CUDA not available" when only XPU was present, which prevented
#   functionalize_rng_ops=True from working with dropout on XPU.
#   Additionally, decompositions_for_rng.rand_like and philox_rand
#   rejected non-CUDA devices.
#
# Run with:
#   pytest test/repro/test_activation_checkpointing_xpu.py -v


import pytest
import torch

xpu_available = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU not available"
)

try:
    from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON

    xpu_triton = pytest.mark.skipif(
        not HAS_XPU_AND_TRITON, reason="XPU and Triton not available"
    )
except ImportError:
    xpu_triton = pytest.mark.skip(reason="inductor_utils not available")


@xpu_available
def test_cuda_rng_helper_supports_xpu():
    """
    Regression: CUDARngStateHelper.get_torch_state_as_tuple must NOT raise
    RuntimeError('CUDA not available') when only XPU is present.
    """
    # The XPU test module applies the required monkey-patches on import.
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../xpu/dynamo"))
    import test_higher_order_ops_xpu  # noqa: F401 – side-effect: patches applied

    from torch._prims_common import CUDARngStateHelper

    seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
    assert seed is not None
    assert offset is not None
    assert seed.numel() == 1, f"expected scalar seed, got {seed.shape}"
    assert offset.numel() == 1, f"expected scalar offset, got {offset.shape}"


@xpu_triton
def test_dropout_activation_checkpointing_xpu():
    """
    Regression: dropout inside torch.utils.checkpoint.checkpoint must work
    on XPU with functionalize_rng_ops=True.

    This is the XPU adaptation of upstream PyTorch's:
      test/dynamo/test_higher_order_ops.py::ActivationCheckpointingTests::test_dropout
    """
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../xpu/dynamo"))
    import test_higher_order_ops_xpu as m

    t = m.ActivationCheckpointingTests()
    t.test_dropout()
