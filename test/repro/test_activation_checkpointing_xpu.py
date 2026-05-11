# Owner(s): ["module: intel"]

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

import os
import sys
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON
except ImportError:
    HAS_XPU_AND_TRITON = False

_XPU_DYNAMO_DIR = os.path.join(os.path.dirname(__file__), "../xpu/dynamo")
if _XPU_DYNAMO_DIR not in sys.path:
    sys.path.insert(0, _XPU_DYNAMO_DIR)


@unittest.skipUnless(torch.xpu.is_available(), "XPU not available")
class TestCUDARngHelperXPUSupport(TestCase):
    def test_cuda_rng_helper_supports_xpu(self):
        """
        Regression: CUDARngStateHelper.get_torch_state_as_tuple must NOT raise
        RuntimeError('CUDA not available') when only XPU is present.
        """
        # Importing the XPU test module applies the required monkey-patches.
        import test_higher_order_ops_xpu  # noqa: F401 - side-effect: patches applied
        from torch._prims_common import CUDARngStateHelper

        seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
        self.assertIsNotNone(seed)
        self.assertIsNotNone(offset)
        self.assertEqual(seed.numel(), 1)
        self.assertEqual(offset.numel(), 1)


@unittest.skipUnless(HAS_XPU_AND_TRITON, "XPU and Triton not available")
class TestDropoutActivationCheckpointingXPU(TestCase):
    def test_dropout_activation_checkpointing_xpu(self):
        """
        Regression: dropout inside torch.utils.checkpoint.checkpoint must work
        on XPU with functionalize_rng_ops=True.

        This is the XPU adaptation of upstream PyTorch's:
          test/dynamo/test_higher_order_ops.py::ActivationCheckpointingTests::test_dropout
        """
        import test_higher_order_ops_xpu as m

        t = m.ActivationCheckpointingTests()
        t.test_dropout()


if __name__ == "__main__":
    run_tests()
