# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Reproducer for issue #4723 (pytorch/pytorch#190139): NLLLoss2d backward
silently zeroed the gradient of the tail samples for large inputs.

The backward kernel computed the per-sample base offsets
(sample, toffset = sample * map_nelem, ioffset = sample * map_nelem * n_classes)
in 32-bit int. Once the flattened extent reaches 2**31 the multiplication
overflows and wraps the base pointer, so the trailing samples receive no
gradient (grad == 0) instead of the correct value. The fix widens those offsets
to int64_t, matching the forward kernel's index type selection.
"""

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestNllLoss2dBackwardOffsetOverflow(TestCase):
    def test_backward_offset_no_overflow(self):
        # (2**16 + 1) samples of 2**15 classes: the last sample's input offset
        # sample * map_nelem * n_classes exceeds 2**31, overflowing int32.
        x = torch.zeros(
            (2**16 + 1, 2**15, 1, 1),
            device="xpu",
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.zeros((2**16 + 1, 1, 1), device="xpu", dtype=torch.long)

        F.nll_loss(x, target, reduction="sum").backward()

        # With reduction="sum" and target class 0, every selected gradient is -1.
        # Pre-fix, the wrapped offset leaves the tail sample's gradient at 0.
        self.assertEqual(x.grad[0, 0, 0, 0].item(), -1.0)
        self.assertEqual(x.grad[-1, 0, 0, 0].item(), -1.0)


if __name__ == "__main__":
    run_tests()
