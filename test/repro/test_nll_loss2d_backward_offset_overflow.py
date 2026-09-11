# Copyright 2020-2025 Intel Corporation
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

The backward kernel computed per-sample offsets in 32-bit arithmetic. Once the
flattened extent reaches 2**31, the multiplication overflows and the trailing
samples receive no gradient. The fix selects 32- or 64-bit indexing based on
the input tensor, matching the forward kernel and pytorch/pytorch#190144.
"""

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestNllLoss2dBackwardOffsetOverflow(TestCase):
    @largeTensorTest("5GB", device="xpu")
    def test_backward_offset_no_overflow(self):
        batch_size = 2**16 + 1
        num_classes = 2**15
        ignore_index = -100

        # Reduced backward only uses input metadata. Expanding a scalar avoids
        # materializing another four-GiB tensor.
        input = torch.empty(
            (),
            device="xpu",
            dtype=torch.float16,
        ).expand(batch_size, num_classes, 1, 1)
        target = torch.full(
            (batch_size, 1, 1),
            ignore_index,
            dtype=torch.int64,
            device="xpu",
        )
        target[-1] = 0
        one = torch.ones((), dtype=torch.float16, device="xpu")

        grad_input = torch.ops.aten.nll_loss2d_backward.default(
            one,
            input,
            target,
            None,
            F._Reduction.get_enum("sum"),
            ignore_index,
            one,
        )

        torch.xpu.synchronize()
        self.assertEqual(grad_input[-1, 0, 0, 0], -1)


if __name__ == "__main__":
    run_tests()
