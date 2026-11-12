# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

# grad_input is (2**16 + 1) x 2**15 float16 == 4.3 GiB; leave headroom.
_REQUIRED_BYTES = 6 * 1024**3


def _has_memory():
    return torch.xpu.is_available() and torch.xpu.mem_get_info(0)[0] >= _REQUIRED_BYTES


class TestNllLoss2dBackwardLargeOffset(TestCase):
    @unittest.skipUnless(_has_memory(), "needs 6GB of free XPU memory")
    def test_nll_loss2d_backward_large_sample_offset(self):
        # Regression test for the 32-bit overflow of the per-sample offset in
        # nll_loss2d_backward's reduced kernel: the last sample's
        # `sample * map_nelem * n_classes` reaches 2**31, so a 32-bit offset
        # wraps and the kernel writes ~4GiB below grad_input. Depending on what
        # the allocator has mapped there this is either a silently missing
        # gradient or a GPU page fault that aborts the process.
        batch_size = 2**16 + 1
        num_classes = 2**15
        ignore_index = -100

        # The reduced backward only reads input's metadata, so expanding a
        # scalar avoids materializing a second 4GiB tensor.
        input = torch.empty((), device="xpu", dtype=torch.float16).expand(
            batch_size, num_classes, 1, 1
        )
        target = torch.full(
            (batch_size, 1, 1), ignore_index, dtype=torch.int64, device="xpu"
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
        self.assertEqual(grad_input[-1, 0, 0, 0].item(), -1.0)


if __name__ == "__main__":
    run_tests()
