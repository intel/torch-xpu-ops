# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import math

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestXpuTransposeAlignment(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("src_offset, dst_offset", [(0, 1), (1, 0), (1, 1)])
    def test_transpose_copy_unaligned(self, device, dtype, src_offset, dst_offset):
        # The batch dim is oversized on purpose: detect_batch_transpose only
        # takes the tiled kernel when total subgroups fill every hardware
        # thread slot, and a shape that merely meets that bound on one device
        # falls back to copy_kernel on a larger one, testing nothing.
        shape = (64, 64, 32, 32)

        def offset_view(offset, view_shape):
            base = torch.empty(
                offset + math.prod(view_shape), device=device, dtype=dtype
            )
            return base[offset:].view(view_shape)

        src = offset_view(src_offset, (64, 32, 32, 64)).permute(0, 3, 1, 2)
        dst = offset_view(dst_offset, shape)
        src.copy_(torch.randn(shape, device=device, dtype=dtype))

        # The width is the min over both base pointers, so each combination
        # leaves a different one of the two terms as the only thing clamping it.
        if src_offset:
            self.assertNotEqual(src.data_ptr() % (4 * src.element_size()), 0)
        if dst_offset:
            self.assertNotEqual(dst.data_ptr() % (4 * dst.element_size()), 0)

        dst.copy_(src)

        self.assertEqual(dst, src)


instantiate_device_type_tests(
    TestXpuTransposeAlignment, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
