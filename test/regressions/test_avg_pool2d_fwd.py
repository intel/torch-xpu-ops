# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for the channels-last avg_pool2d forward kernels on XPU.

The launcher picks a kernel from channel divisibility, pointer alignment and
work-item counts: VEC (vec 8/4/2), W-tile (k=3, stride_w=1), HW-tile (vec 2,
k=3, stride=1) or the scalar fallback. Its thresholds derive from the device's
resident-thread capacity, so the shapes below are sized for BMG, where op
regressions run in CI; each case is named after the kernel it reaches there.
"""

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

PATH_CASES = [
    subtest(((16, 64, 64, 64), 2, 2, 0), name="vec8_k2"),
    subtest(((16, 64, 64, 64), 3, 2, 1), name="vec8_k3"),
    subtest(((16, 64, 64, 64), (2, 3), 2, 1), name="vec8_generic_window"),
    subtest(((16, 36, 64, 64), 2, 2, 0), name="vec4"),
    subtest(((16, 26, 64, 64), 3, 2, 1), name="vec2"),
    subtest(((8, 3, 64, 64), 3, 2, 1), name="scalar_odd_channels"),
    subtest(((2, 16, 8, 8), 3, 1, 1), name="scalar_low_occupancy"),
    subtest(((16, 64, 64, 64), 3, 1, 1), name="wtile_vec4"),
    subtest(((16, 64, 63, 63), 3, 1, 1), name="wtile_vec4_odd_width"),
    subtest(((64, 5, 64, 64), 3, 1, 1), name="wtile_vec1"),
    subtest(((32, 26, 64, 64), 3, 1, 1), name="hwtile"),
    subtest(((32, 26, 63, 63), 3, 1, 1), name="hwtile_odd_size"),
    # HW-tile reuses rows across adjacent output rows, which is only valid for
    # stride_h == 1; other strides must take the VEC path.
    subtest(((64, 26, 64, 64), 3, (2, 1), 1), name="hwtile_stride_h2_uses_vec2"),
    # Work-item counts above the launch cap, so each work-item loops.
    subtest(((48, 64, 64, 64), 3, 1, 1), name="wtile_vec4_grid_stride"),
    subtest(((128, 26, 64, 64), 3, 1, 1), name="hwtile_grid_stride"),
]

KERNEL_CASES = [
    subtest(((16, 64, 64, 64), 3, 2, 1), name="vec8"),
    subtest(((16, 64, 64, 63), 3, (2, 1), 1), name="wtile"),
    subtest(((32, 26, 63, 63), 3, 1, 1), name="hwtile"),
]


class TestAvgPool2dFwdChannelsLast(TestCase):
    def _check(
        self,
        shape,
        kernel_size,
        stride,
        padding,
        dtype=torch.float,
        storage_offset=0,
        **kwargs,
    ):
        n, c, h, w = shape
        buf = torch.randn(n * c * h * w + storage_offset, dtype=dtype, device="xpu")
        x = buf[storage_offset:].view(n, h, w, c).permute(0, 3, 1, 2)
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

        actual = F.avg_pool2d(x, kernel_size, stride, padding, **kwargs)
        expected = F.avg_pool2d(
            x.cpu().double(), kernel_size, stride, padding, **kwargs
        ).to(dtype)

        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(actual.cpu(), expected)

    @parametrize("shape,kernel_size,stride,padding", PATH_CASES)
    def test_kernel_paths(self, shape, kernel_size, stride, padding):
        self._check(shape, kernel_size, stride, padding)

    @parametrize("shape,kernel_size,stride,padding", KERNEL_CASES)
    @parametrize(
        "options",
        [
            subtest({"ceil_mode": True}, name="ceil_mode"),
            subtest({"count_include_pad": False}, name="exclude_pad"),
            subtest({"divisor_override": 5}, name="divisor_override"),
        ],
    )
    def test_pooling_options(self, shape, kernel_size, stride, padding, options):
        self._check(shape, kernel_size, stride, padding, **options)

    @parametrize("shape,kernel_size,stride,padding", KERNEL_CASES)
    @parametrize("dtype", [torch.half, torch.bfloat16, torch.double])
    def test_dtypes(self, shape, kernel_size, stride, padding, dtype):
        self._check(shape, kernel_size, stride, padding, dtype=dtype)

    # An element offset below the vector width caps the vector size from the
    # pointer alignment check.
    @parametrize(
        "shape,kernel_size,stride,padding,storage_offset",
        [
            subtest(((16, 64, 64, 64), 2, 2, 0, 4), name="vec4"),
            subtest(((16, 64, 64, 64), 2, 2, 0, 1), name="scalar"),
            subtest(((16, 64, 64, 64), 3, 1, 1, 1), name="wtile_vec1"),
        ],
    )
    def test_misaligned_input(
        self, shape, kernel_size, stride, padding, storage_offset
    ):
        self._check(shape, kernel_size, stride, padding, storage_offset=storage_offset)


instantiate_parametrized_tests(TestAvgPool2dFwdChannelsLast)


if __name__ == "__main__":
    run_tests()
