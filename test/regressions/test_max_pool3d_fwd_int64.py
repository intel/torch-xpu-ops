# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for int32 overflow in max_pool3d forward kernel.

The forward kernel computes strides (e.g. in_batch_stride = C * D * H * W)
that can exceed INT32_MAX for large tensors. This test exercises shapes
where batch strides overflow int32 in both channels_last_3d and contiguous
memory formats.
"""

import torch
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


class TestMaxPool3dFwdInt64(TestCase):
    """Test that the forward pass of max_pool3d handles int64 strides correctly.

    Exercises two overflow scenarios:
    - channels_last_3d: in_batch_stride = C * D * H * W > INT32_MAX
      (shape [2, 2200, 100, 100, 100] -> stride 2.2B).
    - contiguous: in_cf_c_stride = D * H * W > INT32_MAX
      (shape [1, 2, 1300, 1300, 1300] -> stride 2.197B).
    """

    @largeTensorTest("20GB", device="cpu")
    @largeTensorTest("12GB", device="xpu")
    @parametrize(
        "shape,memory_format",
        [
            subtest(
                ((2, 2200, 100, 100, 100), torch.channels_last_3d),
                name="channels_last_3d",
            ),
            subtest(
                ((1, 2, 1300, 1300, 1300), torch.contiguous_format), name="contiguous"
            ),
        ],
    )
    def test_pool3d_fwd_large_size_int64(self, shape, memory_format):
        x = torch.empty(
            *shape, dtype=torch.half, device="xpu", memory_format=memory_format
        ).normal_()
        y = torch.nn.functional.max_pool3d(x, 5)
        torch.xpu.synchronize()

        ref_x = x.detach().cpu().float().contiguous()
        ref_y = torch.nn.functional.max_pool3d(ref_x, 5)

        self.assertEqual(y.cpu(), ref_y, exact_dtype=False)
        del x, y, ref_x, ref_y


instantiate_parametrized_tests(TestMaxPool3dFwdInt64)


if __name__ == "__main__":
    run_tests()
