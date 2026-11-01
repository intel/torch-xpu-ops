# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestAvgPool2dBackwardStrided(TestCase):
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize(
        "channels,height,width,pixel_stride,mixed,ceil_mode,divisor",
        [
            (14, 56, 56, 112, False, False, None),
            (26, 56, 56, 104, False, False, None),
            (32, 56, 56, 128, False, False, None),
            (384, 14, 14, 384, True, False, None),
            (32, 55, 57, 128, False, False, None),
            (13, 15, 17, 52, False, False, None),
            (26, 15, 17, 104, False, True, None),
            (26, 15, 17, 104, False, False, 7),
        ],
    )
    @parametrize("include_pad", [True, False])
    def test_avg_pool2d_backward_low_precision_strided(
        self,
        device,
        dtype,
        channels,
        height,
        width,
        pixel_stride,
        mixed,
        ceil_mode,
        divisor,
        include_pad,
    ):
        # XPU rounds each normalized contribution to the input dtype before
        # accumulating in float32. CPU low-precision backward rounds each sum.
        shape = (2, channels, height, width)
        strides = (
            height * width * pixel_stride, 1, width * pixel_stride, pixel_stride
        )
        input = (
            torch.empty(shape, device=device, dtype=dtype)
            if mixed
            else torch.empty_strided(shape, strides, device=device, dtype=dtype)
        )
        input.zero_()
        grad_values = make_tensor(shape, device="cpu", dtype=dtype, low=-2, high=2)
        grad_output = torch.empty_strided(shape, strides, device=device, dtype=dtype)
        grad_output.copy_(grad_values)
        expected = torch.zeros(shape, dtype=torch.float32)
        for y in range(height):
            y0, y1 = max(y - 1, 0), min(y + 2, height)
            for x in range(width):
                x0, x1 = max(x - 1, 0), min(x + 2, width)
                denominator = (
                    divisor
                    if divisor is not None
                    else (9 if include_pad else (y1 - y0) * (x1 - x0))
                )
                contribution = (
                    (grad_values[:, :, y, x].float() / denominator).to(dtype).float()
                )
                expected[:, :, y0:y1, x0:x1] += contribution[:, :, None, None]
        actual = torch.ops.aten.avg_pool2d_backward(
            grad_output, input, [3, 3], [1, 1], [1, 1], ceil_mode, include_pad, divisor
        )
        self.assertEqual(actual.cpu(), expected.to(dtype), rtol=0, atol=0)
        memory_format = torch.contiguous_format if mixed else torch.channels_last
        self.assertTrue(actual.is_contiguous(memory_format=memory_format))


instantiate_device_type_tests(
    TestAvgPool2dBackwardStrided, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
