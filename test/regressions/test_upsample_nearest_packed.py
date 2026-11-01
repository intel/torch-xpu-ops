# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import math

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestUpsampleNearestPacked(TestCase):
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("channels", [18, 36, 72])
    @parametrize("scale", [2, 4, 8])
    def test_integer_scale(self, device, dtype, channels, scale):
        size = 56 // scale
        x = torch.randn(2, channels, size, size, dtype=dtype)
        x_device = x.to(device).contiguous(memory_format=torch.channels_last)
        actual = torch.ops.aten.upsample_nearest2d(
            x_device, [56, 56], float(scale), float(scale)
        )
        expected = x.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

        grad = torch.randn(2, channels, 56, 56, dtype=dtype)
        grad_device = grad.to(device).contiguous(memory_format=torch.channels_last)
        actual_grad = torch.ops.aten.upsample_nearest2d_backward(
            grad_device, [56, 56], list(x.shape), float(scale), float(scale)
        )
        # Accumulate each replicated block in FP32, rounding only the final sum.
        expected_grad = torch.zeros(x.shape, dtype=torch.float32)
        for h in range(scale):
            for w in range(scale):
                expected_grad += grad[:, :, h::scale, w::scale].float()
        self.assertEqual(actual_grad.cpu(), expected_grad.to(dtype), atol=0, rtol=0)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("mode", ["nearest", "nearest-exact"])
    @parametrize(
        "case",
        [
            "odd_channels",
            "small_channels",
            "contiguous",
            "strided",
            "input_offset",
            "output_offset",
            "fractional_scale",
            "mismatched_scale",
            "unequal_scales",
        ],
    )
    def test_fallbacks(self, device, dtype, mode, case):
        channels = {"odd_channels": 17, "small_channels": 14}.get(case, 18)
        shape = (2, channels, 7, 9)
        output_size, scales = [14, 18], (2.0, 2.0)
        if case == "fractional_scale":
            output_size, scales = [10, 13], (1.5, 1.5)
        elif case == "mismatched_scale":
            # CPU nearest_idx ignores explicit scales for exactly 2x sizes.
            # Use 4x sizes so the CPU reference honors the mismatched scale.
            output_size, scales = [28, 36], (4.1, 4.1)
        elif case == "unequal_scales":
            output_size, scales = [14, 36], (2.0, 4.0)
        layout = case if case in ("contiguous", "strided") else "channels_last"

        def allocate(size, offset=0):
            n, c, h, w = size
            if layout == "contiguous":
                strides = (c * h * w, h * w, w, 1)
            else:
                strides = (h * w * c, 1, w * c, c)
            if layout == "strided":
                strides = tuple(s * 2 for s in strides)
            storage_size = 1 + sum((d - 1) * s for d, s in zip(size, strides))
            storage = torch.empty(storage_size + offset, device=device, dtype=dtype)
            return storage.as_strided(size, strides, offset)

        forward = (
            torch.ops.aten.upsample_nearest2d
            if mode == "nearest"
            else torch.ops.aten._upsample_nearest_exact2d
        )
        backward = (
            torch.ops.aten.upsample_nearest2d_backward
            if mode == "nearest"
            else torch.ops.aten._upsample_nearest_exact2d_backward
        )
        x = torch.randn(shape, dtype=dtype)
        x_device = allocate(shape, int(case == "input_offset")).copy_(x)
        output_shape = (shape[0], shape[1], *output_size)
        out = allocate(output_shape, int(case == "output_offset"))
        forward.out(x_device, output_size, *scales, out=out)
        expected = forward(x.float(), output_size, *scales).to(dtype)
        self.assertEqual(out.cpu(), expected, atol=0, rtol=0)

        # Integer gradients make all reference accumulation orders exact in FP32.
        grad = torch.randint(-4, 5, output_shape).to(dtype)
        grad_device = allocate(output_shape, int(case == "input_offset")).copy_(grad)
        grad_input = allocate(shape, int(case == "output_offset"))
        backward.grad_input(
            grad_device, output_size, list(shape), *scales, grad_input=grad_input
        )
        expected_grad = backward(grad.float(), output_size, list(shape), *scales)
        self.assertEqual(grad_input.cpu(), expected_grad.to(dtype), atol=0, rtol=0)

    @dtypes(torch.float16, torch.bfloat16)
    def test_packed_channel_boundary(self, device, dtype):
        # C=18 crosses channel and row boundaries within a vector.
        shape = (1, 18, 3, 5)
        x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(dtype)
        x_device = x.to(device).contiguous(memory_format=torch.channels_last)
        actual = torch.ops.aten.upsample_nearest2d(x_device, [6, 10])
        expected = x.repeat_interleave(2, 2).repeat_interleave(2, 3)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)


instantiate_device_type_tests(
    TestUpsampleNearestPacked, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
