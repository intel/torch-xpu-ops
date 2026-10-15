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
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestAdaptiveMaxPool3dBackwardLayout(TestCase):
    def _inputs(self, shape, device, dtype):
        if (
            dtype == torch.float64
            and not torch.xpu.get_device_properties(device).has_fp64
        ):
            self.skipTest("XPU device does not support float64")
        generator = torch.Generator().manual_seed(5475)
        x = torch.randperm(math.prod(shape), generator=generator).reshape(shape)
        x = x.to(device=device, dtype=dtype)
        values, indices = F.adaptive_max_pool3d(x, (2, 3, 5), return_indices=True)
        grad = torch.arange(1, values.numel() + 1).reshape(values.shape)
        expected = torch.zeros(shape).flatten(-3)
        expected.scatter_add_(-1, indices.cpu().flatten(-3), grad.float().flatten(-3))
        return (
            x,
            grad.to(device=device, dtype=dtype),
            indices,
            expected.reshape(shape).to(dtype),
        )

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("shape", [(3, 4, 3, 5), (1, 3, 4, 3, 5), (2, 3, 4, 5, 5)])
    @parametrize("index_layout", ["contiguous", "channels_last", "sliced"])
    @parametrize("grad_contiguous", [True, False])
    def test_indices_layout(self, device, dtype, shape, index_layout, grad_contiguous):
        x, grad, indices, expected = self._inputs(shape, device, dtype)
        if index_layout == "channels_last":
            fmt = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
            indices = indices.contiguous(memory_format=fmt)
        elif index_layout == "sliced":
            buffer = torch.zeros(
                (*indices.shape[:-1], indices.shape[-1] * 2 + 1),
                device=device,
                dtype=torch.int64,
            )
            indices_view = buffer[..., 1::2]
            indices_view.copy_(indices)
            indices = indices_view
        if index_layout != "contiguous":
            self.assertFalse(indices.is_contiguous())
        if not grad_contiguous:
            grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
            self.assertFalse(grad.is_contiguous())

        actual = torch.ops.aten.adaptive_max_pool3d_backward(grad, x, indices)
        self.assertEqual(actual.device, x.device)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("shape", [(3, 4, 3, 5), (1, 3, 4, 3, 5), (2, 3, 4, 5, 5)])
    @parametrize("output_layout", ["channels_last", "sliced"])
    @parametrize("indices_contiguous", [True, False])
    def test_grad_input_layout(
        self, device, dtype, shape, output_layout, indices_contiguous
    ):
        x, grad, indices, expected = self._inputs(shape, device, dtype)
        fmt = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
        if not indices_contiguous:
            indices = indices.contiguous(memory_format=fmt)
            self.assertFalse(indices.is_contiguous())
        if output_layout == "channels_last":
            out = torch.empty_like(x, memory_format=fmt)
        else:
            buffer = torch.full(
                (*shape[:-1], shape[-1] * 2 + 1), -999, device=device, dtype=dtype
            )
            out = buffer[..., 1::2]
        self.assertFalse(out.is_contiguous())
        strides, offset, pointer = out.stride(), out.storage_offset(), out.data_ptr()

        actual = torch.ops.aten.adaptive_max_pool3d_backward.grad_input(
            grad, x, indices, grad_input=out
        )
        self.assertEqual(actual.device, x.device)
        self.assertEqual(actual.data_ptr(), pointer)
        self.assertEqual(out.stride(), strides)
        self.assertEqual(out.storage_offset(), offset)
        if output_layout == "sliced":
            self.assertEqual(buffer[..., ::2], torch.full_like(buffer[..., ::2], -999))
        self.assertEqual(out.cpu(), expected, atol=0, rtol=0)


instantiate_device_type_tests(
    TestAdaptiveMaxPool3dBackwardLayout, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
