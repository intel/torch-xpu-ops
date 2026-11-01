# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, subtest, TestCase

aten = torch.ops.aten


class TestMaxPool2dBwd(TestCase):
    def test_max_pool2d_bwd_fp16(self):
        # Regression test: verify correctness under float16 (lowp) inputs.
        # The large window size (13x13) causes window_size > 25, so the lowering
        # falls back to the aten kernel rather than generating a Triton kernel.
        # Reference is computed in float32 (upcast), then downcast to float16 for
        # comparison, matching the behavior of check_model with reference_in_float=True.
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [13, 13], [1, 1], [2, 2], [1, 1], False, c
            )

        x_fp16 = torch.randn([2, 64, 20, 20], dtype=torch.half, device="xpu")
        result, indices = aten.max_pool2d_with_indices(
            x_fp16,
            [13, 13],
            [1, 1],
            2,
            1,
            False,
        )
        grad_output_fp16 = torch.randn_like(result)

        # fp32 reference
        x_fp32 = x_fp16.float()
        grad_output_fp32 = grad_output_fp16.float()
        expected_fp32 = fn(grad_output_fp32, x_fp32, indices)
        expected = expected_fp32.half()

        actual = fn(grad_output_fp16, x_fp16, indices)

        self.assertEqual(actual, expected)

    def test_max_pool2d_bwd_channels_last(self):
        # Exercises the channels-last (NHWC) vectorized backward kernel against
        # the CPU reference. (The int64 offset overflow it guards against only
        # manifests at >2**31-element offsets, which is not allocatable in CI.)
        kernel, stride, pad, dil = [3, 3], [2, 2], [1, 1], [1, 1]
        x = torch.randn([2, 16, 12, 12]).to(memory_format=torch.channels_last)
        result, indices = aten.max_pool2d_with_indices(
            x, kernel, stride, pad, dil, False
        )
        grad_output = torch.randn_like(result)

        def bwd(grad_output, x, indices):
            return aten.max_pool2d_with_indices_backward(
                grad_output, x, kernel, stride, pad, dil, False, indices
            )

        expected = bwd(grad_output, x, indices)
        x_xpu = x.xpu()
        _, indices_xpu = aten.max_pool2d_with_indices(
            x_xpu, kernel, stride, pad, dil, False
        )
        actual = bwd(grad_output.xpu(), x_xpu, indices_xpu)

        self.assertEqual(actual.cpu(), expected)
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))


class TestMaxPool2dLayouts(TestCase):
    @dtypes(torch.float16)
    @parametrize("pattern", ["ties", "nan", "infinity"])
    @parametrize(
        "shape,kernel,stride,padding,memory_format",
        [
            subtest(((2, 24, 17, 17), 3, 2, 0, torch.channels_last), name="odd_spatial"),
            subtest(((2, 32, 16, 16), 3, 2, 1, torch.channels_last), name="padded"),
            subtest(((2, 32, 16, 16), 3, 2, 1, torch.contiguous_format), name="nchw"),
            subtest(((2, 32, 12, 16), 5, 1, 2, torch.channels_last), name="window5"),
            subtest(((2, 32, 12, 16), 9, 1, 4, torch.channels_last), name="window9"),
            subtest(((2, 32, 12, 16), 13, 1, 6, torch.channels_last), name="window13"),
            subtest(((2, 216, 42, 42), 3, 1, 1, torch.channels_last), name="stride1"),
            subtest(((2, 25, 17, 17), 3, 2, 1, torch.channels_last), name="odd_channels"),
        ],
    )
    def test_max_pool2d_forward_indices(
        self, device, dtype, pattern, shape, kernel, stride, padding, memory_format
    ):
        # Repeated maxima and multiple NaNs exercise both the first-maximum
        # tie rule and the last-NaN index rule, including padded windows.
        x = torch.zeros(shape, dtype=dtype, device="cpu").contiguous(
            memory_format=memory_format
        )
        x[..., 1::3, 1::3] = 1
        if pattern == "nan":
            x[..., 1::3, 1::3] = float("nan")
            x[..., 2::3, 2::3] = float("nan")
        elif pattern == "infinity":
            x.fill_(-float("inf"))
            x[..., 1::3, 1::3] = float("inf")
        args = ([kernel, kernel], [stride, stride], [padding, padding], [1, 1], False)
        expected, expected_indices = aten.max_pool2d_with_indices(x.float(), *args)
        actual, actual_indices = aten.max_pool2d_with_indices(x.to(device), *args)
        self.assertEqual(actual.cpu(), expected.to(dtype), rtol=0, atol=0)
        self.assertEqual(actual_indices.cpu(), expected_indices)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize(
        "shape,kernel,stride,padding,dilation,ceil_mode,gap",
        [
            subtest(((2, 24, 56, 56), 3, 2, 1, 1, False, 8), name="packed_channels"),
            subtest(((2, 26, 56, 56), 3, 2, 1, 1, False, 6), name="unaligned_channels"),
            subtest(((2, 32, 56, 56), 2, 2, 0, 1, False, 8), name="nonoverlap"),
            subtest(((2, 128, 17, 17), 3, 2, 0, 1, False, 8), name="odd_spatial"),
            subtest(((2, 216, 42, 42), 3, 1, 1, 1, False, 8), name="compact42"),
            subtest(((2, 432, 21, 21), 3, 1, 1, 1, False, 8), name="compact21"),
            subtest(((2, 864, 11, 11), 3, 1, 1, 1, False, 8), name="compact11"),
            subtest(((1, 24, 256, 256), 3, 2, 1, 1, False, 8), name="large_spatial"),
            subtest(((2, 25, 17, 17), 3, 2, 1, 1, False, 7), name="odd_channels"),
            subtest(((2, 24, 17, 17), 3, 2, 1, 2, False, 8), name="dilation"),
            subtest(((2, 24, 16, 16), 3, 2, 1, 1, True, 8), name="ceil_mode"),
        ],
    )
    def test_max_pool2d_backward_gapped_gradient(
        self, device, dtype, shape, kernel, stride, padding, dilation, ceil_mode, gap
    ):
        # Keep the original channel/spatial families with smaller batches.
        # The backward fast path accepts NCHW indices and a channels-last
        # gradient view with gaps between pixels; .to() on the view would
        # compact those gaps, so allocate the backing storage on the device.
        x = torch.testing.make_tensor(shape, dtype=dtype, device="cpu").contiguous(
            memory_format=torch.channels_last
        )
        args = (
            [kernel, kernel], [stride, stride], [padding, padding],
            [dilation, dilation], ceil_mode,
        )
        values, indices = aten.max_pool2d_with_indices(x.float(), *args)
        n, c, h, w = values.shape
        storage = torch.empty((n, h, w, c + gap), dtype=dtype, device=device)
        grad = storage[..., :c].permute(0, 3, 1, 2)
        # Small integers make accumulation exactly representable in both dtypes.
        grad_cpu = torch.randint(-2, 3, values.shape, device="cpu").to(dtype)
        grad.copy_(grad_cpu)
        indices = indices.contiguous()
        expected = aten.max_pool2d_with_indices_backward(
            grad_cpu.float(), x.float(), *args, indices
        ).to(dtype)
        self.assertFalse(grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(grad.stride(), (h * w * (c + gap), 1, w * (c + gap), c + gap))
        actual = aten.max_pool2d_with_indices_backward(
            grad, x.to(device), *args, indices.to(device)
        )
        self.assertEqual(actual.cpu(), expected, rtol=0, atol=0)
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))


instantiate_device_type_tests(
    TestMaxPool2dLayouts, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
