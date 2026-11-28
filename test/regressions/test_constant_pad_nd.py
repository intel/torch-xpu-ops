# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

DTYPES = (torch.bool, torch.int8, torch.float16, torch.float32, torch.complex128)


def make(shape, dtype):
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype)
    return torch.randn(shape).mul(10).to(dtype)


def to_xpu(t):
    # Same sizes, strides and storage offset as t.
    n = t.untyped_storage().nbytes() // t.element_size()
    flat = torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0, (n,), (1,))
    return flat.to("xpu").as_strided(t.shape, t.stride(), t.storage_offset())


class TestConstantPadNd(TestCase):
    def test_constant_pad_nd_layouts(self):
        # Pad, crop and mixed pads in one pass; channels-last, strided and
        # misaligned inputs; rows that are not a multiple of the store width.
        pads = [[1, 2], [0, 1, 2, 0], [1, 1, 1, 1, 1, 1], [-1, 2, 3, -2]]
        pads += [[-1, -2, 0, -1], [3, 3, 3, 3, 0, 0, 1, 0], [0, 0, -2, 0]]
        for dtype in DTYPES:
            value = True if dtype == torch.bool else 2
            x = make((2, 6, 5, 7), dtype)
            inputs = [x, x.contiguous(memory_format=torch.channels_last)]
            inputs += [x[:, :, 1:, 2:], x.transpose(2, 3)]
            for inp in inputs:
                for pad in pads:
                    ref = torch.constant_pad_nd(inp, pad, value)
                    res = torch.constant_pad_nd(to_xpu(inp), pad, value)
                    self.assertEqual(res.stride(), ref.stride())
                    self.assertEqual(res.cpu(), ref, atol=0, rtol=0)

    def test_constant_pad_nd_autograd_and_errors(self):
        x = torch.randn(3, 4, 5, device="xpu", requires_grad=True)
        torch.constant_pad_nd(x, [1, 2, -1, 1]).sum().backward()
        expected = torch.ones(3, 4, 5)
        expected[:, 0] = 0
        self.assertEqual(x.grad.cpu(), expected)
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x.detach(), [1])
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x.detach(), [-3, -3])


if __name__ == "__main__":
    run_tests()
