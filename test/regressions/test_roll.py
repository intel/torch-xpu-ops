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
    largeTensorTest,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def make(shape, dtype):
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype)
    return torch.randn(shape).mul(10).to(dtype)


def to_device(t, device):
    # Same sizes, strides and storage offset as t.
    n = t.untyped_storage().nbytes() // t.element_size()
    flat = torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0, (n,), (1,))
    return flat.to(device).as_strided(t.shape, t.stride(), t.storage_offset())


class TestRoll(TestCase):
    @dtypes(torch.bool, torch.int8, torch.float16, torch.float32, torch.complex128)
    def test_roll_layouts(self, device, dtype):
        # Several rolled dims in one pass, input read in place (strided,
        # expanded, channels-last, misaligned), rows that are not a multiple
        # of the store width.
        args = [((1,), (0,)), ((2, -3), (1, 3)), ((5, 1, 2, -9), (0, 1, 2, 3))]
        args += [((3, 4), (2, 2)), ((-1,), (-1,)), ((7,), (1,))]
        x = make((4, 6, 5, 7), dtype)
        inputs = [x, x.permute(0, 2, 3, 1), x[:, 1:, :, 2:]]
        inputs += [x[:1].expand(3, 6, 5, 7)]
        inputs += [x.contiguous(memory_format=torch.channels_last)]
        for inp in inputs:
            for shifts, dims in args:
                ref = torch.roll(inp, shifts, dims)
                res = torch.roll(to_device(inp, device), shifts, dims)
                self.assertTrue(res.is_contiguous())
                self.assertEqual(res.cpu(), ref, atol=0, rtol=0)
        self.assertEqual(torch.roll(x.to(device), 3).cpu(), torch.roll(x, 3))

    @largeTensorTest("4GB", device="xpu")
    def test_roll_index_overflow(self, device):
        # The last coordinate plus the shift is 2^31 and overflows int32.
        x = torch.randint(-128, 128, (2**30 + 1,), dtype=torch.int8, device=device)
        res = torch.roll(x, 1, 0)
        self.assertEqual(res[0], x[-1])
        self.assertTrue(torch.equal(res[1:], x[:-1]))

    @largeTensorTest("8GB", device="xpu")
    def test_roll_fallback_layout(self, device):
        # Beyond 32-bit indexing the native fallback is used; it must also
        # return a contiguous result for a channels-last input.
        x = torch.empty(
            (2, 1024, 1024, 1024),
            dtype=torch.int8,
            device=device,
            memory_format=torch.channels_last,
        ).random_()
        res = torch.roll(x, 1, 0)
        self.assertTrue(res.is_contiguous())
        self.assertTrue(torch.equal(res[0], x[1]))
        self.assertTrue(torch.equal(res[1], x[0]))

    def test_roll_errors(self, device):
        x = torch.randn(3, 4, device=device)
        with self.assertRaises(RuntimeError):
            torch.roll(x, (1, 2), (0,))
        with self.assertRaises(IndexError):
            torch.roll(x, 1, 2)


instantiate_device_type_tests(TestRoll, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
