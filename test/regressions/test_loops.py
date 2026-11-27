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
from torch.testing._internal.common_utils import parametrize, TestCase

test_shapes = [
    [[23, 72, 72], [5184, 72, 1], [23, 72, 72], [5184, 72, 1]],
    [[23, 16, 16], [23, 1, 16]],
    [[23, 16, 17], [23, 1, 17]],
    [[1, 72, 72], [23, 72, 72]],
    [[23, 72, 1], [23, 72, 72]],
    [[23000, 72, 72], [5184, 72, 1], [23000, 72, 72], [5184, 72, 1]],
    [[16, 16, 256, 256], [16, 16, 256, 256]],
    [[16, 16, 512, 512], [16, 1, 1, 512]],
    [[4, 15000, 3], [105000, 1, 15000], [4, 1, 3], [3, 3, 1]],
    [[16, 16, 512, 513], [16, 1, 1, 513]],
    [[28, 4096, 9], [36864, 9, 1], [28, 4096, 1], [4096, 1, 1]],
]


class TestLoopsKernel(TestCase):
    def _test_loops(self, dtype=torch.float):
        for shape in test_shapes:
            if len(shape) == 2:
                a = torch.randn(shape[0], dtype=dtype)
                b = torch.randn(shape[1], dtype=dtype)
            elif len(shape) == 4:
                a = torch.as_strided(
                    torch.randn(shape[0][0] * shape[1][0]), shape[0], shape[1]
                )
                b = torch.as_strided(
                    torch.randn(shape[2][0] * shape[3][0]), shape[2], shape[3]
                )
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            c = a + b + 1
            c_xpu = a_xpu + b_xpu + 1
            self.assertEqual(c, c_xpu.cpu())

    def test_loops_float(self):
        self._test_loops(torch.float)

    def test_loops_half(self):
        self._test_loops(torch.half)

    def test_loops_bfloat16(self):
        self._test_loops(torch.bfloat16)

    def test_loops_dynamic_cast(self):
        for shape in test_shapes:
            if len(shape) == 2:
                a = torch.randn(shape[0], dtype=torch.float)
                b = torch.randn(shape[1], dtype=torch.half)
            elif len(shape) == 4:
                a = torch.as_strided(
                    torch.randn(shape[0][0] * shape[1][0], dtype=torch.float),
                    shape[0],
                    shape[1],
                )
                b = torch.as_strided(
                    torch.randn(shape[2][0] * shape[3][0], dtype=torch.half),
                    shape[2],
                    shape[3],
                )
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            print(
                f"a_xpu:{a_xpu.dtype}, {a_xpu.shape}, {a.stride()}; b_xpu:{b_xpu.dtype}, {b_xpu.shape}, {b_xpu.stride()}",
                flush=True,
            )
            c = a + b + 1
            c_xpu = a_xpu + b_xpu + 1
            self.assertEqual(c, c_xpu.cpu())

    def test_bc_vec_large_tensor(self):
        raw_data = torch.rand(48, 64, 64, 64, 64)
        a = raw_data.xpu().transpose(0, 1).contiguous().transpose(0, 1)
        self.assertEqual(a, raw_data.xpu())


class TestStridedUnary(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_broadcast_row_crossing_and_tail(self, device, dtype):
        rows = torch.tensor([[-3], [1], [7]], dtype=dtype)
        # Nine elements per row cross vec4 and vec8 boundaries; 27 leave a tail.
        x = rows.to(device).expand(3, 9)
        ref = rows.expand(3, 9)
        self.assertEqual(torch.neg(x).cpu(), torch.neg(ref))
        self.assertEqual((x - 2).cpu(), ref - 2)
        self.assertEqual((2 - x).cpu(), 2 - ref)

    @parametrize(
        "shape,order", [((24, 16), (1, 0)), ((2, 3, 4, 16), (0, 3, 1, 2))]
    )
    def test_padded_rows(self, device, shape, order):
        rows = torch.arange(-192, 192, dtype=torch.float32).view(shape)
        ref = rows[..., :8].permute(order)
        x = rows.to(device)[..., :8].permute(order)
        for offset in (0, 1):
            # Offset one float misaligns the output while preserving its strides.
            buf = torch.full((ref.numel() + offset,), 777.0, device=device)
            out = buf[offset:].view(*shape[:-1], 8).permute(order)
            torch.neg(x, out=out)
            self.assertEqual(out.cpu(), -ref)
            if offset:
                self.assertEqual(buf[0].item(), 777.0)
        # Eight bool outputs would require a 32-byte load from each FP32 row.
        self.assertEqual(torch.signbit(x).cpu(), torch.signbit(ref))


instantiate_device_type_tests(
    TestStridedUnary, globals(), only_for="xpu", allow_xpu=True
)
