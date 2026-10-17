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

# (float dtype, matching signed-int dtype for bit inspection)
_DTYPE_CONFIGS = [
    (torch.float16, torch.int16),
    (torch.bfloat16, torch.int16),
    (torch.float32, torch.int32),
    (torch.float64, torch.int64),
]


class TestNextafter(TestCase):
    def _neg_min_subnormal(self, dtype, int_dtype, device):
        # Bit pattern 0x80...01 is the smallest-magnitude negative subnormal.
        int_min = torch.iinfo(int_dtype).min
        return torch.tensor([int_min + 1], dtype=int_dtype, device=device).view(dtype)

    def test_nextafter_crosses_to_negative_zero(self):
        # Stepping from -min_subnormal toward any larger value must land on
        # -0.0: the magnitude is zero but the sign bit must stay set, matching
        # CPU / IEEE. The device builtin used to drop the sign and return +0.0
        # for fp32/fp64.
        for dtype, int_dtype in _DTYPE_CONFIGS:
            neg_min = self._neg_min_subnormal(dtype, int_dtype, "xpu")
            for to in (0.0, 1.0, float("inf")):
                to_t = torch.tensor([to], dtype=dtype, device="xpu")
                out = torch.nextafter(neg_min, to_t)
                self.assertEqual(out.item(), 0.0)
                self.assertTrue(
                    torch.signbit(out).item(),
                    msg=f"{dtype}: nextafter(-min_subnormal, {to}) must be -0.0",
                )

    def test_nextafter_crosses_to_positive_zero(self):
        # The mirror case: stepping from +min_subnormal toward a smaller value
        # lands on +0.0 (sign bit clear).
        for dtype, int_dtype in _DTYPE_CONFIGS:
            pos_min = torch.tensor([1], dtype=int_dtype, device="xpu").view(dtype)
            for to in (0.0, -1.0, float("-inf")):
                to_t = torch.tensor([to], dtype=dtype, device="xpu")
                out = torch.nextafter(pos_min, to_t)
                self.assertEqual(out.item(), 0.0)
                self.assertFalse(
                    torch.signbit(out).item(),
                    msg=f"{dtype}: nextafter(+min_subnormal, {to}) must be +0.0",
                )

    def test_nextafter_bitexact_vs_cpu(self):
        # Bit-exact parity with CPU over a grid that includes the signed-zero
        # crossing cases the device builtin previously flushed.
        for dtype, int_dtype in _DTYPE_CONFIGS:
            int_min = torch.iinfo(int_dtype).min
            subnormals = torch.tensor([int_min + 1, 1], dtype=int_dtype).view(dtype)
            base = torch.tensor([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0], dtype=dtype)
            values = torch.cat((base, subnormals))
            x = values[:, None].repeat(1, values.numel())
            y = values[None, :].repeat(values.numel(), 1)
            ref = torch.nextafter(x, y)
            out = torch.nextafter(x.xpu(), y.xpu()).cpu()
            self.assertEqual(out.view(int_dtype), ref.view(int_dtype))


if __name__ == "__main__":
    run_tests()
