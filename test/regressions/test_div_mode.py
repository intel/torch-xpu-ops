# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import TestCase


class TestDivMode(TestCase):
    def test_div_true_dtype(self):
        claimed_dtypes = get_all_dtypes()
        for dtype in claimed_dtypes:
            a_cpu = torch.randint(1, 100, [8, 8]).to(dtype)
            a_xpu = a_cpu.to("xpu")
            ref = torch.ops.aten.div(a_cpu * 2, a_cpu, rounding_mode=None)
            res = torch.ops.aten.div(a_xpu * 2, a_xpu, rounding_mode=None)
            self.assertEqual(ref, res.to("cpu"))

    def test_div_trunc_scalar_divisor_exact_quotient(self):
        # A CPU-scalar divisor must give the same result as on CPU; computing
        # a * (1 / b) turned 7.625 / 3.8125 == 2 into 1.
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            a = torch.tensor([7.625, -7.625, 6.0, 0.75], dtype=dtype)
            for b in (3.8125, 0.375):
                expected = torch.div(a, b, rounding_mode="trunc")
                actual = torch.div(a.to("xpu"), b, rounding_mode="trunc")
                self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    def test_div_trunc_reduced_precision_quotient(self):
        # With a single-element divisor CPU truncates the 16-bit quotient in
        # float, reading a scalar divisor at float precision and a scalar
        # numerator at the input dtype. Rounding the quotient first gives
        # 5.90625 / -0.1376953125 = -42.89 -> -43; reading the divisor 0.1 as
        # bf16 gives 1.0 / 0.1 = 9.99 -> 9; reading the numerator 0.1 as float
        # gives 0.1 / bf16(0.010009765625) = 9.99 -> 9 (CPU: 10).
        for dtype in (torch.float16, torch.bfloat16):
            a = torch.tensor([5.90625, 5.5, -6.1875], dtype=dtype)
            b = torch.tensor(-0.1376953125, dtype=dtype)
            for divisor in (b, b.to("xpu")):
                expected = torch.div(a, b, rounding_mode="trunc")
                actual = torch.div(a.to("xpu"), divisor, rounding_mode="trunc")
                self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)
            a = torch.tensor([1.0, 2.0, 3.0, -3.0], dtype=dtype)
            for scalar in (0.1, 0.3, 0.7):
                expected = torch.div(a, scalar, rounding_mode="trunc")
                actual = torch.div(a.to("xpu"), scalar, rounding_mode="trunc")
                self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)
            b = torch.tensor([0.010009765625, -0.0301513671875], dtype=dtype)
            for scalar in (0.1, 0.3):
                expected = torch.div(scalar, b, rounding_mode="trunc")
                actual = torch.div(scalar, b.to("xpu"), rounding_mode="trunc")
                self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)
