# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Reproducer for: RuntimeError: iter.device(arg).is_xpu() in Loops.h
when torch.addcmul is called with a CPU scalar tensor (use_cpu_scalar=True).

The fix adds handling for is_cpu_scalar(3) in addcmul_kernel, extracting the
scalar value and running a 2-argument kernel on the XPU tensors.
"""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestAddcmulCpuScalar(TestCase):
    def _run(self, dtype):
        device = "xpu"
        if dtype.is_floating_point or dtype.is_complex:
            a = torch.rand(3, 3, dtype=dtype, device=device)
            b = torch.rand(3, 3, dtype=dtype, device=device)
            c = torch.tensor(2.0, dtype=dtype, device="cpu")
            alpha = 0.5
        else:
            a = torch.randint(1, 5, (3, 3), dtype=dtype, device=device)
            b = torch.randint(1, 5, (3, 3), dtype=dtype, device=device)
            c = torch.tensor(2, dtype=dtype, device="cpu")
            alpha = 3

        actual = torch.addcmul(a, b, c, value=alpha)
        expected = torch.addcmul(a.cpu(), b.cpu(), c, value=alpha).to(device)
        if dtype.is_floating_point or dtype.is_complex:
            self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        else:
            self.assertEqual(actual, expected)

    def test_float32(self):
        self._run(torch.float32)

    def test_float64(self):
        self._run(torch.float64)

    def test_complex64(self):
        self._run(torch.complex64)

    def test_complex128(self):
        self._run(torch.complex128)

    def test_int8(self):
        self._run(torch.int8)

    def test_int16(self):
        self._run(torch.int16)

    def test_int32(self):
        self._run(torch.int32)

    def test_int64(self):
        self._run(torch.int64)

    def test_uint8(self):
        self._run(torch.uint8)


if __name__ == "__main__":
    run_tests()
