# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# Compare exact GELU against a cancellation-free float64 reference rather than
# relying on the CPU GELU implementation's negative-tail accuracy.
import math

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGeluTailAccuracy(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_forward_tail_reference(self, device, dtype):
        x = torch.arange(-12, 12, 2**-6, dtype=torch.float64).to(
            device=device, dtype=dtype
        )
        reference_x = x.cpu().double()
        expected = (reference_x * 0.5 * torch.erfc(-reference_x * math.sqrt(0.5))).to(
            dtype
        )
        tolerance = {
            torch.float16: 0.0015,
            torch.bfloat16: 0.009,
            torch.float32: 3e-5,
            torch.float64: 3e-13,
        }[dtype]
        subnormal = torch.nextafter(
            torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
        ).item()
        self.assertEqual(F.gelu(x).cpu(), expected, atol=subnormal, rtol=tolerance)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_backward_tail_reference(self, device, dtype):
        x = torch.arange(-12, 12, 2**-6, dtype=torch.float64).to(
            device=device, dtype=dtype
        )
        reference_x = x.cpu().double()
        grad = torch.randn_like(x)
        cdf = 0.5 * torch.erfc(-reference_x * math.sqrt(0.5))
        derivative = cdf + reference_x * math.sqrt(0.5 / math.pi) * torch.exp(
            -0.5 * reference_x.square()
        )
        expected = (derivative * grad.cpu().double()).to(dtype)
        tolerance = {
            torch.float16: 0.0015,
            torch.bfloat16: 0.009,
            torch.float32: 3e-5,
            torch.float64: 3e-13,
        }[dtype]
        subnormal = torch.nextafter(
            torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
        ).item()
        self.assertEqual(
            torch.ops.aten.gelu_backward(grad, x).cpu(),
            expected,
            atol=subnormal,
            rtol=tolerance,
        )


instantiate_device_type_tests(
    TestGeluTailAccuracy, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
