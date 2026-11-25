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


class TestBessel(TestCase):
    # Cover both the |x| <= 5 rational approximation and the |x| > 5
    # asymptotic expansion; the denominator polynomials of both branches
    # were previously seeded with 0 instead of 1.
    def _check(self, op, x_cpu):
        for dtype in (torch.float32, torch.float64):
            x = x_cpu.to(dtype)
            self.assertEqual(op(x.xpu()).cpu(), op(x), msg=f"{op.__name__} {dtype}")

    def test_bessel_j1(self):
        x = torch.cat((torch.linspace(-20.0, 20.0, 401), torch.tensor([1e-3, 5.0])))
        self._check(torch.special.bessel_j1, x)

    def test_bessel_y1(self):
        x = torch.cat((torch.linspace(0.1, 20.0, 200), torch.tensor([5.0, 30.0])))
        self._check(torch.special.bessel_y1, x)


if __name__ == "__main__":
    run_tests()
