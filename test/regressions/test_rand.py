# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_distribution_normal(self, dtype=torch.float):
        tol = 1e-2
        x_xpu = torch.tensor(list(range(10000)), device=xpu_device, dtype=dtype)
        x_xpu.normal_(2.0, 0.5)
        self.assertEqual(x_xpu.cpu().mean(), 2.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 0.5, rtol=tol, atol=tol)
        x_xpu = torch.normal(
            mean=-5.0, std=0.2, size=(10000,), device=xpu_device, dtype=dtype
        )
        self.assertEqual(x_xpu.cpu().mean(), -5.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 0.2, rtol=tol, atol=tol)
        torch.normal(
            mean=-3.0, std=1.2, size=(10000,), device=xpu_device, dtype=dtype, out=x_xpu
        )
        self.assertEqual(x_xpu.cpu().mean(), -3.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 1.2, rtol=tol, atol=tol)

    def test_distribution_uniform(self, dtype=torch.float):
        tol = 1e-2
        x_xpu = torch.tensor(list(range(10000)), device=xpu_device, dtype=dtype)
        x_xpu.uniform_(0, 10)
        self.assertEqual(x_xpu.cpu().mean(), 5.0, rtol=tol, atol=tol)
        x_xpu = torch.rand(size=(2000, 5), device=xpu_device, dtype=dtype)
        self.assertEqual(x_xpu.cpu().view(-1).mean(), 0.5, rtol=tol, atol=tol)


class TestNullaryDistribution(TestCase):
    @parametrize("numel", [7, 231, 640001, 1 << 20])
    @parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
    def test_normal_moments_and_determinism(self, numel, dtype):
        x = torch.empty(numel, dtype=dtype, device=xpu_device)
        torch.manual_seed(1234)
        x.normal_()
        y = torch.empty_like(x)
        torch.manual_seed(1234)
        y.normal_()
        self.assertEqual(x, y)
        x = x.float().cpu()
        self.assertFalse(x.isnan().any())
        self.assertEqual(x.mean().item(), 0.0, atol=5.0 / numel**0.5, rtol=0)
        if numel > 1:
            self.assertEqual(
                x.std().item(), 1.0, atol=5.0 / (2.0 * numel) ** 0.5 + 1e-2, rtol=0
            )

    def test_normal_tails(self):
        # Box-Muller uses fast sin/cos; check the tails of a large sample
        # against the standard normal, P(|z| > 4) ~= 6.3e-5.
        n = 1 << 24
        torch.manual_seed(0)
        z = torch.empty(n, device=xpu_device).normal_().cpu().double()
        self.assertFalse(z.isnan().any() or z.isinf().any())
        for k, p in ((2.0, 4.550026e-2), (3.0, 2.699796e-3), (4.0, 6.334248e-5)):
            observed = (z.abs() > k).sum().item()
            expected = p * n
            self.assertLess(abs(observed - expected), 5 * expected**0.5)

    @parametrize("start", [0, 1])
    @parametrize("numel", [7, 231, 640001])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_normal_stays_in_bounds(self, start, numel, dtype):
        sentinel = 100.0
        buf = torch.full(
            (start + numel + 64,), sentinel, dtype=dtype, device=xpu_device
        )
        buf[start : start + numel].normal_()
        buf = buf.cpu()
        self.assertTrue((buf[:start] == sentinel).all())
        self.assertTrue((buf[start + numel :] == sentinel).all())
        self.assertFalse((buf[start : start + numel] == sentinel).any())


instantiate_parametrized_tests(TestNullaryDistribution)


if __name__ == "__main__":
    run_tests()
