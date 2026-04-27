# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

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

    def test_std_large_float32_no_inf(self):
        torch.manual_seed(0)
        x = torch.randn(1000, dtype=torch.float32) * 1e19 + 1e20

        cpu_result = torch.std(x)
        xpu_result = torch.std(x.to(xpu_device)).cpu()
        cpu_var = torch.var(x)
        xpu_var = torch.var(x.to(xpu_device)).cpu()

        self.assertTrue(torch.isfinite(cpu_result))
        self.assertTrue(torch.isfinite(xpu_result))
        self.assertTrue(torch.isfinite(cpu_var))
        self.assertTrue(torch.isfinite(xpu_var))
        self.assertEqual(xpu_result, cpu_result, rtol=1e-4, atol=1e-4)
        self.assertEqual(xpu_var, cpu_var, rtol=1e-4, atol=1e-4)
