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
