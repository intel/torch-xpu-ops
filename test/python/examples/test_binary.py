import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleBinary(TestCase):
    def test_add(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu + b_xpu
        c_cpu = a_cpu + b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_add_scalar(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b = 1.11
        a_xpu = a_cpu.to(xpu_device)
        c_xpu = a_xpu + b
        c_cpu = a_cpu + b
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_sub(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu - b_xpu
        c_cpu = a_cpu - b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_mul(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu * b_xpu
        c_cpu = a_cpu * b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_div(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu / b_xpu
        c_cpu = a_cpu / b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))
