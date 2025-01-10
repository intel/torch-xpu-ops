# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleCopy(TestCase):
    def test_copy_and_clone(self, dtype=torch.float):
        a_cpu = torch.randn(16, 64, 28, 28)
        b_cpu = torch.randn(16, 64, 28, 28)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        # naive
        b_cpu.copy_(a_cpu)
        b_xpu.copy_(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
        # clone + permutation
        b_cpu = a_cpu.clone(memory_format=torch.channels_last)
        b_xpu = a_xpu.clone(memory_format=torch.channels_last)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
