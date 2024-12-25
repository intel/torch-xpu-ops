import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSafeSoftMax(TestCase):
    def test_sm(self):
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
            x_cpu = torch.randn(128,128,128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, -1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, -1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[0,0,:] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, -1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, -1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)

            x_cpu = torch.randn(128,128,128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[0,:,0] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)

            x_cpu = torch.randn(128,128,128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 0)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 0)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[:,0,0] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 0)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 0)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)


