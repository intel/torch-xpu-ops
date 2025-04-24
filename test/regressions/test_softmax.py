# Owner(s): ["module: intel"]
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_softmax_half_to_float(self):
        shape = [
            [8],
            [7, 8],
            [8192, 64],
            [8192, 8192],
            [7, 8, 512],
            [7, 8, 11],
            [16, 7, 8, 512],
            [16, 7, 8, 512, 35],
            [117, 7, 9, 513, 35],
        ]
        input_type = torch.float16
        output_type = torch.float
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                dim = j - 1
                x = torch.randn(shape[i]).to(input_type)
                grad = torch.randn(shape[i]).to(output_type)
                x_cpu = x.clone().requires_grad_()
                y_cpu = F.softmax(x_cpu, dim, dtype=output_type)
                y_cpu.backward(grad.clone())

                x_xpu = x.clone().to(xpu_device).requires_grad_()
                y_xpu = F.softmax(x_xpu, dim, dtype=output_type)
                y_xpu.backward(grad.clone().to(xpu_device))
                self.assertEqual(y_cpu, y_xpu.cpu())
                self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
