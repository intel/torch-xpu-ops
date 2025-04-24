import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


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

                x_dpcpp = x.clone().to(dpcpp_device).requires_grad_()
                y_dpcpp = F.softmax(x_dpcpp, dim, dtype=output_type)
                y_dpcpp.backward(grad.clone().to(dpcpp_device))
                self.assertEqual(y_cpu, y_dpcpp.cpu())
                self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
