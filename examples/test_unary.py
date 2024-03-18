import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleUnary(TestCase):
    def test_abs(self, dtype=torch.float):
        data = [
            [
                -0.2911,
                -1.3204,
                -2.6425,
                -2.4644,
                -0.6018,
                -0.0839,
                -0.1322,
                -0.4713,
                -0.3586,
                -0.8882,
                0.0000,
                0.0000,
                1.1111,
                2.2222,
                3.3333,
            ]
        ]
        excepted = [
            [
                0.2911,
                1.3204,
                2.6425,
                2.4644,
                0.6018,
                0.0839,
                0.1322,
                0.4713,
                0.3586,
                0.8882,
                0.0000,
                0.0000,
                1.1111,
                2.2222,
                3.3333,
            ]
        ]
        x_dpcpp = torch.tensor(data, device=xpu_device)
        y = torch.tensor(excepted, device=xpu_device)
        y_dpcpp = torch.abs(x_dpcpp)
        self.assertEqual(y.to(cpu_device), y_dpcpp.to(cpu_device))
