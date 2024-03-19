import torch
from torch.testing._internal.common_utils import TestCase

class TestFoldUnfold(TestCase):
    def test_im2col(self):
        def helper(x):
            return torch.nn.functional.unfold(x, kernel_size=(10, 15), dilation=2, padding=5, stride=3)
        x_cpu = torch.rand(1, 1, 200, 100)
        x = x_cpu.detach().clone().to('xpu')
        self.assertEqual(helper(x_cpu), helper(x))

    def test_unfold(self):
        x_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True)
        x_xpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True, device="xpu"
        )
        y_cpu = x_cpu.unfold(0, 5, 1)
        y_cpu = torch.mul(y_cpu, y_cpu)

        y_xpu = x_xpu.unfold(0, 5, 1)
        y_xpu = torch.mul(y_xpu, y_xpu)
        self.assertEqual(y_xpu, y_cpu)

    def test_col2im(self):
        def helper(x):
            return torch.nn.functional.fold(x, output_size=(224, 224), kernel_size=(7, 7), dilation=(6,6), padding=1, stride=1)
        x_cpu = torch.rand(1, 147, 36100)
        x = x_cpu.detach().clone().to('xpu')
        self.assertEqual(helper(x_cpu), helper(x))
