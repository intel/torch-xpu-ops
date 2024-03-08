import torch
from torch.testing._internal.common_utils import TestCase


test_shapes = [
    # instruction bound nobc
    [[23, 72, 72], [5184, 72, 1], [23, 72, 72], [5184, 72, 1]],
    # instruction bound bc
    [[23, 16, 16], [23, 1, 16]],
    [[23, 16, 17], [23, 1, 17]],
    [[1, 72, 72], [23, 72, 72]],
    [[23, 72, 1], [23, 72, 72]],
    # memory bound nobc
    [[23000, 72, 72], [5184, 72, 1], [23000, 72, 72], [5184, 72, 1]],
    [[16, 16, 256, 256], [16, 16, 256, 256]],
    # memory bound bc
    [[16, 16, 512, 512], [16, 1, 1, 512]],
    [[4, 15000, 3], [105000, 1, 15000], [4, 1, 3], [3, 3, 1]],
    # memory bound bc no vectorized (launch_legancy_kernel)
    [[16, 16, 512, 513], [16, 1, 1, 513]],
    [[28, 4096, 9], [36864, 9, 1], [28, 4096, 1], [4096, 1, 1]],
]


class TestTensorMethod(TestCase):
    def test_loops_backbone(self, dtype=torch.float):
        for shape in test_shapes:
            if len(shape) == 2:
                a = torch.randn(shape[0], dtype=dtype)
                b = torch.randn(shape[1], dtype=dtype)
            elif len(shape) == 4:
                a = torch.as_strided(
                    torch.randn(shape[0][0] * shape[1][0]), shape[0], shape[1]
                )
                b = torch.as_strided(
                    torch.randn(shape[2][0] * shape[3][0]), shape[2], shape[3]
                )
            print(shape, flush=True)
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            c = a + b
            c_xpu = a_xpu + b_xpu
            self.assertEqual(c, c_xpu.cpu())
