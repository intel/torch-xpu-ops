# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

test_shapes = [
    [[23, 72, 72], [5184, 72, 1], [23, 72, 72], [5184, 72, 1]],
    [[23, 16, 16], [23, 1, 16]],
    [[23, 16, 17], [23, 1, 17]],
    [[1, 72, 72], [23, 72, 72]],
    [[23, 72, 1], [23, 72, 72]],
    [[23000, 72, 72], [5184, 72, 1], [23000, 72, 72], [5184, 72, 1]],
    [[16, 16, 256, 256], [16, 16, 256, 256]],
    [[16, 16, 512, 512], [16, 1, 1, 512]],
    [[4, 15000, 3], [105000, 1, 15000], [4, 1, 3], [3, 3, 1]],
    [[16, 16, 512, 513], [16, 1, 1, 513]],
    [[28, 4096, 9], [36864, 9, 1], [28, 4096, 1], [4096, 1, 1]],
]


class TestLoopsKernel(TestCase):
    def _test_loops(self, dtype=torch.float):
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
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            c = a + b + 1
            c_xpu = a_xpu + b_xpu + 1
            self.assertEqual(c, c_xpu.cpu())

    def test_loops_float(self):
        self._test_loops(torch.float)

    def test_loops_half(self):
        self._test_loops(torch.half)

    def test_loops_bfloat16(self):
        self._test_loops(torch.bfloat16)

    def test_loops_dynamic_cast(self):
        for shape in test_shapes:
            if len(shape) == 2:
                a = torch.randn(shape[0], dtype=torch.float)
                b = torch.randn(shape[1], dtype=torch.half)
            elif len(shape) == 4:
                a = torch.as_strided(
                    torch.randn(shape[0][0] * shape[1][0], dtype=torch.float),
                    shape[0],
                    shape[1],
                )
                b = torch.as_strided(
                    torch.randn(shape[2][0] * shape[3][0], dtype=torch.half),
                    shape[2],
                    shape[3],
                )
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            print(
                f"a_xpu:{a_xpu.dtype}, {a_xpu.shape}, {a.stride()}; b_xpu:{b_xpu.dtype}, {b_xpu.shape}, {b_xpu.stride()}",
                flush=True,
            )
            c = a + b + 1
            c_xpu = a_xpu + b_xpu + 1
            self.assertEqual(c, c_xpu.cpu())
