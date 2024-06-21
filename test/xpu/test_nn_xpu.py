# Owner(s): ["module: intel"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests
from torch.testing._internal.common_cuda import tf32_on_and_off

def grid_sample_bfloat16_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ('bilinear', 'nearest', 'bicubic'):
            if len(shape_in) != 4 and mode == 'bicubic':
                continue
            data = torch.randn(shape_in, device='xpu', dtype=torch.bfloat16)
            grid = torch.rand(shape_out, device='xpu', dtype=torch.bfloat16) * 2.0 - 1.0

            out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
            out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                        align_corners=align_corners)

            self.assertEqual(out_half, out_double.bfloat16(), msg=f"grid_sample with mode = {mode} doesn't match")

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu

def grid_sample_half_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ('bilinear', 'nearest', 'bicubic'):
            if len(shape_in) != 4 and mode == 'bicubic':
                continue
            data = torch.randn(shape_in, device='xpu', dtype=torch.half)
            grid = torch.rand(shape_out, device='xpu', dtype=torch.half) * 2.0 - 1.0

            out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
            out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                        align_corners=align_corners)

            self.assertEqual(out_half, out_double.half(), msg=f"grid_sample with mode = {mode} doesn't match")

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu

@tf32_on_and_off(0.005)
def grid_sample_large(self, device=torch.device('xpu')):
    def issue_35202():
        input_tensor = torch.rand(1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True)
        coords = torch.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=torch.float, device=device)
        coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        result = torch.nn.functional.grid_sample(input_tensor, coords)
        self.assertEqual(result, torch.tensor([[[[0., 0.]]]], dtype=torch.float, device=device))
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()
    issue_35202()

    def issue_24823_1(dtype):
        image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
        image.requires_grad_()
        grid = torch.nn.functional.affine_grid(
            torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
            (1, 1, 3, 3, 3))
        grid[:, 1, 1, 1, 0] = float('inf')
        result = torch.nn.functional.grid_sample(image, grid, padding_mode='zeros')
        tol_override = {'atol': 0.005, 'rtol': 0} if dtype == torch.half else {}
        self.assertEqual(result, torch.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                    [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                    [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                device=device, dtype=dtype), **tol_override)
        result.backward(torch.ones_like(result))
        expected_grad = torch.ones_like(image)
        expected_grad[0, 0, 1, 1, 1] = 0
        self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
    # grid_sampler_3d is not supported in xpu
    # issue_24823_1(torch.half)
    # issue_24823_1(torch.float)
    # issue_24823_1(torch.double)

    def issue_24823_2():
        param = torch.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=torch.float, device=device)
        img = torch.zeros((1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True)
        grid = torch.nn.functional.affine_grid(param, img.size())
        result = torch.nn.functional.grid_sample(img, grid)
        self.assertEqual(result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float))
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()
    issue_24823_2()

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_nn import TestNNDeviceType, TestNN

    TestNNDeviceType.test_grid_sample_bfloat16_precision = grid_sample_bfloat16_precision
    TestNNDeviceType.test_grid_sample_half_precision = grid_sample_half_precision
    TestNNDeviceType.test_grid_sample_large = grid_sample_large


instantiate_device_type_tests(TestNNDeviceType, globals(), only_for="xpu", allow_xpu=True)
instantiate_parametrized_tests(TestNN)


if __name__ == "__main__":
    run_tests()
