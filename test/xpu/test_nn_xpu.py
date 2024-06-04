# Owner(s): ["module: intel"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests

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

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_nn import TestNNDeviceType, TestNN

    TestNNDeviceType.test_grid_sample_bfloat16_precision = grid_sample_bfloat16_precision
    TestNNDeviceType.test_grid_sample_half_precision = grid_sample_half_precision


instantiate_device_type_tests(TestNNDeviceType, globals(), only_for="xpu")
instantiate_parametrized_tests(TestNN)


if __name__ == "__main__":
    run_tests()
