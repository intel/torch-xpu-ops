# Owner(s): ["module: intel"]

from unittest import SkipTest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests, \
    parametrize as parametrize_test
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

@parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
@parametrize_test("mode", ["bilinear", "bicubic"])
@parametrize_test("antialias", [True, False])
@parametrize_test("align_corners", [True, False])
@parametrize_test("num_channels", [3, 5])
@parametrize_test("output_size", [32, 600])
@parametrize_test("check_as_unsqueezed_3d_tensor", [True, False])
@parametrize_test("non_contig", [False, "sliced", "restrided"])
@parametrize_test("batch_size", [1, 5])
def upsamplingBiMode2d_consistency(
    self,
    device,
    memory_format,
    mode,
    antialias,
    align_corners,
    num_channels,
    output_size,
    check_as_unsqueezed_3d_tensor,
    non_contig,
    batch_size,
):
    # Check output value consistency between resized_input_uint8 and resized input_float
    if torch.device(device).type == "xpu":
        raise SkipTest("XPU implementation is not yet supporting uint8")

    torch.manual_seed(0)

    # - input range is set to [30, 220] for bicubic mode, because the bicubic kernel may create
    #   [intermediate] values outside of the [0, 255] range, which need
    #   to be clipped in uint8 path, but not in float path. This isn't
    #   an issue with bilinear kernel.
    input_range = (30, 220) if mode == "bicubic" else (0, 256)
    input_ui8 = torch.randint(*input_range, size=(batch_size, num_channels, 400, 400), dtype=torch.uint8, device=device)
    input_ui8 = input_ui8.contiguous(memory_format=memory_format)

    if non_contig == "sliced":
        input_ui8 = input_ui8[:, :, 10:-10, 10:-10]
    elif non_contig == "restrided":
        input_ui8 = input_ui8[:, :, ::2, ::2]

    if batch_size == 1 and check_as_unsqueezed_3d_tensor:
        input_ui8 = input_ui8[0, ...]
        input_ui8 = input_ui8[None, ...]

    input_f32 = input_ui8.float()

    output_f32 = F.interpolate(
        input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
    ).round().clip(0, 255)
    output_ui8 = F.interpolate(
        input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
    )

    if non_contig is False:
        self.assertTrue(input_ui8.is_contiguous(memory_format=memory_format))

    # FIXME if-clause shows the current behaviour which is definitely unexpected.
    # Ideally we want to fix it such that both the ui8 and f32 outputs are also channels_last
    # See for more details: https://github.com/pytorch/pytorch/pull/100373
    if batch_size == 1 and check_as_unsqueezed_3d_tensor and memory_format == torch.channels_last:
        self.assertTrue(output_ui8.is_contiguous())
        self.assertTrue(output_f32.is_contiguous())
    else:
        self.assertTrue(output_ui8.is_contiguous(memory_format=memory_format))
        self.assertTrue(output_f32.is_contiguous(memory_format=memory_format))

    if mode == "bilinear":
        torch.testing.assert_close(output_f32, output_ui8.float(), rtol=0, atol=1)
    else:
        diff = (output_f32 - output_ui8.float()).abs()
        self.assertLess(diff.max(), 15)

        threshold = 2
        percent = 3
        self.assertLess((diff > threshold).float().mean(), percent / 100)

        threshold = 5
        percent = 1
        self.assertLess((diff > threshold).float().mean(), percent / 100)

        self.assertLess(diff.mean(), 0.4)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_nn import TestNNDeviceType, TestNN, TestAddRelu

    TestNNDeviceType.test_grid_sample_bfloat16_precision = grid_sample_bfloat16_precision
    TestNNDeviceType.test_grid_sample_half_precision = grid_sample_half_precision
    TestNNDeviceType.test_grid_sample_large = grid_sample_large
    TestNNDeviceType.test_upsamplingBiMode2d_consistency = upsamplingBiMode2d_consistency


instantiate_device_type_tests(TestNNDeviceType, globals(), only_for="xpu", allow_xpu=True)
instantiate_parametrized_tests(TestNN)
instantiate_parametrized_tests(TestAddRelu)


if __name__ == "__main__":
    run_tests()
