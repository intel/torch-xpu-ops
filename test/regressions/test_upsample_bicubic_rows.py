# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestBicubicRows(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_batch_partitions(self, device, dtype):
        source = torch.randn((10, 3, 240, 240), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        reference = source if dtype == torch.float64 else source.float()
        expected = torch.ops.aten.upsample_bicubic2d.default(
            reference, [224, 224], False
        ).to(dtype)
        actual = torch.ops.aten.upsample_bicubic2d.default(
            source.to(device), [224, 224], False
        )
        tolerances = {
            torch.float16: 1e-3,
            torch.bfloat16: 1.6e-2,
            torch.float32: 1e-5,
            torch.float64: 1e-12,
        }
        self.assertEqual(
            actual.cpu(),
            expected,
            rtol=tolerances[dtype],
            atol=1e-12 if dtype == torch.float64 else 1e-5,
        )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("channels", [1, 3, 7])
    @parametrize(
        "geometry",
        [
            (240, 240, 224, 224),
            (408, 31, 224, 27),
            (9, 13, 19, 21),
            (1, 5, 9, 7),
            (15, 17, 7, 9),
            (15, 17, 1, 1),
            (8, 9, 8, 13),
        ],
    )
    @parametrize("layout", ["contiguous", "channels_last", "sliced_channels_last"])
    @parametrize("align_corners", [False, True])
    def test_rows(self, device, dtype, channels, geometry, layout, align_corners):
        height, width, out_height, out_width = geometry
        channel_step = 2 if layout == "sliced_channels_last" else 1
        source = torch.randn(2, channels * channel_step, height, width, dtype=dtype)
        if layout != "contiguous":
            source = source.contiguous(memory_format=torch.channels_last)
        source_device = source.to(device)[:, ::channel_step]
        source = source[:, ::channel_step]
        reference = source if dtype == torch.float64 else source.float()
        expected = torch.ops.aten.upsample_bicubic2d.default(
            reference, [out_height, out_width], align_corners
        ).to(dtype)
        actual = torch.ops.aten.upsample_bicubic2d.default(
            source_device,
            [out_height, out_width],
            align_corners,
        )
        tolerances = {
            torch.float16: 1e-3,
            torch.bfloat16: 1.6e-2,
            torch.float32: 1e-5,
            torch.float64: 1e-12,
        }
        self.assertEqual(
            actual.cpu(), expected, rtol=tolerances[dtype], atol=tolerances[dtype]
        )
        if layout == "contiguous":
            self.assertTrue(actual.is_contiguous())
        elif layout == "channels_last":
            self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("channels_last", [False, True])
    def test_strided_output(self, device, dtype, channels_last):
        source = torch.randn(3, 6, 19, 23, dtype=dtype)
        if channels_last:
            source = source.contiguous(memory_format=torch.channels_last)
        source_device = source.to(device)[:, ::2]
        source = source[:, ::2]
        reference = source if dtype == torch.float64 else source.float()
        expected = torch.ops.aten.upsample_bicubic2d.default(
            reference, [15, 17], False
        ).to(dtype)
        backing = torch.full((3, 6, 15, 34), 42, device=device, dtype=dtype)
        if channels_last:
            backing = backing.contiguous(memory_format=torch.channels_last)
        output = backing[:, ::2, :, ::2]
        torch.ops.aten.upsample_bicubic2d.out(
            source_device, [15, 17], False, out=output
        )
        tolerances = {
            torch.float16: 1e-3,
            torch.bfloat16: 1.6e-2,
            torch.float32: 1e-5,
            torch.float64: 1e-12,
        }
        self.assertEqual(
            output.cpu(), expected, rtol=tolerances[dtype], atol=tolerances[dtype]
        )
        self.assertEqual(backing[:, 1::2], torch.full_like(backing[:, 1::2], 42))
        self.assertEqual(
            backing[:, ::2, :, 1::2], torch.full_like(backing[:, ::2, :, 1::2], 42)
        )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "layout", ["contiguous", "sliced", "channels_last", "sliced_channels_last"]
    )
    @parametrize("output_size", [(13, 11), (17, 19)])
    def test_layout_and_scales(self, device, dtype, layout, output_size):
        source = torch.randn(2, 6, 17, 19, dtype=dtype)
        if "channels_last" in layout:
            source = source.contiguous(memory_format=torch.channels_last)
        source_device = source.to(device)
        if "sliced" in layout:
            source = source[:, ::2]
            source_device = source_device[:, ::2]
        reference = source if dtype == torch.float64 else source.float()
        scales = (None, None) if output_size == (17, 19) else (0.75, 0.6)
        expected = torch.ops.aten.upsample_bicubic2d.default(
            reference, output_size, False, *scales
        ).to(dtype)
        actual = torch.ops.aten.upsample_bicubic2d.default(
            source_device, output_size, False, *scales
        )
        tolerances = {
            torch.float16: 1e-3,
            torch.bfloat16: 1.6e-2,
            torch.float32: 1e-5,
            torch.float64: 1e-12,
        }
        self.assertEqual(
            actual.cpu(), expected, rtol=tolerances[dtype], atol=tolerances[dtype]
        )


instantiate_device_type_tests(
    TestBicubicRows, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
