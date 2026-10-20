# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.autograd import Variable
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import run_tests, TestCase

device = torch.device("xpu")
cpu_device = torch.device("cpu")

# Per batch 1024 * 1024 * 800 = 838,860,800 elements (< INT32_MAX),
# total 3 * 838,860,800 = 2,516,582,400 elements (> INT32_MAX).
_SMALL_SIZE = (3, 1, 512, 512, 400)
_LARGE_SIZE = (3, 1, 1024, 1024, 800)


class TestTorchMethod(TestCase):
    def test_upsample_nearest(self):
        test_dtypes = [torch.float, torch.bfloat16, torch.float16]

        def test_upsample_nearest_exact(dtype):
            # #### upsample nearest exact 1D #####
            input_cpu = torch.randn((2, 3, 5), dtype=dtype, device=cpu_device)
            input_xpu = input_cpu.to("xpu")
            scales = [6]
            input_cpu.requires_grad = True
            input_xpu.requires_grad = True
            rsf = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            output_xpu = torch.nn.functional.interpolate(
                input_xpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_xpu.cpu())

            grad_out_cpu = torch.ones_like(output_cpu)
            grad_out_xpu = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_xpu.backward(grad_out_xpu)
            grad_cpu = input_cpu.grad
            grad_xpu = input_xpu.grad
            self.assertEqual(grad_cpu, grad_xpu.cpu())

            # #### upsample nearest exact 2D #####
            input_cpu = torch.randn(
                (2, 3, 5, 5), dtype=torch.float32, device=cpu_device
            )
            input_dpcpp = input_cpu.to("xpu")
            scales = [6, 8]
            input_cpu.requires_grad = True
            input_dpcpp.requires_grad = True
            alc = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            output_dpcpp = torch.nn.functional.interpolate(
                input_dpcpp,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_dpcpp.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30, 40), dtype=torch.float32, device=cpu_device
            )
            grad_out_dpcpp = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_dpcpp.backward(grad_out_dpcpp)
            grad_cpu = input_cpu.grad
            grad_dpcpp = input_dpcpp.grad
            self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        def test_upsamle_nearest(dtype):
            # #### upsample nearest 1D #####
            input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
            input_xpu = input_cpu.to("xpu")
            scales = [6]
            input_cpu.requires_grad = True
            input_xpu.requires_grad = True
            rsf = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            output_xpu = torch.nn.functional.interpolate(
                input_xpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_xpu.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30), dtype=torch.float32, device=cpu_device
            )
            grad_out_xpu = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_xpu.backward(grad_out_xpu)
            grad_cpu = input_cpu.grad
            grad_xpu = input_xpu.grad
            self.assertEqual(grad_cpu, grad_xpu.cpu())

            # #### upsample nearest 2D #####
            input_cpu = torch.randn(
                (2, 3, 5, 5), dtype=torch.float32, device=cpu_device
            )
            input_dpcpp = input_cpu.to("xpu")
            scales = [6, 8]
            input_cpu.requires_grad = True
            input_dpcpp.requires_grad = True
            alc = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            output_dpcpp = torch.nn.functional.interpolate(
                input_dpcpp,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_dpcpp.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30, 40), dtype=torch.float32, device=cpu_device
            )
            grad_out_dpcpp = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_dpcpp.backward(grad_out_dpcpp)
            grad_cpu = input_cpu.grad
            grad_dpcpp = input_dpcpp.grad
            self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        for dtype in test_dtypes:
            test_upsamle_nearest(dtype)
            test_upsample_nearest_exact(dtype)

    def test_upsample_nearest2d_channels_last_many_channels(self):
        # 8x8 -> 16x16 gives 256 outputs per channel, so C > 256 exceeds the
        # 65536 work-items the channels_last kernel launches (intel/torch-xpu-ops#5701).
        for mode in ("nearest", "nearest-exact"):
            for dtype in (torch.float, torch.bfloat16, torch.float16):
                for channels in (255, 256, 257, 512):
                    with self.subTest(mode=mode, dtype=dtype, channels=channels):
                        x = torch.randn(1, channels, 8, 8).to(dtype)
                        x = x.contiguous(memory_format=torch.channels_last)
                        out = torch.nn.functional.interpolate(
                            x.to(device), scale_factor=2.0, mode=mode
                        )
                        ref = torch.nn.functional.interpolate(
                            x.float(), scale_factor=2.0, mode=mode
                        ).to(dtype)
                        self.assertEqual(out.cpu(), ref)

    """
    Regression tests for large-tensor indexing in upsample_nearest3d forward/backward.

    *_large_size_int64: the kernels walk the batch dim with dst_idx += C * D * H * W,
    so the flat index exceeds INT32_MAX once N * C * D * H * W > INT32_MAX even if a
    single batch fits in int32. Each batch is compared against a single-batch launch,
    which stays on the 32-bit index path.

    *_single_batch_over_uint32: a single batch has more than UINT32_MAX elements, so
    the launch size (ceil_div of the per-batch numel) must be computed in 64 bits or
    part of the output is never written.
    """

    @largeTensorTest("8GB", device="xpu")
    def test_upsample_nearest3d_fwd_large_size_int64(self):
        x = torch.randint(0, 256, _SMALL_SIZE, dtype=torch.uint8, device="xpu")
        y = torch.nn.functional.interpolate(x, size=_LARGE_SIZE[2:], mode="nearest")
        self.assertEqual(y.shape, _LARGE_SIZE)

        for b in range(_SMALL_SIZE[0]):
            ref = torch.nn.functional.interpolate(
                x[b : b + 1], size=_LARGE_SIZE[2:], mode="nearest"
            )
            self.assertEqual(y[b : b + 1], ref)
            del ref

    @largeTensorTest("8GB", device="xpu")
    def test_upsample_nearest3d_bwd_large_size_int64(self):
        # Downsampling backward makes grad_input the large tensor that
        # selects the int64 path via canUse32BitIndexMath(grad_input).
        grad_output = torch.randint(
            0, 256, _SMALL_SIZE, dtype=torch.uint8, device="xpu"
        )
        grad_input = torch.ops.aten.upsample_nearest3d_backward(
            grad_output, _SMALL_SIZE[2:], _LARGE_SIZE
        )
        self.assertEqual(grad_input.shape, _LARGE_SIZE)

        for b in range(_SMALL_SIZE[0]):
            ref = torch.ops.aten.upsample_nearest3d_backward(
                grad_output[b : b + 1], _SMALL_SIZE[2:], (1, *_LARGE_SIZE[1:])
            )
            self.assertEqual(grad_input[b : b + 1], ref)
            del ref

    @largeTensorTest("8GB", device="xpu")
    def test_upsample_nearest3d_fwd_single_batch_over_uint32(self):
        # Per-batch output numel 1024 * 1024 * 4100 > UINT32_MAX, so the launch
        # size must not be computed in 32 bits.
        out_size = (1024, 1024, 4100)
        x = torch.randint(
            0, 256, (1, 1, 1024, 1024, 2050), dtype=torch.uint8, device="xpu"
        )
        y = torch.zeros((1, 1, *out_size), dtype=torch.uint8, device="xpu")
        torch.ops.aten.upsample_nearest3d.out(x, out_size, None, None, None, out=y)

        chunk = 16
        for d in range(0, out_size[0], chunk):
            ref = x[:, :, d : d + chunk].repeat_interleave(2, dim=-1)
            self.assertEqual(y[:, :, d : d + chunk], ref)
            del ref

    @largeTensorTest("8GB", device="xpu")
    def test_upsample_nearest3d_bwd_single_batch_over_uint32(self):
        # Per-batch grad_input numel 1024 * 1024 * 4100 > UINT32_MAX. Backward of
        # a 2x downsample along W: even columns get grad_output, odd columns get 0.
        in_size = (1, 1, 1024, 1024, 4100)
        out_size = (1024, 1024, 2050)
        grad_output = torch.randint(
            0, 256, (1, 1, *out_size), dtype=torch.uint8, device="xpu"
        )
        grad_input = torch.full(in_size, 255, dtype=torch.uint8, device="xpu")
        torch.ops.aten.upsample_nearest3d_backward.grad_input(
            grad_output, out_size, in_size, None, None, None, grad_input=grad_input
        )

        chunk = 16
        for d in range(0, in_size[2], chunk):
            ref = torch.zeros_like(grad_input[:, :, d : d + chunk])
            ref[..., ::2] = grad_output[:, :, d : d + chunk]
            self.assertEqual(grad_input[:, :, d : d + chunk], ref)
            del ref


if __name__ == "__main__":
    run_tests()
