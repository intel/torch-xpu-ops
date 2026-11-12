# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for the batch_norm_mean_var contiguity fix.

The fix ensures that when batch_norm_choose_impl selects Impl::Contiguous
(input is contiguous with strides()[1] != 1, i.e. standard NCHW), the kernel
checks whether save_mean / save_var output tensors are contiguous before
using batch_norm_stats_template.  If they are not, it falls through to the
General implementation which handles non-contiguous outputs correctly via
var_mean_out.

These tests exercise the fix by calling the .out variant of the batch norm op
with **non-contiguous** save_mean / save_invstd output tensors.  Without the
fix, batch_norm_stats_template writes to non-contiguous memory as though it
were contiguous, producing incorrect values.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

xpu_device = torch.device("xpu")


def _make_noncontiguous_1d(size, device, dtype=torch.float32):
    """Return a non-contiguous 1-D tensor of length *size* (stride-2 view)."""
    return torch.empty(size * 2, device=device, dtype=dtype)[::2]


class TestBatchNormContiguous(TestCase):
    # ------------------------------------------------------------------
    # Core regression: non-contiguous save_mean / save_invstd on the
    # Impl::Contiguous path (contiguous NCHW input, strides()[1] != 1).
    # ------------------------------------------------------------------
    def _test_noncontiguous_save_tensors(self, input_shape, dtype):
        """
        Call _native_batch_norm_legit.out with a contiguous input that
        selects Impl::Contiguous, but pre-allocate non-contiguous
        save_mean / save_invstd output tensors.

        Without the fix the contiguous kernel writes to non-contiguous
        memory as if it were contiguous, producing wrong values.
        """
        C = input_shape[1]
        eps = 1e-5
        momentum = 0.1

        torch.manual_seed(42)
        input_data = torch.randn(*input_shape)
        weight = torch.ones(C)
        bias = torch.zeros(C)
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        # CPU reference (allocates contiguous save tensors internally)
        ref_out, ref_mean, ref_invstd = torch.ops.aten._native_batch_norm_legit(
            input_data.float(),
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            momentum,
            eps,
        )

        # XPU with non-contiguous save tensors
        input_xpu = input_data.to(dtype).to(xpu_device)
        self.assertTrue(input_xpu.is_contiguous())
        self.assertNotEqual(input_xpu.stride(1), 1)

        acc_dtype = torch.float32
        save_mean_xpu = _make_noncontiguous_1d(C, xpu_device, acc_dtype)
        save_invstd_xpu = _make_noncontiguous_1d(C, xpu_device, acc_dtype)
        out_xpu = torch.empty_like(input_xpu)

        self.assertFalse(save_mean_xpu.is_contiguous())
        self.assertFalse(save_invstd_xpu.is_contiguous())

        torch.ops.aten._native_batch_norm_legit.out(
            input_xpu,
            weight.to(dtype).to(xpu_device),
            bias.to(dtype).to(xpu_device),
            running_mean.clone().to(xpu_device),
            running_var.clone().to(xpu_device),
            True,
            momentum,
            eps,
            out=out_xpu,
            save_mean=save_mean_xpu,
            save_invstd=save_invstd_xpu,
        )

        atol, rtol = (8e-3, 8e-3) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        self.assertEqual(ref_out, out_xpu.cpu().float(), atol=atol, rtol=rtol)
        # bfloat16 input reduces precision of accumulated mean/invstd
        save_atol, save_rtol = (5e-4, 5e-4) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        self.assertEqual(ref_mean, save_mean_xpu.cpu(), atol=save_atol, rtol=save_rtol)
        self.assertEqual(
            ref_invstd, save_invstd_xpu.cpu(), atol=save_atol, rtol=save_rtol
        )

    # -- BatchNorm2d (N,C,H,W) -- stride[1] = H*W > 1
    def test_bn2d_noncontiguous_save_float32(self):
        self._test_noncontiguous_save_tensors((4, 8, 7, 7), torch.float32)

    def test_bn2d_noncontiguous_save_float16(self):
        self._test_noncontiguous_save_tensors((4, 8, 7, 7), torch.float16)

    def test_bn2d_noncontiguous_save_bfloat16(self):
        self._test_noncontiguous_save_tensors((4, 8, 7, 7), torch.bfloat16)

    # -- BatchNorm1d on 3-D input (N,C,L) -- stride[1] = L > 1
    def test_bn1d_noncontiguous_save_float32(self):
        self._test_noncontiguous_save_tensors((4, 8, 16), torch.float32)

    # -- BatchNorm3d (N,C,D,H,W) -- stride[1] = D*H*W > 1
    def test_bn3d_noncontiguous_save_float32(self):
        self._test_noncontiguous_save_tensors((2, 4, 3, 5, 5), torch.float32)

    # ------------------------------------------------------------------
    # Non-contiguous output tensor: verify the output buffer itself can
    # be non-contiguous without corruption.
    # ------------------------------------------------------------------
    def test_bn2d_noncontiguous_output(self):
        N, C, H, W = 4, 8, 7, 7
        eps = 1e-5
        momentum = 0.1

        torch.manual_seed(42)
        input_data = torch.randn(N, C, H, W)
        weight = torch.ones(C)
        bias = torch.zeros(C)
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        ref_out, _, _ = torch.ops.aten._native_batch_norm_legit(
            input_data,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            momentum,
            eps,
        )

        input_xpu = input_data.to(xpu_device)
        # Non-contiguous output: allocate double-width and slice
        out_buf = torch.empty(N, C * 2, H, W, device=xpu_device)
        out_xpu = out_buf[:, ::2, :, :]
        self.assertFalse(out_xpu.is_contiguous())

        save_mean_xpu = _make_noncontiguous_1d(C, xpu_device)
        save_invstd_xpu = _make_noncontiguous_1d(C, xpu_device)

        torch.ops.aten._native_batch_norm_legit.out(
            input_xpu,
            weight.to(xpu_device),
            bias.to(xpu_device),
            running_mean.clone().to(xpu_device),
            running_var.clone().to(xpu_device),
            True,
            momentum,
            eps,
            out=out_xpu,
            save_mean=save_mean_xpu,
            save_invstd=save_invstd_xpu,
        )

        self.assertEqual(ref_out, out_xpu.cpu(), atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Running statistics update with non-contiguous save tensors:
    # verify running_mean / running_var are correctly updated.
    # ------------------------------------------------------------------
    def test_bn2d_noncontiguous_save_running_stats(self):
        N, C, H, W = 4, 8, 7, 7
        eps = 1e-5
        momentum = 0.1

        torch.manual_seed(42)
        input_data = torch.randn(N, C, H, W)
        weight = torch.ones(C)
        bias = torch.zeros(C)

        # CPU reference
        rm_cpu = torch.zeros(C)
        rv_cpu = torch.ones(C)
        torch.ops.aten._native_batch_norm_legit(
            input_data,
            weight,
            bias,
            rm_cpu,
            rv_cpu,
            True,
            momentum,
            eps,
        )

        # XPU with non-contiguous save tensors
        input_xpu = input_data.to(xpu_device)
        rm_xpu = torch.zeros(C, device=xpu_device)
        rv_xpu = torch.ones(C, device=xpu_device)
        save_mean_xpu = _make_noncontiguous_1d(C, xpu_device)
        save_invstd_xpu = _make_noncontiguous_1d(C, xpu_device)
        out_xpu = torch.empty_like(input_xpu)

        torch.ops.aten._native_batch_norm_legit.out(
            input_xpu,
            weight.to(xpu_device),
            bias.to(xpu_device),
            rm_xpu,
            rv_xpu,
            True,
            momentum,
            eps,
            out=out_xpu,
            save_mean=save_mean_xpu,
            save_invstd=save_invstd_xpu,
        )

        self.assertEqual(rm_cpu, rm_xpu.cpu(), atol=1e-4, rtol=1e-4)
        self.assertEqual(rv_cpu, rv_xpu.cpu(), atol=1e-4, rtol=1e-4)


def _channels_last_with_offset(src, offset):
    """Copy *src* (NCHW logical) to an XPU channels-last tensor whose
    storage_offset is *offset* elements, so offset=1 breaks 2-element
    alignment for 16-bit dtypes and forces the scalar fallback."""
    numel = src.numel()
    buf = torch.empty(numel + offset, dtype=src.dtype, device=xpu_device)
    N, C, H, W = src.shape
    t = buf.as_strided(src.shape, (H * W * C, 1, W * C, C), offset)
    t.copy_(src.to(xpu_device))
    assert t.is_contiguous(memory_format=torch.channels_last)
    assert t.storage_offset() == offset
    return t


class TestBatchNormBackwardElemtChannelsLast(TestCase):
    """Numerical coverage for the 1-D channels-last backward elementwise kernel
    (batch_norm_backward_elemt_channels_last_template, with and without count).

    Shapes cover: aligned C, even non-aligned C, odd C with odd total element
    count (vec2 channel wrap-around + scalar tail), and C >= 1024 (non-SLM
    paths). storage_offset=1 on input / grad_output independently exercises
    the unaligned scalar fallback.
    """

    SHAPES = [
        (2, 32, 5, 5),  # aligned C
        (2, 24, 7, 7),  # even, non cache-line aligned C
        (3, 27, 3, 3),  # odd C, odd total -> wrap-around + tail
        (1, 61, 3, 3),  # odd C, odd total
        (2, 1026, 3, 3),  # even C >= 1024 -> non-SLM vec2
        (1, 1025, 3, 3),  # odd C >= 1024, odd total -> scalar
    ]
    DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    def _tol(self, dtype):
        if dtype == torch.float32:
            return 1e-4, 1e-4
        if dtype == torch.float16:
            return 2e-3, 2e-3
        return 2e-2, 2e-2

    def _inputs(self, shape, dtype):
        torch.manual_seed(0)
        C = shape[1]
        x = torch.randn(shape).to(dtype)
        go = torch.randn(shape).to(dtype)
        # Distinct per-channel parameters so a wrong channel index is visible.
        w = (torch.arange(C, dtype=torch.float32) / C + 0.5).to(dtype)
        b = torch.randn(C).to(dtype)
        return x, go, w, b

    def _check_native_backward(self, shape, dtype, in_off, go_off):
        x, go, w, b = self._inputs(shape, dtype)

        # Reference: CPU fp64 NCHW.
        xr = x.double().requires_grad_()
        out = torch.nn.functional.batch_norm(
            xr, None, None, w.double(), b.double(), training=True, eps=1e-5
        )
        out.backward(go.double())
        ref = xr.grad

        xx = _channels_last_with_offset(x, in_off).requires_grad_()
        gx = _channels_last_with_offset(go, go_off)
        out = torch.nn.functional.batch_norm(
            xx,
            None,
            None,
            w.to(xpu_device),
            b.to(xpu_device),
            training=True,
            eps=1e-5,
        )
        out.backward(gx)
        atol, rtol = self._tol(dtype)
        self.assertEqual(xx.grad.cpu().double(), ref, atol=atol, rtol=rtol)

    def _check_backward_elemt_count(self, shape, dtype, in_off, go_off):
        x, go, w, _ = self._inputs(shape, dtype)
        N, C = shape[0], shape[1]
        dims = [0, 2, 3]

        xd, god, wd = x.double(), go.double(), w.double()
        mean = xd.mean(dims)
        invstd = 1.0 / torch.sqrt(xd.var(dims, unbiased=False) + 1e-5)
        sum_dy = god.sum(dims)
        sum_dy_xmu = (god * (xd - mean.view(1, C, 1, 1))).sum(dims)
        # Simulate 2 ranks each contributing the local count.
        count = torch.tensor([x.numel() // C] * 2, dtype=torch.int32)
        total = float(count.sum())
        c4 = lambda t: t.view(1, C, 1, 1)
        ref = (
            god
            - c4(sum_dy) / total
            - (xd - c4(mean)) * c4(invstd * invstd * sum_dy_xmu / total)
        ) * c4(wd * invstd)

        xx = _channels_last_with_offset(x, in_off)
        gx = _channels_last_with_offset(go, go_off)
        f32 = lambda t: t.float().to(xpu_device)
        gi = torch.batch_norm_backward_elemt(
            gx,
            xx,
            f32(mean),
            f32(invstd),
            w.to(xpu_device),
            f32(sum_dy),
            f32(sum_dy_xmu),
            count.to(xpu_device),
        )
        atol, rtol = self._tol(dtype)
        self.assertEqual(gi.cpu().double(), ref, atol=atol, rtol=rtol)

    def _run(self, checker):
        for shape in self.SHAPES:
            for dtype in self.DTYPES:
                for in_off, go_off in [(0, 0), (1, 0), (0, 1)]:
                    with self.subTest(
                        shape=shape, dtype=dtype, in_off=in_off, go_off=go_off
                    ):
                        checker(shape, dtype, in_off, go_off)

    def test_native_backward_channels_last(self):
        self._run(self._check_native_backward)

    def test_backward_elemt_count_channels_last(self):
        self._run(self._check_backward_elemt_count)


if __name__ == "__main__":
    run_tests()
