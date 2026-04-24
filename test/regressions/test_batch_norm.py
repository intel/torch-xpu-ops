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
        self.assertEqual(ref_invstd, save_invstd_xpu.cpu(), atol=save_atol, rtol=save_rtol)

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


if __name__ == "__main__":
    run_tests()
