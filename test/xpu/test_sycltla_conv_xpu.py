# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Unit tests for sycltla conv2d/conv3d: fprop, dgrad, wgrad with BF16/FP16.

Correctness is verified by comparing sycltla results against:
  - PyTorch XPU conv (oneDNN backend) for fprop (same dtype, same device)
  - FP32 CPU reference for dgrad/wgrad (gold standard)

Both comparisons use the SAME input tensors to ensure determinism.
"""

import unittest
import torch
import torch.nn.functional as F


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestSycltlaConv(unittest.TestCase):
    """Test sycltla_conv ops against oneDNN XPU reference and FP32 CPU gold."""

    # Tolerance: sycltla vs oneDNN (both low-precision on XPU)
    # should be very tight since both operate on same dtype
    ATOL_ONEDNN = {torch.bfloat16: 1.0, torch.float16: 0.125}
    RTOL_ONEDNN = {torch.bfloat16: 0.02, torch.float16: 0.01}

    # Tolerance: sycltla vs FP32 CPU gold (cross-precision comparison)
    ATOL_GOLD = {torch.bfloat16: 4.0, torch.float16: 0.6}
    RTOL_GOLD = {torch.bfloat16: 0.05, torch.float16: 0.02}

    def _check(self, actual, expected, dtype, tol_type="onednn", msg=""):
        if tol_type == "onednn":
            atol, rtol = self.ATOL_ONEDNN[dtype], self.RTOL_ONEDNN[dtype]
        else:
            atol, rtol = self.ATOL_GOLD[dtype], self.RTOL_GOLD[dtype]
        maxdiff = (actual.float() - expected.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol),
            f"{msg}: maxdiff={maxdiff:.4f}, atol={atol}, rtol={rtol} (vs {tol_type})"
        )

    # ========================================================================
    # Conv2d fprop: compare vs oneDNN XPU (same dtype)
    # ========================================================================
    def _test_conv2d_fprop(self, dtype):
        torch.manual_seed(42)
        N, C, H, W, K, kH, kW = 2, 64, 32, 32, 128, 3, 3
        s, p, d = [1, 1], [1, 1], [1, 1]
        x = torch.randn(N, C, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kH, kW, device="xpu", dtype=dtype)
        b = torch.randn(K, device="xpu", dtype=dtype)

        # oneDNN reference (same dtype, same device)
        ref = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
        out = torch.ops.sycltla_conv.conv2d_fprop(x, w, b, s, p, d, 1)
        self._check(out, ref, dtype, "onednn", "conv2d_fprop")

        # Also without bias
        ref_nb = F.conv2d(x, w, None, stride=s, padding=p, dilation=d)
        out_nb = torch.ops.sycltla_conv.conv2d_fprop(x, w, None, s, p, d, 1)
        self._check(out_nb, ref_nb, dtype, "onednn", "conv2d_fprop_no_bias")

    def test_conv2d_fprop_bf16(self):
        self._test_conv2d_fprop(torch.bfloat16)

    def test_conv2d_fprop_fp16(self):
        self._test_conv2d_fprop(torch.float16)

    # ========================================================================
    # Conv2d dgrad: compare vs FP32 CPU gold
    # ========================================================================
    def _test_conv2d_dgrad(self, dtype):
        torch.manual_seed(42)
        N, C, H, W, K, kH, kW = 2, 64, 32, 32, 128, 3, 3
        s, p, d = [1, 1], [1, 1], [1, 1]
        x = torch.randn(N, C, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kH, kW, device="xpu", dtype=dtype)

        # FP32 CPU gold reference
        x_cpu = x.float().cpu().requires_grad_(True)
        w_cpu = w.float().cpu()
        out_cpu = F.conv2d(x_cpu, w_cpu, stride=s, padding=p, dilation=d)
        grad_out_cpu = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out_cpu)
        gold = x_cpu.grad  # FP32 CPU

        # sycltla
        grad_out_xpu = grad_out_cpu.to(dtype).to("xpu")
        result = torch.ops.sycltla_conv.conv2d_dgrad(
            grad_out_xpu, w, list(x.shape), s, p, d, 1)

        # Compare sycltla (converted to float) vs FP32 gold
        self._check(result.cpu(), gold.to(dtype), dtype, "gold", "conv2d_dgrad")

    def test_conv2d_dgrad_bf16(self):
        self._test_conv2d_dgrad(torch.bfloat16)

    def test_conv2d_dgrad_fp16(self):
        self._test_conv2d_dgrad(torch.float16)

    # ========================================================================
    # Conv2d wgrad: compare vs FP32 CPU gold
    # ========================================================================
    def _test_conv2d_wgrad(self, dtype):
        torch.manual_seed(42)
        N, C, H, W, K, kH, kW = 2, 64, 32, 32, 128, 3, 3
        s, p, d = [1, 1], [1, 1], [1, 1]
        x = torch.randn(N, C, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kH, kW, device="xpu", dtype=dtype)

        x_cpu = x.float().cpu()
        w_cpu = w.float().cpu().requires_grad_(True)
        out_cpu = F.conv2d(x_cpu, w_cpu, stride=s, padding=p, dilation=d)
        grad_out_cpu = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out_cpu)
        gold = w_cpu.grad

        grad_out_xpu = grad_out_cpu.to(dtype).to("xpu")
        result = torch.ops.sycltla_conv.conv2d_wgrad(
            grad_out_xpu, x, [kH, kW], s, p, d, 1)

        self._check(result.cpu(), gold.to(dtype), dtype, "gold", "conv2d_wgrad")

    def test_conv2d_wgrad_bf16(self):
        self._test_conv2d_wgrad(torch.bfloat16)

    def test_conv2d_wgrad_fp16(self):
        self._test_conv2d_wgrad(torch.float16)

    # ========================================================================
    # Conv3d fprop: compare vs oneDNN XPU
    # ========================================================================
    def _test_conv3d_fprop(self, dtype):
        torch.manual_seed(42)
        N, C, D, H, W, K = 2, 32, 8, 16, 16, 64
        kD, kH, kW = 3, 3, 3
        s, p, d = [1, 1, 1], [1, 1, 1], [1, 1, 1]
        x = torch.randn(N, C, D, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kD, kH, kW, device="xpu", dtype=dtype)
        b = torch.randn(K, device="xpu", dtype=dtype)

        ref = F.conv3d(x, w, b, stride=s, padding=p, dilation=d)
        out = torch.ops.sycltla_conv.conv3d_fprop(x, w, b, s, p, d, 1)
        self._check(out, ref, dtype, "onednn", "conv3d_fprop")

    def test_conv3d_fprop_bf16(self):
        self._test_conv3d_fprop(torch.bfloat16)

    def test_conv3d_fprop_fp16(self):
        self._test_conv3d_fprop(torch.float16)

    # ========================================================================
    # Conv3d dgrad: compare vs FP32 CPU gold
    # ========================================================================
    def _test_conv3d_dgrad(self, dtype):
        torch.manual_seed(42)
        N, C, D, H, W, K = 2, 32, 8, 16, 16, 64
        kD, kH, kW = 3, 3, 3
        s, p, d = [1, 1, 1], [1, 1, 1], [1, 1, 1]
        x = torch.randn(N, C, D, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kD, kH, kW, device="xpu", dtype=dtype)

        x_cpu = x.float().cpu().requires_grad_(True)
        w_cpu = w.float().cpu()
        out_cpu = F.conv3d(x_cpu, w_cpu, stride=s, padding=p, dilation=d)
        grad_out_cpu = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out_cpu)
        gold = x_cpu.grad

        grad_out_xpu = grad_out_cpu.to(dtype).to("xpu")
        result = torch.ops.sycltla_conv.conv3d_dgrad(
            grad_out_xpu, w, list(x.shape), s, p, d, 1)

        self._check(result.cpu(), gold.to(dtype), dtype, "gold", "conv3d_dgrad")

    def test_conv3d_dgrad_bf16(self):
        self._test_conv3d_dgrad(torch.bfloat16)

    def test_conv3d_dgrad_fp16(self):
        self._test_conv3d_dgrad(torch.float16)

    # ========================================================================
    # Conv3d wgrad: compare vs FP32 CPU gold
    # ========================================================================
    def _test_conv3d_wgrad(self, dtype):
        torch.manual_seed(42)
        N, C, D, H, W, K = 2, 32, 8, 16, 16, 64
        kD, kH, kW = 3, 3, 3
        s, p, d = [1, 1, 1], [1, 1, 1], [1, 1, 1]
        x = torch.randn(N, C, D, H, W, device="xpu", dtype=dtype)
        w = torch.randn(K, C, kD, kH, kW, device="xpu", dtype=dtype)

        x_cpu = x.float().cpu()
        w_cpu = w.float().cpu().requires_grad_(True)
        out_cpu = F.conv3d(x_cpu, w_cpu, stride=s, padding=p, dilation=d)
        grad_out_cpu = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out_cpu)
        gold = w_cpu.grad

        grad_out_xpu = grad_out_cpu.to(dtype).to("xpu")
        result = torch.ops.sycltla_conv.conv3d_wgrad(
            grad_out_xpu, x, [kD, kH, kW], s, p, d, 1)

        self._check(result.cpu(), gold.to(dtype), dtype, "gold", "conv3d_wgrad")

    def test_conv3d_wgrad_bf16(self):
        self._test_conv3d_wgrad(torch.bfloat16)

    def test_conv3d_wgrad_fp16(self):
        self._test_conv3d_wgrad(torch.float16)


if __name__ == "__main__":
    unittest.main()
