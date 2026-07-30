# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# Regression test for the XPU exact-gelu tail accuracy bug.
#
# Exact gelu previously computed 0.5 * x * (1 + erf(x / sqrt(2))).  For large
# negative x, erf(x / sqrt(2)) -> -1, so "1 + erf" cancels catastrophically in
# fp32 and gelu flushes to exactly 0.0 from x ~ -6 on.  The fix rewrites the
# normal CDF subterm as the cancellation-free 0.5 * erfc(-x / sqrt(2)) in both
# the forward and backward XPU kernels, aligning XPU with the upstream
# CPU/CUDA/MPS fix in pytorch/pytorch#189234.  See intel/torch-xpu-ops#4753.
#
# Baseline (pre-fix) XPU gelu forward had ~100% relative error in the tail;
# with the fix it matches the float64 reference to fp32 rounding.
import math

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

xpu_device = torch.device("xpu")


class TestGeluTailAccuracy(TestCase):
    def test_gelu_forward_tail_accuracy_xpu(self):
        dtype = torch.float32
        x = torch.arange(-12.0, 12.0, 2**-6, dtype=dtype, device=xpu_device)
        xref = x.cpu().double()

        kAlpha = math.sqrt(0.5)
        # Cancellation-free float64 reference for exact gelu.
        expected = 0.5 * xref * torch.erfc(-xref * kAlpha)

        actual = F.gelu(x).cpu().double()

        tail = xref.abs() >= 4.0
        rel_err = (
            (actual - expected)[tail] / expected[tail].abs().clamp(min=1e-30)
        ).abs().max().item()
        # Pre-fix this was ~1.0 (100% off, flushed to zero); fp32 rounding of
        # x / sqrt(2) leaves a residual well under 1e-3 after the fix.
        self.assertLess(rel_err, 1e-3)

    def test_gelu_backward_tail_accuracy_xpu(self):
        dtype = torch.float32
        x = torch.arange(-12.0, 12.0, 2**-6, dtype=dtype, device=xpu_device)
        xref = x.cpu().double()

        kAlpha = math.sqrt(0.5)
        kBeta = math.sqrt(2.0 / math.pi) * 0.5
        expected = 0.5 * torch.erfc(-xref * kAlpha) + xref * kBeta * torch.exp(
            -0.5 * xref * xref
        )

        grad = torch.ones_like(x)
        actual = torch.ops.aten.gelu_backward(grad, x).cpu().double()

        tail = xref.abs() >= 4.0
        rel_err = (
            (actual - expected)[tail] / expected[tail].abs().clamp(min=1e-30)
        ).abs().max().item()
        self.assertLess(rel_err, 1e-3)

    def test_gelu_matches_cpu_tail_xpu(self):
        # CPU exact gelu is already accurate in the tail; XPU should match it.
        dtype = torch.float32
        x = torch.tensor(
            [-10.0, -8.0, -6.0, -5.5, -5.0, -2.0, 3.0], dtype=dtype
        )
        y_cpu = F.gelu(x)
        y_xpu = F.gelu(x.to(xpu_device)).cpu()
        self.assertEqual(y_cpu, y_xpu, atol=1e-6, rtol=1e-5)
