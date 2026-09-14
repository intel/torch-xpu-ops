# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

xpu_device = torch.device("xpu")


def _assert_close_relative_l2(actual, expected, tol, msg=""):
    """Compare a column reduction by relative L2 norm rather than elementwise.

    The two-stage path accumulates each row tile in fp32 but stores the per-tile
    partial into a `dgamma_blocks` buffer that inherits dgamma's dtype, so a
    bf16 dgamma rounds every one of the ~65 partials before the final sum. That
    is inherent to the kernel, not a defect, and it puts a handful of the 128
    columns outside any elementwise tolerance tight enough to still be
    meaningful. The relative L2 norm is the right measure here: it stays at
    ~2e-3 for a healthy reduction while every structural corruption (a dropped
    tile, a missing rstd scale, mean subtraction, sum replaced by mean) lands at
    8e-2 or above, so a 2e-2 threshold keeps a wide margin in both directions.
    """
    diff = (actual.float() - expected.float()).norm()
    rel = (diff / expected.float().norm()).item()
    if rel > tol:
        raise AssertionError(f"{msg}relative L2 error {rel:.3e} exceeds {tol:.3e}")


def _over_threshold_rows():
    """Row count above the kernel's two-stage column-reduction threshold.

    The kernel gates on `M > xe_core_count * 1024`, where its `xe_core_count` is
    `gpu_eu_count / gpu_eu_count_per_subslice` -- the same quotient PyTorch
    exposes as `gpu_subslice_count`. One extra tile of rows clears it.
    """
    xe_core_count = torch.xpu.get_device_properties(0).gpu_subslice_count
    return xe_core_count * 1024 + 1024


class TestRMSNorm(TestCase):
    def test_rms_norm_backward_weight_grad_large_rows(self):
        """Weight gradient above the two-stage column-reduction threshold.

        rms_norm_backward_kernel used to alias dgamma into the shared impl's
        dbeta slot. Above `xe_core_count * 1024` rows the impl takes a two-stage
        column reduction whose both-defined arm reads a dbeta_blocks buffer that
        the rms_norm specialization never allocates, so the call died with
        "tensor does not have a device". Only the weight-gradient path was
        affected, and only for normalized_shape < 2048.
        """
        torch.manual_seed(42)
        norm = 128
        rows = _over_threshold_rows()

        x = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        weight = torch.randn(norm, device=xpu_device, dtype=torch.bfloat16)
        grad_out = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        rstd = torch.rand(rows, 1, device=xpu_device, dtype=torch.float32) + 0.5

        grad_input, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, x, [norm], rstd, weight, [True, True]
        )

        self.assertEqual(grad_input.shape, x.shape)
        self.assertEqual(grad_weight.shape, weight.shape)
        self.assertTrue(torch.isfinite(grad_input).all())
        self.assertTrue(torch.isfinite(grad_weight).all())

        # dgamma = sum over rows of (dY * x * rstd). This is the check that would
        # catch a silently wrong reduction, not just a crash.
        expected = (grad_out.float() * x.float() * rstd.float()).sum(0)
        _assert_close_relative_l2(grad_weight, expected, 2e-2, "dgamma: ")

    def test_rms_norm_backward_weight_grad_matches_small_rows(self):
        """The below-threshold path must be unchanged by the fix."""
        torch.manual_seed(42)
        norm = 128
        rows = 1024

        x = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        weight = torch.randn(norm, device=xpu_device, dtype=torch.bfloat16)
        grad_out = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        rstd = torch.rand(rows, 1, device=xpu_device, dtype=torch.float32) + 0.5

        _, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, x, [norm], rstd, weight, [True, True]
        )
        expected = (grad_out.float() * x.float() * rstd.float()).sum(0)
        _assert_close_relative_l2(grad_weight, expected, 2e-2, "dgamma: ")

    def test_rms_norm_autograd_large_rows(self):
        """nn.RMSNorm with a trainable weight, above the threshold.

        This is how the bug reached users: a per-head QK-norm (normalized_shape
        = head_dim) on a 4-D tensor has batch * seq * num_heads rows, which
        crosses the threshold at modest batch sizes.
        """
        norm = 128
        rows = _over_threshold_rows()

        layer = nn.RMSNorm(norm, eps=1e-6, dtype=torch.bfloat16).to(xpu_device)
        x = torch.randn(
            rows, norm, device=xpu_device, dtype=torch.bfloat16, requires_grad=True
        )
        layer(x).sum().backward()

        self.assertIsNotNone(layer.weight.grad)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rms_norm_backward_input_grad_only_large_rows(self):
        """output_mask=[True, False] always worked; keep it that way."""
        norm = 128
        rows = _over_threshold_rows()

        x = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        weight = torch.randn(norm, device=xpu_device, dtype=torch.bfloat16)
        grad_out = torch.randn(rows, norm, device=xpu_device, dtype=torch.bfloat16)
        rstd = torch.rand(rows, 1, device=xpu_device, dtype=torch.float32) + 0.5

        grad_input, _ = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, x, [norm], rstd, weight, [True, False]
        )
        self.assertEqual(grad_input.shape, x.shape)
        self.assertTrue(torch.isfinite(grad_input).all())

    def test_layer_norm_backward_unaffected(self):
        """LayerNorm shares the impl and must keep producing a real dbeta.

        The fix only changes the rms_norm=true instantiation, but LayerNorm is
        the path that would break if the dbeta plumbing were disturbed.
        """
        norm = 128
        rows = _over_threshold_rows()

        layer = nn.LayerNorm(norm, eps=1e-6, dtype=torch.float32).to(xpu_device)
        x = torch.randn(
            rows, norm, device=xpu_device, dtype=torch.float32, requires_grad=True
        )
        layer(x).sum().backward()

        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.bias.grad).all())
        # d(sum)/d(bias) is 1 per row, so dbeta is exactly the row count.
        self.assertEqual(
            layer.bias.grad,
            torch.full_like(layer.bias.grad, float(rows)),
            atol=1e-1,
            rtol=1e-3,
        )
