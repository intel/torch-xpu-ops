# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import math
from functools import partial

import torch
from torch.testing._internal.common_utils import TestCase


class TestGroupNorm(TestCase):
    """Regression test for native_group_norm fp16 accuracy on XPU.

    Verifies eager and torch.compile produce consistent results.
    See https://github.com/pytorch/pytorch/issues/185367.
    """

    def _make_sample_inputs(self):
        """Generate the same sample inputs as sample_inputs_native_group_norm."""
        device = "xpu"
        dtype = torch.float16
        make_arg = partial(
            torch.testing.make_tensor, device=device, dtype=dtype, requires_grad=False
        )

        # Same cases as sample_inputs_group_norm
        cases = (
            ((1, 6, 3), 2, {"eps": 0.5}),
            ((2, 6, 3), 2, {"eps": -0.5}),
            ((1, 3), 1, {"eps": 1e-5}),
            ((0, 2), 1, {"eps": 1e-5}),
            ((5, 5, 5), 1, {"eps": 0.5}),
        )

        samples = []
        for input_shape, num_groups, kw in cases:
            channels = input_shape[1] if len(input_shape) > 1 else 0
            weight_tensor = make_arg(channels)
            bias_tensor = make_arg(channels)

            for weight in [weight_tensor, None]:
                for bias in [bias_tensor, None]:
                    inp = make_arg(input_shape)
                    N = inp.shape[0]
                    C = inp.shape[1]
                    HxW = math.prod(inp.shape[2:])
                    eps = kw["eps"]
                    samples.append((inp, weight, bias, N, C, HxW, num_groups, eps))

        # Without any optional args
        inp = make_arg((1, 2))
        samples.append((inp, None, None, 1, 2, 1, 1, 1e-5))
        return samples

    def test_group_norm_eager_vs_compile_float16(self):
        @torch.compile
        def compiled_group_norm(x, weight, bias, N, C, HxW, group, eps):
            return torch.native_group_norm(x, weight, bias, N, C, HxW, group, eps)

        for inp, weight, bias, N, C, HxW, group, eps in self._make_sample_inputs():
            eager_out = torch.native_group_norm(
                inp, weight, bias, N, C, HxW, group, eps
            )
            compiled_out = compiled_group_norm(inp, weight, bias, N, C, HxW, group, eps)
            # Same tolerance as TestInductorOpInfoXPU: float16 default
            for e, c in zip(eager_out, compiled_out):
                self.assertEqual(e, c)


class TestGroupNormFusedForward(TestCase):
    """Correctness tests for the fused GroupNorm forward kernel.

    Compares XPU results against CPU fp32 reference across all three
    fused kernel dispatch paths (SmallDS, MediumDS, LargeDS) plus the
    D=1 edge case that previously triggered an OOB bug.
    """

    # (N, C, *spatial, G) tuples covering all dispatch paths:
    #   SmallDS: DS <= 128, pow2, G and HxW pow2
    #   MediumDS: DS in {128,256,384,512}, DS % 128 == 0
    #   LargeDS: DS > 512 or non-aligned
    #   D=1 edge case: G == C
    SHAPES = [
        # --- SmallDS path (DS = D*HxW <= 128, pow2) ---
        # DS=16: 1 lane/group (LANES=4, GROUPS_PER_SG=8)
        (32, 128, 4, 4, 32),  # D=4, HxW=16 -> DS=64 (original comment wrong)
        # DS=32
        (64, 64, 2, 4, 32),  # D=2, HxW=8 -> DS=16... let me recalc
        # Actually: D=C/G, DS=D*HxW
        # (N=32, C=128, H=2, W=2, G=32): D=4, HxW=4, DS=16
        (32, 128, 2, 2, 32),
        # (N=32, C=64, H=4, W=4, G=32): D=2, HxW=16, DS=32
        (32, 64, 4, 4, 32),
        # (N=32, C=32, H=8, W=4, G=32): D=1, HxW=32, DS=32
        (32, 32, 8, 4, 32),
        # DS=128: (N=1024, C=1024, H=2, W=2, G=32): D=32, HxW=4, DS=128
        (64, 1024, 2, 2, 32),
        # --- MediumDS path (DS % 128 == 0, DS/128 in [1..4]) ---
        # DS=128: (N=32, C=64, H=8, W=8, G=2): D=32, HxW=64... DS=2048 too big
        # DS=256: (N=32, C=128, H=2, W=1, G=1): D=128, HxW=2, DS=256
        (32, 128, 2, 1, 1),
        # DS=512: (N=32, C=64, H=4, W=2, G=1): D=64, HxW=8, DS=512
        (32, 64, 4, 2, 1),
        # --- LargeDS path ---
        # (N=32, C=64, H=14, W=14, G=1): D=64, HxW=196, DS=12544
        (32, 64, 14, 14, 1),
        # (N=4, C=192, H=28, W=28, G=1): D=192, HxW=784, DS=150528
        (4, 192, 28, 28, 1),
        # Non-pow2 HxW to test head/tail scalar paths
        (8, 96, 7, 7, 3),  # D=32, HxW=49, DS=1568
        # --- D=1 edge case (G == C, exercises c_head clamp fix) ---
        (32, 512, 20, 20, 512),  # D=1, HxW=400, DS=400
        (16, 64, 5, 5, 64),  # D=1, HxW=25, DS=25
    ]

    def _ref_group_norm(self, x, weight, bias, G, eps):
        """Compute GroupNorm in fp32 on CPU as reference."""
        x_fp32 = x.float().cpu()
        N, C = x_fp32.shape[:2]
        D = C // G
        HxW = x_fp32[0, 0].numel()
        x_flat = x_fp32.view(N, G, D * HxW)
        mean = x_flat.mean(dim=2, keepdim=True)
        var = x_flat.var(dim=2, unbiased=False, keepdim=True)
        x_norm = (x_flat - mean) / (var + eps).sqrt()
        x_norm = x_norm.view_as(x_fp32)
        if weight is not None:
            w = weight.float().cpu().view(1, C, *([1] * (x.dim() - 2)))
            x_norm = x_norm * w
        if bias is not None:
            b = bias.float().cpu().view(1, C, *([1] * (x.dim() - 2)))
            x_norm = x_norm + b
        mean_out = mean.squeeze(2)  # (N, G)
        rstd_out = (1.0 / (var + eps).sqrt()).squeeze(2)  # (N, G)
        return x_norm, mean_out, rstd_out

    def test_fused_forward_fp16(self):
        """Test fused kernel correctness for fp16 across all dispatch paths."""
        for N, C, H, W, G in self.SHAPES:
            with self.subTest(N=N, C=C, H=H, W=W, G=G):
                x = torch.randn(N, C, H, W, device="xpu", dtype=torch.float16)
                weight = torch.randn(C, device="xpu", dtype=torch.float16)
                bias = torch.randn(C, device="xpu", dtype=torch.float16)
                eps = 1e-5

                y, mean, rstd = torch.native_group_norm(
                    x, weight, bias, N, C, H * W, G, eps
                )
                y_ref, mean_ref, rstd_ref = self._ref_group_norm(
                    x, weight, bias, G, eps
                )

                self.assertEqual(
                    y.float().cpu(), y_ref, atol=2e-3, rtol=1e-3,
                    msg=f"Y mismatch for shape ({N},{C},{H},{W}) G={G}",
                )
                self.assertEqual(
                    mean.float().cpu(), mean_ref, atol=2e-3, rtol=1e-3,
                    msg=f"mean mismatch for shape ({N},{C},{H},{W}) G={G}",
                )
                self.assertEqual(
                    rstd.float().cpu(), rstd_ref, atol=2e-3, rtol=1e-3,
                    msg=f"rstd mismatch for shape ({N},{C},{H},{W}) G={G}",
                )

    def test_fused_forward_bf16(self):
        """Test fused kernel correctness for bf16."""
        # Subset of shapes to keep runtime reasonable
        bf16_shapes = [
            (32, 128, 2, 2, 32),  # SmallDS
            (32, 128, 2, 1, 1),   # MediumDS
            (8, 96, 7, 7, 3),     # LargeDS non-aligned
            (16, 64, 5, 5, 64),   # D=1
        ]
        for N, C, H, W, G in bf16_shapes:
            with self.subTest(N=N, C=C, H=H, W=W, G=G):
                x = torch.randn(N, C, H, W, device="xpu", dtype=torch.bfloat16)
                weight = torch.randn(C, device="xpu", dtype=torch.bfloat16)
                bias = torch.randn(C, device="xpu", dtype=torch.bfloat16)
                eps = 1e-5

                y, mean, rstd = torch.native_group_norm(
                    x, weight, bias, N, C, H * W, G, eps
                )
                y_ref, mean_ref, rstd_ref = self._ref_group_norm(
                    x, weight, bias, G, eps
                )

                self.assertEqual(
                    y.float().cpu(), y_ref, atol=2e-2, rtol=5e-3,
                    msg=f"Y mismatch for shape ({N},{C},{H},{W}) G={G}",
                )

    def test_fused_forward_no_affine(self):
        """Test fused kernel with weight=None and/or bias=None."""
        shape = (16, 64, 14, 14)  # LargeDS
        G = 4
        N, C, H, W = shape
        eps = 1e-5

        for use_weight, use_bias in [(True, False), (False, True), (False, False)]:
            with self.subTest(weight=use_weight, bias=use_bias):
                x = torch.randn(*shape, device="xpu", dtype=torch.float16)
                weight = (
                    torch.randn(C, device="xpu", dtype=torch.float16)
                    if use_weight
                    else None
                )
                bias = (
                    torch.randn(C, device="xpu", dtype=torch.float16)
                    if use_bias
                    else None
                )

                y, mean, rstd = torch.native_group_norm(
                    x, weight, bias, N, C, H * W, G, eps
                )
                y_ref, _, _ = self._ref_group_norm(x, weight, bias, G, eps)

                self.assertEqual(
                    y.float().cpu(), y_ref, atol=2e-3, rtol=1e-3,
                )

    def test_fused_forward_fp32(self):
        """Test fused kernel correctness for fp32 (no accumulation mismatch)."""
        shapes = [
            (8, 64, 4, 4, 32),    # SmallDS
            (4, 192, 28, 28, 1),  # LargeDS
        ]
        for N, C, H, W, G in shapes:
            with self.subTest(N=N, C=C, H=H, W=W, G=G):
                x = torch.randn(N, C, H, W, device="xpu", dtype=torch.float32)
                weight = torch.randn(C, device="xpu", dtype=torch.float32)
                bias = torch.randn(C, device="xpu", dtype=torch.float32)
                eps = 1e-5

                y, mean, rstd = torch.native_group_norm(
                    x, weight, bias, N, C, H * W, G, eps
                )
                y_ref, mean_ref, rstd_ref = self._ref_group_norm(
                    x, weight, bias, G, eps
                )

                self.assertEqual(
                    y.cpu(), y_ref, atol=1e-5, rtol=1e-5,
                    msg=f"Y mismatch for shape ({N},{C},{H},{W}) G={G}",
                )
