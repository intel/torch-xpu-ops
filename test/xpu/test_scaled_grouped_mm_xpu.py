# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Unit tests for XPU scaled_grouped_mm kernel.

Tests FP8 x FP8 -> BF16 grouped GEMM with rowwise float32 scaling.
All 4 input modes: 2D×2D, 2D×3D, 3D×3D, 3D×2D.

Two-tier reference strategy:
  1. BF16-path reference — mirrors kernel's dequantize-then-GEMM approach
     (FP8→BF16, scale→BF16, float32 matmul→BF16). Catches layout and
     scaling bugs with tight tolerances (≤2 ULP).
  2. Float32-path reference — gold-standard precision (all computation in
     float32). Documents the expected precision gap from BF16 intermediate
     dequantization.
"""

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def reference_f32_per_group(a_fp8, b_fp8_t, scale_a, scale_b):
    """Float32 reference: all computation in float32, cast to BF16 at the end.

    Args:
        a_fp8: (M, K) FP8 tensor, row-major
        b_fp8_t: (K, N) FP8 tensor — transposed view (physical (N, K) row-major)
        scale_a: (M,) float32 rowwise scale for A
        scale_b: (N,) float32 rowwise scale for B
    Returns:
        (M, N) BF16 tensor
    """
    a_f32 = a_fp8.float() * scale_a.unsqueeze(-1)
    b_phys = b_fp8_t.t().contiguous()  # (N, K) row-major
    b_f32 = b_phys.float() * scale_b.unsqueeze(-1)  # (N, K) * (N, 1)
    out = a_f32 @ b_f32.t()
    return out.to(torch.bfloat16)


def reference_bf16_per_group(a_fp8, b_fp8_t, scale_a, scale_b):
    """BF16-path reference: mirrors kernel's dequantize-then-GEMM path.

    Dequantize FP8→BF16, apply scale (promotes to float32, cast back to BF16),
    then matmul in float32 (matching sycl-tla's float32 accumulation).

    Args:
        a_fp8: (M, K) FP8 tensor, row-major
        b_fp8_t: (K, N) FP8 tensor — transposed view
        scale_a: (M,) float32 rowwise scale for A
        scale_b: (N,) float32 rowwise scale for B
    Returns:
        (M, N) BF16 tensor
    """
    a_bf16 = (a_fp8.to(torch.bfloat16) * scale_a.unsqueeze(-1)).to(torch.bfloat16)
    b_contig = b_fp8_t.contiguous()  # (K, N) row-major
    b_bf16 = (b_contig.to(torch.bfloat16) * scale_b.unsqueeze(0)).to(torch.bfloat16)
    out = a_bf16.float() @ b_bf16.float()
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScaledGroupedMMXPU(TestCase):
    fp8_dtype = torch.float8_e4m3fn

    # BF16-path tolerances: tight, catches real bugs.
    # 2 ULP absolute + 1% relative covers hardware DPAS accumulation ordering.
    bf16_atol = 0.125
    bf16_rtol = 1e-2

    # F32-path tolerances: wider, documents BF16 dequant precision gap.
    f32_atol = 0.5
    f32_rtol = 5e-2

    def _make_fp8(self, *shape, device):
        return torch.randn(*shape, device=device).to(self.fp8_dtype)

    def _make_scale(self, *shape, device, low=0.1, high=1.1):
        return (
            torch.rand(*shape, device=device, dtype=torch.float32) * (high - low) + low
        )

    # ====================================================================
    # BF16-path tests — tight tolerances catching real bugs
    # ====================================================================

    @onlyXPU
    def test_bf16_3d_3d(self, device):
        """3D x 3D: batched grouped GEMM, BF16-path reference."""
        m, n, k, G = 16, 32, 64, 4
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(G, n, device=device)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
        self.assertEqual(out.shape, (G, m, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        for g in range(G):
            ref = reference_bf16_per_group(a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"3D×3D group {g}",
            )

    @onlyXPU
    def test_bf16_3d_3d_various_shapes(self, device):
        """3D x 3D with multiple shape configurations."""
        shapes = [
            (16, 16, 16, 2),
            (32, 64, 128, 4),
            (64, 32, 128, 4),
            (128, 256, 64, 8),
        ]
        for m, n, k, G in shapes:
            with self.subTest(M=m, N=n, K=k, G=G):
                a = self._make_fp8(G, m, k, device=device)
                b_phys = self._make_fp8(G, n, k, device=device)
                b_t = b_phys.transpose(-2, -1)
                scale_a = self._make_scale(G, m, device=device)
                scale_b = self._make_scale(G, n, device=device)

                out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
                for g in range(G):
                    ref = reference_bf16_per_group(
                        a[g], b_t[g], scale_a[g], scale_b[g]
                    )
                    torch.testing.assert_close(
                        out[g],
                        ref,
                        atol=self.bf16_atol,
                        rtol=self.bf16_rtol,
                        msg=f"shape ({m},{n},{k},{G}) group {g}",
                    )

    @onlyXPU
    def test_bf16_2d_3d(self, device):
        """2D x 3D: ragged A (MoE pattern), BF16-path reference."""
        m, n, k, G = 16, 32, 64, 4
        total_M = m * G
        a = self._make_fp8(total_M, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(total_M, device=device)
        scale_b = self._make_scale(G, n, device=device)
        offs = torch.arange(m, total_M + 1, m, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (total_M, n))

        # BF16-path: global A dequant then slice (mirrors kernel)
        a_bf16_full = (a.to(torch.bfloat16) * scale_a.unsqueeze(-1)).to(
            torch.bfloat16
        )
        row_start = 0
        for g in range(G):
            row_end = offs[g].item()
            a_g = a_bf16_full[row_start:row_end]
            b_contig_g = b_t[g].contiguous()
            b_bf16_g = (b_contig_g.to(torch.bfloat16) * scale_b[g].unsqueeze(0)).to(
                torch.bfloat16
            )
            ref = (a_g.float() @ b_bf16_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[row_start:row_end],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"2D×3D group {g}",
            )
            row_start = row_end

    @onlyXPU
    def test_bf16_2d_3d_nonuniform(self, device):
        """2D x 3D with non-uniform group sizes."""
        k, n, G = 64, 32, 4
        group_sizes = [16, 48, 32, 64]
        total_M = sum(group_sizes)
        a = self._make_fp8(total_M, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(total_M, device=device)
        scale_b = self._make_scale(G, n, device=device)

        cumulative = []
        s = 0
        for sz in group_sizes:
            s += sz
            cumulative.append(s)
        offs = torch.tensor(cumulative, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (total_M, n))

        a_bf16_full = (a.to(torch.bfloat16) * scale_a.unsqueeze(-1)).to(
            torch.bfloat16
        )
        row_start = 0
        for g in range(G):
            row_end = offs[g].item()
            a_g = a_bf16_full[row_start:row_end]
            b_contig_g = b_t[g].contiguous()
            b_bf16_g = (b_contig_g.to(torch.bfloat16) * scale_b[g].unsqueeze(0)).to(
                torch.bfloat16
            )
            ref = (a_g.float() @ b_bf16_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[row_start:row_end],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"2D×3D nonuniform group {g}",
            )
            row_start = row_end

    @onlyXPU
    def test_bf16_3d_2d(self, device):
        """3D x 2D: ragged B, BF16-path reference."""
        m, n, k, G = 16, 32, 64, 4
        total_N = n * G
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(total_N, k, device=device)
        b_t = b_phys.t()
        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(total_N, device=device)
        offs = torch.arange(n, total_N + 1, n, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (m, total_N))

        # BF16-path: global dequant then column-slice (mirrors kernel)
        a_bf16 = (a.to(torch.bfloat16) * scale_a.unsqueeze(-1)).to(torch.bfloat16)
        b_contig = b_t.contiguous()
        b_bf16 = (b_contig.to(torch.bfloat16) * scale_b.unsqueeze(0)).to(
            torch.bfloat16
        )

        col_start = 0
        for g in range(G):
            col_end = offs[g].item()
            a_g = a_bf16[g]
            b_g = b_bf16[:, col_start:col_end].contiguous()
            ref = (a_g.float() @ b_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[:, col_start:col_end],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"3D×2D group {g}",
            )
            col_start = col_end

    @onlyXPU
    def test_bf16_3d_2d_nonuniform(self, device):
        """3D x 2D with non-uniform group sizes."""
        m, k, G = 16, 64, 4
        group_sizes = [32, 64, 16, 48]
        total_N = sum(group_sizes)
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(total_N, k, device=device)
        b_t = b_phys.t()
        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(total_N, device=device)

        cumulative = []
        s = 0
        for sz in group_sizes:
            s += sz
            cumulative.append(s)
        offs = torch.tensor(cumulative, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (m, total_N))

        a_bf16 = (a.to(torch.bfloat16) * scale_a.unsqueeze(-1)).to(torch.bfloat16)
        b_contig = b_t.contiguous()
        b_bf16 = (b_contig.to(torch.bfloat16) * scale_b.unsqueeze(0)).to(
            torch.bfloat16
        )

        col_start = 0
        for g in range(G):
            col_end = offs[g].item()
            a_g = a_bf16[g]
            b_g = b_bf16[:, col_start:col_end].contiguous()
            ref = (a_g.float() @ b_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[:, col_start:col_end],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"3D×2D nonuniform group {g}",
            )
            col_start = col_end

    @onlyXPU
    def test_bf16_2d_2d(self, device):
        """2D x 2D: ragged K, BF16-path reference."""
        m, n, k, G = 16, 32, 64, 4
        total_K = k * G
        a = self._make_fp8(m, total_K, device=device)
        b_phys = self._make_fp8(n, total_K, device=device)
        b_t = b_phys.t()
        scale_a = self._make_scale(m * G, device=device)
        scale_b = self._make_scale(n * G, device=device)
        offs = torch.arange(k, total_K + 1, k, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (G, m, n))

        # BF16-path: per-group dequant (mirrors kernel's 2D×2D path)
        a_bf16 = a.to(torch.bfloat16)
        b_contig = b_t.contiguous()
        b_bf16 = b_contig.to(torch.bfloat16)

        k_start = 0
        for g in range(G):
            k_end = offs[g].item()
            sa_g = scale_a[g * m : (g + 1) * m]
            sb_g = scale_b[g * n : (g + 1) * n]
            a_g = (a_bf16[:, k_start:k_end] * sa_g.unsqueeze(-1)).to(
                torch.bfloat16
            ).contiguous()
            b_g = (b_bf16[k_start:k_end] * sb_g.unsqueeze(0)).to(
                torch.bfloat16
            ).contiguous()
            ref = (a_g.float() @ b_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"2D×2D group {g}",
            )
            k_start = k_end

    @onlyXPU
    def test_bf16_2d_2d_nonuniform(self, device):
        """2D x 2D with non-uniform group sizes."""
        m, n, G = 16, 32, 4
        group_sizes = [32, 64, 48, 112]
        total_K = sum(group_sizes)
        a = self._make_fp8(m, total_K, device=device)
        b_phys = self._make_fp8(n, total_K, device=device)
        b_t = b_phys.t()
        scale_a = self._make_scale(m * G, device=device)
        scale_b = self._make_scale(n * G, device=device)

        cumulative = []
        s = 0
        for sz in group_sizes:
            s += sz
            cumulative.append(s)
        offs = torch.tensor(cumulative, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
        self.assertEqual(out.shape, (G, m, n))

        a_bf16 = a.to(torch.bfloat16)
        b_contig = b_t.contiguous()
        b_bf16 = b_contig.to(torch.bfloat16)

        k_start = 0
        for g in range(G):
            k_end = offs[g].item()
            sa_g = scale_a[g * m : (g + 1) * m]
            sb_g = scale_b[g * n : (g + 1) * n]
            a_g = (a_bf16[:, k_start:k_end] * sa_g.unsqueeze(-1)).to(
                torch.bfloat16
            ).contiguous()
            b_g = (b_bf16[k_start:k_end] * sb_g.unsqueeze(0)).to(
                torch.bfloat16
            ).contiguous()
            ref = (a_g.float() @ b_g.float()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg=f"2D×2D nonuniform group {g}",
            )
            k_start = k_end

    # ====================================================================
    # Float32-path tests — documents BF16 dequant precision gap
    # ====================================================================

    @onlyXPU
    def test_f32_3d_3d(self, device):
        """3D x 3D accuracy vs float32 reference."""
        m, n, k, G = 16, 32, 64, 4
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(G, n, device=device)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
        for g in range(G):
            ref = reference_f32_per_group(a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.f32_atol,
                rtol=self.f32_rtol,
                msg=f"3D×3D f32-ref group {g}",
            )

    @onlyXPU
    def test_f32_accuracy_large(self, device):
        """Larger shapes stress test vs float32 reference."""
        G, m, n, k = 8, 64, 128, 128
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(G, n, device=device)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
        for g in range(G):
            ref = reference_f32_per_group(a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.f32_atol,
                rtol=self.f32_rtol,
                msg=f"large shape f32-ref group {g}",
            )

    # ====================================================================
    # Edge case tests
    # ====================================================================

    @onlyXPU
    def test_single_group_3d_3d(self, device):
        """Single group (G=1)."""
        m, n, k = 16, 32, 64
        a = self._make_fp8(1, m, k, device=device)
        b_phys = self._make_fp8(1, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = self._make_scale(1, m, device=device)
        scale_b = self._make_scale(1, n, device=device)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
        self.assertEqual(out.shape, (1, m, n))
        ref = reference_bf16_per_group(a[0], b_t[0], scale_a[0], scale_b[0])
        torch.testing.assert_close(
            out[0], ref, atol=self.bf16_atol, rtol=self.bf16_rtol
        )

    @onlyXPU
    def test_unit_scales(self, device):
        """Scale=1.0 gives near-exact results vs float32 reference."""
        m, n, k, G = 16, 32, 64, 4
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)
        scale_a = torch.ones(G, m, device=device, dtype=torch.float32)
        scale_b = torch.ones(G, n, device=device, dtype=torch.float32)

        out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
        for g in range(G):
            ref = reference_f32_per_group(a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g],
                ref,
                atol=self.bf16_atol,
                rtol=self.bf16_rtol,
                msg="Unit scales should give near-exact results",
            )

    @onlyXPU
    def test_scale_magnitudes(self, device):
        """Various scale magnitudes; atol scales with output magnitude."""
        m, n, k, G = 16, 32, 64, 2
        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)

        for scale_val in [0.01, 0.1, 1.0, 10.0]:
            with self.subTest(scale=scale_val):
                sa = self._make_scale(
                    G, m, device=device, low=scale_val * 0.5, high=scale_val * 1.5
                )
                sb = self._make_scale(
                    G, n, device=device, low=scale_val * 0.5, high=scale_val * 1.5
                )
                out = torch._scaled_grouped_mm(a, b_t, sa, sb)
                out_mag = max(out.float().abs().max().item(), 1.0)
                dynamic_atol = max(self.f32_atol, out_mag * 0.02)
                for g in range(G):
                    ref = reference_f32_per_group(a[g], b_t[g], sa[g], sb[g])
                    torch.testing.assert_close(
                        out[g],
                        ref,
                        atol=dynamic_atol,
                        rtol=self.f32_rtol,
                        msg=f"scale={scale_val} group {g}",
                    )


instantiate_device_type_tests(
    TestScaledGroupedMMXPU, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
