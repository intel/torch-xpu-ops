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
"""

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_utils import run_tests, TestCase

TEST_XPU = torch.xpu.is_available()


def reference_scaled_mm_per_group(a_fp8, b_fp8_t, scale_a, scale_b):
    """Reference: dequantize FP8 -> float32, apply scales, matmul, cast to BF16.

    Args:
        a_fp8: (M, K) FP8 tensor, row-major
        b_fp8_t: (K, N) FP8 tensor — transposed view (physical (N, K) row-major)
        scale_a: (M,) float32 rowwise scale for A
        scale_b: (N,) float32 rowwise scale for B (one per column of logical B)
    Returns:
        (M, N) BF16 tensor
    """
    a_f32 = a_fp8.float() * scale_a.unsqueeze(-1)
    b_phys = b_fp8_t.t().contiguous()  # (N, K) row-major
    b_f32 = b_phys.float() * scale_b.unsqueeze(-1)  # (N, K) * (N, 1)
    out = a_f32 @ b_f32.t()
    return out.to(torch.bfloat16)


class TestScaledGroupedMMXPU(TestCase):
    # Tolerances are wider than CUDA because the dequantize-then-GEMM approach
    # truncates to BF16 before the matrix multiply.
    atol = 2e-1
    rtol = 5e-2
    fp8_dtype = torch.float8_e4m3fn

    def _make_fp8(self, *shape, device):
        return torch.randn(*shape, device=device).to(self.fp8_dtype)

    def _make_scale(self, *shape, device):
        return torch.rand(*shape, device=device, dtype=torch.float32) + 0.1

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_scaled_grouped_gemm_3d_3d(self, device, dtype):
        """3D x 3D: batched grouped GEMM, no offsets."""
        m, n, k, G = 16, 32, 64, 4

        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(G, n, device=device)

        out = torch._scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        self.assertEqual(out.shape, (G, m, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        for g in range(G):
            ref = reference_scaled_mm_per_group(
                a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"3D x 3D group {g} mismatch")

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_scaled_grouped_gemm_2d_3d(self, device, dtype):
        """2D x 3D: ragged A (MoE pattern) with offsets along M."""
        m, n, k, G = 16, 32, 64, 4
        total_M = m * G

        a = self._make_fp8(total_M, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(total_M, device=device)
        scale_b = self._make_scale(G, n, device=device)

        offs = torch.arange(
            m, total_M + 1, m, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (total_M, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        row_start = 0
        for g in range(G):
            row_end = offs[g].item()
            ref = reference_scaled_mm_per_group(
                a[row_start:row_end], b_t[g],
                scale_a[row_start:row_end], scale_b[g])
            torch.testing.assert_close(
                out[row_start:row_end], ref, atol=self.atol, rtol=self.rtol,
                msg=f"2D x 3D group {g} mismatch")
            row_start = row_end

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_scaled_grouped_gemm_3d_2d(self, device, dtype):
        """3D x 2D: ragged B with offsets along N."""
        m, n, k, G = 16, 32, 64, 4
        total_N = n * G

        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(total_N, k, device=device)
        b_t = b_phys.t()

        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(total_N, device=device)

        offs = torch.arange(
            n, total_N + 1, n, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (m, total_N))
        self.assertEqual(out.dtype, torch.bfloat16)

        col_start = 0
        for g in range(G):
            col_end = offs[g].item()
            ref = reference_scaled_mm_per_group(
                a[g], b_t[:, col_start:col_end],
                scale_a[g], scale_b[col_start:col_end])
            torch.testing.assert_close(
                out[:, col_start:col_end], ref, atol=self.atol, rtol=self.rtol,
                msg=f"3D x 2D group {g} mismatch")
            col_start = col_end

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_scaled_grouped_gemm_2d_2d(self, device, dtype):
        """2D x 2D: ragged K with offsets along K dimension."""
        m, n, k, G = 16, 32, 64, 4
        total_K = k * G

        a = self._make_fp8(m, total_K, device=device)
        b_phys = self._make_fp8(n, total_K, device=device)
        b_t = b_phys.t()

        scale_a = self._make_scale(m * G, device=device)
        scale_b = self._make_scale(n * G, device=device)

        offs = torch.arange(
            k, total_K + 1, k, device=device, dtype=torch.int32)

        out = torch._scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (G, m, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        k_start = 0
        for g in range(G):
            k_end = offs[g].item()
            a_g = a[:, k_start:k_end]
            b_g_phys = b_phys[:, k_start:k_end]
            b_g_t = b_g_phys.t()
            sa_g = scale_a[g * m:(g + 1) * m]
            sb_g = scale_b[g * n:(g + 1) * n]
            ref = reference_scaled_mm_per_group(a_g, b_g_t, sa_g, sb_g)
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"2D x 2D group {g} mismatch")
            k_start = k_end

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_scaled_grouped_gemm_accuracy_large(self, device, dtype):
        """Test with larger sizes to stress the kernel."""
        G, m, n, k = 8, 64, 128, 128

        a = self._make_fp8(G, m, k, device=device)
        b_phys = self._make_fp8(G, n, k, device=device)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(G, m, device=device)
        scale_b = self._make_scale(G, n, device=device)

        out = torch._scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        for g in range(G):
            ref = reference_scaled_mm_per_group(
                a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"Large shape group {g} mismatch")


instantiate_device_type_tests(
    TestScaledGroupedMMXPU, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
