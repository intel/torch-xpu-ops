# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTriangularSolveSparseCSR(TestCase):
    """Test torch.triangular_solve with sparse CSR coefficient matrix on XPU.

    Regression test for issue #3167: triangular_solve should work when the
    coefficient matrix A is in sparse CSR layout on XPU.
    """

    def _make_triangular_csr(self, n, upper=True, dtype=torch.float64):
        """Create a non-singular triangular matrix in sparse CSR format."""
        A = torch.randn(n, n, dtype=dtype)
        if upper:
            A = torch.triu(A)
        else:
            A = torch.tril(A)
        # Ensure non-singular by clamping diagonal magnitudes away from zero
        A.diagonal().abs_().clamp_(min=1.0)
        return A.to_sparse_csr()

    def _assert_triangular_solve_matches(
        self,
        A_csr,
        B,
        upper=True,
        transpose=False,
        unitriangular=False,
        atol=1e-4,
        rtol=1e-4,
    ):
        """Verify sparse CSR XPU result matches dense CPU result."""
        A_dense = A_csr.to_dense()
        expected, _ = torch.triangular_solve(
            B,
            A_dense,
            upper=upper,
            transpose=transpose,
            unitriangular=unitriangular,
        )
        result, cloned_A = torch.triangular_solve(
            B.to(xpu_device),
            A_csr.to(xpu_device),
            upper=upper,
            transpose=transpose,
            unitriangular=unitriangular,
        )
        # Validate solution
        self.assertEqual(expected, result.cpu(), atol=atol, rtol=rtol)
        # Validate cloned_coefficient: sparse CSR triangular solve should NOT clone A
        self.assertEqual(cloned_A.numel(), 0, "cloned_coefficient should be empty for sparse CSR")
        self.assertEqual(
            cloned_A.device,
            result.device,
            "cloned_coefficient should be on same device",
        )
        self.assertEqual(cloned_A.dtype, result.dtype, "cloned_coefficient should have same dtype")

    def test_upper_triangular(self):
        """Solve with upper triangular sparse CSR matrix."""
        A_csr = self._make_triangular_csr(5, upper=True)
        B = torch.randn(5, 1, dtype=torch.float64)
        self._assert_triangular_solve_matches(A_csr, B, upper=True)

    def test_lower_triangular(self):
        """Solve with lower triangular sparse CSR matrix."""
        A_csr = self._make_triangular_csr(5, upper=False)
        B = torch.randn(5, 1, dtype=torch.float64)
        self._assert_triangular_solve_matches(A_csr, B, upper=False)

    def test_multiple_rhs(self):
        """Solve with multiple right-hand-side columns."""
        A_csr = self._make_triangular_csr(6, upper=True)
        B = torch.randn(6, 3, dtype=torch.float64)
        self._assert_triangular_solve_matches(A_csr, B, upper=True)

    def test_transpose(self):
        """Solve with transposed coefficient matrix."""
        A_csr = self._make_triangular_csr(5, upper=True)
        B = torch.randn(5, 2, dtype=torch.float64)
        self._assert_triangular_solve_matches(A_csr, B, upper=True, transpose=True)

    def test_unitriangular(self):
        """Solve assuming unit diagonal."""
        A_csr = self._make_triangular_csr(4, upper=True)
        B = torch.randn(4, 1, dtype=torch.float64)
        self._assert_triangular_solve_matches(
            A_csr,
            B,
            upper=True,
            unitriangular=True,
        )

    def test_float32(self):
        """Solve with float32 precision."""
        A_csr = self._make_triangular_csr(5, upper=True, dtype=torch.float32)
        B = torch.randn(5, 2, dtype=torch.float32)
        self._assert_triangular_solve_matches(A_csr, B, upper=True, atol=1e-3, rtol=1e-3)

    def test_empty_rhs(self):
        """Solve with zero-element B should not error and preserve empty output shape."""
        A_csr = self._make_triangular_csr(3, upper=True)
        B = torch.empty(3, 0, dtype=torch.float64)
        X, cloned_A = torch.triangular_solve(
            B.to(xpu_device),
            A_csr.to(xpu_device),
            upper=True,
        )
        self.assertEqual(X.shape, (3, 0))
        self.assertEqual(cloned_A.numel(), 0, "cloned_coefficient should be empty for sparse CSR")

    def test_zero_nnz(self):
        """Solve with an all-zero sparse CSR matrix should return NaN-filled output."""
        n = 4
        A_csr = torch.sparse_csr_tensor(
            torch.zeros(n + 1, dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float64),
            size=(n, n),
        )
        B = torch.randn(n, 2, dtype=torch.float64)
        X, cloned_A = torch.triangular_solve(
            B.to(xpu_device),
            A_csr.to(xpu_device),
            upper=True,
        )
        self.assertEqual(X.shape, (n, 2))
        self.assertTrue(X.isnan().all(), "Expected all NaN output for zero-nnz A")
        self.assertEqual(cloned_A.numel(), 0, "cloned_coefficient should be empty for sparse CSR")


if __name__ == "__main__":
    run_tests()
