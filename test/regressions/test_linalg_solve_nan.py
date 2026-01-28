# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestLinalgSolveNaN(TestCase):
    """Test torch.linalg.solve with NaN handling.
    
    Regression test: single NaN in matrix would crash oneMKL's iamax during pivoting.
    """

    def _assert_results_match(self, A, b, atol=1e-4, rtol=1e-4):
        """Verify CPU and XPU results match (including NaN patterns)."""
        result_cpu = torch.linalg.solve(A, b)
        result_xpu = torch.linalg.solve(A.to(xpu_device), b.to(xpu_device))
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=atol, rtol=rtol, equal_nan=True)

    def test_solve_nan_variants(self):
        """Test NaN in matrix (all/partial) and in b vector."""
        # All NaN
        self._assert_results_match(torch.full((3, 3), float('nan')), torch.randn(3))
        
        # Single NaN in matrix
        A = torch.randn(4, 4)
        A[0, 0] = float('nan')
        self._assert_results_match(A, torch.randn(4))
        
        # NaN in b vector
        self._assert_results_match(torch.randn(3, 3), torch.tensor([1., float('nan'), 2.]))

    def test_solve_batch_mixed_nan(self):
        """Test batch: some with NaN, others without."""
        A = torch.randn(4, 3, 3)
        A[[0, 2], 0, 0] = float('nan')  # Batches 0, 2 have NaN
        self._assert_results_match(A, torch.randn(4, 3))

    def test_solve_cayley_transform(self):
        """Test issues 2667"""
        data = torch.randn(4, 4, 4)
        data[0, 0, 0] = float('nan')
        
        S = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(4).unsqueeze(0).expand(4, 4, 4)
        self._assert_results_match(I + S, I - S)
