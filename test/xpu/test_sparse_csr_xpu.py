# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: sparse"]

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_sparse_csr import (
        TestSparseCompressedDevice,
        TestSparseCompressedTritonKernelsDevice,
        TestSparseCSRDevice,
    )


# ======================================================================
# XPU-only tests for torch.sparse.spsolve (aten::_spsolve)
# ======================================================================


@onlyXPU
@dtypes(*floating_types())
def _test_spsolve_xpu(self, device, dtype):
    def make_spd(n):
        # Diagonally dominant, so the system has a unique solution.
        m = make_tensor((n, n), dtype=dtype, device=device, low=-1, high=1)
        return m @ m.mT + n * torch.eye(n, dtype=dtype, device=device)

    test_matrices = [
        torch.eye(5, dtype=dtype, device=device),  # identity
        make_spd(1),  # 1x1
        make_spd(4),  # dense-ish
        make_spd(16),  # larger
        # diagonal (extreme sparsity)
        torch.diag(make_tensor((7,), dtype=dtype, device=device, low=1, high=2)),
        # tridiagonal (banded sparsity pattern)
        torch.diag(torch.full((6,), 4.0, dtype=dtype, device=device))
        + torch.diag(torch.full((5,), -1.0, dtype=dtype, device=device), 1)
        + torch.diag(torch.full((5,), -1.0, dtype=dtype, device=device), -1),
    ]

    for A_dense in test_matrices:
        n = A_dense.size(0)
        b = make_tensor((n,), dtype=dtype, device=device, low=-1, high=1)
        A_csr = A_dense.to_sparse_csr()

        x = torch.sparse.spsolve(A_csr, b)
        expect = torch.linalg.solve(A_dense, b)
        self.assertEqual(x, expect)

        # Residual check: A @ x should reconstruct b.
        self.assertEqual(A_dense @ x, b)

        # Correctness against the CPU reference.
        x_cpu = torch.linalg.solve(A_dense.cpu(), b.cpu())
        self.assertEqual(x.cpu(), x_cpu)

    # Empty system: a 0x0 matrix with an empty rhs yields an empty result.
    empty_A = torch.zeros(0, 0, dtype=dtype, device=device).to_sparse_csr()
    empty_b = torch.zeros(0, dtype=dtype, device=device)
    empty_x = torch.sparse.spsolve(empty_A, empty_b)
    self.assertEqual(empty_x.shape, torch.Size([0]))


@onlyXPU
@dtypes(*floating_types())
def _test_spsolve_xpu_errors(self, device, dtype):
    A_dense = torch.eye(4, dtype=dtype, device=device)
    A_csr = A_dense.to_sparse_csr()
    b = make_tensor((4,), dtype=dtype, device=device, low=-1, high=1)

    # left=False is not supported by the Sparse CSR backend (matches CUDA).
    with self.assertRaisesRegex(RuntimeError, "only left == true is supported"):
        torch.sparse.spsolve(A_csr, b, left=False)

    B2d = make_tensor((4, 2), dtype=dtype, device=device, low=-1, high=1)
    with self.assertRaisesRegex(RuntimeError, "b must be a 1D tensor"):
        torch.sparse.spsolve(A_csr, B2d)

    b_bad = make_tensor((5,), dtype=dtype, device=device, low=-1, high=1)
    with self.assertRaisesRegex(RuntimeError, "linear system size mismatch"):
        torch.sparse.spsolve(A_csr, b_bad)

    singular = torch.zeros(4, 4, dtype=dtype, device=device).to_sparse_csr()
    with self.assertRaises(torch.linalg.LinAlgError):
        torch.sparse.spsolve(singular, b)


TestSparseCSRDevice.test_spsolve_xpu = _test_spsolve_xpu
TestSparseCSRDevice.test_spsolve_xpu_errors = _test_spsolve_xpu_errors


# ======================================================================
# Instantiate test classes for XPU execution
# ======================================================================

instantiate_device_type_tests(
    TestSparseCompressedDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestSparseCSRDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestSparseCompressedTritonKernelsDevice, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
