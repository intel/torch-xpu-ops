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

import functools

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
# Sparse CSR ops with non-contiguous values views (gh-201268 / issue #6173)
#
# The XPU kernels used to allocate fresh indices/values and drop a
# non-contiguous `out.values()` view instead of writing the result through
# it.
#
# The write-through helpers (_with_strided_values, _check_written_through_view,
# etc.) are inherited from upstream TestSparseCSRDevice (imported above from
# test_sparse_csr), so they are not duplicated here.
# ======================================================================


@onlyXPU
@dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
def _test_sparse_csr_add_noncontiguous_xpu(self, device, dtype):
    fill = -99

    def gen(nnz, index_dtype):
        return self.genSparseCSRTensor(
            [5, 5], nnz, dtype=dtype, device=device, index_dtype=index_dtype
        )

    for index_dtype in (torch.int32, torch.int64):
        # Identical operands with a distinct strided output. The guard runs
        # before the trivial-case handler (deliberately, unlike CPU/CUDA;
        # tracked by https://github.com/intel/torch-xpu-ops/issues/6333), so
        # the result is written through the view.
        a = gen(7, index_dtype)
        out, buf = self._with_strided_values(gen(7, index_dtype), fill)
        values = out.values()
        torch.add(a, a, alpha=0.5, out=out)
        self._check_written_through_view(out, buf, values, (1.5 * a.to_dense()), fill)

        # nnz change: raising is the other new behaviour here (CPU/CUDA
        # silently mis-handle this case).
        out, buf = self._with_strided_values(gen(3, index_dtype), fill)
        self._check_nnz_change_rejected(
            out, buf, lambda o: torch.add(a, a, alpha=0.5, out=o), fill
        )

        # Result nnz 0.
        z = gen(0, index_dtype)
        out, buf = self._with_strided_values(gen(7, index_dtype), fill)
        self._check_nnz_zero_result(out, buf, lambda o: torch.add(z, z, out=o))


# TODO: delete this test when https://github.com/intel/torch-xpu-ops/issues/6173
# is fully resolved and the upstream skipXPUIf on
# test_mm_addmm_out_noncontiguous_values is removed. It is a temporary copy of
# that upstream test so that the XPU path is covered in this CI until then.
@onlyXPU
@dtypes(torch.float64)
def _test_sparse_csr_mm_addmm_noncontiguous_out_xpu(self, device, dtype):
    fill = -99.0
    for index_dtype in (torch.int32, torch.int64):
        gen = functools.partial(
            self.genSparseCSRTensor,
            dtype=dtype,
            device=device,
            index_dtype=index_dtype,
        )
        a, b = gen((4, 6), 9), gen((6, 5), 11)
        nnz = torch.mm(a, b)._nnz()
        expected_mm = a.to_dense() @ b.to_dense()
        spy_a, spy_b = (
            torch.sparse_csr_tensor(
                t.crow_indices(),
                t.col_indices(),
                torch.ones_like(t.values()),
                size=t.shape,
            ).to_dense()
            for t in (a, b)
        )
        pattern = (spy_a @ spy_b).to_sparse_csr()
        c = torch.sparse_csr_tensor(
            pattern.crow_indices().to(index_dtype),
            pattern.col_indices().to(index_dtype),
            make_tensor((nnz,), dtype=dtype, device=device),
            size=pattern.shape,
        )
        expected_addmm = 0.5 * c.to_dense() + 2 * expected_mm

        for fn, expected in (
            (lambda o: torch.mm(a, b, out=o), expected_mm),
            (lambda o: torch.addmm(c, a, b, beta=0.5, alpha=2, out=o), expected_addmm),
        ):
            out, buf = self._with_strided_values(gen((4, 5), nnz), fill)
            values = out.values()
            fn(out)
            self._check_written_through_view(out, buf, values, expected, fill)

            out, buf = self._with_strided_values(
                gen((4, 5), 2 if nnz != 2 else 3), fill
            )
            self._check_nnz_change_rejected(out, buf, fn, fill)

        a0 = gen((4, 6), 0)
        for fn in (
            lambda o: torch.mm(a0, b, out=o),
            lambda o: torch.addmm(gen((4, 5), 0), a0, b, out=o),
        ):
            out, buf = self._with_strided_values(gen((4, 5), 3), fill)
            self._check_nnz_zero_result(out, buf, fn)


TestSparseCSRDevice.test_sparse_csr_add_noncontiguous_xpu = (
    _test_sparse_csr_add_noncontiguous_xpu
)
TestSparseCSRDevice.test_sparse_csr_mm_addmm_noncontiguous_out_xpu = (
    _test_sparse_csr_mm_addmm_noncontiguous_out_xpu
)


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
