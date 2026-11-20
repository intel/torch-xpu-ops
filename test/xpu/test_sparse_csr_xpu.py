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
# ======================================================================


def _with_strided_values(t, fill):
    values = t.values()
    buf = torch.full(
        (2 * values.numel(),), fill, dtype=values.dtype, device=values.device
    )
    strided_values = buf[::2].view(values.shape)
    strided_values.copy_(values)
    out = torch.sparse_csr_tensor(
        t.crow_indices().clone(),
        t.col_indices().clone(),
        strided_values,
        size=t.shape,
    )
    return out, buf


def _check_written_through_view(self, out, buf, values, expected, fill):
    # `values` is out.values() from before the op: the result must have been
    # written through that view, and the storage between its elements must
    # be intact.
    self.assertEqual(out.to_dense(), expected)
    self.assertEqual(out.values().data_ptr(), values.data_ptr())
    self.assertEqual(out.values().stride(), values.stride())
    self.assertEqual(out.values().shape, values.shape)
    self.assertEqual(buf[::2], out.values().flatten())
    self.assertEqual(buf[1::2], torch.full_like(buf[1::2], fill))


def _check_nnz_change_rejected(self, out, buf, fn, fill):
    # A view cannot hold a result with a different nnz; resizing it would
    # overwrite the storage between its elements, so fn(out) must raise and
    # leave out and the buffer untouched.
    values, before = out.values(), out.to_dense()
    with self.assertRaisesRegex(RuntimeError, "resizing the non-contiguous values"):
        fn(out)
    self.assertEqual(out.values().data_ptr(), values.data_ptr())
    self.assertEqual(out.values().stride(), values.stride())
    self.assertEqual(out.values().shape, values.shape)
    self.assertEqual(buf[1::2], torch.full_like(buf[1::2], fill))
    torch._validate_sparse_compressed_tensor_args(
        out.crow_indices(), out.col_indices(), out.values(), out.shape, out.layout
    )
    self.assertEqual(out.to_dense(), before)


def _check_nnz_zero_result(self, out, buf, fn):
    # A result with nnz 0 writes nothing, so fn(out) may shrink the values
    # view to zero elements, as out.zero_() does, and leaves the buffer
    # untouched.
    buf_before = buf.clone()
    fn(out)
    torch._validate_sparse_compressed_tensor_args(
        out.crow_indices(), out.col_indices(), out.values(), out.shape, out.layout
    )
    self.assertEqual(out._nnz(), 0)
    self.assertEqual(
        out.to_dense(), torch.zeros(out.shape, dtype=out.dtype, device=out.device)
    )
    self.assertEqual(buf, buf_before)


@onlyXPU
@dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
def _test_sparse_csr_add_noncontiguous_xpu(self, device, dtype):
    fill = -99

    def gen(nnz, index_dtype):
        return self.genSparseCSRTensor(
            [5, 5], nnz, dtype=dtype, device=device, index_dtype=index_dtype
        )

    for index_dtype in (torch.int32, torch.int64):
        # Non-contiguous *input* values must be read in element order.
        s1, _ = _with_strided_values(gen(4, index_dtype), fill)
        s2, _ = _with_strided_values(gen(6, index_dtype), fill)
        out = gen(4, index_dtype)
        torch.add(s1, s2, out=out)
        self.assertEqual(out.to_dense(), s1.to_dense() + s2.to_dense())

        # b has a's sparsity pattern, so a + 0.5 * b has a's nnz and a
        # stride-2 out.values() can hold the result.
        a = gen(7, index_dtype)
        b = torch.sparse_csr_tensor(
            a.crow_indices().clone(),
            a.col_indices().clone(),
            make_tensor((7,), dtype=dtype, device=device),
            size=a.shape,
        )
        expected = a.to_dense() + 0.5 * b.to_dense()
        for fn, out_like in (
            (lambda o: torch.add(a, b, alpha=0.5, out=o), gen(7, index_dtype)),
            (lambda o: o.add_(b, alpha=0.5), a),
        ):
            out, buf = _with_strided_values(out_like, fill)
            values = out.values()
            fn(out)
            _check_written_through_view(self, out, buf, values, expected, fill)

        # a + 0.5 * b has nnz 7 while out has nnz 13: must raise.
        out, buf = _with_strided_values(gen(13, index_dtype), fill)
        _check_nnz_change_rejected(
            self, out, buf, lambda o: torch.add(a, b, alpha=0.5, out=o), fill
        )

        # The result has nnz 0.
        out, buf = _with_strided_values(gen(7, index_dtype), fill)
        _check_nnz_zero_result(
            self,
            out,
            buf,
            lambda o: torch.add(gen(0, index_dtype), gen(0, index_dtype), out=o),
        )


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
        # c has the product's sparsity pattern, so addmm's result has its
        # nnz.
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
            out, buf = _with_strided_values(gen((4, 5), nnz), fill)
            values = out.values()
            fn(out)
            _check_written_through_view(self, out, buf, values, expected, fill)

            out, buf = _with_strided_values(gen((4, 5), 2 if nnz != 2 else 3), fill)
            _check_nnz_change_rejected(self, out, buf, fn, fill)

        # The result has nnz 0.
        a0 = gen((4, 6), 0)
        for fn in (
            lambda o: torch.mm(a0, b, out=o),
            lambda o: torch.addmm(gen((4, 5), 0), a0, b, out=o),
        ):
            out, buf = _with_strided_values(gen((4, 5), 3), fill)
            _check_nnz_zero_result(self, out, buf, fn)


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
