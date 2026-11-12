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

# Owner(s): ["module: intel"]
# ruff: noqa: F401, F841

import os
import unittest

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfMPS,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    coalescedonoff,
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    subtest,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TEST_XPU,
)

try:
    from xpu_test_utils import ensure_pytorch_test_path, XPUImportCtx
except Exception:
    from .xpu_test_utils import ensure_pytorch_test_path, XPUImportCtx

with XPUImportCtx(False) as patcher:
    from test_sparse import (
        all_sparse_layouts,
        gradcheck_semantics,
        TestSparse,
        TestSparseAny,
        TestSparseLegacyAndDeprecation,
        TestSparseMaskedReductions,
        TestSparseMeta,
        TestSparseOneOff,
        TestSparseUnaryUfuncs,
    )

# XPUImportCtx restores sys.path on exit; keep test/ on it permanently so
# multiprocessing.spawn (Windows) can re-import test_sparse for the DataLoader
# workers spawned by TestSparseAny.test_dataloader.
ensure_pytorch_test_path(os.path.abspath(patcher.test_package[0]))


# ======================================================================
# Missing largeTensorTest decorators
# ======================================================================


TestSparse.test_coalesce_accepts_large_tensor = largeTensorTest("30GB", "xpu")(
    TestSparse.test_coalesce_accepts_large_tensor
)


# ======================================================================
# Method overrides (body changes for device generalization)
# ======================================================================

# TestSparse


# Modify the skipIf - test works on XPU
@unittest.skipIf(
    not TEST_XPU and (IS_LINUX or IS_MACOS or TEST_WITH_ROCM or IS_WINDOWS),
    "https://github.com/pytorch/pytorch/issues/174389",
)
@coalescedonoff
@dtypes(torch.double)
@dtypesIfMPS(torch.float32)
@unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
@gradcheck_semantics()
def _test_sparse_mul(self, device, dtype, coalesced, gradcheck):
    # https://github.com/pytorch/pytorch/issues/79914
    a = (
        torch.tensor([[0.0, 1]], dtype=dtype, device=device)
        .to_sparse()
        .requires_grad_(True)
    )
    b = (
        torch.tensor([[0.0, 1]], dtype=dtype, device=device)
        .to_sparse()
        .requires_grad_(True)
    )
    gradcheck(
        lambda x, y: torch.sparse.sum(x * y).to_dense(masked_grad=gradcheck.masked),
        [a, b],
        check_batched_grad=False,
    )

    def test_shape(sparse_dims, nnz, with_shape):
        a = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[
            0
        ].requires_grad_(True)
        b = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[
            0
        ].requires_grad_(True)

        self.assertEqual((a * b).to_dense(), a.to_dense() * b.to_dense())
        gradcheck(lambda x, y: (x * y).to_dense(), [a, b], check_batched_grad=False)
        # Issues with 0-dim indices/values
        gradcheck(
            lambda x, y: torch.sparse.sum(x * y).to_dense(),
            [a, b],
            masked=True,
            check_batched_grad=False,
        )

    test_shape(2, 3, [2, 3, 4, 5])
    test_shape(2, 3, [2, 2, 0])
    test_shape(2, 3, [4, 5])


TestSparse.test_sparse_mul = _test_sparse_mul


# TestSparseAny


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda/xpu")
@onlyCPU
@all_sparse_layouts("layout", include_strided=True)
def _test_constructor_pin_memory(self, device, layout):
    """Tests sparse_xyz_tensor(indices, values, pin_memory=True)."""
    self.assertEqual(device, "cpu")
    for t in self.generate_simple_inputs(
        layout,
        device=device,
        dtype=torch.float64,
        enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
        pin_memory=True,
    ):
        if layout is torch.sparse_coo:
            self.assertTrue(t._indices().is_pinned())
            self.assertTrue(t._values().is_pinned())
        elif layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.assertTrue(t.crow_indices().is_pinned())
            self.assertTrue(t.col_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
        elif layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.assertTrue(t.ccol_indices().is_pinned())
            self.assertTrue(t.row_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
        elif layout is torch.strided:
            pass
        else:
            raise AssertionError(f"unreachable: layout={layout}")
        self.assertTrue(t.is_pinned())


TestSparseAny.test_constructor_pin_memory = _test_constructor_pin_memory


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda/xpu")
@onlyCPU
@all_sparse_layouts("layout", include_strided=True)
def _test_method_pin_memory(self, device, layout):
    """Tests sparse_xyz_tensor(indices, values, pin_memory=False).pin_memory()."""

    for t_ in self.generate_simple_inputs(
        layout,
        device=device,
        dtype=torch.float64,
        enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
        pin_memory=False,  # no pinning
    ):
        t = t_.pin_memory()
        self.assertTrue(t.is_pinned())

        # registering a non-pinned tensor with CUDA/XPU memory is a clone operation
        self.assertFalse(t_.is_pinned())

        # registering already pinned tensor with CUDA/XPU memory is an identity operation
        t2 = t.pin_memory()
        self.assertTrue(t2 is t)

        if layout is torch.sparse_coo:
            self.assertTrue(t._indices().is_pinned())
            self.assertTrue(t._values().is_pinned())
            self.assertFalse(t_._indices().is_pinned())
            self.assertFalse(t_._values().is_pinned())
        elif layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.assertTrue(t.crow_indices().is_pinned())
            self.assertTrue(t.col_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
            self.assertFalse(t_.crow_indices().is_pinned())
            self.assertFalse(t_.col_indices().is_pinned())
            self.assertFalse(t_.values().is_pinned())
        elif layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.assertTrue(t.ccol_indices().is_pinned())
            self.assertTrue(t.row_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
            self.assertFalse(t_.ccol_indices().is_pinned())
            self.assertFalse(t_.row_indices().is_pinned())
            self.assertFalse(t_.values().is_pinned())
        elif layout is torch.strided:
            pass
        else:
            raise AssertionError(f"unreachable: layout={layout}")


TestSparseAny.test_method_pin_memory = _test_method_pin_memory


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda/xpu")
@onlyCPU
@all_sparse_layouts("layout", include_strided=True)
def _test_constructor_pinned_memory(self, device, layout):
    """Tests sparse_xyz_tensor(indices.pin_memory(device), values.pin_memory(device))."""
    for t in self.generate_simple_inputs(
        layout,
        device=device,
        dtype=torch.float64,
        enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
        pin_memory=None,  # constructor does not specify pin_memory=...
        members_pin_memory=True,  # indices and values are pinned
    ):
        if layout is torch.sparse_coo:
            self.assertTrue(t._indices().is_pinned())
            self.assertTrue(t._values().is_pinned())
        elif layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.assertTrue(t.crow_indices().is_pinned())
            self.assertTrue(t.col_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
        elif layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.assertTrue(t.ccol_indices().is_pinned())
            self.assertTrue(t.row_indices().is_pinned())
            self.assertTrue(t.values().is_pinned())
        elif layout is torch.strided:
            pass
        else:
            raise AssertionError(f"unreachable: layout={layout}")
        self.assertTrue(t.is_pinned())


TestSparseAny.test_constructor_pinned_memory = _test_constructor_pinned_memory


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda/xpu")
@onlyCPU
@all_sparse_layouts("layout", include_strided=False)
def _test_constructor_mismatched_pinned_memory(self, device, layout):
    """Test failure for constructing sparse tensors from mismatched pinned inputs."""

    def generic_constructor(*args, **kwargs):
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            kwargs.update(layout=layout)
            return torch.sparse_compressed_tensor(*args, **kwargs)
        if layout is torch.sparse_coo:
            return torch.sparse_coo_tensor(*args, **kwargs)
        raise NotImplementedError(layout)

    for args, kwargs in self.generate_simple_inputs(
        layout,
        device=device,
        dtype=torch.float64,
        enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
        output_tensor=False,
    ):
        # indices are pinned while values are not pinned
        args1 = (args[0].pin_memory(), *args[1:])

        # indices are not pinned while values are pinned
        args2 = (*args[:-1], args[-1].pin_memory())

        with self.assertRaisesRegex(
            RuntimeError,
            r"memory pinning of \w*indices \(=1\) must match memory pinning of values \(=0\)",
        ):
            generic_constructor(*args1, **kwargs)

        with self.assertRaisesRegex(
            RuntimeError,
            r"memory pinning of \w*indices \(=0\) must match memory pinning of values \(=1\)",
        ):
            generic_constructor(*args2, **kwargs)


TestSparseAny.test_constructor_mismatched_pinned_memory = (
    _test_constructor_mismatched_pinned_memory
)


# Widen regex in assertRaisesRegex to include XPU.
# NOTE: `self.skipTest('NOT IMPL')` were removed as the `assertRaisesRegex`
# passes on XPU. Need to evaluate if skips are needed in upstream anyway.
@onlyNativeDeviceTypes
@all_sparse_layouts("layout", include_strided=not True)
@dtypes(torch.float64, torch.cdouble)
@parametrize("masked", [subtest(False, name="sparse"), subtest(True, name="masked")])
@parametrize("fast_mode", [subtest(False, name="slow"), subtest(True, name="fast")])
def _test_gradcheck_mm(self, layout, dtype, device, masked, fast_mode):
    # This function does not check the following cases:
    # - batch or hybrid tensors because addmm does not support
    #   such inputs yet
    # - check_forward_ad=True because of the lack of sparse tensor
    #   support in aten::view_as_real, torch._VF._make_dual, etc.

    ref_x = torch.tensor(
        [[1, 2, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0], [13, 14, 0, 15]],
        dtype=dtype,
        device=device,
    )
    ref_y = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]],
        dtype=dtype,
        device=device,
    )

    mm = torch.sparse.mm if masked else torch.mm

    blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
    x = ref_x.to_sparse(layout=layout, blocksize=blocksize).requires_grad_(True)
    y = ref_y.requires_grad_(True)

    if layout is torch.sparse_bsr and not masked or layout is torch.sparse_bsc:
        with self.assertRaisesRegex(
            RuntimeError,
            r"addmm: computation on (CPU|CUDA|XPU) is not implemented for Strided \+ Sparse(Bsr|Bsc) @ Strided",
        ):
            torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
    elif layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and masked:
        with self.assertRaisesRegex(
            RuntimeError,
            r"(sparse_addmm_sparse_backward: unsupported combination of layouts,"
            r" grad: Strided, mat1: Sparse(Csc|Bsr|Bsc), mat2: Strided"
            r"|addmm: computation on (CPU|CUDA|XPU) is not implemented for "
            r"Strided \+ Sparse(Csc|Bsr|Bsc) @ Strided without MKL)",
        ):
            torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
    else:
        torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)


TestSparseAny.test_gradcheck_mm = _test_gradcheck_mm


# ======================================================================
# New XPU-only tests
# ======================================================================


@onlyOn("xpu")
@coalescedonoff
@dtypes(torch.double)
def _test_hspmm_out(self, device, dtype, coalesced):
    x = self._gen_sparse(2, 20, [7, 5], dtype, device, coalesced)[0]
    y = self.randn(5, 3, dtype=dtype, device=device)
    out = torch.empty(0, dtype=dtype, device=device).to_sparse()

    result = torch.ops.aten.hspmm.out(x, y, out=out)
    expected = torch.mm(self.safeToDense(x), y)

    self.assertIs(result, out)
    self.assertEqual(result.to_dense(), expected)


TestSparse.test_hspmm_out = _test_hspmm_out


@onlyOn("xpu")
@dtypes(torch.double)
def _test_hspmm_out_errors(self, device, dtype):
    x = self._gen_sparse(2, 10, [4, 3], dtype, device, coalesced=True)[0]
    out = torch.empty(0, dtype=dtype, device=device).to_sparse()

    with self.assertRaisesRegex(RuntimeError, "Expected dim 0 size 3"):
        torch.ops.aten.hspmm.out(
            x, self.randn(2, 5, dtype=dtype, device=device), out=out
        )

    with self.assertRaisesRegex(
        RuntimeError,
        "expected 'mat2' to be XPU|mat2 is on cpu",
    ):
        torch.ops.aten.hspmm.out(
            x, self.randn(3, 5, dtype=dtype, device="cpu"), out=out
        )

    with self.assertRaisesRegex(
        RuntimeError,
        "expected 'out' to be XPU|out is on cpu|different from other tensors on cpu",
    ):
        torch.ops.aten.hspmm.out(
            x,
            self.randn(3, 5, dtype=dtype, device=device),
            out=torch.empty(0, dtype=dtype, device="cpu").to_sparse(),
        )


TestSparse.test_hspmm_out_errors = _test_hspmm_out_errors


# ======================================================================
# Instantiate tests for XPU
# ======================================================================

instantiate_device_type_tests(
    TestSparseUnaryUfuncs, globals(), allow_mps=True, allow_xpu=True, except_for="meta"
)

instantiate_device_type_tests(
    TestSparseMaskedReductions, globals(), allow_xpu=True, except_for="meta"
)

instantiate_device_type_tests(
    TestSparse, globals(), allow_mps=True, allow_xpu=True, except_for="meta"
)

instantiate_device_type_tests(
    TestSparseAny, globals(), allow_xpu=True, except_for="meta"
)

instantiate_device_type_tests(
    TestSparseOneOff, globals(), only_for="xpu", allow_xpu=True
)

instantiate_parametrized_tests(TestSparseMeta)

instantiate_parametrized_tests(TestSparseLegacyAndDeprecation)


if __name__ == "__main__":
    run_tests()
