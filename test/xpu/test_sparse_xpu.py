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

import itertools
import os
import unittest

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfMPS,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyOn,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import (
    coalescedonoff,
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    subtest,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_XPU,
)

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
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


# ======================================================================
# Module-level XPU additions
# ======================================================================

TEST_MULTIACCELERATOR = torch.accelerator.device_count() >= 2

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


# ======================================================================
# Class-level utility patches
# ======================================================================

# Local expect-file fallback for XPU print tests.
# When upstreaming, copy over:
# - TestSparseXPU.test_print_coalesced_xpu_float64.expect
# - TestSparseXPU.test_print_uncoalesced_xpu_float64.expect
_orig_assert_expected = TestSparse.assertExpected


def _assert_expected_with_local_xpu_fallback(self, output, *args, **kwargs):
    if self.__class__.__name__ == "TestSparseXPU" and self._testMethodName in {
        "test_print_coalesced_xpu_float64",
        "test_print_uncoalesced_xpu_float64",
    }:
        local_expect = os.path.join(
            os.path.dirname(__file__),
            "expect",
            f"{self.__class__.__name__}.{self._testMethodName}.expect",
        )
        with open(local_expect) as f:
            expected = f.read()
        self.assertEqual(output, expected)
        return

    return _orig_assert_expected(self, output, *args, **kwargs)


TestSparse.assertExpected = _assert_expected_with_local_xpu_fallback


# ======================================================================
# Retargeting @onlyCUDA -> @onlyOn(["cuda", "xpu"])  (no body change)
# ======================================================================


TestSparse.test_cuda_empty = retarget_outermost_onlycuda_to_onlyon(
    TestSparse.test_cuda_empty
)

TestSparse.test_storage_not_null = retarget_outermost_onlycuda_to_onlyon(
    TestSparse.test_storage_not_null
)

TestSparse.test_coalesce_accepts_large_tensor = largeTensorTest("30GB", "xpu")(
    retarget_outermost_onlycuda_to_onlyon(TestSparse.test_coalesce_accepts_large_tensor)
)


# ======================================================================
# Decorator additions (no body change)
# ======================================================================

TestSparse.test_bmm = unittest.skipIf(
    IS_WINDOWS and TEST_XPU, "bmm sparse-dense XPU is not yet supported"
)(TestSparse.test_bmm)

TestSparse.test_sparse_matmul = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16, torch.complex64, torch.complex128)
)(TestSparse.test_sparse_matmul)

TestSparse.test_sparse_addmm = toleranceOverride(
    {torch.double: tol(atol=2e-6, rtol=1e-6)}
)(TestSparse.test_sparse_addmm)


# ======================================================================
# Method overrides (body changes for device generalization)
# ======================================================================

# TestSparse


@coalescedonoff
@unittest.skipIf(not TEST_MULTIACCELERATOR, "multi-GPU not supported")
@dtypes(torch.double, torch.cdouble)
def _test_Sparse_to_Sparse_copy_multi_gpu(self, device, dtype, coalesced):
    # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
    device_type = torch.device(device).type

    sparse_dims = 3
    nnz = 10
    sizes = [2, 3, 4, 5]  # hybrid sparse
    x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
    x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
    x1 = x1.to(f"{device_type}:0")

    def test_cross_device(x1, x2):
        x1_device = x1.device
        x1.copy_(x2)
        self.assertEqual(x2.to(f"{device_type}:0").to_dense(), x1.to_dense())
        self.assertEqual(x1_device, x1.device)

    test_cross_device(x1, x2.to(f"{device_type}:1"))  # test across gpu devices
    test_cross_device(x1, x2.to("cpu"))  # test between cpu and gpu

    # test autograd
    x2 = x2.to(f"{device_type}:1")
    x2.requires_grad_(True)
    x1.copy_(x2)
    y = x1 * 2
    x2_clone = x2.clone().to(f"{device_type}:0")
    y.backward(x2_clone)
    expected_grad = x2_clone * 2
    self.assertEqual(
        expected_grad.to_dense(), x2.grad.to(f"{device_type}:0").to_dense()
    )
    self.assertEqual(None, x1.grad)


TestSparse.test_Sparse_to_Sparse_copy_multi_gpu = _test_Sparse_to_Sparse_copy_multi_gpu


def _test_new_device(self, size, device, gpu_id=0):
    device_type = torch.device(device).type
    with torch.get_device_module(device_type).device(gpu_id):
        x = torch.sparse_coo_tensor(size, device=device_type, dtype=torch.float64)
    self.assertEqual(x.get_device(), gpu_id)
    x1 = x.new()
    x2 = x.new(2, 3)
    self.assertEqual(x1.get_device(), gpu_id)
    self.assertEqual(x2.get_device(), gpu_id)


TestSparse._test_new_device = _test_new_device


@onlyOn(["cuda", "xpu"])
def _test_new_device_single_gpu(self, device):
    self._test_new_device((), device, 0)
    self._test_new_device((30, 20), device, 0)
    self._test_new_device((30, 20, 10), device, 0)
    self._test_new_device((30, 20, 10, 0), device, 0)


TestSparse.test_new_device_single_gpu = _test_new_device_single_gpu


@onlyOn(["cuda", "xpu"])
@unittest.skipIf(not TEST_MULTIACCELERATOR, "only one GPU detected")
def _test_new_device_multi_gpu(self, device):
    self._test_new_device((), device, 1)
    self._test_new_device((30, 20), device, 1)
    self._test_new_device((30, 20, 10), device, 1)
    self._test_new_device((30, 20, 10, 0), device, 1)


TestSparse.test_new_device_multi_gpu = _test_new_device_multi_gpu


@onlyCPU  # not really, but we only really want to run this once
@dtypes(torch.float64, torch.float32, torch.float16, torch.cfloat, torch.cdouble)
def _test_factory(self, device, dtype):
    for test_empty_tensor in [True, False]:
        if test_empty_tensor:
            default_size = torch.Size([1, 3, 0])
            size = torch.Size([3, 3, 0])
        else:
            default_size = torch.Size([1, 3])
            size = torch.Size([3, 3])
        for include_size in [True, False]:
            for use_tensor_idx in [True, False]:
                for use_tensor_val in [True, False]:
                    for use_accelerator in (
                        [False]
                        if not torch.accelerator.is_available()
                        else [True, False]
                    ):
                        # have to include size with cuda sparse tensors
                        include_size = include_size or use_accelerator
                        long_dtype = torch.int64
                        device = (
                            torch.device("cpu")
                            if not use_accelerator
                            else torch.device(torch.accelerator.device_count() - 1)
                        )
                        indices = (
                            torch.tensor(([0], [2]), dtype=long_dtype)
                            if use_tensor_idx
                            else ([0], [2])
                        )
                        if test_empty_tensor:
                            values = torch.empty(1, 0).to(dtype)
                        else:
                            if use_tensor_val:
                                values = torch.tensor([1.0], dtype=dtype)
                            else:
                                values = 1.0
                        if include_size:
                            sparse_tensor = torch.sparse_coo_tensor(
                                indices,
                                values,
                                size,
                                dtype=dtype,
                                device=device,
                                requires_grad=True,
                            )
                        else:
                            sparse_tensor = torch.sparse_coo_tensor(
                                indices,
                                values,
                                dtype=dtype,
                                device=device,
                                requires_grad=True,
                            )
                        self.assertEqual(indices, sparse_tensor._indices())
                        self.assertEqual(values, sparse_tensor._values())
                        self.assertEqual(
                            size if include_size else default_size, sparse_tensor.size()
                        )
                        self.assertEqual(dtype, sparse_tensor.dtype)
                        if use_accelerator:
                            self.assertEqual(device, sparse_tensor._values().device)
                        self.assertEqual(True, sparse_tensor.requires_grad)


TestSparse.test_factory = _test_factory


@onlyOn(["cuda", "xpu"])
def _test_factory_device_type_inference(self, device):
    # both indices/values are CUDA/XPU
    device_type = torch.device(device).type

    cpu_gpu = ("cpu", device_type)
    cpu_gpu_none = cpu_gpu + (None,)
    for indices_device, values_device, _device in itertools.product(
        cpu_gpu, cpu_gpu, cpu_gpu_none
    ):
        indices = torch.tensor(([0], [2]), device=indices_device)
        values = torch.tensor([1.0], device=values_device)
        empty_values = torch.empty(1, 0).to(values_device)
        shape = (1, 3)
        empty_shape = (1, 3, 0)
        if _device is None and indices_device != values_device:
            with self.assertRaises(RuntimeError):
                torch.sparse_coo_tensor(indices, values, shape, device=_device)
            with self.assertRaises(RuntimeError):
                torch.sparse_coo_tensor(
                    indices, empty_values, empty_shape, device=_device
                )
        else:
            t = torch.sparse_coo_tensor(indices, values, shape, device=_device)
            t_empty = torch.sparse_coo_tensor(
                indices, empty_values, empty_shape, device=_device
            )
            should_be_gpu = _device == device_type or (
                _device is None and values_device == device_type
            )
            self.assertEqual(should_be_gpu, t.device.type == device_type)
            self.assertEqual(
                t.device.type == device_type, t_empty.device.type == device_type
            )


TestSparse.test_factory_device_type_inference = _test_factory_device_type_inference


@onlyOn(["cuda", "xpu"])
def _test_legacy_new_device(self, device):
    device_type = torch.device(device).type

    i = torch.tensor([[0, 1, 1], [2, 0, 2]])
    v = torch.tensor([3.0, 4.0, 5.0])
    size = torch.Size([2, 3])

    x = torch.sparse_coo_tensor(i, v, size, device="cpu")
    self.assertRaises(RuntimeError, lambda: x.new(device=device_type))
    self.assertRaises(RuntimeError, lambda: x.new(i, v, device=device_type))
    self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device=device_type))
    self.assertRaises(
        RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device=device_type)
    )

    x = torch.sparse_coo_tensor(i, v, size, device=device_type)
    self.assertRaises(RuntimeError, lambda: x.new(device="cpu"))
    self.assertRaises(RuntimeError, lambda: x.new(i, v, device="cpu"))
    self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device="cpu"))
    self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device="cpu"))


TestSparse.test_legacy_new_device = _test_legacy_new_device


@onlyCPU  # not really, but we only really want to run this once
def _test_dtypes(self, device):
    from torch.testing._internal.common_dtype import all_types_and_complex_and
    from torch.testing._internal.common_utils import do_test_dtypes

    all_sparse_dtypes = all_types_and_complex_and(
        torch.half, torch.bool, torch.bfloat16
    )
    do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device("cpu"))
    if TEST_CUDA or TEST_XPU:
        do_test_dtypes(
            self,
            all_sparse_dtypes,
            torch.sparse_coo,
            torch.device(f"{device_type}:0"),
        )


TestSparse.test_dtypes = _test_dtypes


@onlyOn(["cuda", "xpu"])
@coalescedonoff
@dtypes(torch.double)
@unittest.skipIf(
    IS_WINDOWS,
    "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1",
)
def _test_bmm_deterministic(self, device, dtype, coalesced):
    device_type = torch.device(device).type

    def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
        a_list = []
        b_list = []
        for _ in range(num_mats):
            a_list.append(
                self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0]
            )
            b_list.append(torch.randn([dim_j, dim_k], dtype=dtype, device=device))

        a = torch.stack(a_list).to(device_type)
        b = torch.stack(b_list).to(device_type)
        with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
            torch.use_deterministic_algorithms(False)
            ab_nondeterministic = torch.bmm(a, b)
            torch.use_deterministic_algorithms(True)
            ab_deterministic = torch.bmm(a, b)
        diff_abs = (ab_deterministic - ab_nondeterministic).abs()
        diff_rel = diff_abs / ab_deterministic.abs()
        diff_rel[torch.isnan(diff_rel)] = 0

        # deterministic and non-deterministic results should either be
        # equal or within a small relative difference
        equal_abs_or_rel = diff_abs.eq(0).logical_or(diff_rel.lt(0.001))
        self.assertTrue(equal_abs_or_rel.all())

    test_shape(10, 10, 100, 99, 20)
    test_shape(10, 100, 1000, 200, 20)
    test_shape(10, 64, 10000, 300, 20)
    test_shape(10, 0, 100, 99, 0)
    test_shape(10, 10, 0, 100, 0)
    test_shape(10, 10, 100, 0, 0)
    test_shape(10, 10, 100, 0, 20)
    test_shape(10, 10, 100, 0, 20)


TestSparse.test_bmm_deterministic = _test_bmm_deterministic


@onlyOn(["cuda", "xpu"])
@unittest.skipIf(
    IS_WINDOWS and TEST_CUDA,
    "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1",
)
@unittest.skipIf(IS_WINDOWS and TEST_XPU, "bmm sparse-dense XPU is not yet supported")
def _test_bmm_oob(self, device):
    # Targets an out of bounds error when the sparse tensor has no non-zero
    # values in the first batch dimension (#131977).
    torch.accelerator.empty_cache()
    indices = torch.tensor([[1], [0], [0]], device=device)
    values = torch.tensor([1.0], device=device)
    a = torch.sparse_coo_tensor(indices, values, size=(2, 1, 1))
    b = torch.zeros((2, 1, 1), device=device)
    ab = torch.bmm(a, b)
    self.assertEqual(ab, torch.zeros((2, 1, 1), device=device))


TestSparse.test_bmm_oob = _test_bmm_oob


# Strip outermost skipIf decorator (test passes on XPU).
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


# Widen backend regex to include XPU.
def _test_empty_like(self, sparse_tensor, dtype, device, coalesced):
    result = torch.empty_like(sparse_tensor)
    self.assertTrue(result.is_sparse)
    self._assert_sparse_invars(result)
    self.assertEqual(result.shape, sparse_tensor.shape)
    self.assertEqual(result.dtype, sparse_tensor.dtype)
    self.assertEqual(result.device, sparse_tensor.device)
    self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
    self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

    sparse_tensor, _, _ = self._gen_sparse(
        len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced
    )
    data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
    mem_formats = [
        torch.channels_last,
        torch.contiguous_format,
        torch.preserve_format,
        torch.channels_last_3d,
    ]
    for x, mem_format in zip(data, mem_formats):
        with self.assertRaisesRegex(
            RuntimeError, "memory format option is only supported by strided tensors"
        ):
            result = torch.empty_like(x, memory_format=mem_format)

        result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
        self.assertTrue(result.layout == torch.strided)

    with self.assertRaisesRegex(
        RuntimeError,
        r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA|MPS|XPU)' backend",
    ):
        dense_tensor = sparse_tensor.to_dense()
        result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)


TestSparse._test_empty_like = _test_empty_like


# TestSparseOneOff


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA/XPU not available")
def _test_cuda_from_cpu(self):
    with self.assertRaisesRegex(
        RuntimeError, "Expected all tensors to be on the same device"
    ):
        torch.sparse_coo_tensor(
            torch.zeros(1, 4).long().to(device_type),
            torch.randn(4, 4, 4),
            [3, 4, 4],
        )

    with self.assertRaisesRegex(
        RuntimeError, "Expected all tensors to be on the same device"
    ):
        torch.sparse_coo_tensor(
            torch.zeros(1, 4).long().to(device_type),
            torch.randn(4, 4, 4, 0),
            [3, 4, 4, 0],
        )

    with self.assertRaisesRegex(
        RuntimeError, "Expected all tensors to be on the same device"
    ):
        torch.sparse_coo_tensor(
            torch.empty(1, 0).long().to(device_type),
            torch.randn(0, 4, 4, 0),
            [0, 4, 4, 0],
        )


TestSparseOneOff.test_cuda_from_cpu = _test_cuda_from_cpu


@unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA/XPU not available")
def _test_cuda_sparse_cpu_dense_add(self):
    x = torch.zeros(3, 4, 4)
    sparse_y = torch.sparse_coo_tensor(
        torch.zeros(1, 4).long().to(device_type),
        torch.randn(4, 4, 4).to(device_type),
        [3, 4, 4],
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "add: expected 'self' to be a "
        + device_type.upper()
        + " tensor, but got a CPU tensor",
    ):
        x + sparse_y

    x = torch.zeros(3, 4, 4, 0)
    sparse_y = torch.sparse_coo_tensor(
        torch.zeros(1, 4).long().to(device_type),
        torch.randn(4, 4, 4, 0).to(device_type),
        [3, 4, 4, 0],
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "add: expected 'self' to be a "
        + device_type.upper()
        + " tensor, but got a CPU tensor",
    ):
        x + sparse_y

    x = torch.zeros(0, 4, 4, 0)
    sparse_y = torch.sparse_coo_tensor(
        torch.empty(1, 0).long().to(device_type),
        torch.randn(0, 4, 4, 0).to(device_type),
        [0, 4, 4, 0],
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "add: expected 'self' to be a "
        + device_type.upper()
        + " tensor, but got a CPU tensor",
    ):
        x + sparse_y


TestSparseOneOff.test_cuda_sparse_cpu_dense_add = _test_cuda_sparse_cpu_dense_add


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
@unittest.skipIf(
    not IS_WINDOWS,
    "Windows-specific error check; skipping on non-Windows",
)
@dtypes(torch.double)
def _test_bmm_windows_error(self, device, dtype):
    self.assertTrue(device.startswith("xpu"))
    a = torch.rand(2, 2, 2, dtype=dtype).to_sparse().to(device)
    b = torch.rand(2, 2, 2, dtype=dtype).to(device)
    # XPU supports sparse-dense bmm; verify result matches dense reference
    ab = a.bmm(b)
    self.assertEqual(ab, torch.bmm(a.to_dense(), b))


TestSparse.test_bmm_windows_error = _test_bmm_windows_error


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

instantiate_parametrized_tests(TestSparseMeta)

instantiate_parametrized_tests(TestSparseLegacyAndDeprecation)


if __name__ == "__main__":
    run_tests()
