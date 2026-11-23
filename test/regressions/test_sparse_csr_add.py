# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import unittest

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _make_csr(values):
    return torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2], device="xpu"),
        torch.tensor([0, 1], device="xpu"),
        values,
        size=(2, 2),
        device="xpu",
    )


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestSparseCsrAdd(TestCase):
    def test_strided_inputs_with_sparse_out_raise(self):
        dense = torch.ones((2, 2), device="xpu")
        out = _make_csr(torch.zeros(2, device="xpu"))

        for self_input, other_input in ((dense, dense), (dense, out), (out, dense)):
            with self.assertRaisesRegex(RuntimeError, "out.*strided"):
                torch.add(self_input, other_input, out=out)

    def test_sparse_and_dense_inputs_with_strided_out(self):
        dense = torch.ones((2, 2), device="xpu")
        sparse = _make_csr(torch.full((2,), 2.0, device="xpu"))
        expected = torch.tensor([[3.0, 1.0], [1.0, 3.0]])

        for self_input, other_input in ((sparse, dense), (dense, sparse)):
            out = torch.empty((2, 2), device="xpu")
            torch.add(self_input, other_input, out=out)
            self.assertEqual(out.cpu(), expected)

    def test_sparse_inputs_with_sparse_out(self):
        first = _make_csr(torch.ones(2, device="xpu"))
        second = _make_csr(torch.full((2,), 2.0, device="xpu"))
        out = _make_csr(torch.zeros(2, device="xpu"))

        torch.add(first, second, out=out)

        self.assertEqual(out.to_dense().cpu(), torch.tensor([[3.0, 0.0], [0.0, 3.0]]))


class TestSparseCompressedMixedAdd(TestCase):
    @dtypes(
        torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.complex64
    )
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("batch_shape", [(), (2,), (2, 3)])
    @parametrize("reverse", [False, True])
    @parametrize("noncontiguous", [False, True])
    @parametrize("alpha", [0, 1, 2, -1])
    def test_mixed_add(
        self, device, dtype, layout, batch_shape, reverse, noncontiguous, alpha
    ):
        dense_cpu = (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            .expand(*batch_shape, 2, 3)
            .clone()
        )
        sparse_cpu = (
            torch.tensor([[2, 0, 1], [0, 3, 0]], dtype=dtype)
            .expand(*batch_shape, 2, 3)
            .clone()
        )
        dense = dense_cpu.to(device)
        if noncontiguous:
            dense = dense.transpose(-1, -2).contiguous().transpose(-1, -2)
        sparse = sparse_cpu.to_sparse(layout=layout).to(device)
        inputs = (sparse, dense) if reverse else (dense, sparse)
        out = torch.empty(dense.shape, dtype=dtype, device=device)

        returned = torch.add(*inputs, alpha=alpha, out=out)

        self.assertIs(returned, out)
        self.assertEqual(out.cpu(), dense_cpu + alpha * sparse_cpu)
        self.assertEqual(dense.cpu(), dense_cpu)

    @dtypes(torch.float32, torch.float64)
    @parametrize("output_dtype", [torch.float32, torch.float64])
    @parametrize("empty", [False, True])
    def test_output_dtype(self, device, dtype, output_dtype, empty):
        dense_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        sparse_cpu = (
            torch.zeros((2, 3))
            if empty
            else torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
        )
        dense = dense_cpu.to(device)
        sparse = sparse_cpu.to_sparse_csr().to(device)
        out = torch.empty((2, 3), dtype=output_dtype, device=device)
        common_dtype = torch.promote_types(dtype, sparse_cpu.dtype)
        expected = (dense_cpu.to(common_dtype) + 2 * sparse_cpu.to(common_dtype)).to(
            output_dtype
        )

        torch.add(dense, sparse, alpha=2, out=out)

        self.assertEqual(out.cpu(), expected)
        self.assertEqual(dense.cpu(), dense_cpu)

    @dtypes(torch.float16, torch.bfloat16, torch.complex32)
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("reverse", [False, True])
    @parametrize("case", ["large_alpha", "cancellation", "large_product"])
    @parametrize("output", ["functional", "out", "inplace", "converted"])
    def test_low_precision_opmath(self, device, dtype, layout, reverse, case, output):
        if case == "large_alpha":
            dense_value, sparse_value, alpha = 1, 0.001, 70000
        elif case == "cancellation":
            dense_value, sparse_value = 1, -10
            alpha = 0.1001 if dtype == torch.bfloat16 else 0.10001
            if dtype.is_complex:
                dense_value, alpha = 1 + 1j, alpha * (1 + 1j)
        else:
            value = 3.375e38 if dtype == torch.bfloat16 else 40000
            alpha = 1.0078125 if dtype == torch.bfloat16 else 2
            dense_value, sparse_value = -value, value
        dense_cpu = torch.full((1, 1), dense_value, dtype=dtype)
        values_cpu = torch.tensor([sparse_value], dtype=dtype)
        oracle_dtype = torch.complex128 if dtype.is_complex else torch.float64
        expected = (
            dense_cpu.to(oracle_dtype) + alpha * values_cpu.to(oracle_dtype)
        ).to(dtype)
        dense = dense_cpu.to(device)
        sparse = torch.sparse_compressed_tensor(
            torch.tensor([0, 1], device=device),
            torch.tensor([0], device=device),
            values_cpu.to(device),
            size=(1, 1),
            layout=layout,
            device=device,
            check_invariants=True,
        )
        inputs = (sparse, dense) if reverse else (dense, sparse)
        if output == "functional":
            actual = torch.add(*inputs, alpha=alpha)
        else:
            output_dtype = dtype
            if output == "converted":
                output_dtype = torch.complex64 if dtype.is_complex else torch.float32
            out = (
                dense
                if output == "inplace"
                else torch.empty((1, 1), dtype=output_dtype, device=device)
            )
            actual = torch.add(*inputs, alpha=alpha, out=out)
            self.assertIs(actual, out)
        self.assertEqual(actual.cpu().to(oracle_dtype), expected.to(oracle_dtype))
        if output != "inplace":
            self.assertEqual(dense.cpu().to(oracle_dtype), dense_cpu.to(oracle_dtype))

    @dtypes(torch.float32)
    def test_mixed_layout_checks(self, device, dtype):
        dense = torch.ones((2, 3), dtype=dtype, device=device)
        sparse = (
            torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]]).to_sparse_csr().to(device)
        )
        out = torch.empty_like(dense)

        with self.assertRaisesRegex(RuntimeError, "broadcasting"):
            torch.add(dense[:1], sparse, out=out)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch.add(dense, sparse, out=torch.empty((3, 2), device=device).t())
        with self.assertRaisesRegex(RuntimeError, "convert result type"):
            torch.add(
                dense, sparse, out=torch.empty((2, 3), dtype=torch.int32, device=device)
            )

    @dtypes(torch.float32)
    def test_out_is_dense(self, device, dtype):
        dense = torch.ones((2, 3), dtype=dtype, device=device)
        sparse = (
            torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]]).to_sparse_csr().to(device)
        )

        torch.add(dense, sparse, alpha=2, out=dense)

        self.assertEqual(dense.cpu(), torch.tensor([[5.0, 1.0, 3.0], [1.0, 7.0, 1.0]]))

    @dtypes(torch.float32, torch.int64)
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("reverse", [False, True])
    @parametrize("inplace", [False, True])
    def test_out_aliases_sparse_values(self, device, dtype, layout, reverse, inplace):
        out = torch.tensor([[2, 3], [0, 0]], dtype=dtype, device=device)
        sparse = torch.sparse_compressed_tensor(
            torch.tensor([0, 1, 2], device=device),
            torch.tensor([0, 1], device=device),
            out.reshape(-1)[:2],
            size=(2, 2),
            layout=layout,
            device=device,
            check_invariants=True,
        )
        self.assertEqual(
            out.untyped_storage().data_ptr(),
            sparse.values().untyped_storage().data_ptr(),
        )
        dense = out if inplace else torch.ones_like(out)
        expected = dense.cpu().clone() + 2 * torch.diag(
            torch.tensor([2, 3], dtype=dtype)
        )
        inputs = (sparse, dense) if reverse else (dense, sparse)

        returned = torch.add(*inputs, alpha=2, out=out)

        self.assertIs(returned, out)
        self.assertEqual(out.cpu(), expected)

    @dtypes(torch.int64)
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("reverse", [False, True])
    @parametrize("component", ["compressed", "plain"])
    def test_out_aliases_sparse_indices(
        self, device, dtype, layout, reverse, component
    ):
        out = torch.tensor([[0, 1], [2, 0]], dtype=dtype, device=device)
        compressed = torch.tensor([0, 1, 2], device=device)
        plain = torch.tensor([0, 1], device=device)
        if component == "compressed":
            compressed = out.reshape(-1)[:3]
        else:
            plain = out.reshape(-1)[:2]
        sparse = torch.sparse_compressed_tensor(
            compressed,
            plain,
            torch.tensor([2, 3], dtype=dtype, device=device),
            size=(2, 2),
            layout=layout,
            device=device,
            check_invariants=True,
        )
        if component == "compressed":
            aliased_indices = (
                sparse.crow_indices()
                if layout == torch.sparse_csr
                else sparse.ccol_indices()
            )
        else:
            aliased_indices = (
                sparse.col_indices()
                if layout == torch.sparse_csr
                else sparse.row_indices()
            )
        self.assertEqual(
            out.untyped_storage().data_ptr(),
            aliased_indices.untyped_storage().data_ptr(),
        )
        dense = torch.ones_like(out)
        inputs = (sparse, dense) if reverse else (dense, sparse)

        torch.add(*inputs, alpha=2, out=out)

        self.assertEqual(out.cpu(), torch.tensor([[5, 1], [1, 7]], dtype=dtype))

    @dtypes(torch.int64)
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("reverse", [False, True])
    @parametrize("component", ["values", "compressed", "plain"])
    def test_out_is_sparse_constructor_argument(
        self, device, dtype, layout, reverse, component
    ):
        compressed = torch.tensor([0, 1, 2], device=device)
        plain = torch.tensor([0, 1], device=device)
        values = torch.tensor([2, 3], dtype=dtype, device=device)
        out = {"compressed": compressed, "plain": plain, "values": values}[component]
        sparse = torch.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            size=(2, 2),
            layout=layout,
            device=device,
            check_invariants=True,
        )
        dense = torch.ones((2, 2), dtype=dtype, device=device)
        inputs = (sparse, dense) if reverse else (dense, sparse)

        with self.assertWarnsRegex(UserWarning, "output.*resized"):
            returned = torch.add(*inputs, alpha=2, out=out)

        self.assertIs(returned, out)
        self.assertEqual(out.cpu(), torch.tensor([[5, 1], [1, 7]], dtype=dtype))
        self.assertEqual(dense.cpu(), torch.ones((2, 2), dtype=dtype))

    @dtypes(torch.int32, torch.float32)
    @parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
    @parametrize("reverse", [False, True])
    @parametrize("empty", [False, True])
    def test_invalid_alpha(self, device, dtype, layout, reverse, empty):
        dense = torch.ones((2, 2), dtype=dtype, device=device)
        sparse_dense = torch.zeros_like(dense) if empty else dense
        sparse = sparse_dense.to_sparse(layout=layout)
        inputs = (sparse, dense) if reverse else (dense, sparse)
        invalid_alphas = [(True, "Boolean alpha"), (1j, "integral input|non-complex")]
        if dtype == torch.int32:
            invalid_alphas.append((0.5, "integral input"))
        for alpha, error in invalid_alphas:
            out = torch.full((1,), 17, dtype=dtype, device=device)
            original = out.clone()
            with self.assertRaisesRegex(RuntimeError, error):
                torch.add(*inputs, alpha=alpha, out=out)
            self.assertEqual(out, original)


instantiate_device_type_tests(
    TestSparseCompressedMixedAdd, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
