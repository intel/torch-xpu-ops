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


instantiate_device_type_tests(
    TestSparseCompressedMixedAdd, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
