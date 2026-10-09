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
from torch.testing._internal.common_utils import run_tests, TestCase


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


if __name__ == "__main__":
    run_tests()
