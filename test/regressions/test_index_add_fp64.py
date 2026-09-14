# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Regression test for https://github.com/intel/torch-xpu-ops/issues/4184
#
# IndexFuncLargeIndexFunctor coalesces index_add_ writes through shared local
# memory using work-group-scope 64-bit floating-point atomics. On BMG that
# combination hangs the device, so the coalescing path is disabled for double
# and complex<double> (src/ATen/native/xpu/sycl/Indexing.cpp).
#
# Both cases below drive the large-index kernel (numIndex > 16) with heavy
# destination-index duplication, which is what forces repeated atomics on the
# same shared-memory slot. A regression re-introduces a hang, so these tests
# fail by timing out rather than by a wrong result; the numeric checks guard
# against the global-memory fallback accumulating incorrectly.

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestIndexAddFp64(TestCase):
    @parametrize("dtype", [torch.float64, torch.complex128])
    def test_index_add_duplicate_indices(self, device, dtype):
        src = torch.randn((65536, 32), dtype=dtype, device=device)
        index = torch.randint(0, 8, (65536,), device=device)
        out = torch.zeros((8, 32), dtype=dtype, device=device)
        out.index_add_(0, index, src)

        ref = torch.zeros((8, 32), dtype=dtype)
        ref.index_add_(0, index.cpu(), src.cpu())
        self.assertEqual(out.cpu(), ref)

    @parametrize("dtype", [torch.float64, torch.complex128])
    def test_sparse_index_select_to_dense(self, device, dtype):
        t = torch.randn((100, 50, 3, 3), dtype=dtype, device=device)
        index = torch.tensor([0, 1, 2], device=device)
        got = t.to_sparse().index_select(2, index).to_dense()
        self.assertEqual(got, t.index_select(2, index))


instantiate_device_type_tests(
    TestIndexAddFp64, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
