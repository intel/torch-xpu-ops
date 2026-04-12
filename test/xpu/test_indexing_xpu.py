# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    dtypesIfXPU,
)
from torch.testing._internal.common_utils import DeterministicGuard, run_tests
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import all_types_complex_float8_and
from itertools import product
import numpy as np

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import torch
    from test_indexing import NumpyTests, TestIndexing

    torch.Tensor.is_cuda = torch.Tensor.is_xpu

    def __test_index_put_deterministic_with_optional_tensors(self, device):
        def func(x, i, v):
            with DeterministicGuard(True):
                x[..., i] = v
            return x

        def func1(x, i, v):
            with DeterministicGuard(True):
                x[i] = v
            return x

        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])
        values2d = torch.randn(n, 1)

        for val in (value0d, value1d, values2d):
            out_cuda = func(t_dev, indices_dev, val.to(device))
            out_cpu = func(t, indices, val)
            self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros((5, 4))
        t_dev = t.to(device)
        indices = torch.tensor([1, 4, 3])
        indices_dev = indices.to(device)
        val = torch.randn(4)
        out_cuda = func1(t_dev, indices_dev, val.xpu())
        out_cpu = func1(t, indices, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros(2, 3, 4)
        ind = torch.tensor([0, 1])
        val = torch.randn(6, 2)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            func(t, ind, val)

        with self.assertRaisesRegex(RuntimeError, "must match"):
            func(t.to(device), ind.to(device), val.to(device))

        val = torch.randn(2, 3, 1)
        out_cuda = func1(t.to(device), ind.to(device), val.to(device))
        out_cpu = func1(t, ind, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

    @dtypes(
        torch.cfloat, torch.cdouble, torch.float, torch.bfloat16, torch.long, torch.bool
    )
    @dtypesIfXPU(
        torch.cfloat,
        torch.cdouble,
        torch.half,
        torch.long,
        torch.bool,
        torch.bfloat16,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    )
    def index_put_src_datatype(self, device, dtype):
        src = torch.ones(3, 2, 4, device=device, dtype=dtype)
        vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
        indices = (torch.tensor([0, 2, 1]),)
        res = src.index_put_(indices, vals, accumulate=True)
        self.assertEqual(res.shape, src.shape)

    @dtypes(*all_types_complex_float8_and(torch.half, torch.bool, torch.bfloat16))
    def index_select(self, device, dtype):
        num_src, num_out = 3, 5

        def make_arg(batch_sizes, n, dim, contig):
            size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
            return make_tensor(
                size_arg,
                dtype=dtype,
                device=device,
                low=None,
                high=None,
                noncontiguous=not contig,
            )

        def ref_index_select(src, dim, idx):
            # some types not supported on numpy
            not_np_dtypes = (
                torch.bfloat16,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            )
            if dtype in not_np_dtypes:
                src = src.float()
            out = torch.from_numpy(
                np.take(src.cpu().numpy(), idx.cpu().numpy(), axis=dim)
            )
            if dtype in not_np_dtypes:
                out = out.to(device=device, dtype=dtype)
            return out

        for src_contig, idx_contig in product([True, False], repeat=2):
            for other_sizes in ((), (4, 5)):
                for dim in range(len(other_sizes)):
                    src = make_arg(other_sizes, num_src, dim, src_contig)
                    idx = make_tensor(
                        (num_out,),
                        dtype=torch.int64,
                        device=device,
                        low=0,
                        high=num_src,
                        noncontiguous=not idx_contig,
                    )
                    out = torch.index_select(src, dim, idx)
                    out2 = ref_index_select(src, dim, idx)
                    self.assertEqual(out, out2)

        for idx_type in (torch.int32, torch.int64):
            other_sizes = (3, 2)
            dim = 1
            src = make_arg(other_sizes, num_src, dim, True)
            idx = make_tensor(
                (num_out,),
                dtype=idx_type,
                device=device,
                low=0,
                high=num_src,
                noncontiguous=False,
            )
            out = torch.index_select(src, dim, idx)
            out2 = ref_index_select(src, dim, idx)
            self.assertEqual(out, out2)

        # Create the 4 possible combinations of scalar sizes for index / source
        scalars = (
            (
                make_tensor(size_s, dtype=dtype, device=device),
                torch.zeros(size_i, dtype=torch.int64, device=device),
            )
            for size_s, size_i in product([(), (1,)], repeat=2)
        )
        for source, idx in scalars:
            out = source.index_select(0, idx)
            self.assertEqual(out.item(), source.item())

    TestIndexing.test_index_put_deterministic_with_optional_tensors = (
        __test_index_put_deterministic_with_optional_tensors
    )
    TestIndexing.test_index_put_src_datatype = (index_put_src_datatype)
    TestIndexing.test_index_select = (index_select)

instantiate_device_type_tests(NumpyTests, globals(), only_for=("xpu"), allow_xpu=True)

instantiate_device_type_tests(TestIndexing, globals(), only_for=("xpu"), allow_xpu=True)
if __name__ == "__main__":
    run_tests()
