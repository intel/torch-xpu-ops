# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import TestCase


class TestNNMethod(TestCase):
    def test_sort_large_slice(self, device=torch.device("xpu")):
        x = torch.randn(4, 1024000, device=device)
        res1val, res1ind = torch.sort(x, stable=True)
        torch.xpu.synchronize()
        # assertIsOrdered is too slow, so just compare to cpu
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())
        res1val, res1ind = torch.sort(x, descending=True, stable=True)
        torch.xpu.synchronize()
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), descending=True, stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())

    def test_sort_large_bool(self):
        tensor_dtype = torch.bool
        value_range = 2
        a = torch.randint(value_range, (22099,)).to(dtype=tensor_dtype).xpu()
        for dim in reversed(range(a.dim())):
            sorted_cpu, indices = torch.sort(a.cpu())
            sorted, indices = torch.sort(a)
            self.assertEqual(sorted.cpu(), sorted_cpu)
            sorted, indices = a.sort()
            self.assertEqual(sorted.cpu(), sorted_cpu)
            sorted, indices = a.sort(stable=True)
            self.assertEqual(sorted.cpu(), sorted_cpu)

    @largeTensorTest("48GB", device="xpu")
    def test_topk_num_tiles_no_overflow(self):
        n = 2**31 - 1
        # k > 256 routes through topk_out_with_sort -> segmented_radix_sort
        k = 300
        data = torch.zeros((1, n), device="xpu", dtype=torch.float16)
        values, indices = torch.topk(data, k, dim=1, largest=True, sorted=False)
        self.assertEqual(values.shape, (1, k))
        self.assertEqual(indices.shape, (1, k))
        # All input values are 0.0, so every top-k value must also be 0.0.
        self.assertTrue((values == 0.0).all())
        # Indices must be valid positions within the input dimension.
        self.assertTrue((indices >= 0).all() and (indices < n).all())

    @largeTensorTest("8GB", device="xpu")
    def test_topk_dimension_larger_than_int_max(self):
        n = 2**31
        k = 10
        data = torch.zeros((1, n), device="xpu", dtype=torch.float16)
        values, indices = torch.topk(data, k, dim=1, sorted=False)
        self.assertEqual(values.shape, (1, k))
        self.assertEqual(indices.shape, (1, k))
        self.assertTrue((values == 0.0).all())
        self.assertTrue((indices >= 0).all() and (indices < n).all())
