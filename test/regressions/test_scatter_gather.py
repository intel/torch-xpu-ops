# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import float8_types_and
from torch.testing._internal.common_utils import run_tests, TestCase

xpu_device = torch.device("xpu")


class TestScatterGatherBComplex32(TestCase):
    def test_gather_bcomplex32(self):
        # bcomplex32 CPU support is limited; compare XPU results directly
        x = torch.randn(4, 8).to(dtype=torch.bcomplex32, device=xpu_device)
        # gather column 0 from all rows via a (4, 1) index tensor
        idx = torch.zeros(4, 1, dtype=torch.long, device=xpu_device)
        result = torch.gather(x, 1, idx)
        self.assertEqual(result.dtype, torch.bcomplex32)
        self.assertEqual(result[:, 0], x[:, 0])

    def test_scatter_bcomplex32(self):
        x = torch.zeros(4, 8, dtype=torch.bcomplex32, device=xpu_device)
        idx = torch.tensor([[0, 2, 4]], dtype=torch.long, device=xpu_device)
        src = torch.randn(1, 3).to(dtype=torch.bcomplex32, device=xpu_device)
        x.scatter_(1, idx, src)
        self.assertEqual(x.dtype, torch.bcomplex32)
        self.assertEqual(x[0, 0], src[0, 0])
        self.assertEqual(x[0, 2], src[0, 1])
        self.assertEqual(x[0, 4], src[0, 2])


class TestScatterFillFloat8(TestCase):
    @dtypes(*float8_types_and(torch.float8_e8m0fnu))
    def test_scatter_fill_float8(self, dtype):
        # scatter_ with a scalar value hits ScatterFillBaseKernel's
        # TensorAssign overload, which must support fp8 dtypes.
        expected_one = torch.tensor(1.0, device=xpu_device)
        expected_zero = torch.tensor(0.0, device=xpu_device)
        x = torch.zeros(4, 8, dtype=dtype, device=xpu_device)
        idx = torch.tensor([[0, 2, 4]], dtype=torch.long, device=xpu_device)
        x.scatter_(1, idx, 1.0)
        self.assertEqual(x.dtype, dtype)
        x_cpu = x.float().cpu()
        self.assertEqual(x_cpu[0, 0], expected_one)
        self.assertEqual(x_cpu[0, 2], expected_one)
        self.assertEqual(x_cpu[0, 4], expected_one)
        self.assertEqual(x_cpu[0, 1], expected_zero)


instantiate_device_type_tests(
    TestScatterFillFloat8, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
