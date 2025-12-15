# Copyright 2020-2025 Intel Corporation
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
from torch.testing._internal.common_dtype import float8_types
from torch.testing._internal.common_utils import run_tests, TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleConversion(TestCase):
    def _compare_convert_with_cpu(self, src_cpu, dtype):
        src_xpu = src_cpu.to(xpu_device)
        dst_cpu = src_cpu.to(dtype)
        dst_xpu = src_xpu.to(dtype)
        self.assertEqual(dst_xpu.to(cpu_device), dst_cpu)

    @dtypes(*float8_types())
    def test_half_zero(self, dtype):
        pos_zero_fp16_cpu = torch.zeros((5, 6), dtype=torch.float16)
        self._compare_convert_with_cpu(pos_zero_fp16_cpu, dtype)

        neg_zero_fp16_cpu = torch.full((5, 6), -0.0, dtype=torch.float16)
        self._compare_convert_with_cpu(neg_zero_fp16_cpu, dtype)

    @dtypes(*float8_types())
    def test_half_nonzero(self, dtype):
        x_fp16_cpu = torch.arange(-100.0, 101.0, dtype=torch.float16)
        self._compare_convert_with_cpu(x_fp16_cpu, dtype)


instantiate_device_type_tests(
    TestSimpleConversion, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
