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
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestSimpleReduce(TestCase):
    def test_aminmax_int64_large(self):
        # Regression for intel/torch-xpu-ops#4435: the int64 aminmax reduction
        # over a large input reaches the combined group-x/group-y group_reduce
        # path and used to abort the process.
        cpu_input = torch.randint(0, 1000, (16384,), dtype=torch.int64)
        cpu_min, cpu_max = torch.aminmax(cpu_input)

        xpu_input = cpu_input.xpu()
        xpu_min, xpu_max = torch.aminmax(xpu_input)
        torch.xpu.synchronize()

        self.assertEqual(xpu_min.cpu(), cpu_min)
        self.assertEqual(xpu_max.cpu(), cpu_max)


class TestMinMaxReduce(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("op", ("argmax", "argmin", "max", "min"))
    @parametrize("layout", ("contiguous", "offset", "strided", "output_vectorized"))
    def test_minmax_first_index(self, device, dtype, op, layout):
        # Odd row widths exercise alignment heads and vector tails.
        length = 4099
        step = 2 if layout == "strided" else 1
        offset = 1 if layout == "offset" else 0
        extremum = 1.0 if op in ("argmax", "max") else -1.0
        cpu_base = torch.full((12, length * step + offset), -extremum, dtype=dtype)
        cpu_input = cpu_base[:, offset::step]
        cpu_input[0].fill_(extremum)
        cpu_input[1, [0, 1, 2, 1024, -1]] = extremum
        cpu_input[2, [1, 2, 1025, -1]] = extremum
        cpu_input[3, [1, 2, 1025, -1]] = float("nan")
        cpu_input[4, [33, 34, 1057, -1]] = float("nan")
        cpu_input[5].fill_(float("inf"))
        cpu_input[6].fill_(-float("inf"))
        cpu_input[7:9].zero_()
        cpu_input[7, 0] = -0.0
        cpu_input[8, 1:] = -0.0
        cpu_input[9, [-2, -1]] = extremum
        cpu_input[10, [-2, -1]] = float("nan")
        cpu_input[11, [31, 32, 1055, -1]] = extremum

        # Transfer the base before slicing to preserve offsets and strides.
        device_input = cpu_base.to(device)[:, offset::step]
        dim = 1
        if layout == "output_vectorized":
            cpu_input = cpu_input.t().contiguous()
            device_input = device_input.t().contiguous()
            dim = 0

        reduce = getattr(torch, op)
        expected = reduce(cpu_input, dim=dim)
        actual = reduce(device_input, dim=dim)
        if op in ("argmax", "argmin"):
            self.assertEqual(actual.cpu(), expected)
        else:
            actual_values = actual.values.cpu()
            self.assertEqual(actual_values, expected.values, atol=0, rtol=0)
            self.assertEqual(actual.indices.cpu(), expected.indices)
            zeros = expected.values == 0
            self.assertEqual(
                actual_values[zeros].signbit(), expected.values[zeros].signbit()
            )


instantiate_device_type_tests(
    TestMinMaxReduce, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
