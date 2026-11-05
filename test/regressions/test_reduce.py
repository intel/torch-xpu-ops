# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


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


if __name__ == "__main__":
    run_tests()
