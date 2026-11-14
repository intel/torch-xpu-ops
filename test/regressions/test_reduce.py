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

    def test_global_reduce_staging_visibility(self):
        # Regression for intel/torch-xpu-ops#6033: the last work-group could
        # read stale staging slots after prior device work.
        torch.manual_seed(6033)
        for name in ("sum", "mean", "amax"):
            for _ in range(12):
                rows = 64 * int(torch.randint(300, 760, ()))
                cpu_input = torch.randn(rows, 256)
                cpu_result = getattr(cpu_input.double(), name)(0)

                xpu_input = cpu_input.xpu()
                xpu_result = getattr(xpu_input, name)(0).cpu().double()

                relative_error = (
                    (xpu_result - cpu_result).norm()
                    / cpu_result.norm().clamp_min(1e-12)
                ).item()
                self.assertLess(relative_error, 1e-4)


if __name__ == "__main__":
    run_tests()
