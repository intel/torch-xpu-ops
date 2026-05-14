# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

# Regression test for: https://github.com/intel/torch-xpu-ops/issues/2953
# torch.full with out-of-range values for Half/BFloat16 on XPU should not
# raise RuntimeError but instead saturate to -inf (matching CPU behavior).

import math

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFillOverflow(TestCase):
    def test_fill_overflow_saturates_to_inf_float16(self):
        """torch.full with torch.finfo(float32).min into float16 must not raise."""
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        fill_val = torch.finfo(torch.float32).min
        t = torch.full((4, 4), fill_val, dtype=torch.float16, device="xpu")
        self.assertEqual(t.shape, (4, 4))
        self.assertTrue(math.isinf(t[0, 0].item()) and t[0, 0].item() < 0)

    def test_fill_overflow_saturates_to_inf_bfloat16(self):
        """torch.full with torch.finfo(float32).min into bfloat16 must not raise."""
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        fill_val = torch.finfo(torch.float32).min
        t = torch.full((4, 4), fill_val, dtype=torch.bfloat16, device="xpu")
        self.assertEqual(t.shape, (4, 4))
        self.assertTrue(math.isinf(t[0, 0].item()) and t[0, 0].item() < 0)

    def test_fill_finfo_min_matches_cpu_float16(self):
        """XPU and CPU must produce the same result for float16 out-of-range fill."""
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        fill_val = torch.finfo(torch.float32).min
        cpu_t = torch.full((4, 4), fill_val, dtype=torch.float16, device="cpu")
        xpu_t = torch.full((4, 4), fill_val, dtype=torch.float16, device="xpu")
        self.assertTrue(torch.equal(cpu_t, xpu_t.cpu()))

    def test_fill_finfo_min_matches_cpu_bfloat16(self):
        """XPU and CPU must produce the same result for bfloat16 out-of-range fill."""
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        fill_val = torch.finfo(torch.float32).min
        cpu_t = torch.full((4, 4), fill_val, dtype=torch.bfloat16, device="cpu")
        xpu_t = torch.full((4, 4), fill_val, dtype=torch.bfloat16, device="xpu")
        self.assertTrue(torch.equal(cpu_t, xpu_t.cpu()))


if __name__ == "__main__":
    run_tests()
