# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Reproducer for: long float32 torch.cumprod numerical parity on XPU
# Without the float64-intermediate path in ScanKernels.cpp, cumprod on
# float32 tensors with scan length >= 1<<16 accumulates significant
# numerical drift vs the CPU reference.

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCumprodLongFloat32Precision(TestCase):
    def test_cumprod_long_float32_1d(self):
        long_scan_len = (1 << 16) + 1
        with torch.random.fork_rng():
            torch.manual_seed(0)
            x = 1.0 + 0.0001 * torch.randn(long_scan_len, dtype=torch.float32)
            cpu_ref = torch.cumprod(x, dim=0)
            xpu_out = torch.cumprod(x.xpu(), dim=0).cpu()
            torch.testing.assert_close(xpu_out, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_cumprod_long_float32_2d(self):
        long_scan_len = (1 << 16) + 1
        with torch.random.fork_rng():
            torch.manual_seed(0)
            y = 1.0 + 0.0001 * torch.randn(2, long_scan_len, dtype=torch.float32)
            cpu_ref = torch.cumprod(y, dim=1)
            xpu_out = torch.cumprod(y.xpu(), dim=1).cpu()
            torch.testing.assert_close(xpu_out, cpu_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
