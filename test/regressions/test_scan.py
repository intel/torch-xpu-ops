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

# dispatch_to_loop_scan_kernel (ScanUtils.h) selects the loop-scan kernel when
# scanning the contiguous last dim (stride == 1) with batch > 128 and
# problem < 16384. Scanning dim 0 of these 2D tensors is strided, so it exercises
# the segment-scan fallback instead. Shapes below cover both sides of the
# batch/problem thresholds plus the batch == 1 corner.
_shapes = [
    (256, 1000),  # loop scan: batch and problem both in range
    (1000, 512),  # loop scan
    (200, 8192),  # loop scan: multiple problem chunks per row
    (129, 33),  # loop scan: just above the batch threshold
    (64, 4096),  # segment scan: batch <= 128
    (1, 5000),  # segment scan: single batch
]


class TestScan(TestCase):
    def test_cumsum_loop_scan(self):
        for r, c in _shapes:
            x = torch.randn(r, c, dtype=torch.float32)
            x_xpu = x.xpu()
            for dim in (0, 1):
                # Relaxed tolerance: XPU and CPU accumulate in a different order.
                self.assertEqual(
                    torch.cumsum(x, dim),
                    torch.cumsum(x_xpu, dim).cpu(),
                    atol=1e-3,
                    rtol=1e-3,
                )

    def _test_cumminmax(self, op):
        for r, c in _shapes:
            x = torch.randn(r, c, dtype=torch.float32)
            x_xpu = x.xpu()
            for dim in (0, 1):
                ref_val, _ = op(x, dim)
                val, idx = op(x_xpu, dim)
                self.assertEqual(ref_val, val.cpu())
                # Indices may tie-break differently across backends; assert the
                # returned indices reproduce the returned values instead.
                self.assertEqual(val.cpu(), x.gather(dim, idx.cpu()))

    def test_cummax_loop_scan(self):
        self._test_cumminmax(torch.cummax)

    def test_cummin_loop_scan(self):
        self._test_cumminmax(torch.cummin)


if __name__ == "__main__":
    run_tests()
