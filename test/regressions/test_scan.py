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

_loop_scan_shapes = [
    (256, 5),  # short scan
    (129, 31),
    (129, 32),
    (129, 33),  # non-multiple of 32, just over one subgroup
    (129, 63),
    (129, 64),
    (129, 65),
    (129, 97),  # multiple subgroup-sized chunks
    (129, 257),
    (129, 4097),
]


class TestScan(TestCase):
    def _assert_loop_scan(self, op, x, dtype, atol, rtol, equal_nan=False):
        # All inputs are contiguous 2D tensors scanned along the last dimension:
        # stride == 1, batch > 128, and problem < 16384 select LoopScanKernel.
        x_xpu = x.xpu()
        if dtype is None:
            expected = op(x, -1)
            actual = op(x_xpu, -1).cpu()
        else:
            expected = op(x, -1, dtype=dtype)
            actual = op(x_xpu, -1, dtype=dtype).cpu()
        self.assertEqual(expected, actual, atol=atol, rtol=rtol, equal_nan=equal_nan)

    def _supported_float_dtypes(self):
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if torch.xpu.get_device_properties(0).has_fp64:
            dtypes.append(torch.float64)
        return dtypes

    def test_cumsum_loop_scan(self):
        for dtype in self._supported_float_dtypes():
            atol, rtol = {
                torch.float16: (1e-2, 1e-2),
                torch.bfloat16: (2e-2, 2e-2),
                torch.float32: (1e-4, 1e-4),
                torch.float64: (1e-10, 1e-10),
            }[dtype]
            for rows, columns in _loop_scan_shapes:
                if dtype in (torch.float16, torch.bfloat16):
                    pattern = torch.tensor([1 / 256, -1 / 256], dtype=torch.float32)
                    x = pattern.repeat(rows, (columns + 1) // 2)[:, :columns].to(dtype)
                else:
                    x = torch.randn(rows, columns, dtype=dtype) * 0.01
                self._assert_loop_scan(torch.cumsum, x, dtype, atol, rtol)

        for dtype in (torch.int32, torch.int64):
            for rows, columns in _loop_scan_shapes:
                pattern = torch.tensor([1, -2, 3, 0], dtype=dtype)
                x = pattern.repeat(rows, (columns + 3) // 4)[:, :columns]
                self._assert_loop_scan(torch.cumsum, x, dtype, 0, 0)

    def test_cumprod_loop_scan(self):
        shapes = _loop_scan_shapes[:-1]
        for dtype in self._supported_float_dtypes():
            atol, rtol = {
                torch.float16: (1e-2, 1e-2),
                torch.bfloat16: (2e-2, 2e-2),
                torch.float32: (1e-5, 1e-5),
                torch.float64: (1e-10, 1e-10),
            }[dtype]
            pattern = torch.tensor([1.125, 0.875], dtype=torch.float32)
            for rows, columns in shapes:
                x = pattern.repeat(rows, (columns + 1) // 2)[:, :columns].to(dtype)
                self._assert_loop_scan(torch.cumprod, x, dtype, atol, rtol)

        pattern = torch.tensor([1, -1], dtype=torch.int32)
        for dtype in (torch.int32, torch.int64):
            for rows, columns in shapes:
                x = pattern.to(dtype).repeat(rows, (columns + 1) // 2)[:, :columns]
                self._assert_loop_scan(torch.cumprod, x, dtype, 0, 0)

    def test_logcumsumexp_loop_scan(self):
        for dtype in self._supported_float_dtypes():
            atol, rtol = {
                torch.float16: (2e-2, 2e-2),
                torch.bfloat16: (1e-1, 1e-1),
                torch.float32: (1e-4, 1e-4),
                torch.float64: (1e-10, 1e-10),
            }[dtype]
            for rows, columns in _loop_scan_shapes:
                x = torch.randn(rows, columns, dtype=torch.float32).to(dtype)
                self._assert_loop_scan(torch.logcumsumexp, x, None, atol, rtol)

    def test_logcumsumexp_loop_scan_special_values(self):
        for dtype in self._supported_float_dtypes():
            x = torch.zeros(129, 65, dtype=dtype)
            x[0, 0] = -float("inf")
            x[1, 1] = float("inf")
            x[2, 2] = float("nan")
            self._assert_loop_scan(
                torch.logcumsumexp,
                x,
                None,
                atol=1e-1 if dtype is torch.bfloat16 else 1e-2,
                rtol=1e-1 if dtype is torch.bfloat16 else 1e-2,
                equal_nan=True,
            )

    def test_cumsum_loop_scan_grid_stride(self):
        properties = torch.xpu.get_device_properties(0)
        rows = max(
            131073,
            properties.gpu_eu_count * max(properties.sub_group_sizes) * 32,
        )
        columns = 5
        x = torch.randn(rows, columns, dtype=torch.float32) * 0.01
        self._assert_loop_scan(torch.cumsum, x, torch.float32, 1e-5, 1e-5)

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
