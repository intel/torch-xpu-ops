# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for FP8 e8m0fnu (UE8M0) index operations on XPU.

e8m0fnu is the scale dtype used in MX/MXFP8 block-floating-point formats.
CUDA supports a limited set of memory ops for this type; XPU must align with
that support so that operations like printing an e8m0fnu tensor work correctly.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
)


class TestFP8E8M0IndexOps(TestCase):
    """Tests for float8_e8m0fnu (and other FP8) index operations on XPU."""

    def test_index_tensor_all_fp8(self):
        """Test basic tensor indexing (tensor[i]) for all FP8 dtypes.

        This is required for printing fp8 tensors (Python calls __getitem__).
        """
        for dtype in FLOAT8_DTYPES:
            with self.subTest(dtype=dtype):
                a_cpu = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).to(dtype)
                a_xpu = a_cpu.to("xpu")
                # Basic element access via indexing
                row_cpu = a_cpu[0]
                row_xpu = a_xpu[0]
                self.assertEqual(row_cpu, row_xpu.cpu())

    def test_index_put_no_accumulate_all_fp8(self):
        """Test non-accumulate index_put_ for all FP8 dtypes."""
        for dtype in FLOAT8_DTYPES:
            with self.subTest(dtype=dtype):
                a_cpu = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).to(dtype)
                a_xpu = a_cpu.to("xpu")
                val_cpu = torch.tensor([10, 11, 12], dtype=torch.float32).to(dtype)
                val_xpu = val_cpu.to("xpu")
                idx = torch.tensor([1])
                a_cpu.index_put_([idx], val_cpu, accumulate=False)
                a_xpu.index_put_([idx.to("xpu")], val_xpu, accumulate=False)
                self.assertEqual(a_cpu, a_xpu.cpu())

    def test_index_select_e8m0fnu(self):
        """Test index_select for float8_e8m0fnu (UE8M0 scale dtype)."""
        dtype = torch.float8_e8m0fnu
        a_cpu = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).to(dtype)
        a_xpu = a_cpu.to("xpu")
        idx = torch.tensor([0, 2])
        out_cpu = a_cpu.index_select(1, idx)
        out_xpu = a_xpu.index_select(1, idx.to("xpu"))
        self.assertEqual(out_cpu, out_xpu.cpu())

    def test_where_e8m0fnu(self):
        """Test torch.where for float8_e8m0fnu."""
        dtype = torch.float8_e8m0fnu
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=dtype)
        cond = torch.tensor([[True, False], [False, True]])
        out_cpu = torch.where(cond, x, y)
        out_xpu = torch.where(cond.xpu(), x.xpu(), y.xpu())
        self.assertEqual(out_cpu, out_xpu.cpu())

    def test_print_e8m0fnu_tensor(self):
        """Test that printing an e8m0fnu tensor works (uses index_kernel internally)."""
        dtype = torch.float8_e8m0fnu
        a = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).to(dtype).to("xpu")
        # str() triggers element-wise indexing; should not raise
        s = str(a)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)


if __name__ == "__main__":
    run_tests()
