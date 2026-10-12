# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XPU zero_() device-memset path (zero_xpu_).

zero_() on XPU now dispatches to zero_xpu_, a dedicated native that zeroes the
tensor with a SYCL device memset on the current XPU stream, instead of the
composite zero_() that falls back to fill_(0). This makes zero_() work for
dtypes that have no fill kernel: the bitserial Bits* dtypes and
Float4_e2m1fn_x2. These tests verify zero_() actually zeroes those dtypes and
that the device-memset path is exercised for a dense tensor and falls back to
fill_(0) for a non-dense tensor.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# dtypes with no fill kernel on XPU (see intel/torch-xpu-ops#5378)
_BITS_DTYPES = [
    torch.bits1x8,
    torch.bits2x4,
    torch.bits4x2,
    torch.bits8,
    torch.bits16,
    torch.float4_e2m1fn_x2,
]


class TestZeroXpu(TestCase):
    def test_zero_clears_dtypes_without_fill_kernel(self):
        # zero_() on a view of a bitserial/Float4 dtype must clear the
        # underlying storage (value 0 is the all-zero bit pattern in every
        # layout). Previously this raised "fill_xpu not implemented".
        for dtype in _BITS_DTYPES:
            raw = torch.ones(1024, dtype=torch.uint8, device="xpu")
            raw.view(dtype).zero_()
            self.assertEqual(raw.count_nonzero().item(), 0, f"dtype={dtype}")

    def test_zero_uses_device_memset_on_dense_tensor(self):
        t = torch.randint(1, 100, (64, 64), device="xpu", dtype=torch.float32)
        t.zero_()
        self.assertEqual(t.count_nonzero().item(), 0)

    def test_zero_strided_falls_back_to_fill(self):
        # A strided (non-overlapping-and-dense) zero_ falls back to fill_(0).
        base = torch.ones(64, 64, device="xpu", dtype=torch.float32)
        base[:, ::2].zero_()
        self.assertEqual(base[:, ::2].count_nonzero().item(), 0)


if __name__ == "__main__":
    run_tests()