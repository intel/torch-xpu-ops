# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

xpu_device = torch.device("xpu")

# dtypes with no fill kernel on XPU (see intel/torch-xpu-ops#5378). zero_()
# on these dispatches to the XPU device-memset path (zero_xpu_) instead of a
# fill kernel, which would otherwise raise "fill_xpu not implemented".
BITS_DTYPES = [
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
        # layout).
        for dtype in BITS_DTYPES:
            raw = torch.ones(1024, dtype=torch.uint8, device=xpu_device)
            raw.view(dtype).zero_()
            self.assertEqual(raw.count_nonzero().item(), 0, f"dtype={dtype}")

    def test_zero_clears_dense_tensor(self):
        t = torch.randint(1, 100, (64, 64), device=xpu_device, dtype=torch.float32)
        t.zero_()
        self.assertEqual(t.count_nonzero().item(), 0)


if __name__ == "__main__":
    run_tests()