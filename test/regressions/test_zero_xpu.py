# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

"""Tests for ``zero_()`` on XPU.

The cases come from ``test_zero_*`` in pytorch ``test/test_torch.py``. The
bitserial and Float4 dtypes have no fill kernel. Until ``zero_xpu_`` existed,
``zero_()`` on them raised ``"fill_xpu" not implemented``
(intel/torch-xpu-ops#5378). This copy omits the upstream ``@expectedFailureXPU``
for that case, because the module must catch the failure.
"""

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import make_tensor, run_tests, TestCase


class TestZero(TestCase):
    @dtypes(torch.float32, torch.complex64, torch.complex128)
    def test_zero_dense_view_with_storage_offset(self, device, dtype):
        base = make_tensor((64, 96), dtype=dtype, device=device, low=1, high=2)
        original = base.clone()
        base[16:32].zero_()
        self.assertEqual(base[16:32].count_nonzero().item(), 0)
        self.assertEqual(base[:16], original[:16])
        self.assertEqual(base[32:], original[32:])

    def test_zero_transposed_dense_view(self, device):
        base = torch.ones(64, 96, device=device)
        base[16:32].t().zero_()
        self.assertEqual(base[16:32].count_nonzero().item(), 0)
        self.assertEqual(base[:16], torch.ones(16, 96, device=device))
        self.assertEqual(base[32:], torch.ones(32, 96, device=device))

    @dtypes(torch.float32, torch.complex64)
    def test_zero_strided_view_with_gaps(self, device, dtype):
        base = make_tensor((64, 96), dtype=dtype, device=device, low=1, high=2)
        original = base.clone()
        base[:, ::2].zero_()
        self.assertEqual(base[:, ::2].count_nonzero().item(), 0)
        self.assertEqual(base[:, 1::2], original[:, 1::2])

    def test_zero_empty_view_leaves_storage_intact(self, device):
        base = torch.ones(4, device=device)
        base[2:2].zero_()
        self.assertEqual(base, torch.ones(4, device=device))

    @dtypes(
        torch.bits1x8,
        torch.bits2x4,
        torch.bits4x2,
        torch.bits8,
        torch.bits16,
        torch.float4_e2m1fn_x2,
    )
    def test_zero_dtypes_without_fill_kernel(self, device, dtype):
        raw = torch.ones(1024, dtype=torch.uint8, device=device)
        raw.view(dtype).zero_()
        self.assertEqual(raw.count_nonzero().item(), 0)


instantiate_device_type_tests(TestZero, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
