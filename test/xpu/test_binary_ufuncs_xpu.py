# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_binary_ufuncs import (
        TestBinaryUfuncsDevice,
        TestChebyshevNanPropagationDevice,
    )


instantiate_device_type_tests(
    TestBinaryUfuncsDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestChebyshevNanPropagationDevice, globals(), only_for="xpu", allow_xpu=True
)


class TestMixedDtypeElementwise(TestCase):
    """Test mixed-precision elementwise ops (bf16/fp16 + fp32) correctness.

    These exercise the LoadWithCastFP/StoreWithCastFP fast path in
    gpu_kernel_impl when inputs have mixed floating-point dtypes.
    """

    def _test_mixed_dtype_binary(self, op, shape, low_dtype, device):
        a_fp32 = torch.randn(shape, device=device, dtype=torch.float32)
        b_fp32 = torch.randn(shape, device=device, dtype=torch.float32)
        a_low = a_fp32.to(low_dtype)
        expected = op(a_low.float(), b_fp32)
        result = op(a_low, b_fp32)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result, expected)
        result_rev = op(b_fp32, a_low)
        expected_rev = op(b_fp32, a_low.float())
        self.assertEqual(result_rev, expected_rev)

    def test_mixed_fp_small(self):
        """GPT2 residual shapes: [4, 1024, 768]"""
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        for op in ops:
            for dt in [torch.bfloat16, torch.float16]:
                self._test_mixed_dtype_binary(op, [4, 1024, 768], dt, "xpu")

    def test_mixed_fp_large(self):
        """GPT2 MLP / Albert MLP shapes: [4, 1024, 3072], [16, 512, 3072]"""
        ops = [torch.add, torch.mul]
        for op in ops:
            for dt in [torch.bfloat16, torch.float16]:
                for shape in [[4, 1024, 3072], [16, 512, 3072]]:
                    self._test_mixed_dtype_binary(op, shape, dt, "xpu")


if __name__ == "__main__":
    run_tests()
