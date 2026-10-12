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

import unittest

import numpy as np
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import os
    import sys

    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_quantized_tensor import TestQuantizedTensor
    from torch.testing._internal.common_cuda import TEST_CUDA

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def _test_compare_per_channel_device_numerics(self):
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        axis = 1
        device = torch.device("xpu")
        for i in range(20):
            for dtype, zero_type in dtype_and_zero_types:
                r = torch.rand(2, 2) * 10
                r[0, 0] = 2.5
                scales = torch.rand(2).abs()
                zero_points = (torch.rand(2) * 10).round().to(zero_type)

                qr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
                dqr = qr.dequantize()
                qr_cuda = torch.quantize_per_channel(
                    r.to(device), scales.to(device), zero_points.to(device), axis, dtype
                )
                dqr_cuda = qr_cuda.dequantize()
                self.assertEqual(qr.int_repr(), qr_cuda.int_repr())
                self.assertTrue(np.allclose(dqr, dqr_cuda.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def _test_compare_per_tensor_device_numerics(self):
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        device = torch.device("xpu")
        for dtype in dtypes:
            r = torch.rand(2, 2) * 10
            r[0, 0] = 2.5
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            qtr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            dqtr = qtr.dequantize()
            qtr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            dqtr_cuda = qtr_cuda.dequantize()
            self.assertEqual(qtr.int_repr(), qtr_cuda.int_repr())
            self.assertTrue(np.allclose(dqtr, dqtr_cuda.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def _test_cuda_quantization_does_not_pin_memory(self):
        # Context - https://github.com/pytorch/pytorch/issues/41115
        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

        q_int = torch.randint(0, 100, [1, 2, 3], device="xpu", dtype=torch.uint8)
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=0.1, zero_point=0)

        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_dequantize_fp16_cuda(self):
        self._test_dequantize_fp16(torch.device("xpu"))


TestQuantizedTensor.test_compare_per_channel_device_numerics = (
    _test_compare_per_channel_device_numerics
)
TestQuantizedTensor.test_compare_per_tensor_device_numerics = (
    _test_compare_per_tensor_device_numerics
)
TestQuantizedTensor.test_cuda_quantization_does_not_pin_memory = (
    _test_cuda_quantization_does_not_pin_memory
)
TestQuantizedTensor.test_dequantize_fp16_cuda = _test_dequantize_fp16_cuda

instantiate_device_type_tests(
    TestQuantizedTensor, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
