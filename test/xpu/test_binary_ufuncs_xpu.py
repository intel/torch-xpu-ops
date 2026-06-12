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
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import integral_types
from torch.testing._internal.common_utils import make_tensor, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_binary_ufuncs import TestBinaryUfuncsDevice
<<<<<<< dev/jmamzax/2630_add_test_fmod_remainder_by_zero_integral_xpu


@dtypes(*integral_types())
def _test_fmod_remainder_by_zero_integral(self, device, dtype):
    fn_list = (torch.fmod, torch.remainder)
    for fn in fn_list:
        # check integral tensor fmod/remainder to zero
        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        zero = torch.zeros_like(x)
        # RuntimeError on CPU
        if self.device_type == "cpu":
            with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
                fn(x, zero)
        elif torch.version.hip is not None:
            # ROCm behavior: x % 0 is a no-op; x is returned
            self.assertEqual(fn(x, zero), x)
        elif self.device_type == "cuda" and dtype == torch.int64:
            # CUDA behavior: Different value for different dtype
            # Due to it's an undefined behavior, CUDA returns a pattern of all 1s
            # for integral dividend (other than int64) divided by zero. For int64,
            # CUDA returns all 1s for negative dividend, half 1s for positive dividend.
            # uint8: 0xff -> 255
            # int32: 0xffffffff -> -1
            if dtype == torch.int64:
                self.assertEqual(fn(x, zero) == 4294967295, x >= 0)
                self.assertEqual(fn(x, zero) == -1, x < 0)
        else:
            value = 255 if dtype == torch.uint8 else -1
            self.assertTrue(torch.all(fn(x, zero) == value))


TestBinaryUfuncsDevice.test_fmod_remainder_by_zero_integral = (
    _test_fmod_remainder_by_zero_integral
)

=======
>>>>>>> main

instantiate_device_type_tests(
    TestBinaryUfuncsDevice, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
