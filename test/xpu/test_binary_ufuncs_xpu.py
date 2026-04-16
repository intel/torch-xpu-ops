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
from torch.testing._internal.common_utils import make_tensor, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_binary_ufuncs import integral_types, TestBinaryUfuncs


@dtypes(*integral_types())
def _test_fmod_remainder_by_zero_integral(self, device, dtype):
    fn_list = (torch.fmod, torch.remainder)
    for fn in fn_list:
        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        zero = torch.zeros_like(x)
        value = 255 if dtype == torch.uint8 else -1
        self.assertTrue(torch.all(fn(x, zero) == value))


TestBinaryUfuncs.test_fmod_remainder_by_zero_integral = (
    _test_fmod_remainder_by_zero_integral
)


instantiate_device_type_tests(
    TestBinaryUfuncs, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
