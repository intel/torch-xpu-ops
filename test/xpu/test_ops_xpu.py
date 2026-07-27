# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


from functools import wraps

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import (
        fake_autocast_device_skips,
        TestCommon,
        TestCompositeCompliance,
        TestFakeTensor,
        TestForwardADWithScalars,
        TestMathBits,
    )

fake_autocast_device_skips["xpu"] = {"linalg.pinv", "pinverse"}


def _is_problematic_fft_case(dtype, op):
    return op.name.startswith("_refs.fft.") and dtype is torch.half


_original_test_python_ref = TestCommon.test_python_ref
_original_test_python_ref_torch_fallback = TestCommon.test_python_ref_torch_fallback
_original_test_python_ref_executor = TestCommon.test_python_ref_executor


@wraps(_original_test_python_ref)
def _test_python_ref_xpu(self, device, dtype, op):
    if _is_problematic_fft_case(dtype, op):
        self.skipTest("Skipped on XPU: python ref FFT mismatch for half precision")
    return _original_test_python_ref(self, device, dtype, op)


@wraps(_original_test_python_ref_torch_fallback)
def _test_python_ref_torch_fallback_xpu(self, device, dtype, op):
    if _is_problematic_fft_case(dtype, op):
        self.skipTest("Skipped on XPU: python ref FFT mismatch for half precision")
    return _original_test_python_ref_torch_fallback(self, device, dtype, op)


@wraps(_original_test_python_ref_executor)
def _test_python_ref_executor_xpu(self, device, dtype, op, executor):
    if _is_problematic_fft_case(dtype, op):
        self.skipTest("Skipped on XPU: python ref FFT mismatch for half precision")
    return _original_test_python_ref_executor(self, device, dtype, op, executor)


TestCommon.test_python_ref = _test_python_ref_xpu
TestCommon.test_python_ref_torch_fallback = _test_python_ref_torch_fallback_xpu
TestCommon.test_python_ref_executor = _test_python_ref_executor_xpu
instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu", allow_xpu=True)
# in finegrand
instantiate_device_type_tests(
    TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True
)
# only CPU
# instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu", allow_xpu=True)
# not important
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestForwardADWithScalars, globals(), only_for="xpu", allow_xpu=True
)
# instantiate_device_type_tests(TestTags, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
