# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import unittest
from itertools import product

import numpy as np
import torch
from packaging import version
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import (
    spectral_funcs,
    SpectralFuncType,
)
from torch.testing._internal.common_utils import run_tests

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_spectral_ops import TestFFT

has_scipy_fft = False
try:
    import scipy.fft

    has_scipy_fft = True
except ModuleNotFoundError:
    pass


@ops(spectral_funcs, allowed_dtypes=(torch.half, torch.chalf))
@toleranceOverride(
    {
        torch.half: tol(1e-2, 1e-2),
        torch.chalf: tol(1e-2, 1e-2),
    }
)
def _test_fft_half_and_chalf_not_power_of_two(self, device, dtype, op):
    t = torch.randn(13, 13, device=device, dtype=dtype)

    if op.ndimensional in (SpectralFuncType.ND, SpectralFuncType.TwoD):
        kwargs = {"s": (12, 12)}
    else:
        kwargs = {"n": 12}

    # Promote to higher precision for CPU reference calculations.
    cpu_input = t.to(torch.complex64 if dtype.is_complex else torch.float32).cpu()

    # Validate default call
    cpu_default = op(cpu_input)
    xpu_default = op(t)
    self._compare_xpu_cpu(xpu_default, cpu_default, t)

    # Validate sized call
    cpu_sized = op(cpu_input, **kwargs)
    xpu_sized = op(t, **kwargs)
    self._compare_xpu_cpu(xpu_sized, cpu_sized, t)


def _compare_xpu_cpu(self, xpu_result, cpu_result, t):
    self.assertEqual(xpu_result.device, t.device)
    self.assertEqual(xpu_result.is_complex(), cpu_result.is_complex())
    self.assertEqual(xpu_result, cpu_result, exact_dtype=False)


TestFFT._compare_xpu_cpu = _compare_xpu_cpu
TestFFT.test_fft_half_and_chalf_not_power_of_two_error = (
    _test_fft_half_and_chalf_not_power_of_two
)

instantiate_device_type_tests(TestFFT, globals(), only_for=("xpu"), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
