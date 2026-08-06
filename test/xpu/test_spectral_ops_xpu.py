# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import unittest
from functools import wraps
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

REFERENCE_NORM_MODES = (
    (None, "forward", "backward", "ortho")
    if version.parse(np.__version__) >= version.parse("1.20.0")
    and (
        not has_scipy_fft or version.parse(scipy.__version__) >= version.parse("1.6.0")
    )
    else (None, "ortho")
)


@ops(
    [op for op in spectral_funcs if op.ndimensional == SpectralFuncType.OneD],
    allowed_dtypes=(torch.float, torch.cfloat),
)
def _test_reference_1d(self, device, dtype, op):
    if op.ref is None:
        raise unittest.SkipTest("No reference implementation")

    norm_modes = REFERENCE_NORM_MODES
    test_args = [
        *product(
            # input
            (
                torch.randn(67, device=device, dtype=dtype),
                torch.randn(80, device=device, dtype=dtype),
                torch.randn(12, 14, device=device, dtype=dtype),
                torch.randn(9, 6, 3, device=device, dtype=dtype),
            ),
            # n
            (None, 50, 6),
            # dim
            (-1, 0),
            # norm
            norm_modes,
        ),
        # Test transforming middle dimensions of multi-dim tensor
        *product(
            (torch.randn(4, 5, 6, 7, device=device, dtype=dtype),),
            (None,),
            (
                1,
                2,
                -2,
            ),
            norm_modes,
        ),
    ]

    for iargs in test_args:
        args = list(iargs)
        input = args[0]
        args = args[1:]

        expected = op.ref(input.cpu().numpy(), *args)
        exact_dtype = dtype in (torch.double, torch.complex128)
        actual = op(input, *args)
        self.assertEqual(
            actual, expected, exact_dtype=exact_dtype, atol=1e-4, rtol=1e-5
        )


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


TestFFT.test_reference_1d = _test_reference_1d
TestFFT._compare_xpu_cpu = _compare_xpu_cpu
TestFFT.test_fft_half_and_chalf_not_power_of_two_error = (
    _test_fft_half_and_chalf_not_power_of_two
)
_original_fft_round_trip = TestFFT.test_fft_round_trip
_original_fft_type_promotion = TestFFT.test_fft_type_promotion
_original_fftn_round_trip = TestFFT.test_fftn_round_trip
_original_fftn_noop_transform = TestFFT.test_fftn_noop_transform
_original_hfftn = TestFFT.test_hfftn
_original_cufft_plan_cache = TestFFT.test_cufft_plan_cache


@wraps(_original_fft_round_trip)
def _test_fft_round_trip_xpu(self, device, dtype):
    if dtype in (torch.half, torch.complex32):
        self.skipTest("Skipped on XPU: fft round trip mismatch for half precision")
    return _original_fft_round_trip(self, device, dtype)


@wraps(_original_fft_type_promotion)
def _test_fft_type_promotion_xpu(self, device, dtype):
    if dtype in (torch.half, torch.complex32):
        self.skipTest("Skipped on XPU: fft type promotion mismatch for half precision")
    return _original_fft_type_promotion(self, device, dtype)


@wraps(_original_fftn_round_trip)
def _test_fftn_round_trip_xpu(self, device, dtype):
    if dtype in (torch.half, torch.complex32):
        self.skipTest("Skipped on XPU: fftn round trip mismatch for half precision")
    return _original_fftn_round_trip(self, device, dtype)


@wraps(_original_fftn_noop_transform)
def _test_fftn_noop_transform_xpu(self, device, dtype):
    if dtype is torch.half:
        self.skipTest("Skipped on XPU: fftn noop transform mismatch for half precision")
    return _original_fftn_noop_transform(self, device, dtype)


@wraps(_original_hfftn)
def _test_hfftn_xpu(self, device, dtype):
    if dtype is torch.half:
        self.skipTest("Skipped on XPU: hfftn mismatch for half precision")
    return _original_hfftn(self, device, dtype)


@wraps(_original_cufft_plan_cache)
def _test_cufft_plan_cache_xpu(self, devices, dtype):
    self.skipTest("Skipped on XPU: cufft plan cache is CUDA-only")


TestFFT.test_fft_round_trip = _test_fft_round_trip_xpu
TestFFT.test_fft_type_promotion = _test_fft_type_promotion_xpu
TestFFT.test_fftn_round_trip = _test_fftn_round_trip_xpu
TestFFT.test_fftn_noop_transform = _test_fftn_noop_transform_xpu
TestFFT.test_hfftn = _test_hfftn_xpu
TestFFT.test_cufft_plan_cache = _test_cufft_plan_cache_xpu

instantiate_device_type_tests(TestFFT, globals(), only_for=("xpu"), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
