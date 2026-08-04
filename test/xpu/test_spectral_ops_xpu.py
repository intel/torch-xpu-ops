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


def _test_fft_out_variants(self, device):
    dims = [0, 1, 2, 3]
    real_cpu = torch.randn(2, 3, 4, 5)
    complex_cpu = torch.complex(real_cpu, torch.randn_like(real_cpu))

    complex_input = complex_cpu.to(device)
    c2c_expected = torch.ops.aten._fft_c2c.default(complex_cpu, dims, 0, True)
    c2c_out = torch.empty_like(complex_input)
    c2c_result = torch.ops.aten._fft_c2c.out(complex_input, dims, 0, True, out=c2c_out)
    self.assertIs(c2c_result, c2c_out)
    self.assertEqual(c2c_out.cpu(), c2c_expected)

    real_input = real_cpu.to(device)
    r2c_expected = torch.ops.aten._fft_r2c.default(real_cpu, dims, 0, True)
    r2c_out = torch.empty_like(r2c_expected, device=device)
    r2c_result = torch.ops.aten._fft_r2c.out(real_input, dims, 0, True, out=r2c_out)
    self.assertIs(r2c_result, r2c_out)
    self.assertEqual(r2c_out.cpu(), r2c_expected)

    c2r_input = r2c_expected.to(device)
    c2r_expected = torch.ops.aten._fft_c2r.default(
        r2c_expected, dims, 0, real_cpu.size(-1)
    )
    c2r_out = torch.empty_like(real_input)
    c2r_result = torch.ops.aten._fft_c2r.out(
        c2r_input, dims, 0, real_cpu.size(-1), out=c2r_out
    )
    self.assertIs(c2r_result, c2r_out)
    self.assertEqual(c2r_out.cpu(), c2r_expected, atol=2e-5, rtol=2e-5)

    noncontiguous_out = torch.empty(
        5, 4, 3, 2, device=device, dtype=complex_input.dtype
    ).permute(3, 2, 1, 0)
    original_stride = noncontiguous_out.stride()
    result = torch.ops.aten._fft_c2c.out(
        complex_input, dims, 0, True, out=noncontiguous_out
    )
    self.assertIs(result, noncontiguous_out)
    self.assertEqual(noncontiguous_out.stride(), original_stride)
    self.assertEqual(noncontiguous_out.cpu(), c2c_expected)


TestFFT.test_reference_1d = _test_reference_1d
TestFFT._compare_xpu_cpu = _compare_xpu_cpu
TestFFT.test_fft_half_and_chalf_not_power_of_two_error = (
    _test_fft_half_and_chalf_not_power_of_two
)
TestFFT.test_fft_out_variants = _test_fft_out_variants

instantiate_device_type_tests(TestFFT, globals(), only_for=("xpu"), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
