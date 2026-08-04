# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Regression test for the SYCL FFT kernel path.
#
# Tests 2D complex-to-complex FFTs at sizes supported by the SYCL kernel
# (512, 768) with both the SYCL path (USE_SYCL_SPECTRAL=1) and the XPU MKL
# path (USE_SYCL_SPECTRAL unset).

import os
import unittest
from itertools import product
from unittest import mock

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import spectral_funcs
from torch.testing._internal.common_utils import run_tests, TestCase


SUPPORTED_SIZES = [512, 768]

fft2_ops = [op for op in spectral_funcs if op.name in ("fft.fft2", "fft.ifft2")]


class TestFftC2CSyclKernel(TestCase):
    @ops(fft2_ops, allowed_dtypes=(torch.cfloat, torch.cdouble))
    @toleranceOverride(
        {
            torch.cfloat: tol(1e-3, 1e-5),
            torch.cdouble: tol(1e-7, 1e-10),
        }
    )
    @mock.patch.dict("os.environ", {"USE_SYCL_SPECTRAL": "1"})
    def test_fft2_sycl_path(self, device, dtype, op):
        """2D complex FFT/IFFT through the SYCL kernel path."""
        self._run_fft2(device, dtype, op)

    @ops(fft2_ops, allowed_dtypes=(torch.cfloat, torch.cdouble))
    @toleranceOverride(
        {
            torch.cfloat: tol(1e-3, 1e-5),
            torch.cdouble: tol(1e-7, 1e-10),
        }
    )
    @mock.patch.dict("os.environ", {}, clear=False)
    def test_fft2_mkl_path(self, device, dtype, op):
        """2D complex FFT/IFFT through the MKL path (USE_SYCL_SPECTRAL unset)."""
        os.environ.pop("USE_SYCL_SPECTRAL", None)
        self._run_fft2(device, dtype, op)

    def _run_fft2(self, device, dtype, op):
        dim = (-2, -1)

        for s0, s1 in product(SUPPORTED_SIZES, SUPPORTED_SIZES):
            input = torch.randn(4, s0, s1, device=device, dtype=dtype)
            input_t = input.transpose(-2, -1)
            for x in (input, input_t):
                expected = op.ref(x.cpu().numpy(), axes=dim)
                actual = op(x, dim=dim)
                self.assertEqual(
                    actual,
                    expected,
                    exact_dtype=False,
                )


instantiate_device_type_tests(
    TestFftC2CSyclKernel, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
