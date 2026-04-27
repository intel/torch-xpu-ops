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

import torch
from torch.testing._internal.common_utils import run_tests, TEST_NUMPY

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

if TEST_NUMPY:
    import numpy

# numba.cuda may fail to import due to numpy ABI. Detect at module scope.
try:
    import numba.cuda  # noqa: F401

    TEST_NUMBA_XPU = True
except Exception:
    TEST_NUMBA_XPU = False

with XPUPatchForImport(False):
    from test_numba_integration import TestNumbaIntegration


@unittest.skipIf(not TEST_NUMPY, "No numpy")
@unittest.skipIf(not torch.xpu.is_available(), "No xpu")
def _test_cuda_array_interface(self):
    """XPU tensors do not expose __cuda_array_interface__.

    __cuda_array_interface__ is a CUDA-specific protocol. XPU has no
    equivalent interface exposed on torch.Tensor; this test tracks the
    gap by asserting the CUDA-only tensor types behave as expected on
    the XPU backend.
    """
    types = [
        torch.DoubleTensor,
        torch.FloatTensor,
        torch.HalfTensor,
        torch.LongTensor,
        torch.IntTensor,
        torch.ShortTensor,
        torch.CharTensor,
        torch.ByteTensor,
    ]
    for tp in types:
        cput = tp(10)
        self.assertFalse(hasattr(cput, "__cuda_array_interface__"))
        self.assertRaises(AttributeError, lambda: cput.__cuda_array_interface__)

        # XPU tensors do not expose __cuda_array_interface__ (unlike CUDA).
        xput = tp(10).xpu()
        self.assertFalse(hasattr(xput, "__cuda_array_interface__"))
        self.assertRaises(AttributeError, lambda: xput.__cuda_array_interface__)


@unittest.skipIf(not torch.xpu.is_available(), "No xpu")
@unittest.skipIf(not TEST_NUMBA_XPU, "No numba.cuda")
def _test_array_adaptor(self):
    """numba.cuda.as_cuda_array rejects XPU tensors.

    XPU tensors are not CUDA tensors and therefore are not cuda arrays.
    """
    torch_dtypes = [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ]
    for dt in torch_dtypes:
        cput = torch.arange(10).to(dt)
        self.assertFalse(numba.cuda.is_cuda_array(cput))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput)

        xput = cput.to(device="xpu")
        self.assertFalse(numba.cuda.is_cuda_array(xput))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(xput)


@unittest.skipIf(not torch.xpu.is_available(), "No xpu")
@unittest.skipIf(not TEST_NUMBA_XPU, "No numba.cuda")
def _test_conversion_errors(self):
    """numba.cuda detects that XPU / CPU / sparse tensors are not cuda arrays."""
    # CPU tensors are not cuda arrays.
    cput = torch.arange(100)
    self.assertFalse(numba.cuda.is_cuda_array(cput))
    with self.assertRaises(TypeError):
        numba.cuda.as_cuda_array(cput)

    # Sparse tensors are not cuda arrays, regardless of device.
    sparset = torch.sparse_coo_tensor(cput[None, :], cput)
    self.assertFalse(numba.cuda.is_cuda_array(sparset))
    with self.assertRaises(TypeError):
        numba.cuda.as_cuda_array(sparset)

    sparset.xpu()
    self.assertFalse(numba.cuda.is_cuda_array(sparset))
    with self.assertRaises(TypeError):
        numba.cuda.as_cuda_array(sparset)

    # CPU+gradient isn't a cuda array.
    cpu_gradt = torch.zeros(100).requires_grad_(True)
    self.assertFalse(numba.cuda.is_cuda_array(cpu_gradt))
    with self.assertRaises(TypeError):
        numba.cuda.as_cuda_array(cpu_gradt)

    # XPU+gradient is also not a cuda array (XPU has no __cuda_array_interface__).
    xpu_gradt = torch.zeros(100).requires_grad_(True).xpu()
    self.assertFalse(numba.cuda.is_cuda_array(xpu_gradt))
    with self.assertRaises(TypeError):
        numba.cuda.as_cuda_array(xpu_gradt)


TestNumbaIntegration.test_cuda_array_interface = _test_cuda_array_interface
TestNumbaIntegration.test_array_adaptor = _test_array_adaptor
TestNumbaIntegration.test_conversion_errors = _test_conversion_errors


if __name__ == "__main__":
    run_tests()
