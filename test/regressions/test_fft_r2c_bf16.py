# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Regression test for https://github.com/intel/torch-xpu-ops/issues/4279
#
# bfloat16 real-to-complex (R2C) FFT previously failed on XPU with
# "MKL FFT doesn't support tensor of type". The XPU MKL backend now promotes
# BFloat16 input to Float32 in promote_fft_input
# (src/ATen/native/xpu/mkl/SpectralOps.cpp).
#
# This test monitors that the R2C output dtype does not silently change:
# for a bfloat16 input the R2C FFT output must be ComplexFloat (complex64),
# matching the CUDA contract (bf16 promotes to fp32, not fp16).

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFftR2cBFloat16(TestCase):
    def test_fft_r2c_bfloat16_output_dtype(self, device):
        # torch.fft.rfft is the R2C FFT
        x = torch.randn(100, dtype=torch.bfloat16, device=device)
        out = torch.fft.rfft(x, n=10)
        # bf16 promotes to fp32 -> R2C output must be complex64
        self.assertEqual(out.dtype, torch.complex64)


instantiate_device_type_tests(
    TestFftR2cBFloat16, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
