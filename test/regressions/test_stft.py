# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSTFT(TestCase):
    """Regression tests for stft accuracy on XPU.

    See https://github.com/intel/torch-xpu-ops/issues/3277
    The stft operation in float16 on XPU has an accuracy gap that exceeds
    inductor test tolerances (atol=1e-5, rtol=1e-3). XPU should not claim
    float16 support for stft, aligning with CUDA which only supports
    floating_and_complex_types() (f32/f64/c64/c128).
    """

    def test_stft_float32_cpu_xpu_match(self):
        """Verify stft in float32 produces matching results on CPU and XPU."""
        torch.manual_seed(0)
        x = torch.randn(100)
        n_fft = 10

        result_cpu = torch.stft(
            x.to(cpu_device), n_fft, return_complex=True
        )
        result_xpu = torch.stft(
            x.to(xpu_device), n_fft, return_complex=True
        )
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-5, rtol=1e-5)

    def test_stft_float64_cpu_xpu_match(self):
        """Verify stft in float64 produces matching results on CPU and XPU."""
        torch.manual_seed(0)
        x = torch.randn(100, dtype=torch.float64)
        n_fft = 10

        result_cpu = torch.stft(
            x.to(cpu_device), n_fft, return_complex=True
        )
        result_xpu = torch.stft(
            x.to(xpu_device), n_fft, return_complex=True
        )
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-7, rtol=1e-7)

    def test_stft_float16_accuracy_gap(self):
        """Verify stft in float16 has an accuracy gap exceeding inductor tolerances.

        This test reproduces the original failure from
        test_comprehensive_stft_xpu_float16, confirming that float16 precision
        is insufficient for stft. The inductor test uses atol=1e-5, rtol=1e-3.
        """
        torch.manual_seed(0)
        x_f32 = torch.randn(100)
        n_fft = 10

        # Compute reference in float32 on CPU
        ref_cpu = torch.stft(
            x_f32.to(cpu_device), n_fft, return_complex=True
        )

        # Compute in float16 on XPU (cast input to float16, then back to
        # compare against float32 reference)
        x_f16_xpu = x_f32.to(torch.float16).to(xpu_device)
        result_xpu = torch.stft(x_f16_xpu, n_fft, return_complex=True)
        result_f32 = result_xpu.cpu().to(torch.complex64)

        # The float16 result should NOT meet the tight inductor tolerances
        # (atol=1e-5, rtol=1e-3) when compared to the float32 reference.
        # This confirms the accuracy gap that motivated removing float16
        # from stft's supported dtypes on XPU.
        max_abs_diff = (ref_cpu - result_f32).abs().max().item()
        self.assertGreater(
            max_abs_diff,
            1e-5,
            "Expected float16 stft to have accuracy gap > 1e-5 vs float32 reference",
        )

    def test_stft_with_window_float32(self):
        """Verify stft with a Hann window in float32 matches CPU."""
        torch.manual_seed(42)
        x = torch.randn(200)
        n_fft = 20
        window = torch.hann_window(n_fft)

        result_cpu = torch.stft(
            x.to(cpu_device),
            n_fft,
            window=window.to(cpu_device),
            return_complex=True,
        )
        result_xpu = torch.stft(
            x.to(xpu_device),
            n_fft,
            window=window.to(xpu_device),
            return_complex=True,
        )
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-5, rtol=1e-5)
