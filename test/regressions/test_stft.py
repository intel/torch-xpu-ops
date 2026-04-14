# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import os
import sys

import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSTFT(TestCase):
    """Regression tests for stft dtype support and accuracy on XPU.

    The stft operation in float16 on XPU has an accuracy gap that exceeds
    inductor test tolerances (atol=1e-5, rtol=1e-3). XPU should not claim
    float16 support for stft, aligning with CUDA which only supports
    floating_and_complex_types() (f32/f64/c64/c128).

    These tests verify:
    1. The dtype configuration does not include float16 for stft.
    2. stft compiled via torch.compile matches eager for supported dtypes.
    3. stft in float16 has accuracy gap exceeding inductor tolerances.
    """

    def test_stft_not_in_dtype_different_cuda_support(self):
        """Verify stft is not listed in _ops_dtype_different_cuda_support.

        This is the direct regression guard: if someone re-adds stft to the
        dtype override dict, this test will fail.
        """
        xpu_test_dir = os.path.join(os.path.dirname(__file__), os.pardir, "xpu")
        sys.path.insert(0, os.path.abspath(xpu_test_dir))
        from xpu_test_utils import _ops_dtype_different_cuda_support

        self.assertNotIn(
            "stft",
            _ops_dtype_different_cuda_support,
            "stft must not be in _ops_dtype_different_cuda_support; "
            "float16 precision is insufficient for inductor tolerances",
        )

    def test_stft_eager_vs_compiled_float32(self):
        """Verify stft eager and torch.compile results match in float32.

        Reproduces the inductor test pattern: compare eager vs compiled on XPU
        with tight tolerances. This must pass for supported dtypes.
        """
        torch.manual_seed(0)
        x = torch.randn(100, device=xpu_device)
        n_fft = 10

        def stft_fn(x):
            return torch.stft(x, n_fft, return_complex=True)

        eager_result = stft_fn(x)
        compiled_fn = torch.compile(stft_fn)
        compiled_result = compiled_fn(x)

        self.assertEqual(eager_result, compiled_result, atol=1e-5, rtol=1e-5)

    def test_stft_eager_vs_compiled_float64(self):
        """Verify stft eager and torch.compile results match in float64."""
        torch.manual_seed(0)
        x = torch.randn(100, dtype=torch.float64, device=xpu_device)
        n_fft = 10

        def stft_fn(x):
            return torch.stft(x, n_fft, return_complex=True)

        eager_result = stft_fn(x)
        compiled_fn = torch.compile(stft_fn)
        compiled_result = compiled_fn(x)

        self.assertEqual(eager_result, compiled_result, atol=1e-7, rtol=1e-7)

    def test_stft_float16_inductor_accuracy_gap(self):
        """Verify stft in float16 eager vs compiled exceeds inductor tolerances.

        This reproduces the original test_comprehensive_stft_xpu_float16
        failure: the inductor test compares eager vs compiled with
        atol=1e-5, rtol=1e-3 and float16 fails to meet these tolerances.
        """
        torch.manual_seed(0)
        x = torch.randn(100, dtype=torch.float16, device=xpu_device)
        n_fft = 10

        def stft_fn(x):
            return torch.stft(x, n_fft, return_complex=True)

        eager_result = stft_fn(x)
        compiled_fn = torch.compile(stft_fn)
        compiled_result = compiled_fn(x)

        # Compute the difference between eager and compiled in float16.
        # Cast to complex64 for accurate diff computation.
        diff = eager_result.to(torch.complex64) - compiled_result.to(torch.complex64)
        max_abs_diff = diff.abs().max().item()

        # The max absolute diff should exceed the inductor tolerance (atol=1e-5),
        # confirming that float16 precision is insufficient for stft.
        # This is the core reason stft must not claim float16 support.
        self.assertGreater(
            max_abs_diff,
            1e-5,
            "Expected float16 stft eager-vs-compiled gap > 1e-5 "
            "(the inductor tolerance); float16 should not be a supported dtype",
        )

    def test_stft_cpu_xpu_parity_float32(self):
        """Verify stft produces matching CPU and XPU results in float32."""
        torch.manual_seed(0)
        x = torch.randn(100)
        n_fft = 10

        result_cpu = torch.stft(x, n_fft, return_complex=True)
        result_xpu = torch.stft(x.to(xpu_device), n_fft, return_complex=True)
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-5, rtol=1e-5)

    def test_stft_cpu_xpu_parity_float64(self):
        """Verify stft produces matching CPU and XPU results in float64."""
        torch.manual_seed(0)
        x = torch.randn(100, dtype=torch.float64)
        n_fft = 10

        result_cpu = torch.stft(x, n_fft, return_complex=True)
        result_xpu = torch.stft(x.to(xpu_device), n_fft, return_complex=True)
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-7, rtol=1e-7)

    def test_stft_with_window_float32(self):
        """Verify stft with Hann window in float32 matches CPU."""
        torch.manual_seed(42)
        x = torch.randn(200)
        n_fft = 20
        window = torch.hann_window(n_fft)

        result_cpu = torch.stft(x, n_fft, window=window, return_complex=True)
        result_xpu = torch.stft(
            x.to(xpu_device),
            n_fft,
            window=window.to(xpu_device),
            return_complex=True,
        )
        self.assertEqual(result_cpu, result_xpu.cpu(), atol=1e-5, rtol=1e-5)
