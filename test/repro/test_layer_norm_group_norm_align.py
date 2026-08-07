# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

# Regression tests for XPU LayerNorm and GroupNorm kernels.
# These tests validate XPU correctness by comparing against CPU reference
# implementations, covering the same kernel paths as the CUDA/ROCm regression
# tests in stock PyTorch (small-batch, large-batch tile-based reduction, etc.).

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

xpu_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestLayerNormAlign(TestCase):
    def _run_layer_norm(self, dtype, N, C, eps=1e-5):
        """Helper: compare XPU LayerNorm forward+backward with CPU."""
        shape = (N, C)
        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        weight_cpu = torch.randn(C, dtype=dtype, requires_grad=True)
        bias_cpu = torch.randn(C, dtype=dtype, requires_grad=True)

        x_xpu = x_cpu.detach().clone().to(xpu_device).requires_grad_(True)
        weight_xpu = weight_cpu.detach().clone().to(xpu_device).requires_grad_(True)
        bias_xpu = bias_cpu.detach().clone().to(xpu_device).requires_grad_(True)

        # Forward
        y_cpu = nn.functional.layer_norm(
            x_cpu, [C], weight=weight_cpu, bias=bias_cpu, eps=eps
        )
        y_xpu = nn.functional.layer_norm(
            x_xpu, [C], weight=weight_xpu, bias=bias_xpu, eps=eps
        )

        atol = 1e-3 if dtype == torch.float16 else 1e-5
        rtol = 1e-3 if dtype == torch.float16 else 1e-5
        self.assertEqual(y_cpu, y_xpu.to(cpu_device), atol=atol, rtol=rtol)

        # Backward
        grad_output = torch.randn_like(y_cpu)
        y_cpu.backward(grad_output)
        y_xpu.backward(grad_output.to(xpu_device))

        self.assertEqual(x_cpu.grad, x_xpu.grad.to(cpu_device), atol=atol, rtol=rtol)
        self.assertEqual(
            weight_cpu.grad, weight_xpu.grad.to(cpu_device), atol=atol, rtol=rtol
        )
        self.assertEqual(
            bias_cpu.grad, bias_xpu.grad.to(cpu_device), atol=atol, rtol=rtol
        )

    def test_layer_norm_float32_vectorized(self):
        # N=32, C=128 (multiple of vec_size=4): exercises vectorized forward/backward path
        self._run_layer_norm(torch.float32, N=32, C=128)

    def test_layer_norm_float16_vectorized(self):
        self._run_layer_norm(torch.float16, N=32, C=128)

    def test_layer_norm_bfloat16_vectorized(self):
        self._run_layer_norm(torch.bfloat16, N=32, C=128)

    def test_layer_norm_float32_non_vectorized(self):
        # C=5: not a multiple of vec_size=4, exercises non-vectorized path
        self._run_layer_norm(torch.float32, N=32, C=5)

    def test_layer_norm_large_m_two_stage_reduction(self):
        # Large M, small N: exercises two-stage column reduction for dgamma/dbeta
        self._run_layer_norm(torch.float32, N=256, C=64)


class TestGroupNormAlign(TestCase):
    def _run_group_norm(self, dtype, N, C, G, HxW=1, eps=1e-5):
        """Helper: compare XPU GroupNorm forward+backward with CPU."""
        shape = (N, C) if HxW == 1 else (N, C, HxW)
        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        weight_cpu = torch.randn(C, dtype=dtype, requires_grad=True)
        bias_cpu = torch.randn(C, dtype=dtype, requires_grad=True)

        x_xpu = x_cpu.detach().clone().to(xpu_device).requires_grad_(True)
        weight_xpu = weight_cpu.detach().clone().to(xpu_device).requires_grad_(True)
        bias_xpu = bias_cpu.detach().clone().to(xpu_device).requires_grad_(True)

        # Forward
        y_cpu = nn.functional.group_norm(
            x_cpu, G, weight=weight_cpu, bias=bias_cpu, eps=eps
        )
        y_xpu = nn.functional.group_norm(
            x_xpu, G, weight=weight_xpu, bias=bias_xpu, eps=eps
        )

        atol = 1e-3 if dtype == torch.float16 else 1e-4
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        self.assertEqual(y_cpu, y_xpu.to(cpu_device), atol=atol, rtol=rtol)

        # Backward
        grad_output = torch.randn_like(y_cpu)
        y_cpu.backward(grad_output)
        y_xpu.backward(grad_output.to(xpu_device))

        self.assertEqual(x_cpu.grad, x_xpu.grad.to(cpu_device), atol=atol, rtol=rtol)
        self.assertEqual(
            weight_cpu.grad, weight_xpu.grad.to(cpu_device), atol=atol, rtol=rtol
        )
        self.assertEqual(
            bias_cpu.grad, bias_xpu.grad.to(cpu_device), atol=atol, rtol=rtol
        )

    def test_group_norm_1d_small_batch_float32(self):
        # N <= 128: exercises GammaBeta1dBackwardSmallKernel path
        self._run_group_norm(torch.float32, N=64, C=32, G=4)

    def test_group_norm_1d_large_batch_float32(self):
        # N > 128: exercises GammaBeta1dBackwardLargeKernel tile-based reduction path
        self._run_group_norm(torch.float32, N=256, C=32, G=4)

    def test_group_norm_1d_large_batch_float16(self):
        # N > 128 with float16
        self._run_group_norm(torch.float16, N=256, C=32, G=4)

    def test_group_norm_1d_large_batch_bfloat16(self):
        # N > 128 with bfloat16
        self._run_group_norm(torch.bfloat16, N=256, C=32, G=4)

    def test_group_norm_2d_small_batch_float32(self):
        # HxW > 1, N <= 128: exercises ComputeInternalGradients + GammaBetaBackwardPlain path
        self._run_group_norm(torch.float32, N=64, C=32, G=4, HxW=16)

    def test_group_norm_2d_large_batch_float32(self):
        # HxW > 1, N > 128: exercises ComputeInternalGradients + GammaBetaBackward tile path
        self._run_group_norm(torch.float32, N=256, C=32, G=4, HxW=16)

    def test_group_norm_2d_large_batch_float16(self):
        # HxW > 1, N > 128, float16
        self._run_group_norm(torch.float16, N=256, C=32, G=4, HxW=16)


if __name__ == "__main__":
    run_tests()
