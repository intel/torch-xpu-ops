# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


@instantiate_parametrized_tests
class TestLayerNorm(TestCase):
    def test_layer_norm_no_nan(self, dtype=torch.float):
        dim = [5]
        x_cpu = torch.tensor([[1e15, 1e15 + 1, 1e15 + 2, 1e15 + 3, 1e15 + 4]])
        layernorm_cpu = nn.LayerNorm(dim)
        y_cpu = layernorm_cpu(x_cpu)

        x_xpu = x_cpu.to(xpu_device)
        layernorm_xpu = nn.LayerNorm(dim).to(xpu_device)
        y_xpu = layernorm_xpu(x_xpu)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_layer_norm_fast_path_forward(self, dtype):
        """Cover the sub-group-0 shuffle fast path in the forward kernel.

        The fast path is selected when the row spans a workgroup that contains
        a full SIMD-width subgroup (for wg_size == SIMD ^ 2).
        """
        torch.manual_seed(0)
        rows, norm = 1024, 8192
        x_cpu = torch.randn(rows, norm, dtype=torch.float32)

        layer_cpu = nn.LayerNorm(norm, dtype=torch.float32)
        y_cpu = layer_cpu(x_cpu)

        layer_xpu = nn.LayerNorm(norm, dtype=dtype).to(xpu_device)
        layer_xpu.weight.data.copy_(layer_cpu.weight.data)
        layer_xpu.bias.data.copy_(layer_cpu.bias.data)
        y_xpu = layer_xpu(x_cpu.to(dtype).to(xpu_device))

        tol = 1e-5 if dtype == torch.float32 else 1e-2
        self.assertEqual(y_xpu.float().cpu(), y_cpu, atol=tol, rtol=tol)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("norm", [220, 244, 221, 256])
    def test_layer_norm_constant_input_is_exactly_zero(self, dtype, norm):
        # 220 and 244 exercise unequal partial counts in the vectorized
        # reduction; 221 and 256 cover the fallback and power-of-two counts.
        x = torch.ones(4, norm, dtype=dtype, device=xpu_device)
        y = nn.functional.layer_norm(x, (norm,), eps=1e-5)
        self.assertEqual(y, torch.zeros_like(y), atol=0, rtol=0)

        _, mean, _ = torch.native_layer_norm(x, (norm,), None, None, 1e-5)
        self.assertEqual(mean, torch.ones_like(mean), atol=0, rtol=0)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("norm", [28, 220, 244, 221, 256, 8192, 8196])
    def test_layer_norm_constant_rows_preserve_mean(self, dtype, norm):
        # Include negative and large constants, and a wide row that exercises
        # merging between subgroups as well as the small-row reduction.
        values = torch.tensor([1.0, -3.0, 1000.0, 0.5], dtype=dtype, device=xpu_device)
        x = values[:, None].expand(-1, norm).contiguous()
        y, mean, _ = torch.native_layer_norm(x, (norm,), None, None, 1e-5)
        self.assertEqual(y, torch.zeros_like(y), atol=0, rtol=0)
        self.assertEqual(mean, values[:, None].to(mean.dtype), atol=0, rtol=0)

        weight = torch.full((norm,), 1.5, dtype=dtype, device=xpu_device)
        bias = torch.linspace(-0.25, 0.25, norm, dtype=dtype, device=xpu_device)
        y = nn.functional.layer_norm(x, (norm,), weight, bias, eps=1e-5)
        self.assertEqual(y, bias.expand_as(y), atol=0, rtol=0)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("norm", [220, 244, 221, 256])
    def test_layer_norm_constant_input_weight_grad_is_zero(self, dtype, norm):
        x = torch.ones(4, norm, dtype=dtype, device=xpu_device, requires_grad=True)
        weight = torch.ones(norm, dtype=dtype, device=xpu_device, requires_grad=True)
        bias = torch.zeros(norm, dtype=dtype, device=xpu_device, requires_grad=True)
        nn.functional.layer_norm(x, (norm,), weight, bias, eps=1e-5).sum().backward()
        self.assertEqual(weight.grad, torch.zeros_like(weight), atol=0, rtol=0)
        self.assertEqual(bias.grad, torch.full_like(bias, 4), atol=0, rtol=0)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_rms_norm_fast_path_forward(self, dtype):
        """Same fast path, rms_norm instantiation, [1024, 8192] fp32/fp16/bf16."""
        torch.manual_seed(0)
        rows, norm = 1024, 8192
        x_cpu = torch.randn(rows, norm, dtype=torch.float32)

        layer_cpu = nn.RMSNorm(norm, eps=1e-6, dtype=torch.float32)
        y_cpu = layer_cpu(x_cpu)

        layer_xpu = nn.RMSNorm(norm, eps=1e-6, dtype=dtype).to(xpu_device)
        layer_xpu.weight.data.copy_(layer_cpu.weight.data)
        y_xpu = layer_xpu(x_cpu.to(dtype).to(xpu_device))

        tol = 1e-5 if dtype == torch.float32 else 1e-2
        self.assertEqual(y_xpu.float().cpu(), y_cpu, atol=tol, rtol=tol)


if __name__ == "__main__":
    run_tests()
