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
