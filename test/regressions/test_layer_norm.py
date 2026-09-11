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
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


def _over_threshold_rows():
    """Row count above the kernel's two-stage column-reduction threshold.

    The kernel gates on `M > xe_core_count * 1024`, where its `xe_core_count` is
    `gpu_eu_count / gpu_eu_count_per_subslice` -- the same quotient PyTorch
    exposes as `gpu_subslice_count`. One extra tile of rows clears it.
    """
    xe_core_count = torch.xpu.get_device_properties(0).gpu_subslice_count
    return xe_core_count * 1024 + 1024


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

    def test_layer_norm_backward_multidim_normalized_shape_large_rows(self):
        """Weight and bias gradients keep normalized_shape above the threshold.

        Above `xe_core_count * 1024` rows the backward takes a two-stage column
        reduction whose accumulator is `{num_tile_m, N}` with N the flattened
        normalized size, so `sum(0)` is a flat `{N}` and assigning it replaced
        the gradient the caller had allocated from the parameter. Autograd then
        saw `[8]` where it expected `[2, 4]`. A one-dimensional normalized_shape
        hides it because `{N}` is already the right shape.
        """
        torch.manual_seed(42)
        dim = [2, 4]
        rows = _over_threshold_rows()

        x_cpu = torch.randn(rows, *dim, dtype=torch.float32, requires_grad=True)
        layernorm_cpu = nn.LayerNorm(dim, dtype=torch.float32)
        layernorm_cpu(x_cpu).sum().backward()

        x_xpu = x_cpu.detach().to(xpu_device).requires_grad_()
        layernorm_xpu = nn.LayerNorm(dim, dtype=torch.float32).to(xpu_device)
        layernorm_xpu.load_state_dict(layernorm_cpu.state_dict())
        layernorm_xpu(x_xpu).sum().backward()

        self.assertEqual(layernorm_xpu.weight.grad.shape, torch.Size(dim))
        self.assertEqual(layernorm_xpu.bias.grad.shape, torch.Size(dim))
        self.assertEqual(
            layernorm_cpu.weight.grad,
            layernorm_xpu.weight.grad.to(cpu_device),
            atol=1e-1,
            rtol=1e-3,
        )
        self.assertEqual(
            layernorm_cpu.bias.grad,
            layernorm_xpu.bias.grad.to(cpu_device),
            atol=1e-1,
            rtol=1e-3,
        )
        self.assertEqual(x_cpu.grad, x_xpu.grad.to(cpu_device))
