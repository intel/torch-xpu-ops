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
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


def _over_threshold_rows(device=xpu_device):
    """Row count above the two-stage column reduction's row threshold.

    The kernel gates on `M > xe_core_count * 1024`, where its `xe_core_count` is
    `gpu_eu_count / gpu_eu_count_per_subslice` -- the same quotient PyTorch
    exposes as `gpu_subslice_count`. One extra tile of rows clears it, but
    dtype, column count, and occupancy also determine which path runs.
    """
    xe_core_count = torch.xpu.get_device_properties(device).gpu_subslice_count
    return xe_core_count * 1024 + 1024


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

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layer_norm_low_precision_accuracy(self, dtype):
        shape = (1024, 32768)
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(shape, generator=generator).to(device=xpu_device, dtype=dtype)
        weight = torch.randn(shape[-1], generator=generator).to(
            device=xpu_device, dtype=dtype
        )
        bias = torch.randn(shape[-1], generator=generator).to(
            device=xpu_device, dtype=dtype
        )
        normalized_shape = (shape[-1],)

        # Promote quantized inputs so the baseline excludes input quantization error.
        baseline = nn.functional.layer_norm(
            x.float(), normalized_shape, weight.float(), bias.float(), eps=1e-5
        ).to(dtype)
        actual = nn.functional.layer_norm(x, normalized_shape, weight, bias, eps=1e-5)

        atol = 1e-5
        rtol = 1e-3 if dtype == torch.float16 else 1e-2
        self.assertEqual(actual, baseline, atol=atol, rtol=rtol)


class TestLayerNormMultidimBackward(TestCase):
    @parametrize("has_weight,has_bias", [(True, True), (True, False), (False, True)])
    def test_layer_norm_backward_multidim_normalized_shape_large_rows(
        self, device, has_weight, has_bias
    ):
        """Both parameter gradients, weight-only, and bias-only retain their shape."""
        torch.manual_seed(42)
        dim = [2, 4]
        rows = _over_threshold_rows(device)
        x = torch.randn(rows, *dim, dtype=torch.float32)
        weight = torch.randn(dim, dtype=torch.float32) if has_weight else None
        bias = torch.randn(dim, dtype=torch.float32) if has_bias else None
        grad_out = torch.randn_like(x)

        grads = []
        for dev in ("cpu", device):
            x_dev = x.to(dev).detach().requires_grad_()
            w_dev = weight.to(dev).detach().requires_grad_() if has_weight else None
            b_dev = bias.to(dev).detach().requires_grad_() if has_bias else None
            out = torch.nn.functional.layer_norm(x_dev, dim, w_dev, b_dev)
            out.backward(grad_out.to(dev))
            grads.append(
                (
                    x_dev.grad,
                    w_dev.grad if has_weight else None,
                    b_dev.grad if has_bias else None,
                )
            )

        for name, ref, grad in zip(("input", "weight", "bias"), grads[0], grads[1]):
            if ref is None:
                self.assertIsNone(grad)
                continue
            expected_shape = x.shape if name == "input" else torch.Size(dim)
            self.assertEqual(grad.shape, expected_shape, name)
            if name == "input":
                self.assertEqual(grad.cpu(), ref, name)
            else:
                self.assertEqual(grad.cpu(), ref, name, atol=1e-3, rtol=1e-3)


instantiate_device_type_tests(
    TestLayerNormMultidimBackward, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
