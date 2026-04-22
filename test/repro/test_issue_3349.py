"""
Reproducer for https://github.com/intel/torch-xpu-ops/issues/3349

torch.native_batch_norm in eval mode returns populated save_mean/save_invstd
on XPU, while CPU returns size-0 tensors for outputs 1 and 2.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required")
def test_native_batch_norm_eval_output_shapes():
    """save_mean and save_invstd should be size-0 tensors in eval mode, matching CPU behavior."""
    torch.manual_seed(0)

    x = torch.randn(2, 3, 4, 4)
    weight = torch.randn(3)
    bias = torch.randn(3)
    running_mean = torch.randn(3)
    running_var = torch.randn(3)

    cpu_out = torch.native_batch_norm(
        x.cpu(),
        weight.cpu(),
        bias.cpu(),
        running_mean.cpu(),
        running_var.cpu(),
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    xpu_out = torch.native_batch_norm(
        x.to("xpu"),
        weight.to("xpu"),
        bias.to("xpu"),
        running_mean.to("xpu"),
        running_var.to("xpu"),
        training=False,
        momentum=0.1,
        eps=1e-5,
    )

    for i, (cpu_item, xpu_item) in enumerate(zip(cpu_out, xpu_out)):
        assert cpu_item.shape == xpu_item.cpu().shape, (
            f"Output {i} shape mismatch: CPU {cpu_item.shape} vs XPU {xpu_item.shape}"
        )
