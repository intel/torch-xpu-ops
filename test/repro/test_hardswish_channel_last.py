# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Regression test for: channel-last aten::hardswish_ triggering extra copy kernel.

When hardswish_ is called in-place on a channel-last tensor the XPU backend
must apply the activation directly in memory-order (via multi_tensor_apply)
instead of going through TensorIterator, which would otherwise introduce an
extra contiguous-copy kernel launch.

The test verifies:
1. Numerical correctness of hardswish_ on channel-last 2-D and 4-D tensors.
2. Memory format is preserved after the in-place call.
"""

import torch
import pytest


def _hardswish_ref(x: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation of hardswish."""
    return x.cpu() * torch.clamp(x.cpu() + 3.0, min=0.0, max=6.0) / 6.0


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_inplace_channels_last_4d(dtype):
    """hardswish_ on a 4-D channel-last tensor must produce the correct result
    and preserve the channels_last memory format."""
    shape = (2, 8, 4, 4)
    x_cpu = torch.randn(shape, dtype=dtype)
    x_ref = x_cpu.clone()

    # Create a channel-last XPU tensor
    x_xpu = x_cpu.to(device="xpu", memory_format=torch.channels_last)
    assert x_xpu.is_contiguous(memory_format=torch.channels_last)
    assert not x_xpu.is_contiguous()  # not C-contiguous

    # Apply in-place hardswish on XPU
    x_xpu.hardswish_()

    # Apply reference on CPU
    ref = _hardswish_ref(x_ref)

    # Numerical check (relaxed tolerance for fp16/bf16)
    tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
    torch.testing.assert_close(x_xpu.cpu().float(), ref.float(), **tol)

    # Memory format must be preserved
    assert x_xpu.is_contiguous(memory_format=torch.channels_last), (
        "hardswish_ must preserve channels_last memory format"
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_inplace_channels_last_3d(dtype):
    """hardswish_ on a 5-D channel-last-3d tensor must produce the correct
    result and preserve the channels_last_3d memory format."""
    shape = (2, 4, 4, 4, 4)
    x_cpu = torch.randn(shape, dtype=dtype)
    x_ref = x_cpu.clone()

    x_xpu = x_cpu.to(device="xpu", memory_format=torch.channels_last_3d)
    assert x_xpu.is_contiguous(memory_format=torch.channels_last_3d)

    x_xpu.hardswish_()
    ref = _hardswish_ref(x_ref)

    tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
    torch.testing.assert_close(x_xpu.cpu().float(), ref.float(), **tol)

    assert x_xpu.is_contiguous(memory_format=torch.channels_last_3d), (
        "hardswish_ must preserve channels_last_3d memory format"
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_inplace_contiguous(dtype):
    """Sanity-check: hardswish_ on a contiguous XPU tensor still works."""
    shape = (4, 8, 8, 8)
    x_cpu = torch.randn(shape, dtype=dtype)
    x_ref = x_cpu.clone()

    x_xpu = x_cpu.to(device="xpu")
    assert x_xpu.is_contiguous()

    x_xpu.hardswish_()
    ref = _hardswish_ref(x_ref)

    tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
    torch.testing.assert_close(x_xpu.cpu().float(), ref.float(), **tol)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_hardswish_inplace_matches_out_of_place_channels_last():
    """In-place and out-of-place hardswish must agree on channel-last input."""
    shape = (2, 16, 8, 8)
    x_cpu = torch.randn(shape, dtype=torch.float32)

    x_inplace = x_cpu.to(device="xpu", memory_format=torch.channels_last)
    x_outplace = x_cpu.to(device="xpu", memory_format=torch.channels_last)

    # Out-of-place (uses TensorIterator with a freshly-allocated output)
    y_out = torch.nn.functional.hardswish(x_outplace)
    # In-place (our fixed multi_tensor_apply path)
    x_inplace.hardswish_()

    torch.testing.assert_close(x_inplace, y_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
