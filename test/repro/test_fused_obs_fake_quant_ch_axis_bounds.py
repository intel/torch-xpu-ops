# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import pytest

import torch


def _make_inputs(device):
    x = torch.randn(2, 3, 4, device=device)
    observer_on = torch.tensor([1], dtype=torch.long, device=device)
    fake_quant_on = torch.tensor([1], dtype=torch.long, device=device)
    running_min = torch.tensor([], dtype=torch.float, device=device)
    running_max = torch.tensor([], dtype=torch.float, device=device)
    scale = torch.tensor([], dtype=torch.float, device=device)
    zero_point = torch.tensor([], dtype=torch.int32, device=device)
    return x, observer_on, fake_quant_on, running_min, running_max, scale, zero_point


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_large_negative_ch_axis_raises():
    """Out-of-range negative ch_axis must raise RuntimeError, not segfault."""
    args = _make_inputs("xpu")
    with pytest.raises(RuntimeError, match="out of range"):
        torch._fused_moving_avg_obs_fq_helper(
            *args,
            averaging_const=0.01,
            quant_min=0,
            quant_max=255,
            ch_axis=-1250999896764,
            per_row_fake_quant=True,
            symmetric_quant=False,
        )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_negative_one_ch_axis_wraps():
    """ch_axis=-1 should wrap to the last dimension without error."""
    args = _make_inputs("xpu")
    # Should not raise
    torch._fused_moving_avg_obs_fq_helper(
        *args,
        averaging_const=0.01,
        quant_min=0,
        quant_max=255,
        ch_axis=-1,
        per_row_fake_quant=True,
        symmetric_quant=False,
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_positive_out_of_range_ch_axis_raises():
    """ch_axis >= x.dim() must raise RuntimeError."""
    args = _make_inputs("xpu")
    with pytest.raises(RuntimeError, match="out of range"):
        torch._fused_moving_avg_obs_fq_helper(
            *args,
            averaging_const=0.01,
            quant_min=0,
            quant_max=255,
            ch_axis=3,
            per_row_fake_quant=True,
            symmetric_quant=False,
        )
