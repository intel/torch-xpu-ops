# Test: torch.full with out-of-range values for Half/BFloat16 on XPU should not
# raise RuntimeError but instead saturate to -inf (matching CPU behavior).
# Regression test for: https://github.com/intel/torch-xpu-ops/issues/2953

import math

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fill_overflow_saturates_to_inf(dtype):
    """torch.full with torch.finfo(float32).min into float16/bfloat16 must not raise."""
    fill_val = torch.finfo(torch.float32).min
    t = torch.full((4, 4), fill_val, dtype=dtype, device="xpu")
    assert t.shape == (4, 4)
    # Value overflows the reduced-precision type, so it should saturate to -inf.
    assert math.isinf(t[0, 0].item()) and t[0, 0].item() < 0


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fill_finfo_min_matches_cpu(dtype):
    """XPU and CPU must produce the same result for out-of-range fill values."""
    fill_val = torch.finfo(torch.float32).min
    cpu_t = torch.full((4, 4), fill_val, dtype=dtype, device="cpu")
    xpu_t = torch.full((4, 4), fill_val, dtype=dtype, device="xpu")
    assert torch.equal(cpu_t, xpu_t.cpu())
