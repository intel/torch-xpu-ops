# Owner(s): ["module: intel"]
"""
Reproducer for: torch.autocast("xpu", dtype=torch.float64) silently disables autocast.

PyTorch's autocast_mode.py restricts non-CUDA devices to
device_supported_dtypes = [bfloat16, float16]. When float64 is passed for XPU,
autocast is silently disabled. As a result torch.mm(float32, float32) returns
float32 instead of float64, causing assertion failures in tests that check the
output dtype.

Fix: skip test_cuda_amp_autocast and test_autocast_float64 for XPU in
test/xpu/dynamo/test_ctx_manager_xpu.py.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_autocast_float64_disabled_on_xpu():
    """
    Verify that torch.autocast with dtype=float64 is silently disabled on XPU.
    float64 is not in device_supported_dtypes for non-CUDA devices, so autocast
    does not promote float32 inputs to float64.
    """
    a = torch.rand((8, 8), device="xpu")
    b = torch.rand((8, 8), device="xpu")
    with torch.autocast(device_type="xpu", dtype=torch.float64):
        enabled = torch.is_autocast_xpu_enabled()
    # autocast is disabled because float64 is not a supported dtype for XPU
    assert not enabled, (
        "Expected autocast to be disabled for XPU with float64, but it was enabled"
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_autocast_bfloat16_enabled_on_xpu():
    """
    Verify that torch.autocast with dtype=bfloat16 works correctly on XPU.
    bfloat16 is in device_supported_dtypes so autocast should be active and
    promote float32 matmul inputs to bfloat16.
    """
    a = torch.rand((8, 8), device="xpu")
    b = torch.rand((8, 8), device="xpu")
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        c = torch.mm(a, b)
    assert c.dtype == torch.bfloat16, (
        f"Expected bfloat16 output with autocast, got {c.dtype}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
