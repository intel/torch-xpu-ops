# Owner(s): ["module: tests"]

# Reproducer for https://github.com/intel/torch-xpu-ops/issues/3599
# XPU upsample_nearest3d kernel should support tensors with > INT32_MAX elements.
# Previously the kernel had an explicit TORCH_CHECK rejecting outputs exceeding
# INT32_MAX elements (~2.1 B). Upsampling (1, 256, 16, 720, 1280) by 2x yields
# ~30 B output elements, triggering that check.

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_upsample_nearest3d_large_tensor_xpu():
    """upsample_nearest3d must not raise when output exceeds INT32_MAX elements."""
    total_bytes = 1 * 256 * 32 * 1440 * 2560 * torch.finfo(torch.bfloat16).bits // 8
    free_mem = torch.xpu.get_device_properties(0).total_memory
    if free_mem < total_bytes:
        pytest.skip(f"Not enough XPU memory ({free_mem} < {total_bytes} bytes needed)")

    x = torch.ones((1, 256, 16, 720, 1280), dtype=torch.bfloat16, device="xpu")
    result = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    assert result.shape == torch.Size([1, 256, 32, 1440, 2560])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
