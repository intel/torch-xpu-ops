"""
Reproducer for: RuntimeError: iter.device(arg).is_xpu() in Loops.h
when torch.addcmul is called with a CPU scalar tensor (use_cpu_scalar=True).

The fix adds handling for is_cpu_scalar(3) in addcmul_kernel, extracting the
scalar value and running a 2-argument kernel on the XPU tensors.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ],
)
def test_addcmul_cpu_scalar_tensor2(dtype):
    """tensor2 as a CPU scalar tensor should work on XPU (use_cpu_scalar=True)."""
    device = "xpu"

    if dtype.is_floating_point or dtype.is_complex:
        a = torch.rand(3, 3, dtype=dtype, device=device)
        b = torch.rand(3, 3, dtype=dtype, device=device)
        # cpu scalar tensor for tensor2
        c = torch.tensor(2.0, dtype=dtype, device="cpu")
    else:
        a = torch.randint(1, 5, (3, 3), dtype=dtype, device=device)
        b = torch.randint(1, 5, (3, 3), dtype=dtype, device=device)
        c = torch.tensor(2, dtype=dtype, device="cpu")

    alpha = 0.5 if dtype.is_floating_point or dtype.is_complex else 3
    actual = torch.addcmul(a, b, c, value=alpha)
    expected = torch.addcmul(
        a.cpu(), b.cpu(), c, value=alpha
    ).to(device)
    assert torch.allclose(actual.float(), expected.float(), atol=1e-4), (
        f"Mismatch for dtype={dtype}: {actual} vs {expected}"
    )
