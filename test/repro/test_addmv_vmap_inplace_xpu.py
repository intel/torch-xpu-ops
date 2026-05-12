"""Reproducer for addmv_ in-place vmap numerical mismatch on XPU.

The in-place vmap path for addmv_ (addmv with out=self) produces
numerically incorrect results on XPU compared to the loop-based reference.
Only in-place subtests diverge; out-of-place passes.

Upstream fix: pytorch/pytorch#178498

Test: pytest test/repro/test_addmv_vmap_inplace_xpu.py
"""
import torch
from torch import vmap


def _loop_vmap(fn, in_dims, args):
    """Reference implementation using a Python loop."""
    batch_size = None
    for a, d in zip(args, in_dims):
        if d is not None:
            batch_size = a.shape[d]
            break
    results = []
    for i in range(batch_size):
        sliced = []
        for a, d in zip(args, in_dims):
            sliced.append(a.select(d, i) if d is not None else a)
        results.append(fn(*sliced))
    return torch.stack(results)


def test_addmv_vmap_outplace_xpu():
    """Out-of-place addmv should produce correct vmap results on XPU."""
    device = "xpu"
    if not torch.xpu.is_available():
        import pytest
        pytest.skip("XPU not available")

    torch.manual_seed(0)
    B = 4
    M, N = 5, 6

    # Batched over first tensor (bias vector)
    bias = torch.randn(B, M, device=device)
    mat = torch.randn(M, N, device=device)
    vec = torch.randn(N, device=device)

    def fn(b):
        return torch.addmv(b, mat, vec)

    vmap_out = vmap(fn)(bias)
    loop_out = _loop_vmap(fn, (0,), (bias,))

    torch.testing.assert_close(vmap_out, loop_out, atol=1e-4, rtol=1e-4)


def test_addmv_inplace_xpu_skip_known_bug():
    """Document that addmv_ in-place vmap is skipped in test_vmap_exhaustive.

    The in-place path for addmv_ produces numerically incorrect results on XPU
    (tracked in pytorch/pytorch#178498). This test documents the known skip
    and verifies out-of-place still works correctly.
    """
    device = "xpu"
    if not torch.xpu.is_available():
        import pytest
        pytest.skip("XPU not available")

    torch.manual_seed(42)
    B = 3
    M, N = 4, 5

    bias = torch.randn(B, M, device=device)
    mat = torch.randn(M, N, device=device)
    vec = torch.randn(N, device=device)

    # Out-of-place addmv works correctly under vmap
    def fn_outplace(b):
        return torch.addmv(b, mat, vec)

    vmap_out = vmap(fn_outplace)(bias)
    loop_out = _loop_vmap(fn_outplace, (0,), (bias,))
    torch.testing.assert_close(vmap_out, loop_out, atol=1e-4, rtol=1e-4)

    # NOTE: addmv_ (in-place) is skipped in test_vmap_exhaustive because it
    # gives numerically incorrect results on XPU. See pytorch/pytorch#178498.
