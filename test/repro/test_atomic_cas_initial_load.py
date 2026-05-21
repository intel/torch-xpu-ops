# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Reproducer for: mixed non-atomic/atomic memory access in CAS loop initial loads.

Exercises AtomicIntegerImpl / AtomicFPImpl CAS-loop code paths in
src/ATen/native/xpu/sycl/Atomics.h via scatter_add_ and histc on XPU,
using dtypes that map to each specialization (uint8/int8/int16/int32/int64
for integer CAS; float16/bfloat16/float32/float64 for FP CAS).
"""

import pytest
import torch


@pytest.fixture
def xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU device not available")
    return torch.device("xpu")


# --- Integer CAS paths (AtomicIntegerImpl<T, N>) ---


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_scatter_add_integer_dtypes(xpu, dtype):
    """Exercises AtomicIntegerImpl 1/2/4/8-byte CAS paths via scatter_add_."""
    n = 128
    src = torch.ones(n, dtype=dtype, device=xpu)
    index = torch.randint(0, n // 2, (n,), device=xpu)
    dst = torch.zeros(n // 2, dtype=dtype, device=xpu)
    dst.scatter_add_(0, index, src)

    # Verify against CPU reference
    src_cpu = src.cpu()
    index_cpu = index.cpu()
    dst_ref = torch.zeros(n // 2, dtype=dtype)
    dst_ref.scatter_add_(0, index_cpu, src_cpu)
    assert torch.equal(dst.cpu(), dst_ref), f"Mismatch for dtype={dtype}"


# --- FP CAS paths (AtomicFPImpl<T>) ---


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_scatter_add_fp_dtypes(xpu, dtype):
    """Exercises AtomicFPImpl CAS paths (half/bfloat16/float/double) via scatter_add_."""
    n = 128
    src = torch.ones(n, dtype=dtype, device=xpu)
    index = torch.randint(0, n // 2, (n,), device=xpu)
    dst = torch.zeros(n // 2, dtype=dtype, device=xpu)
    dst.scatter_add_(0, index, src)

    src_cpu = src.cpu().float()
    index_cpu = index.cpu()
    dst_ref = torch.zeros(n // 2, dtype=torch.float32)
    dst_ref.scatter_add_(0, index_cpu, src_cpu)
    assert torch.allclose(dst.cpu().float(), dst_ref, atol=1e-2), (
        f"Mismatch for dtype={dtype}"
    )


def test_histc_float_atomic(xpu):
    """Exercises float AtomicFPImpl CAS path via histc."""
    x = torch.rand(1024, device=xpu)
    result = torch.histc(x, bins=32, min=0.0, max=1.0)
    assert result.sum().item() == pytest.approx(1024, abs=1)


def test_index_add_concurrent_atomic(xpu):
    """Exercises CAS loops with many collisions (worst-case CAS retry)."""
    n = 1024
    src = torch.ones(n, dtype=torch.float32, device=xpu)
    index = torch.zeros(n, dtype=torch.int64, device=xpu)  # all map to index 0
    dst = torch.zeros(1, dtype=torch.float32, device=xpu)
    dst.index_add_(0, index, src)
    assert dst[0].item() == pytest.approx(n, abs=1e-3)


if __name__ == "__main__":
    raise RuntimeError(
        "This reproducer uses pytest fixtures. Run with: pytest test/repro/test_atomic_cas_initial_load.py"
    )
