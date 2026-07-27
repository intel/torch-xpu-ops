# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Reproducer / regression test for: mixed non-atomic/atomic memory access in
CAS-loop initial loads (Atomics.h).

Background
----------
All CAS-loop implementations in src/ATen/native/xpu/sycl/Atomics.h previously
seeded the ``assumed`` value via a plain (non-atomic) pointer dereference before
constructing ``sycl::atomic_ref``:

    uint32_t assumed = *address_as_ui;                          # plain OpLoad
    sycl_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);
    do { ... } while (!target.compare_exchange_strong(assumed, newval));

Under the SPIR-V / OpenCL Unified Memory Model (section 7 of the SPIR-V spec),
mixing a non-atomic ``OpLoad`` with ``OpAtomicCompareExchange`` on the same
memory location is **undefined behaviour**: the two accesses belong to different
synchronisation domains, so the compiler or runtime is permitted to reorder or
eliminate the plain load relative to the surrounding atomic operations.

Why observable failures are rare / non-deterministic
-----------------------------------------------------
The CAS retry loop is inherently **self-correcting**: when
``compare_exchange_strong`` fails it writes the *current* memory value back into
``assumed`` before the next iteration, so a stale initial load only costs at
most one extra retry.  On today's Intel GPU hardware the UB rarely manifests as
a visible wrong result; the fix is therefore about **spec compliance** (ensuring
every access goes through ``sycl::atomic_ref``) rather than correcting an always-
reproducible numerical error.

What this test verifies
-----------------------
These tests exercise every affected CAS-loop template (AtomicIntegerImpl 1/2/4/8
bytes, AtomicFPImpl half/bfloat16/float/double, and both global/local variants)
under heavy contention and verify that the results match the CPU reference.  They
serve as:

1. A regression guard — the tests must pass with the fixed code.
2. A stress harness — the high-collision variants (all indices → same slot)
   maximise the probability of exposing any future regression where the initial
   speculative value becomes permanently stale (e.g. due to a compiler hoisting
   the plain load out of the loop under a future SPIR-V optimisation).

Because the underlying issue is UB rather than a deterministic bug, running
these tests against the *original* code on current hardware may still pass;
the value of the test is the spec-correct code path it exercises and the
correctness guarantee it provides for future compiler/hardware generations.
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


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_integer_high_collision(xpu, dtype):
    """
    High-collision stress test: all work-items write to the same slot.

    All n elements map to index 0, forcing every CAS invocation to contend on
    the same address.  This maximises the number of CAS retries and is the
    scenario most likely to expose a stale initial load: if the compiler were to
    hoist the initial speculative load out of the kernel dispatch (permitted by
    the SPIR-V spec for plain OpLoad but not for OpAtomicLoad), the ``assumed``
    value could become permanently stale and cause lost updates.
    """
    # Use a smaller n for int8 to avoid overflow (max value 127)
    n = 64 if dtype == torch.int8 else 512
    src = torch.ones(n, dtype=dtype, device=xpu)
    index = torch.zeros(n, dtype=torch.int64, device=xpu)
    dst = torch.zeros(1, dtype=dtype, device=xpu)
    dst.scatter_add_(0, index, src)

    expected = torch.tensor([n], dtype=dtype)
    assert torch.equal(dst.cpu(), expected), (
        f"Lost updates detected for dtype={dtype}: expected {n}, got {dst[0].item()}"
    )


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


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_fp_high_collision(xpu, dtype):
    """
    High-collision stress test for FP CAS paths: all work-items write to the
    same slot.  Any lost update from a stale initial load would cause the sum
    to fall short of the expected value.
    """
    n = 512
    # Use value 1.0 so the expected sum is exactly n (exact in all FP formats)
    src = torch.ones(n, dtype=dtype, device=xpu)
    index = torch.zeros(n, dtype=torch.int64, device=xpu)
    dst = torch.zeros(1, dtype=dtype, device=xpu)
    dst.index_add_(0, index, src)

    # 512 * 1.0 = 512.0 is exactly representable in all FP formats including
    # float16 and bfloat16 (integers up to 2048 are exact in float16), so a
    # tolerance of 0.5 catches any single lost atomic update (which would
    # reduce the sum to at most 511.0).
    atol = 0.5
    assert abs(dst[0].item() - n) <= atol, (
        f"Lost updates detected for dtype={dtype}: expected {n}, got {dst[0].item()}"
    )


def test_histc_float_atomic(xpu):
    """Exercises float AtomicFPImpl CAS path via histc."""
    x = torch.rand(1024, device=xpu)
    result = torch.histc(x, bins=32, min=0.0, max=1.0)
    assert result.sum().item() == pytest.approx(1024, abs=1)


if __name__ == "__main__":
    raise RuntimeError(
        "This reproducer uses pytest fixtures. "
        "Run with: pytest test/repro/test_atomic_cas_initial_load.py"
    )
