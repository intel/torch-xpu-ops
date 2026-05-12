# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Regression test for incorrect strides in addmv/mv XPU ops.

Related issue: https://github.com/intel/torch-xpu-ops/issues/2669
Failing tests:
  - test_ops_xpu.py::TestCommonXPU::test_out_addmv_xpu_float32
  - test_ops_xpu.py::TestCommonXPU::test_out_mv_xpu_float32
  - test_vmap_xpu.py::TestVmapOperatorsOpInfoXPU::test_vmap_exhaustive_addmv_xpu_float32

Root cause: PyTorch core's XPU implementations of addmv.out and mv.out changed
the strides of the output tensor from (2,) to (1,) when called with a
non-contiguous output, causing assertion failures.

Fix: Added addmv_out_xpu in torch-xpu-ops that uses mm + copy_ to correctly
write into non-contiguous output tensors while preserving their strides.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_addmv_out_preserves_strides_xpu():
    """
    Test that addmv.out preserves the strides of a non-contiguous output tensor.
    """
    device = "xpu"
    m, k = 4, 3

    mat = torch.randn(m, k, device=device)
    vec = torch.randn(k, device=device)
    self = torch.randn(m, device=device)

    # Create expected output (contiguous)
    expected = torch.addmv(self, mat, vec)

    # Create a non-contiguous output with stride=2
    buf = torch.zeros(2 * m, device=device)
    out = buf[::2]  # stride=(2,), shape=(m,)
    assert out.stride() == (2,), "pre-condition: out has stride 2"

    # Call addmv with out= parameter
    torch.addmv(self, mat, vec, out=out)

    # Check that strides are preserved
    assert out.stride() == (2,), (
        f"Strides changed! Expected (2,) but got {out.stride()}"
    )

    # Check that values are correct
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_addmv_out_beta_zero_preserves_strides_xpu():
    """
    Test addmv.out with beta=0 preserves strides (avoids NaN from self).
    """
    device = "xpu"
    m, k = 4, 3

    mat = torch.randn(m, k, device=device)
    vec = torch.randn(k, device=device)
    # self with NaN values; beta=0 should make this irrelevant
    self = torch.full((m,), float("nan"), device=device)

    expected = torch.addmv(torch.zeros(m, device=device), mat, vec, beta=0, alpha=2.0)

    buf = torch.zeros(2 * m, device=device)
    out = buf[::2]
    assert out.stride() == (2,)

    torch.addmv(self, mat, vec, beta=0, alpha=2.0, out=out)

    assert out.stride() == (2,), (
        f"Strides changed! Expected (2,) but got {out.stride()}"
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_mv_out_preserves_strides_xpu():
    """
    Test that mv (matrix-vector product) preserves the strides of a
    non-contiguous output tensor when called with out= parameter.
    """
    device = "xpu"
    m, k = 4, 3

    mat = torch.randn(m, k, device=device)
    vec = torch.randn(k, device=device)

    expected = torch.mv(mat, vec)

    # Create a non-contiguous output with stride=2
    buf = torch.zeros(2 * m, device=device)
    out = buf[::2]
    assert out.stride() == (2,), "pre-condition: out has stride 2"

    torch.mv(mat, vec, out=out)

    assert out.stride() == (2,), (
        f"Strides changed! Expected (2,) but got {out.stride()}"
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_addmv_contiguous_output_xpu():
    """
    Test that addmv.out still works correctly for contiguous outputs.
    """
    device = "xpu"
    m, k = 4, 3

    mat = torch.randn(m, k, device=device)
    vec = torch.randn(k, device=device)
    self = torch.randn(m, device=device)

    expected = torch.addmv(self, mat, vec, beta=0.5, alpha=2.0)

    out = torch.empty(m, device=device)
    torch.addmv(self, mat, vec, beta=0.5, alpha=2.0, out=out)

    assert out.stride() == (1,)
    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    run_tests()
