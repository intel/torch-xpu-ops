# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Regression test for: channel-last aten::hardswish_ correctness.

Verifies that hardswish_ called in-place on channel-last tensors produces the
correct numerical result and preserves the memory format.
"""

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _hardswish_ref(x: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation of hardswish."""
    return x.cpu() * torch.clamp(x.cpu() + 3.0, min=0.0, max=6.0) / 6.0


class TestHardswishChannelLast(TestCase):
    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_hardswish_inplace_channels_last_4d(self, device, dtype):
        """hardswish_ on a 4-D channel-last tensor must produce the correct
        result and preserve the channels_last memory format."""
        shape = (2, 8, 4, 4)
        x_cpu = torch.randn(shape, dtype=dtype)
        x_ref = x_cpu.clone()

        # Create a channel-last XPU tensor
        x_dev = x_cpu.to(device=device, memory_format=torch.channels_last)
        self.assertTrue(x_dev.is_contiguous(memory_format=torch.channels_last))
        self.assertFalse(x_dev.is_contiguous())  # not C-contiguous

        # Apply in-place hardswish
        x_dev.hardswish_()

        # Apply reference on CPU
        ref = _hardswish_ref(x_ref)

        # Numerical check (relaxed tolerance for fp16/bf16)
        tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
        torch.testing.assert_close(x_dev.cpu().float(), ref.float(), **tol)

        # Memory format must be preserved
        self.assertTrue(
            x_dev.is_contiguous(memory_format=torch.channels_last),
            "hardswish_ must preserve channels_last memory format",
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_hardswish_inplace_channels_last_3d(self, device, dtype):
        """hardswish_ on a 5-D tensor with channels_last_3d memory format (NDHWC)
        must produce the correct result and preserve the channels_last_3d layout."""
        shape = (2, 4, 4, 4, 4)
        x_cpu = torch.randn(shape, dtype=dtype)
        x_ref = x_cpu.clone()

        x_dev = x_cpu.to(device=device, memory_format=torch.channels_last_3d)
        self.assertTrue(x_dev.is_contiguous(memory_format=torch.channels_last_3d))

        x_dev.hardswish_()
        ref = _hardswish_ref(x_ref)

        tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
        torch.testing.assert_close(x_dev.cpu().float(), ref.float(), **tol)

        self.assertTrue(
            x_dev.is_contiguous(memory_format=torch.channels_last_3d),
            "hardswish_ must preserve channels_last_3d memory format",
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_hardswish_inplace_contiguous(self, device, dtype):
        """Sanity-check: hardswish_ on a contiguous XPU tensor still works."""
        shape = (4, 8, 8, 8)
        x_cpu = torch.randn(shape, dtype=dtype)
        x_ref = x_cpu.clone()

        x_dev = x_cpu.to(device=device)
        self.assertTrue(x_dev.is_contiguous())

        x_dev.hardswish_()
        ref = _hardswish_ref(x_ref)

        tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {}
        torch.testing.assert_close(x_dev.cpu().float(), ref.float(), **tol)

    def test_hardswish_inplace_matches_out_of_place_channels_last(self, device):
        """In-place and out-of-place hardswish must agree on channel-last input."""
        shape = (2, 16, 8, 8)
        x_cpu = torch.randn(shape, dtype=torch.float32)

        x_inplace = x_cpu.to(device=device, memory_format=torch.channels_last)
        x_outplace = x_cpu.to(device=device, memory_format=torch.channels_last)

        # Out-of-place (uses TensorIterator with a freshly-allocated output)
        y_out = F.hardswish(x_outplace)
        # In-place
        x_inplace.hardswish_()

        torch.testing.assert_close(x_inplace, y_out)


instantiate_device_type_tests(
    TestHardswishChannelLast, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
