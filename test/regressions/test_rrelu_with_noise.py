# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for the noise shape check and non-contiguous noise
handling in the XPU rrelu_with_noise kernel.

The training kernel writes one noise value per element of self directly
into the caller's noise tensor. Nothing used to check that noise had the
same shape as self or was contiguous, so a non-contiguous noise (e.g. an
expanded view) was silently written through a discarded contiguous
temporary: the returned output was correct but the caller's noise kept
its old values. Non-contiguous noise now materializes, the same way the
output does.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

xpu_device = torch.device("xpu")


class TestRreluWithNoise(TestCase):
    def test_noise_shape_mismatch(self):
        x = torch.randn(4, 4, device=xpu_device)
        for noise in [
            torch.empty(2, 8, device=xpu_device),
            torch.empty(4, 1, device=xpu_device),
        ]:
            with self.assertRaisesRegex(
                RuntimeError, "noise tensor shape must match self tensor shape"
            ):
                torch.rrelu_with_noise(x, noise, training=True)

    def test_noise_non_contiguous(self):
        x = torch.randn(4, 8, device=xpu_device)
        noise = torch.randn(4, 16, device=xpu_device)[:, ::2]
        self.assertFalse(noise.is_contiguous())
        out = torch.rrelu_with_noise(x, noise, training=True)
        self.assertEqual(out, x * noise)

        # An expanded view: its storage holds one row; writing through its
        # data pointer used to go past the storage. Distinct per-element
        # noise values cannot land there, so the copy back rejects it.
        noise = torch.zeros(1, 8, device=xpu_device).expand(4, 8)
        with self.assertRaisesRegex(RuntimeError, "refers to a single memory location"):
            torch.rrelu_with_noise(x, noise, training=True)

    def test_noise_written(self):
        # Positive path: the kernel must write into the caller's noise.
        x = torch.randn(64, 64, device=xpu_device)
        noise = torch.zeros_like(x)
        out = torch.rrelu_with_noise(x, noise, 0.1, 0.9, training=True)
        self.assertTrue(torch.all(noise[x > 0] == 1))
        self.assertTrue(torch.all((noise >= 0.1) & (noise <= 0.9)))
        self.assertTrue(torch.all(out >= 0))


if __name__ == "__main__":
    run_tests()
