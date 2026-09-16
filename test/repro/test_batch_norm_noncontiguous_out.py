# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Reproducer for: BatchNorm INTERNAL ASSERT when out= tensor is non-contiguous.

When test_out passes a non-standard (e.g., strided/non-contiguous) output
tensor via the out= argument to _native_batch_norm_legit, the XPU kernel
previously hit TORCH_INTERNAL_ASSERT in batch_norm_stats_template because
resize_output does not guarantee contiguity.

Fix: allocate contiguous temporaries when out_mean / out_invstd are
non-contiguous, run the kernel into those temporaries, then copy back.

To run:
    pytest test/repro/test_batch_norm_noncontiguous_out.py
"""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _make_noncontiguous(t):
    """Return a view of *t* with stride-2 layout (non-contiguous)."""
    big = torch.empty(t.numel() * 2, dtype=t.dtype, device=t.device)
    view = big[::2]
    view.copy_(t.flatten())
    return view.reshape(t.shape)


@unittest.skipUnless(torch.xpu.is_available(), "XPU device not available")
class TestBatchNormNoncontiguousOut(TestCase):
    def test_batch_norm_legit_noncontiguous_save_mean_out(self):
        """
        _native_batch_norm_legit with non-contiguous save_mean and save_invstd
        out= tensors must not raise an INTERNAL ASSERT.
        """
        N, C, H, W = 5, 5, 5, 5
        device = "xpu"

        inp = torch.randn(N, C, H, W, device=device)
        weight = torch.randn(C, device=device)
        bias = torch.randn(C, device=device)
        running_mean = torch.zeros(C, device=device)
        running_var = torch.ones(C, device=device)

        # Allocate non-contiguous output tensors.
        acc_dtype = torch.float32
        save_mean_nc = _make_noncontiguous(
            torch.zeros(C, dtype=acc_dtype, device=device)
        )
        save_invstd_nc = _make_noncontiguous(
            torch.zeros(C, dtype=acc_dtype, device=device)
        )
        out_nc = torch.empty_like(inp)

        self.assertFalse(save_mean_nc.is_contiguous())
        self.assertFalse(save_invstd_nc.is_contiguous())

        # This should not raise an INTERNAL ASSERT.
        torch.ops.aten._native_batch_norm_legit(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            True,  # training
            0.1,  # momentum
            1e-5,  # eps
            out=out_nc,
            save_mean=save_mean_nc,
            save_invstd=save_invstd_nc,
        )

        # Cross-check with a fresh contiguous run.
        save_mean_ref = torch.zeros(C, dtype=acc_dtype, device=device)
        save_invstd_ref = torch.zeros(C, dtype=acc_dtype, device=device)
        out_ref = torch.empty_like(inp)
        torch.ops.aten._native_batch_norm_legit(
            inp,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            0.1,
            1e-5,
            out=out_ref,
            save_mean=save_mean_ref,
            save_invstd=save_invstd_ref,
        )

        torch.testing.assert_close(out_nc, out_ref)
        torch.testing.assert_close(save_mean_nc, save_mean_ref)
        torch.testing.assert_close(save_invstd_nc, save_invstd_ref)

    def test_batch_norm_stats_noncontiguous_output(self):
        """
        batch_norm_stats (the standalone stats kernel) should also work when
        returned mean/invstd tensors are later used in non-contiguous form.
        """
        N, C, HW = 4, 8, 16
        device = "xpu"
        inp = torch.randn(N, C, HW, device=device)

        mean, invstd = torch.ops.aten._batch_norm_with_update(
            inp,
            None,
            None,
            torch.zeros(C, device=device),
            torch.ones(C, device=device),
            0.1,
            1e-5,
        )
        # Simply verify shapes — the main goal is no crash.
        self.assertEqual(mean.shape, torch.Size([C]))
        self.assertEqual(invstd.shape, torch.Size([C]))


if __name__ == "__main__":
    run_tests()
