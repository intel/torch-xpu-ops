# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

# Kernels that size a halving tree reduction from the per-kernel max
# work-group size, which is not always a power of two (768 on PTL when a kernel
# uses 160 GRF). These only fail on such hardware/compiler combinations; on
# other machines they guard the numerics of the same launch configs.

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestNonPow2WorkGroup(TestCase):
    @dtypes(torch.float32, torch.bfloat16, torch.float64)
    @parametrize("channels", (128, 256, 512))
    def test_batch_norm_stats_channels_last(self, device, dtype, channels):
        # C >= 128 gives group_x = 32, so a 768 max would give group_y = 24.
        x = torch.randn(32, channels, 32, 32, dtype=dtype)
        x_dev = x.to(device).contiguous(memory_format=torch.channels_last)
        mean, invstd = torch.batch_norm_stats(x_dev, 1e-5)
        x_ref = x.double()
        self.assertEqual(mean.cpu().double(), x_ref.mean((0, 2, 3)), atol=1e-4, rtol=1e-4)
        self.assertEqual(
            invstd.cpu().double(),
            (x_ref.var((0, 2, 3), unbiased=False) + 1e-5).rsqrt(),
            atol=1e-3,
            rtol=1e-3,
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("channels", (128, 256))
    def test_batch_norm_train_channels_last(self, device, dtype, channels):
        # Covers BatchNormCollectStatisticsChannelsLastKernelFunctor.
        x = torch.randn(16, channels, 32, 32, dtype=dtype)
        bn = torch.nn.BatchNorm2d(channels, momentum=1.0).to(device)
        bn(x.to(device).contiguous(memory_format=torch.channels_last))
        x_ref = x.double()
        self.assertEqual(
            bn.running_mean.cpu().double(), x_ref.mean((0, 2, 3)), atol=1e-4, rtol=1e-4
        )
        self.assertEqual(
            bn.running_var.cpu().double(), x_ref.var((0, 2, 3)), atol=1e-3, rtol=1e-3
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("op", ("layer_norm", "rms_norm"))
    @parametrize("n", (384, 768, 1536, 3072, 12288))
    def test_norm_wide_rows(self, device, dtype, op, n):
        # Wide rows reach the SLM subgroup tree in compute_stats.
        x = torch.randn(64, n) * 3 + 1
        fn = getattr(F, op)
        expected = fn(x.to(dtype).float(), (n,))
        actual = fn(x.to(device, dtype), (n,))
        self.assertEqual(actual.cpu().float(), expected, atol=5e-2, rtol=1e-2)

    def test_nonzero_chunked(self, device):
        # N > 2**25 takes the chunked count path.
        x = torch.rand((1 << 25) + 12345) < 0.3
        self.assertEqual(torch.nonzero(x.to(device)).cpu(), torch.nonzero(x))


instantiate_device_type_tests(
    TestNonPow2WorkGroup, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
