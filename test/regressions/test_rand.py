# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_distribution_normal(self, dtype=torch.float):
        tol = 1e-2
        x_xpu = torch.tensor(list(range(10000)), device=xpu_device, dtype=dtype)
        x_xpu.normal_(2.0, 0.5)
        self.assertEqual(x_xpu.cpu().mean(), 2.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 0.5, rtol=tol, atol=tol)
        x_xpu = torch.normal(
            mean=-5.0, std=0.2, size=(10000,), device=xpu_device, dtype=dtype
        )
        self.assertEqual(x_xpu.cpu().mean(), -5.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 0.2, rtol=tol, atol=tol)
        torch.normal(
            mean=-3.0, std=1.2, size=(10000,), device=xpu_device, dtype=dtype, out=x_xpu
        )
        self.assertEqual(x_xpu.cpu().mean(), -3.0, rtol=tol, atol=tol)
        self.assertEqual(x_xpu.cpu().std(), 1.2, rtol=tol, atol=tol)

    def test_distribution_uniform(self, dtype=torch.float):
        tol = 1e-2
        x_xpu = torch.tensor(list(range(10000)), device=xpu_device, dtype=dtype)
        x_xpu.uniform_(0, 10)
        self.assertEqual(x_xpu.cpu().mean(), 5.0, rtol=tol, atol=tol)
        x_xpu = torch.rand(size=(2000, 5), device=xpu_device, dtype=dtype)
        self.assertEqual(x_xpu.cpu().view(-1).mean(), 0.5, rtol=tol, atol=tol)


class TestNullaryDistribution(TestCase):
    # Contiguous, aligned fp32/fp16/bf16 outputs take the vectorized nullary
    # kernel (4 outputs per Philox block per work-item); non-contiguous or
    # misaligned outputs take the strided kernel.

    def _assert_standard_normal(self, x):
        x = x.float().cpu()
        n = x.numel()
        self.assertFalse(x.isnan().any())
        self.assertEqual(x.mean().item(), 0.0, atol=5.0 / n**0.5, rtol=0)
        self.assertEqual(x.std().item(), 1.0, atol=5.0 / (2.0 * n) ** 0.5, rtol=0)

    @parametrize("numel", [7, 231, 640000, 1 << 20])
    @parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
    def test_normal_vectorized(self, numel, dtype):
        x = torch.empty(numel, dtype=dtype, device=xpu_device)
        torch.manual_seed(1234)
        x.normal_()
        self._assert_standard_normal(x)
        y = torch.empty_like(x)
        torch.manual_seed(1234)
        y.normal_()
        self.assertEqual(x, y)

    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_normal_strided_fallback(self, dtype):
        noncontig = torch.empty(10000, 128, dtype=dtype, device=xpu_device)[:, ::2]
        unaligned = torch.empty(10000 * 64 + 1, dtype=dtype, device=xpu_device)[1:]
        for x in (noncontig, unaligned):
            x.normal_()
            self._assert_standard_normal(x)

    @parametrize("start", [0, 1])
    @parametrize("numel", [7, 231, 640001])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_normal_stays_in_bounds(self, start, numel, dtype):
        # start=0 is aligned (vectorized path with a partial tail), start=1 is
        # misaligned (strided path). The guard bands must stay untouched.
        sentinel = 100.0
        buf = torch.full(
            (start + numel + 64,), sentinel, dtype=dtype, device=xpu_device
        )
        buf[start : start + numel].normal_()
        buf = buf.cpu()
        self.assertTrue((buf[:start] == sentinel).all())
        self.assertTrue((buf[start + numel :] == sentinel).all())
        self.assertFalse((buf[start : start + numel] == sentinel).any())

    def test_normal_consecutive_draws_do_not_reuse_counters(self):
        # Each aligned group of 4 outputs comes from one Philox block, so a
        # reused counter (within or across launches) shows up as a repeated row.
        # 1M elements needs several grid-stride iterations per work-item.
        numel = 1 << 20
        torch.manual_seed(0)
        a = torch.empty(numel, device=xpu_device).normal_()
        b = torch.empty(numel, device=xpu_device).normal_()
        rows = torch.cat([a, b]).view(-1, 4).cpu()
        self.assertEqual(torch.unique(rows, dim=0).shape[0], rows.shape[0])

    @parametrize("start", [0, 1])
    @parametrize("numel", [7, 231, 640001])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_random_wide_range_stays_in_bounds(self, start, numel, dtype):
        # A range >= 2^28 takes the 64-bit random path (unroll_factor == 2),
        # a separate instantiation of the vectorized kernel (start=0) with its
        # own tail and alignment handling; start=1 takes the strided path.
        sentinel = -1.0
        buf = torch.full(
            (start + numel + 64,), sentinel, dtype=dtype, device=xpu_device
        )
        buf[start : start + numel].random_(0, 2**30)
        buf = buf.cpu()
        self.assertTrue((buf[:start] == sentinel).all())
        self.assertTrue((buf[start + numel :] == sentinel).all())
        self.assertTrue((buf[start : start + numel] >= 0).all())
        self.assertTrue((buf[start : start + numel] <= 2**30).all())


instantiate_parametrized_tests(TestNullaryDistribution)


if __name__ == "__main__":
    run_tests()
