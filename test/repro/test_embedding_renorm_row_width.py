# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Reproducer for: embedding_renorm_ on XPU derived the embedding row width from
``self.stride(0)`` instead of ``self.size(1)``.

The two differ whenever the weight is not contiguous:

* a ``(20, 0)`` weight has ``stride(0) == 1`` but ``size(1) == 0``, so the
  kernel ran its accumulation loop once and dereferenced a null data pointer,
  faulting the device (https://github.com/intel/torch-xpu-ops/issues/5152);
* a row-slice such as ``base[:, :5]`` has ``stride(0) == 10`` and
  ``size(1) == 5``, so the kernel read 10 elements per row and silently
  computed the norm over neighbouring rows.

Note that a regression in the first case surfaces as an asynchronous device
fault that aborts the whole process, not as a single failing test case.

Mirrors the CUDA-only upstream regression test added in
https://github.com/pytorch/pytorch/pull/195208.
"""

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# XPU reduces the row norm in a float accumulator across a sub-group, so the
# rounding differs from the CPU reference for the narrow types.
TOLERANCES = {
    torch.float32: dict(atol=1e-5, rtol=1.3e-6),
    torch.float16: dict(atol=1e-3, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestEmbeddingRenormRowWidth(TestCase):
    def test_embedding_max_norm_zero_embedding_dim(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                weight = torch.empty((20, 0), dtype=dtype, device="xpu")
                indices = torch.tensor([0], dtype=torch.int64, device="xpu")

                out = F.embedding(indices, weight, max_norm=1.0, norm_type=2.0)
                torch.xpu.synchronize()

                self.assertEqual(out.shape, (1, 0))

    def test_embedding_renorm_non_contiguous_weight(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                base = (torch.randn(20, 10, device="xpu") * 10).to(dtype)
                base_cpu = base.cpu().clone()

                weight = base[:, :5]
                weight_cpu = base_cpu[:, :5]
                self.assertNotEqual(weight.stride(0), weight.size(1))

                indices = torch.tensor([2, 5], dtype=torch.int64, device="xpu")
                torch.embedding_renorm_(weight, indices, 1.0, 2.0)
                torch.xpu.synchronize()
                torch.embedding_renorm_(weight_cpu, indices.cpu(), 1.0, 2.0)

                self.assertEqual(weight.cpu(), weight_cpu, **TOLERANCES[dtype])
                # The columns outside the slice must not be touched.
                self.assertEqual(base.cpu(), base_cpu, **TOLERANCES[dtype])

    def test_embedding_max_norm_contiguous_matches_cpu(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                weight = (torch.randn(20, 8, device="xpu") * 10).to(dtype)
                weight_cpu = weight.cpu().clone()
                indices = torch.tensor([1, 3, 3, 7], dtype=torch.int64, device="xpu")

                out = F.embedding(indices, weight, max_norm=1.0, norm_type=2.0)
                torch.xpu.synchronize()
                out_cpu = F.embedding(
                    indices.cpu(), weight_cpu, max_norm=1.0, norm_type=2.0
                )

                self.assertEqual(weight.cpu(), weight_cpu, **TOLERANCES[dtype])
                self.assertEqual(out.cpu(), out_cpu, **TOLERANCES[dtype])


if __name__ == "__main__":
    run_tests()
