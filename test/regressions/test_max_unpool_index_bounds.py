# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for the max_unpool forward index bounds check on XPU.

The forward kernels used to read the int64 index tensor into a 32-bit integer
before bounds-checking it, so an index above INT_MAX was truncated modulo 2**32,
satisfied the check, and silently wrote to whichever in-range slot the
truncation landed on.

A failing bounds check aborts the process through SYCL_KERNEL_ASSERT, so every
case runs in its own subprocess -- the same approach upstream takes in
test/nn/test_pooling.py::test_MaxUnpool_index_errors.
"""

import subprocess
import sys

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

# Truncates to 3, an in-range slot for every shape below.
INVALID_INDEX = 2**32 + 3

SCRIPT = """
import torch
import torch.nn.functional as F

idx = torch.zeros({shape}, dtype=torch.int64, device="xpu")
idx.flatten()[-1] = {index}
x = torch.ones({shape}, device="xpu")
fmt = {fmt}
if fmt is not None:
    x = x.to(memory_format=fmt)
    idx = idx.to(memory_format=fmt)
F.max_unpool{ndim}d(x, idx, {kernel}, output_size={output_size})
torch.xpu.synchronize()
"""

# ndim, input shape, memory format, kernel, output_size, expected assert text.
# The 3-D contiguous kernel names its index `index`, the others `maxind`.
CASES = [
    subtest(
        (2, (1, 1, 2, 2), "None", 2, (4, 4), "maxind"),
        name="2d_contiguous",
    ),
    subtest(
        (2, (1, 2, 2, 2), "torch.channels_last", 2, (4, 4), "maxind"),
        name="2d_channels_last",
    ),
    subtest(
        (3, (1, 1, 1, 2, 2), "None", (1, 2, 2), (1, 4, 4), "index"),
        name="3d_contiguous",
    ),
    subtest(
        (3, (1, 2, 1, 2, 2), "torch.channels_last_3d", (1, 2, 2), (1, 4, 4), "maxind"),
        name="3d_channels_last",
    ),
]


class TestMaxUnpoolIndexBounds(TestCase):
    def _run(self, ndim, shape, fmt, kernel, output_size, index):
        script = SCRIPT.format(
            shape=shape,
            index=index,
            fmt=fmt,
            ndim=ndim,
            kernel=kernel,
            output_size=output_size,
        )
        p = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        return p.returncode, p.stdout + "\n" + p.stderr

    @parametrize("ndim, shape, fmt, kernel, output_size, var", CASES)
    def test_index_above_int32_is_rejected(
        self, ndim, shape, fmt, kernel, output_size, var
    ):
        rc, output = self._run(ndim, shape, fmt, kernel, output_size, INVALID_INDEX)
        self.assertNotEqual(
            rc, 0, f"max_unpool{ndim}d accepted index {INVALID_INDEX}:\n{output}"
        )
        self.assertIn(
            f"Assertion `{var} >= 0 && {var} < outputImageSize` failed", output
        )

    @parametrize("ndim, shape, fmt, kernel, output_size, var", CASES)
    def test_largest_valid_index_is_accepted(
        self, ndim, shape, fmt, kernel, output_size, var
    ):
        largest = 1
        for size in output_size:
            largest *= size
        rc, output = self._run(ndim, shape, fmt, kernel, output_size, largest - 1)
        self.assertEqual(rc, 0, f"bounds check rejected index {largest - 1}:\n{output}")


instantiate_parametrized_tests(TestMaxUnpoolIndexBounds)


if __name__ == "__main__":
    run_tests()
