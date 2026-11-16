# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TENSOR_COUNTS = (3, 257)
# Includes an empty, a single-element, and a multi-chunk (131073 element)
# tensor so the packed work-group table covers zero, one, and many chunks.
TENSOR_SIZES = (0, 1, 17, 4096, 131073)


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestForeachMetadata(TestCase):
    def _inputs(self, dtype, count):
        tensors = [
            torch.randn(TENSOR_SIZES[i % len(TENSOR_SIZES)], dtype=dtype, device="xpu")
            for i in range(count)
        ]
        return tensors, [torch.randn_like(t) for t in tensors]

    def _check(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_t, expected_t in zip(actual, expected):
            self.assertEqual(actual_t, expected_t)

    def test_foreach_add_tensor_list(self):
        for dtype in DTYPES:
            for count in TENSOR_COUNTS:
                a, b = self._inputs(dtype, count)
                expected = [x + y for x, y in zip(a, b)]

                out = torch._foreach_add(a, b)
                inplace = [x.clone() for x in a]
                torch._foreach_add_(inplace, b)
                # Queue consecutive calls before validating so the pinned host
                # metadata is reused while earlier copies may still be in
                # flight.
                out_again = torch._foreach_add(a, b)
                torch.xpu.synchronize()

                self._check(out, expected)
                self._check(inplace, expected)
                self._check(out_again, expected)

    def test_foreach_add_scalar_list(self):
        for dtype in DTYPES:
            for count in TENSOR_COUNTS:
                a, _ = self._inputs(dtype, count)
                scalars = [float(i % 7) / 2 for i in range(count)]
                expected = [x + scalar for x, scalar in zip(a, scalars)]

                out = torch._foreach_add(a, scalars)
                inplace = [x.clone() for x in a]
                torch._foreach_add_(inplace, scalars)
                out_again = torch._foreach_add(a, scalars)
                torch.xpu.synchronize()

                self._check(out, expected)
                self._check(inplace, expected)
                self._check(out_again, expected)


if __name__ == "__main__":
    run_tests()
