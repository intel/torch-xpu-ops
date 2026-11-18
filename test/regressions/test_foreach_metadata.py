# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestForeachMetadata(TestCase):
    def _inputs(self, dtype, count, device):
        # Empty, single-element, and multi-chunk (131073 element) sizes
        # exercise zero, one, and many work-group table entries.
        sizes = (0, 1, 17, 4096, 131073)
        tensors = [
            torch.randn(sizes[i % len(sizes)], dtype=dtype, device=device)
            for i in range(count)
        ]
        return tensors, [torch.randn_like(t) for t in tensors]

    def _check(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_t, expected_t in zip(actual, expected):
            self.assertEqual(actual_t, expected_t)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("tensor_count", (3, 257))
    def test_foreach_add_tensor_list(self, device, dtype, tensor_count):
        a, b = self._inputs(dtype, tensor_count, device)
        expected = [x + y for x, y in zip(a, b)]

        out = torch._foreach_add(a, b)
        inplace = [x.clone() for x in a]
        torch._foreach_add_(inplace, b)
        # Queue consecutive calls before validating so the pinned host
        # metadata is reused while earlier copies may still be in flight.
        out_again = torch._foreach_add(a, b)
        torch.xpu.synchronize()

        self._check(out, expected)
        self._check(inplace, expected)
        self._check(out_again, expected)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("tensor_count", (3, 257))
    def test_foreach_add_scalar_list(self, device, dtype, tensor_count):
        a, _ = self._inputs(dtype, tensor_count, device)
        scalars = [float(i % 7) / 2 for i in range(tensor_count)]
        expected = [x + scalar for x, scalar in zip(a, scalars)]

        out = torch._foreach_add(a, scalars)
        inplace = [x.clone() for x in a]
        torch._foreach_add_(inplace, scalars)
        out_again = torch._foreach_add(a, scalars)
        torch.xpu.synchronize()

        self._check(out, expected)
        self._check(inplace, expected)
        self._check(out_again, expected)


instantiate_device_type_tests(
    TestForeachMetadata, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
