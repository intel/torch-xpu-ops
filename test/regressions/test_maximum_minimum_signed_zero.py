# Copyright 2026 Intel Corporation
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
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMaximumMinimumSignedZero(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_signed_zero(self, device, dtype):
        a = torch.tensor([-0.0, 0.0, 0.0, -0.0], device=device, dtype=dtype)
        b = torch.tensor([0.0, -0.0, 0.0, -0.0], device=device, dtype=dtype)
        min_signbit = torch.tensor([True, True, False, True])
        max_signbit = torch.tensor([False, False, False, True])

        self.assertEqual(torch.minimum(a, b).signbit().cpu(), min_signbit)
        self.assertEqual(torch.minimum(b, a).signbit().cpu(), min_signbit)
        self.assertEqual(torch.maximum(a, b).signbit().cpu(), max_signbit)
        self.assertEqual(torch.maximum(b, a).signbit().cpu(), max_signbit)


instantiate_device_type_tests(
    TestMaximumMinimumSignedZero, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
