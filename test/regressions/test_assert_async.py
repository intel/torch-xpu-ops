# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAssertAsync(TestCase):
    def test_assert_async_nonzero_dispatched_dtypes(self):
        dtypes = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.bool,
        )
        for dtype in dtypes:
            torch._assert_async(torch.tensor(1, dtype=dtype, device="xpu"))
        torch.xpu.synchronize()


if __name__ == "__main__":
    run_tests()
