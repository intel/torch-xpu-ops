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


class TestFFTC2RValidation(TestCase):
    """Regression test: _fft_c2r must reject inconsistent last_dim_size
    with a user-facing RuntimeError instead of an internal assertion failure.

    See: https://github.com/pytorch/pytorch/issues/141448
    """

    def test_fft_c2r_rejects_oversized_last_dim(self):
        t = torch.full((3, 1, 3, 1), 0.372049, dtype=torch.cfloat, device="xpu")
        with self.assertRaisesRegex(
            RuntimeError, "Expected size of last transformed dimension"
        ):
            torch._fft_c2r(t, [2], 2, 536870912)

    def test_fft_c2r_rejects_zero_last_dim(self):
        t = torch.full((3, 1, 3, 1), 0.372049, dtype=torch.cfloat, device="xpu")
        with self.assertRaisesRegex(RuntimeError, "Invalid number of data points"):
            torch._fft_c2r(t, [2], 2, 0)


if __name__ == "__main__":
    run_tests()
