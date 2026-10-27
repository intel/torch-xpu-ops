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


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestFakeQuantizePerChannel(TestCase):
    def test_float_zero_point_rounds_to_even(self):
        op = torch.ops.aten.fake_quantize_per_channel_affine_cachemask.default
        input_cpu = torch.tensor(
            [[-1.5], [-0.5], [0.5], [1.5], [2.5]], dtype=torch.float32
        )
        scale_cpu = torch.tensor([1.0], dtype=torch.float32)
        zero_point_cpu = torch.tensor([0.0], dtype=torch.float32)

        expected, expected_mask = op(input_cpu, scale_cpu, zero_point_cpu, 1, -10, 10)
        actual, actual_mask = op(
            input_cpu.to("xpu"),
            scale_cpu.to("xpu"),
            zero_point_cpu.to("xpu"),
            1,
            -10,
            10,
        )

        self.assertEqual(actual.cpu(), expected)
        self.assertEqual(actual_mask.cpu(), expected_mask)


if __name__ == "__main__":
    run_tests()
