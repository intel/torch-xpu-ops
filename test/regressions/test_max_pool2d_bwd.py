# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

aten = torch.ops.aten


class TestMaxPool2dBwd(TestCase):
    def test_max_pool2d_bwd_fp16(self):
        # Regression test: verify correctness under float16 (lowp) inputs.
        # The large window size (13x13) causes window_size > 25, so the lowering
        # falls back to the aten kernel rather than generating a Triton kernel.
        # Reference is computed in float32 (upcast), then downcast to float16 for
        # comparison, matching the behavior of check_model with reference_in_float=True.
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [13, 13], [1, 1], [2, 2], [1, 1], False, c
            )

        x_fp16 = torch.randn([2, 64, 20, 20], dtype=torch.half, device="xpu")
        result, indices = aten.max_pool2d_with_indices(
            x_fp16,
            [13, 13],
            [1, 1],
            2,
            1,
            False,
        )
        grad_output_fp16 = torch.randn_like(result)

        # fp32 reference
        x_fp32 = x_fp16.float()
        grad_output_fp32 = grad_output_fp16.float()
        expected_fp32 = fn(grad_output_fp32, x_fp32, indices)
        expected = expected_fp32.half()

        actual = fn(grad_output_fp16, x_fp16, indices)

        self.assertEqual(actual, expected)
