# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Regression test for https://github.com/intel/torch-xpu-ops/issues/3030

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@instantiate_parametrized_tests
class TestConvTransposeComplex32Reference(TestCase):
    @parametrize(
        "conv_op, x_shape",
        [
            subtest((F.conv_transpose1d, (2, 4, 3)), name="1d"),
            subtest((F.conv_transpose2d, (2, 4, 3, 4)), name="2d"),
            subtest((F.conv_transpose3d, (2, 4, 3, 4, 5)), name="3d"),
        ],
    )
    def test_complex32_xpu_close_to_ref64(self, conv_op, x_shape):
        torch.manual_seed(0)
        w_shape = (x_shape[1], 5) + (3,) * (len(x_shape) - 2)

        x_cpu = torch.randn(x_shape, dtype=torch.complex32)
        w_cpu = torch.randn(w_shape, dtype=torch.complex32)
        b_cpu = torch.randn(5, dtype=torch.complex32)

        xpu_out = conv_op(x_cpu.to("xpu"), w_cpu.to("xpu"), b_cpu.to("xpu")).cpu()
        ref64 = conv_op(
            x_cpu.to(torch.complex64),
            w_cpu.to(torch.complex64),
            b_cpu.to(torch.complex64),
        )
        self.assertEqual(xpu_out.to(torch.complex64), ref64, atol=2e-2, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
