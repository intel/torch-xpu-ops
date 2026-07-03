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
from torch.testing._internal.common_utils import run_tests, TestCase


class TestConvTranspose2dComplex32Reference(TestCase):
    def test_conv_transpose2d_complex32_xpu_close_to_ref64(self):
        torch.manual_seed(0)
        atol, rtol = 2e-2, 1e-3

        x_cpu = torch.randn(2, 4, 3, 4, dtype=torch.complex32)
        w_cpu = torch.randn(4, 5, 3, 3, dtype=torch.complex32)
        b_cpu = torch.randn(5, dtype=torch.complex32)

        xpu_out = F.conv_transpose2d(
            x_cpu.to("xpu"), w_cpu.to("xpu"), b_cpu.to("xpu")
        ).cpu()
        ref64 = F.conv_transpose2d(
            x_cpu.to(torch.complex64),
            w_cpu.to(torch.complex64),
            b_cpu.to(torch.complex64),
        )

        self.assertEqual(
            xpu_out.to(torch.complex64),
            ref64,
            atol=atol,
            rtol=rtol,
        )


if __name__ == "__main__":
    run_tests()
