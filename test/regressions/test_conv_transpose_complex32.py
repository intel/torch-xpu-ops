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
from torch import nn
from torch.testing._creation import make_tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@instantiate_parametrized_tests
class TestConvTransposeComplex32Reference(TestCase):
    def _make_inputs(self, module_cls, x_shape):
        in_channels = x_shape[1]
        x_cpu = make_tensor(x_shape, dtype=torch.complex32, device="cpu")
        module_cpu = module_cls(in_channels, 5, 3).to(dtype=torch.complex32)
        return x_cpu, module_cpu.weight.detach(), module_cpu.bias.detach()

    def _assert_conv_transpose_complex32_xpu_close_to_ref64(
        self,
        conv_op,
        x_cpu,
        w_cpu,
        b_cpu,
        *,
        atol=3e-3,
        rtol=1e-3,
    ):
        xpu_out = conv_op(x_cpu.to("xpu"), w_cpu.to("xpu"), b_cpu.to("xpu")).cpu()
        ref64 = conv_op(
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

    @parametrize(
        "conv_op, module_cls, x_shape",
        [
            subtest((F.conv_transpose1d, nn.ConvTranspose1d, (2, 4, 3)), name="1d"),
            subtest((F.conv_transpose2d, nn.ConvTranspose2d, (2, 4, 3, 4)), name="2d"),
            subtest(
                (F.conv_transpose3d, nn.ConvTranspose3d, (2, 4, 3, 4, 5)), name="3d"
            ),
        ],
    )
    def test_complex32_xpu_close_to_ref64(self, conv_op, module_cls, x_shape):
        torch.manual_seed(0)
        x_cpu, w_cpu, b_cpu = self._make_inputs(module_cls, x_shape)
        self._assert_conv_transpose_complex32_xpu_close_to_ref64(
            conv_op, x_cpu, w_cpu, b_cpu
        )


if __name__ == "__main__":
    run_tests()
