# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


@instantiate_parametrized_tests
class TestSafeSoftMax(TestCase):
    def test_sm(self):
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
            x_cpu = torch.randn(128, 128, 128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, -1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, -1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[0, 0, :] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, -1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, -1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)

            x_cpu = torch.randn(128, 128, 128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[0, :, 0] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 1)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 1)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)

            x_cpu = torch.randn(128, 128, 128).to(dtype)
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 0)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 0)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)
            x_cpu[:, 0, 0] = -float("inf")
            x_xpu = x_cpu.to(xpu_device)
            r_cpu = torch.ops.aten._safe_softmax(x_cpu, 0)
            r_xpu = torch.ops.aten._safe_softmax(x_xpu, 0)
            self.assertEqual(r_xpu.to(cpu_device), r_cpu)

    @parametrize("dtype", [torch.float, torch.float16, torch.bfloat16])
    @parametrize("dim_size", [128, 32768])
    def test_sm_nan_in_masked_row(self, dtype, dim_size):
        # A row holding a NaN is not fully masked, so it keeps the NaN rather
        # than being zeroed. The dim sizes straddle the threshold that picks
        # between the two forward kernels, and the NaN sits mid-row so it does
        # not land in the work item that seeds the reduce.
        x_cpu = torch.randn(4, dim_size).to(dtype)
        x_cpu[0, :] = -float("inf")
        x_cpu[0, dim_size // 2] = float("nan")
        x_xpu = x_cpu.to(xpu_device)
        r_cpu = torch.ops.aten._safe_softmax(x_cpu, -1)
        r_xpu = torch.ops.aten._safe_softmax(x_xpu, -1)
        self.assertEqual(r_xpu.to(cpu_device), r_cpu)
