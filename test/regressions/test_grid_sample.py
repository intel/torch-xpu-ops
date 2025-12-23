# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleCopy(TestCase):
    # Refer to https://github.com/pytorch/pytorch/issues/153996
    def test_grid_sample(self, dtype=torch.float):
        input_cpu = torch.rand(1, 2, 5, 5, device=cpu_device)
        grid_cpu = torch.rand(1, 3, 3, 2, device=cpu_device)
        out_cpu = F.grid_sample(input_cpu, grid_cpu, align_corners=False)
        input_xpu = input_cpu.to(xpu_device)
        grid_xpu = grid_cpu.to(xpu_device)
        out_xpu = F.grid_sample(input_xpu, grid_xpu, align_corners=False)
        self.assertEqual(out_cpu, out_xpu.to(cpu_device))
