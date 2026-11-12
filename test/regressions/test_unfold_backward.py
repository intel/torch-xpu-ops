# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestUnfoldBackward(TestCase):
    def test_unfold_backward_work_group_size_limit(self):
        x_cpu = torch.randn(10000, requires_grad=True)
        x_xpu = x_cpu.detach().to("xpu").requires_grad_()

        x_cpu.unfold(0, 2, 1).sum().backward()
        x_xpu.unfold(0, 2, 1).sum().backward()
        torch.xpu.synchronize()

        self.assertEqual(x_xpu.grad.cpu(), x_cpu.grad)


if __name__ == "__main__":
    run_tests()
