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

xpu_device = torch.device("xpu")


class TestScatterGatherBComplex32(TestCase):
    def test_gather_bcomplex32(self):
        # bcomplex32 CPU support is limited; compare XPU results directly
        x = torch.randn(4, 8).to(torch.bcomplex32).to(xpu_device)
        # gather column 0 from all rows via a (4, 1) index tensor
        idx = torch.zeros(4, 1, dtype=torch.long, device=xpu_device)
        result = torch.gather(x, 1, idx)
        self.assertEqual(result[:, 0], x[:, 0])

    def test_scatter_bcomplex32(self):
        x = torch.zeros(4, 8, dtype=torch.bcomplex32, device=xpu_device)
        idx = torch.tensor([[0, 2, 4]], dtype=torch.long, device=xpu_device)
        src = torch.randn(1, 3).to(torch.bcomplex32).to(xpu_device)
        x.scatter_(1, idx, src)
        self.assertEqual(x[0, 0], src[0, 0])
        self.assertEqual(x[0, 2], src[0, 1])
        self.assertEqual(x[0, 4], src[0, 2])


if __name__ == "__main__":
    run_tests()
