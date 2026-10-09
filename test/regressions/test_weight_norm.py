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


class TestWeightNorm(TestCase):
    def test_weight_norm_1d(self):
        # 1-D v with dim == 0: the reduction group is a single element, so the
        # fused kernel has no reduction dim after collapse. It used to index
        # past the TensorInfo and allocate a garbage-sized buffer (OOM); check
        # it now matches CPU for both forward and backward.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            v = torch.randn(8, dtype=dtype)
            g = torch.randn(8, dtype=dtype)
            grad_out = torch.randn(8, dtype=dtype)

            def run(device):
                v_ = v.to(device).requires_grad_()
                g_ = g.to(device).requires_grad_()
                w = torch._weight_norm(v_, g_, 0)
                gv, gg = torch.autograd.grad(w, (v_, g_), grad_out.to(device))
                return w, gv, gg

            w_cpu, gv_cpu, gg_cpu = run("cpu")
            w_xpu, gv_xpu, gg_xpu = run("xpu")
            self.assertEqual(w_xpu.cpu(), w_cpu)
            self.assertEqual(gv_xpu.cpu(), gv_cpu)
            self.assertEqual(gg_xpu.cpu(), gg_cpu)


if __name__ == "__main__":
    run_tests()
