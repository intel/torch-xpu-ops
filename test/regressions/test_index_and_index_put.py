# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_and_index_put(self, dtype=torch.float):
        x_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        y_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        mask_cpu = y_cpu.gt(0)

        # xpu part
        x_xpu = x_cpu.to("xpu")
        mask_xpu = mask_cpu.to("xpu")
        self.assertEqual(mask_cpu.nonzero(), mask_xpu.to(cpu_device).nonzero())
        self.assertEqual(x_cpu[mask_cpu], x_xpu[mask_xpu].to(cpu_device))

        # index put
        input = torch.ones([1], dtype=torch.float, device=cpu_device)
        indcies = torch.tensor([0, 0])
        x_cpu[indcies] = input
        x_cpu.index_put_([indcies], input, True)

        input = input.to("xpu")
        indcies = indcies.to("xpu")
        x_xpu[indcies] = input
        x_xpu.index_put_([indcies], input, True)
        self.assertEqual(x_cpu, x_xpu.to(cpu_device))

    def test_index_put(self, dtype=torch.float32):
        # For half precision, XPU and CUDA produce consistent results, but crash on the following case, so we ignore it.
        cpu_device = torch.device("cpu")
        xpu_device = torch.device("xpu")

        accumulate = True
        x_cpu = torch.zeros([4, 512, 128], dtype=dtype, device=cpu_device)
        y_cpu = torch.ones([4, 15000, 128], dtype=dtype, device=cpu_device)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = y_cpu.to(xpu_device)
        index_cpu = [
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
        ]
        index_xpu = [
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
        ]

        z_cpu = x_cpu.index_put_(index_cpu, y_cpu, accumulate)
        z_xpu = x_xpu.index_put_(index_xpu, y_xpu, accumulate)
        self.assertEqual(z_cpu, z_xpu.cpu())

    def test_index_put_outer_inner(self, dtype=torch.long):
        # XXX using long to avoid accumulate error caused by order of combiniation
        torch.use_deterministic_algorithms(True)
        batch = 15  # outer
        stride = 33  # inner
        numel = 17
        a = torch.randint(
            0, 5, (batch, numel, stride), dtype=dtype, device=torch.device("xpu")
        )
        b = torch.randint(
            0, 5, (batch, numel, stride), dtype=dtype, device=torch.device("xpu")
        )
        idx = a < b
        idx_ = torch.nonzero(idx, as_tuple=True)
        nonzero = torch.nonzero(idx)
        idx_ = (None, idx_[1], None)
        values = torch.randint(
            0,
            5,
            (batch, nonzero.shape[0], stride),
            dtype=dtype,
            device=torch.device("xpu"),
        )
        a_cpu = a.cpu()
        idx_cpu = (None, idx_[1].cpu(), None)
        values_cpu = values.cpu()

        torch.ops.aten._index_put_impl_(a, idx_, values, True)
        torch.ops.aten._index_put_impl_(a_cpu, idx_cpu, values_cpu, True)
        self.assertEqual(a_cpu, a.cpu())
        torch.use_deterministic_algorithms(False)

    def test_index_put_with_zero_shape_dim(self, dtype=torch.bfloat16):
        torch.use_deterministic_algorithms(True)
        a = torch.randn([10, 0], dtype=dtype, device=torch.device("xpu"))
        b = torch.randn([5, 0], dtype=dtype, device=torch.device("xpu"))
        a[:5, :] = a[:5, :] * 2 + b
        torch.use_deterministic_algorithms(False)

    def test_flip_float8(self):
        FLOAT8_DTYPES = (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        )
        for dtype in FLOAT8_DTYPES:
            a_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            a_xpu = a_cpu.to("xpu")
            b_cpu = torch.flip(a_cpu, [0]).to(torch.float32)
            b_xpu = torch.flip(a_xpu, [0]).cpu().to(torch.float32)
            self.assertEqual(b_cpu, b_xpu)

    def test_index_add_empty(self):
        x_cpu = torch.zeros([128], dtype=torch.int32)
        idx_cpu = torch.tensor([], dtype=torch.int32)
        src_cpu = torch.tensor([], dtype=torch.int32)
        y_cpu = x_cpu.index_add(0, idx_cpu, src_cpu)

        x_xpu = x_cpu.xpu()
        idx_xpu = idx_cpu.xpu()
        src_xpu = src_cpu.xpu()
        y_xpu = x_xpu.index_add(0, idx_xpu, src_xpu)

        self.assertEqual(y_xpu.cpu(), y_cpu)

    # Empty inputs below used to divide by zero in BatchKernelConfig (a zero
    # range bound collapses a launch range to 0). Each checks XPU vs CPU.

    def test_index_copy_empty(self):
        # dst.size(dim)==0 with non-empty sliceSize and an empty index: nothing
        # is copied, so the result equals dst.
        dst = torch.randn(2, 0, 3)
        index = torch.empty(0, dtype=torch.long)
        source = torch.randn(2, 0, 3)
        out_cpu = dst.index_copy(1, index, source)
        out_xpu = dst.xpu().index_copy(1, index.xpu(), source.xpu())
        self.assertEqual(out_xpu.cpu(), out_cpu)
        self.assertEqual(out_xpu.cpu(), dst)

    def test_weight_norm_empty(self):
        # v empty along the reduced dim, kept dim non-empty: norm of an empty
        # vector is 0 (forward); grad_g = 0/0 = NaN (backward).
        def fwd(device):
            v = torch.randn(4, 0, device=device)
            g = torch.norm_except_dim(v, 2, 0)
            return torch._weight_norm_interface(v, g, 0)

        w_cpu, n_cpu = fwd("cpu")
        w_xpu, n_xpu = fwd("xpu")
        self.assertEqual(w_xpu.cpu(), w_cpu)
        self.assertEqual(n_xpu.cpu(), n_cpu)
        self.assertEqual(n_xpu.cpu(), torch.zeros_like(n_cpu))

        def bwd(device):
            v = torch.randn(4, 0, device=device, requires_grad=True)
            g = torch.randn(4, 1, device=device, requires_grad=True)
            torch._weight_norm(v, g, 0).sum().backward()
            return v.grad, g.grad

        gv_cpu, gg_cpu = bwd("cpu")
        gv_xpu, gg_xpu = bwd("xpu")
        self.assertEqual(gv_xpu.cpu(), gv_cpu)
        self.assertEqual(gg_xpu.cpu(), gg_cpu)  # assertEqual treats NaN==NaN

    def test_cummax_cummin_empty(self):
        for op in (torch.cummax, torch.cummin):
            for shape, dim in (((0,), 0), ((3, 0, 4), 2)):
                x = torch.randn(shape)
                v_cpu, i_cpu = op(x, dim)
                v_xpu, i_xpu = op(x.xpu(), dim)
                self.assertEqual(v_xpu.cpu(), v_cpu)
                self.assertEqual(i_xpu.cpu(), i_cpu)
