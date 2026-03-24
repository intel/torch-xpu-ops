"""
Unit tests for XPU grouped_mm kernel.

Adapted from pytorch/test/test_matmul_cuda.py grouped_gemm tests.
Tests all 4 input modes: 2D×2D, 2D×3D, 3D×3D, 3D×2D.
"""

import unittest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyXPU,
)


TEST_XPU = torch.xpu.is_available()


class TestGroupedMMXPU(TestCase):

    def grouped_mm_helper(
        self, alist, blist, gOlist, agradlist, bgradlist, outlist
    ):
        for a, b, gO, agrad, bgrad, out in zip(
            alist, blist, gOlist, agradlist, bgradlist, outlist
        ):
            a = a.clone().detach().requires_grad_()
            b = b.clone().detach().requires_grad_()
            out_ref = torch.mm(a, b.t())
            out_ref.backward(gO)
            self.assertEqual(out, out_ref)
            if agrad is not None:
                self.assertEqual(agrad, a.grad)
                self.assertEqual(bgrad, b.grad)

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_grouped_gemm_2d_2d(self, device, dtype):
        m, n, k, n_groups = 16, 32, 64, 4
        a = torch.randn(m, k * n_groups, device=device, dtype=dtype)
        b = torch.randn(n, k * n_groups, device=device, dtype=dtype)

        a.requires_grad_(True)
        b.requires_grad_(True)
        offs = torch.arange(
            k, n_groups * k + 1, k, device=device, dtype=torch.int32
        )

        f = F.grouped_mm
        out = f(a, b.t(), offs=offs, out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        alist, blist, agradlist, bgradlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start : offs_cpu[i]])
            blist.append(b[:, start : offs_cpu[i]])
            agradlist.append(a.grad[:, start : offs_cpu[i]])
            bgradlist.append(b.grad[:, start : offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, blist, gO, agradlist, bgradlist, out)

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_grouped_gemm_2d_3d(self, device, dtype):
        m, n, k, n_groups = 16, 32, 64, 4
        a = torch.randn(m * n_groups, k, device=device, dtype=dtype)
        b = torch.randn(n_groups, k, n, device=device, dtype=dtype)

        a.requires_grad_(True)
        b.requires_grad_(True)

        offs = torch.arange(
            m, n_groups * m + 1, m, device=device, dtype=torch.int32
        )

        f = F.grouped_mm
        out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        alist, agradlist, gOlist, outlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[start : offs_cpu[i]])
            agradlist.append(a.grad[start : offs_cpu[i]])
            outlist.append(out[start : offs_cpu[i]])
            gOlist.append(gO[start : offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(
            alist, b, gOlist, agradlist, b.grad, outlist
        )

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_grouped_gemm_3d_3d(self, device, dtype):
        m, n, k, n_groups = 16, 32, 64, 4
        a = torch.randn(n_groups, m, k, device=device, dtype=dtype)
        b = torch.randn(n_groups, k, n, device=device, dtype=dtype)

        a.requires_grad_(True)
        b.requires_grad_(True)

        f = F.grouped_mm
        out = f(a, b.transpose(-2, -1), out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        self.grouped_mm_helper(a, b, gO, a.grad, b.grad, out)

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_grouped_gemm_3d_2d(self, device, dtype):
        m, n, k, n_groups = 16, 32, 64, 4
        a = torch.randn(n_groups, m, k, device=device, dtype=dtype)
        b = torch.randn(n * n_groups, k, device=device, dtype=dtype)

        a.requires_grad_(True)
        b.requires_grad_(True)

        offs = torch.arange(
            n, n_groups * n + 1, n, device=device, dtype=torch.int32
        )

        f = F.grouped_mm
        out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        blist, outlist, bgradlist, gOlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            blist.append(b[start : offs_cpu[i]])
            bgradlist.append(b.grad[start : offs_cpu[i]])
            outlist.append(out[:, start : offs_cpu[i]])
            gOlist.append(gO[:, start : offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(a, blist, gOlist, a.grad, bgradlist, outlist)

    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_grouped_gemm_accuracy_large(self, device, dtype):
        """Test with larger sizes to stress the sycl-tla kernel."""
        G, M, N, K = 4, 256, 256, 256
        a = torch.randn(G, M, K, device=device, dtype=dtype)
        b = torch.randn(G, K, N, device=device, dtype=dtype)

        out = F.grouped_mm(a, b, out_dtype=dtype)
        ref = torch.bmm(a.float(), b.float()).to(dtype)
        self.assertEqual(out, ref, atol=0.5, rtol=0.1)


instantiate_device_type_tests(TestGroupedMMXPU, globals(), only_for="xpu")

if __name__ == "__main__":
    run_tests()
