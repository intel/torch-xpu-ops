
# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests,dtypes
from torch.testing._internal.common_utils import run_tests,TestCase,setBlasBackendsToDefaultFinally,setLinalgBackendsToDefaultFinally
import torch
import math
try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

def large_bmm_mm_backward(self, device):
    A = torch.randn([1024, 2, 1024], device="xpu").mT.contiguous().mT
    B = torch.randn([1024, 65536], device="xpu", requires_grad=True)
    G = torch.randn([1024, 2, 65536], device="xpu")

    # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
    (A @ B).backward(G)

def large_bmm_backward(self, device):
    A = torch.randn([1024, 2, 1024], device="xpu").mT.contiguous().mT
    B = torch.randn([1, 1024, 65536], device="xpu", requires_grad=True)
    G = torch.randn([1024, 2, 65536], device="xpu")

    # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
    (A @ B).backward(G)

@setBlasBackendsToDefaultFinally
def preferred_blas_library(self):
    # The main purpose of this test is to make sure these "backend" calls work normally without raising exceptions.
    m1 = torch.randint(2, 5, (2048, 2400), device='xpu', dtype=torch.float)
    m2 = torch.randint(2, 5, (128, 2400), device='xpu', dtype=torch.float)

    torch.backends.cuda.preferred_blas_library('cublaslt')
    out1 = torch.nn.functional.linear(m1, m2)

    torch.backends.cuda.preferred_blas_library('cublas')
    out2 = torch.nn.functional.linear(m1, m2)

    # Although blas preferred flags doesn't affect CPU currently,
    # we set this to make sure the flag can switch back to default normally.
    out_ref = torch.nn.functional.linear(m1.cpu(), m2.cpu())

    self.assertEqual(out1, out2)
    self.assertEqual(out_ref, out2.cpu())

@dtypes(torch.float, torch.double)
def eigh_svd_illcondition_matrix_input_should_not_crash(self, device, dtype):
    # See https://github.com/pytorch/pytorch/issues/94772, https://github.com/pytorch/pytorch/issues/105359
    # This test crashes with `cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED` on cuda 11.8,
    # but passes on cuda 12.1 update 1 or later.
    a = torch.ones(512, 512, dtype=dtype, device=device)
    a[0, 0] = 1.0e-5
    a[-1, -1] = 1.0e5

    eigh_out = torch.linalg.eigh(a)
    svd_out = torch.linalg.svd(a)

    # Matrix input a is too ill-conditioned.
    # We'll just compare the first two singular values/eigenvalues. They are 1.0e5 and 511.0
    # The precision override with tolerance of 1.0 makes sense since ill-conditioned inputs are hard to converge
    # to exact values.
    self.assertEqual(eigh_out.eigenvalues.sort(descending=True).values[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)
    self.assertEqual(svd_out.S[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)


def matmul_45724(self, device):
    # https://github.com/pytorch/pytorch/issues/45724
    a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
    b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
    c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
    cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).xpu().half()
    torch.matmul(a, b, out=c)
    self.assertEqual(c, cpu_result)

@setLinalgBackendsToDefaultFinally
def preferred_linalg_library(self):
    # The main purpose of this test is to make sure these "backend" calls work normally without raising exceptions.
    x = torch.randint(2, 5, (2, 4, 4), device='xpu', dtype=torch.double)

    torch.backends.cuda.preferred_linalg_library('cusolver')
    out1 = torch.linalg.inv(x)

    torch.backends.cuda.preferred_linalg_library('magma')
    out2 = torch.linalg.inv(x)

    torch.backends.cuda.preferred_linalg_library('default')
    # Although linalg preferred flags doesn't affect CPU currently,
    # we set this to make sure the flag can switch back to default normally.
    out_ref = torch.linalg.inv(x.cpu())

    self.assertEqual(out_ref, out1.cpu())
    self.assertEqual(out1, out2)

with XPUPatchForImport(False):
    from test_linalg import TestLinalg

TestLinalg.test_large_bmm_mm_backward=large_bmm_mm_backward
TestLinalg.test_large_bmm_backward=large_bmm_backward
TestLinalg.test_preferred_blas_library=preferred_blas_library
TestLinalg.test_eigh_svd_illcondition_matrix_input_should_not_crash=eigh_svd_illcondition_matrix_input_should_not_crash
TestLinalg.test_matmul_45724=matmul_45724
TestLinalg.test_preferred_linalg_library=preferred_linalg_library
TestLinalg._default_dtype_check_enabled = True
instantiate_device_type_tests(TestLinalg, globals(),only_for=("xpu"))

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
