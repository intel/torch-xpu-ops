
# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests,dtypes,precisionOverride
from torch.testing._internal.common_utils import run_tests,TestCase,setBlasBackendsToDefaultFinally,setLinalgBackendsToDefaultFinally,parametrize,IS_WINDOWS
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing._internal.common_mkldnn import bf32_on_and_off
from torch.testing._internal.common_quantization import _dynamically_quantize_per_channel
from torch.testing import make_tensor
import unittest
import itertools
from functools import partial
import torch
import numpy as np
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
@unittest.skip("xpu not support multi blas")
def preferred_blas_library(self):
    pass

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

@unittest.skip("xpu does not support multi linalg")
@setLinalgBackendsToDefaultFinally
def preferred_linalg_library(self):
    pass

@precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
@dtypes(*floating_and_complex_types_and(torch.bfloat16, torch.half))
@tf32_on_and_off(0.05)
@bf32_on_and_off(0.05)
def addbmm(self, device, dtype):
    num_batches = 2
    M, N, O = 16, 17, 18

    def invert_perm(p):
        d = {x: i for i, x in enumerate(p)}
        return (d[0], d[1], d[2])

    def generate_tensor():
        numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32
        # transposed tensors
        for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
            for perm3 in itertools.permutations((0, 1)):
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1) * 0.1
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1) * 0.1
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype).sum(0)
                out_tensor = torch.zeros_like(ref).permute(perm3).contiguous().permute(perm3)
                yield b1, b2, ref, out_tensor
        # broadcasting tensors
        for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
            shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
            shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
            b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N) * 0.1
            b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O) * 0.1
            ref = torch.from_numpy(
                b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
            ).to(device=device, dtype=dtype).sum(0)
            out_tensor = torch.zeros_like(ref)
            yield b1, b2, ref, out_tensor
        # zero-sized tensors
        for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
            shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
            shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
            b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1) * 0.1
            b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1) * 0.1
            ref = torch.from_numpy(
                b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
            ).to(device=device, dtype=dtype).sum(0)
            out_tensor = torch.zeros_like(ref)
            yield b1, b2, ref, out_tensor

    for b1, b2, ref, out_tensor in generate_tensor():
        self._test_addbmm_baddbmm("addbmm", b1, b2, ref, out_tensor)

@unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
@parametrize("k", [16, 32])
@parametrize("n", [16, 32])
@parametrize("use_transpose_a", [True, False])
@parametrize("use_transpose_b", [True, False])
def _int_mm(self, device, k, n, use_transpose_a, use_transpose_b):
    def genf_int_float(x, y, use_transpose):
        if use_transpose:
            x, y = y, x
        x_int8 = torch.randint(-10, 10, (x, y), dtype=torch.int8, device=device)
        x_float = x_int8.to(torch.float32)
        if use_transpose:
            return x_int8.t(), x_float.t()
        return x_int8, x_float

    def _test(m, k, n, transpose_a, transpose_b, test_equal=True):
        a_int8, a_float = genf_int_float(m, k, transpose_a)
        b_int8, b_float = genf_int_float(k, n, transpose_b)
        c_int32 = torch._int_mm(a_int8, b_int8)
        self.assertTrue(c_int32.dtype is torch.int32)
        self.assertEqual(c_int32.device, torch.device(device))
        if test_equal:
            self.assertEqual(c_int32.float(), torch.mm(a_float, b_float))
        else:
            self.assertNotEqual(c_int32.float(), torch.mm(a_float, b_float))
        c_int32_result = c_int32.new_empty(c_int32.size())
        # Checking out variant
        torch._int_mm(a_int8, b_int8, out=c_int32_result)
        if test_equal:
            self.assertEqual(c_int32_result.float(), torch.mm(a_float, b_float))
        else:
            self.assertNotEqual(c_int32_result.float(), torch.mm(a_float, b_float))


    if not use_transpose_a and use_transpose_b:
        _test(17, k, n, use_transpose_a, use_transpose_b)

    if use_transpose_a and not use_transpose_b:
        _test(17, k, n, use_transpose_a, use_transpose_b)

    if use_transpose_a and use_transpose_b:
        _test(17, k, n, use_transpose_a, use_transpose_b)

    if not use_transpose_a and not use_transpose_b:
        _test(17, k, n, use_transpose_a, use_transpose_b)

@unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
@parametrize("m", [1])
@parametrize("k", [32, 64, 128, 256, 512, 1024])
@parametrize("n", [32, 64, 128, 256, 512, 1024])
def _int4_mm(self, device, m, k, n):
    def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
        assert w.dim() == 2
        w = w.transpose(0, 1).contiguous()
        assert q_group_size > 1
        assert w.shape[-1] % q_group_size == 0

        to_quant = w.reshape(-1, q_group_size)
        assert torch.isnan(to_quant).sum() == 0

        max_val = to_quant.amax(dim=1, keepdim=True)
        min_val = to_quant.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-6) / max_int
        assert torch.isnan(scales).sum() == 0
        zeros = min_val + scales * (2 ** (n_bit - 1))
        assert torch.isnan(zeros).sum() == 0

        out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
 
        assert torch.isnan(out).sum() == 0

        out = out.to(dtype=torch.uint8).reshape(w.shape)
        
        if out.device.type == 'xpu':
            out = (out[::, 1::2] << 4 | out[::, ::2]).to(torch.uint8)
        elif out.device != torch.device('cpu'):
            out = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)
        # Scales and zeros for the same q-group should be contiguous, so we can
        # load as a 32-bit word
        scales = scales.view(w.shape[0], -1)
        zeros = zeros.view(w.shape[0], -1)
        scales_and_zeros = (
            torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
        )

        if out.device.type != 'xpu':
            scales_and_zeros = scales_and_zeros.transpose(0, 1).contiguous()
        return out, scales_and_zeros
        
    def convert_weight_to_int4pack(b):
        b_tmp, b_scales_and_zeros = _group_quantize_tensor(
            b, n_bit=4, q_group_size=q_group
        )
        
        if self.device_type == 'cpu':
            b_int4pack = torch._convert_weight_to_int4pack_for_cpu(
                b_tmp, inner_k_tiles
            )
        elif self.device_type == 'xpu':
            b_int4pack = b_tmp.view(torch.int32)
        else:
            b_int4pack = torch._convert_weight_to_int4pack(
                b_tmp, inner_k_tiles
            )

        return b_int4pack, b_scales_and_zeros

    def weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros):
        if self.device_type == 'cpu':
            self.assertTrue(b_int4pack.dtype is torch.uint8)
            self.assertTrue(b_int4pack.dim() == 2)
            return torch._weight_int4pack_mm_for_cpu(
                a, b_int4pack, q_group, b_scales_and_zeros
            )
        elif self.device_type == 'xpu':
            self.assertTrue(b_int4pack.dtype is torch.int32) # or b_int4pack.dtype is torch.uint8)
            self.assertTrue(b_int4pack.dim() == 2)
            return torch._weight_int4pack_mm(
                a, b_int4pack, q_group, b_scales_and_zeros
            )
        else:
            self.assertTrue(b_int4pack.dtype is torch.int32)
            self.assertTrue(b_int4pack.dim() == 4)
            return torch._weight_int4pack_mm(
                a, b_int4pack, q_group, b_scales_and_zeros
            )
    
    q_group = 32
    inner_k_tiles = 2

    torch.manual_seed(1)
    a_bf16 = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    b_bf16 = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    b_int4pack, b_scales_and_zeros_bf16 = convert_weight_to_int4pack(b_bf16)
    for dtype in [torch.bfloat16] + ([torch.float16, torch.float32] if device == "cpu" else [torch.float16] if "xpu" in device else []):
        a = a_bf16.to(dtype=dtype)
        b = b_bf16.to(dtype=dtype)
        b_scales_and_zeros = b_scales_and_zeros_bf16.to(dtype=dtype)
        ref = torch.mm(a, b)
        res = weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros)
        mean_err = ((res - ref).abs() / ref).mean()
        self.assertTrue(mean_err < 0.05)

@dtypes(torch.float, torch.complex64)  # Integer matmul just supported on CPU
@setBlasBackendsToDefaultFinally
def matmul_small_brute_force_1d_Nd(self, device, dtype):
    for backend in ["cublas", "cublaslt"]:
        if torch.device(device).type == 'cuda':
            torch.backends.cuda.preferred_blas_library(backend)

    make_arg = partial(make_tensor, device=device, dtype=dtype)

    for (size_x, size_y), nctg_x, nctg_y in itertools.product(self.gen_sizes_matmul(1), (True, False), (True, False)):
        x = make_arg(size_x, noncontiguous=nctg_x)
        y = make_arg(size_y, noncontiguous=nctg_y)
        self.check_single_matmul(x, y)

@dtypes(torch.float, torch.complex64)  # Integer matmul just supported on CPU
@setBlasBackendsToDefaultFinally
def matmul_small_brute_force_2d_Nd(self, device, dtype):
    for backend in ["cublas", "cublaslt"]:
        if torch.device(device).type == 'cuda':
            torch.backends.cuda.preferred_blas_library(backend)

        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in itertools.product(self.gen_sizes_matmul(2), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

@dtypes(torch.float, torch.complex64)  # Integer matmul just supported on CPU
@setBlasBackendsToDefaultFinally
def matmul_small_brute_force_3d_Nd(self, device, dtype):
    for backend in ["cublas", "cublaslt"]:
        if torch.device(device).type == 'cuda':
            torch.backends.cuda.preferred_blas_library(backend)

        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in itertools.product(self.gen_sizes_matmul(3), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

@setBlasBackendsToDefaultFinally
@unittest.skip("xpu not support ck blas library")
def ck_blas_library(self):
    pass

with XPUPatchForImport(False):
    from test_linalg import TestLinalg

TestLinalg.test_large_bmm_mm_backward=large_bmm_mm_backward
TestLinalg.test_large_bmm_backward=large_bmm_backward
TestLinalg.test_preferred_blas_library=preferred_blas_library
TestLinalg.test_eigh_svd_illcondition_matrix_input_should_not_crash=eigh_svd_illcondition_matrix_input_should_not_crash
TestLinalg.test_matmul_45724=matmul_45724
TestLinalg.test_preferred_linalg_library=preferred_linalg_library
TestLinalg.test_addbmm=addbmm
TestLinalg.test__int_mm=_int_mm
TestLinalg.test__int4_mm=_int4_mm
TestLinalg.test_matmul_small_brute_force_1d_Nd=matmul_small_brute_force_1d_Nd
TestLinalg.test_matmul_small_brute_force_2d_Nd=matmul_small_brute_force_2d_Nd
TestLinalg.test_matmul_small_brute_force_3d_Nd=matmul_small_brute_force_3d_Nd
TestLinalg.test_ck_blas_library = ck_blas_library

TestLinalg._default_dtype_check_enabled = True
instantiate_device_type_tests(TestLinalg, globals(), only_for=("xpu"), allow_xpu=True)

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
