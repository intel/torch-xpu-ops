
# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
    toleranceOverride,
    tol as xtol,
    ) 
from torch.testing._internal.common_utils import run_tests,TestCase,parametrize,IS_WINDOWS
import torch
from functools import partial
from torch.testing import make_tensor
import unittest
from itertools import product

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_matmul_cuda import (
        TestMixedDtypesLinearCuda,
        TestMatmulCuda,
        scaled_mm_supported_device,
        f8_msg,
        e4m3_type,
        e5m2_type,
        tensor_to_scale,
        to_fp8_saturated,
        mm_float8,
        mm_float8_emulated,
        amax_to_scale,
        )

def cublas_addmm(self, size: int, dtype: torch.dtype, reduced_precision: bool = False):
    #
    # Check for catastrophic cuBLAS inaccuracy by measuring the deviation between
    # results from the CUDA invocation of torch.addmm and the CPU invocation
    # (which does not use CUDA backend).
    #
    # Get dims
    n, m, p = (size + 1, size, size + 2)
    # Disable reduced precision reductions in BFloat16 to bypass some kernels
    # which fail the threshold check
    orig_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    orig_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision
    # Make random tensors on CPU (seed set on common_utils.py import)
    # (Not using numpy because it does not support bfloat16)
    make_arg = partial(make_tensor, dtype=dtype, device="cpu")
    m_beta = make_arg(1)
    m_input = make_arg((n, p))
    m_1 = make_arg((n, m))
    m_2 = make_arg((m, p))
    # *(B)FLOAT16 Special Handling*
    # Backend does not tensorize float16 on CPU,
    # and bloat16 may present accuracy issues,
    # so convert to float32 for these cases
    # (but keep same for other types, e.g. float32 and int*)
    if dtype == torch.float16 or dtype == torch.bfloat16:
        m_beta = m_beta.to(dtype=torch.float32)
        m_input = m_input.to(dtype=torch.float32)
        m_1 = m_1.to(dtype=torch.float32)
        m_2 = m_2.to(dtype=torch.float32)
    # Get CPU result
    res_cpu = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
    # *(B)FLOAT16 Special Handling*``
    # Convert back to (b)float16
    if dtype == torch.float16 or dtype == torch.bfloat16:
        m_beta = m_beta.to(dtype=dtype)
        m_input = m_input.to(dtype=dtype)
        m_1 = m_1.to(dtype=dtype)
        m_2 = m_2.to(dtype=dtype)
        res_cpu = res_cpu.to(dtype=dtype)
    # Move arg tensors to CUDA
    m_beta = m_beta.to("xpu")
    m_input = m_input.to("xpu")
    m_1 = m_1.to("xpu")
    m_2 = m_2.to("xpu")
    # Get CUDA result
    res_cuda = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
    # Move to CPU for comparison
    res_cuda = res_cuda.to("cpu")
    # Compare
    self.assertEqual(res_cpu, res_cuda)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16

@toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=2e-3)})
@dtypes(torch.float16)
def cublas_addmm_alignment(self, dtype):
    device = 'xpu'
    # perturb X, A, or B alignment
    for idx in range(0, 3):
        for offset in range(1, 3):
            offsets = [0, 0, 0]
            offsets[idx] = offset
            x_offset, a_offset, b_offset = offsets
            A = torch.rand((5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device)
            A = A[a_offset:].reshape(5120, 2560)
            X = torch.rand((26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device)
            X = X[x_offset:].reshape(26, 1, 2560)
            B = torch.rand((5120 + b_offset), requires_grad=True, dtype=dtype, device=device)
            B = B[b_offset:].reshape(5120)
            out = torch.nn.functional.linear(X, A, B)
            self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)

TestMatmulCuda.cublas_addmm=cublas_addmm
TestMatmulCuda.test_cublas_addmm_alignment=cublas_addmm_alignment

class TestFP8MatmulCuda(TestCase):

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def _test_tautological_mm(self, device: str = "xpu",
                              x_dtype: torch.dtype = e4m3_type,
                              y_dtype: torch.dtype = e4m3_type,
                              out_dtype = None,
                              size: int = 16) -> None:
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        (out_fp8, amax_fp8) = torch._scaled_mm(x_fp8, y_fp8, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        if out_dtype not in [torch.float16, torch.bfloat16, torch.float]:
            self.assertEqual(out_fp32.amax(), amax_fp8)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_basics(self, device) -> None:
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)
        # hipblaslt does not yet support mixed e4m3_type input
        if torch.version.hip is None:
            self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
            self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)
        # According to https://docs.nvidia.com/cuda/cublas/#id99 8F_E5M2 MM is unsupported
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        # hipblaslt does not yet support bfloat16 output
        if torch.version.hip is None:
            self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_scale(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s, amax_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        y = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x, x_scale, e4m3_type)
        y_fp8 = to_fp8_saturated(y, y_scale, e4m3_type)

        # Calculate actual F8 mm
        out_scaled_mm, output_amax_scaled = mm_float8(
            x_fp8,
            y_fp8,
            a_scale=x_scale,
            b_scale=y_scale,
            output_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated, output_amax_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_emulated = out_emulated.to(compare_type)

            out_scaled_mm = out_scaled_mm / amax_to_scale(
                output_amax_scaled, input_dtype
            )
            out_emulated = out_emulated / amax_to_scale(
                output_amax_emulated, input_dtype
            )

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        if torch.version.hip is None:
            self.assertEqual((amaxb_fp8 - amax_fp8).item(), 4.0)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: torch.bool) -> None:
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y, bias=input_bias)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.half)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        self.assertEqual(amaxb_fp8.item(), 3.0)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: torch._scaled_mm(x, y, bias=bias, out_dtype=torch.float32),
        )

    @unittest.skipIf(scaled_mm_supported_device(),
                     "This test is only for devices with compute capability < 8.9")
    def test_error_message_fp8_pre_sm89(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.rand((m, l), device=device).to(e4m3_type).t()
        self.assertRaisesRegex(
            RuntimeError,
            r"torch\.\_scaled\_mm is only supported on CUDA devices with compute capability \>\= 9\.0 or 8\.9, or ROCm MI300\+",
            lambda: torch._scaled_mm(x, y, out_dtype=torch.float32),
        )

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_scale_fast_accum(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y, use_fast_accum=True)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s, amax_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)
        self.assertEqual(out_fp8, out_fp8_s)

TestMixedDtypesLinearCuda._default_dtype_check_enabled = True
TestFP8MatmulCuda._default_dtype_check_enabled = True
TestMatmulCuda._default_dtype_check_enabled = True
instantiate_device_type_tests(TestMixedDtypesLinearCuda, globals(), only_for=("xpu"))

instantiate_device_type_tests(TestFP8MatmulCuda, globals(), only_for=("xpu"))

instantiate_device_type_tests(TestMatmulCuda, globals(), only_for=("xpu"))
if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
