# Owner(s): ["module: intel"]

import re
import unittest
from functools import partial

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    tol as xtol,
    toleranceOverride,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


def get_device_capability(device=None):
    return (9, 0)


torch.cuda.get_device_capability = get_device_capability
torch.testing._internal.common_cuda.SM90OrLater = True

with XPUPatchForImport(False):
    from test_matmul_cuda import (
        e4m3_type,
        e5m2_type,
        f8_msg,
        mm_float8,
        mm_float8_emulated,
        tensor_to_scale,
        TestFP8MatmulCuda,
        TestMatmulCuda,
        TestMixedDtypesLinearCuda,
        to_fp8_saturated,
    )


def cublas_addmm(
    self,
    size: int,
    dtype: torch.dtype,
    reduced_precision: bool = False,
    fp16_accumulate: bool = False,
):
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
    orig_fp16_accumulate = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        reduced_precision
    )
    torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulate
    # Make random tensors on CPU (seed set on common_utils.py import)
    # (Not using numpy because it does not support bfloat16)
    make_arg = partial(make_tensor, dtype=dtype, device="cpu")
    m_beta = make_arg(1)
    m_input = make_arg((n, p))
    m_1 = make_arg((n, m))
    m_2 = make_arg((m, p))
    # scale to abate overflows in fp16 accum
    if fp16_accumulate:
        m_1 = m_1 / 100
        m_2 = m_2 / 100
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
    torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate


@toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=2e-3)})
@dtypes(torch.float16)
def cublas_addmm_alignment(self, dtype):
    device = "xpu"
    # perturb X, A, or B alignment
    for idx in range(0, 3):
        for offset in range(1, 3):
            offsets = [0, 0, 0]
            offsets[idx] = offset
            x_offset, a_offset, b_offset = offsets
            A = torch.rand(
                (5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device
            )
            A = A[a_offset:].reshape(5120, 2560)
            X = torch.rand(
                (26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device
            )
            X = X[x_offset:].reshape(26, 1, 2560)
            B = torch.rand(
                (5120 + b_offset), requires_grad=True, dtype=dtype, device=device
            )
            B = B[b_offset:].reshape(5120)
            out = torch.nn.functional.linear(X, A, B)
            self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)


TestMatmulCuda.cublas_addmm = cublas_addmm
TestMatmulCuda.test_cublas_addmm_alignment = cublas_addmm_alignment


@parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
def _test_scaled_mm_vs_emulated(self, base_dtype):
    torch.manual_seed(42)
    input_dtype = e4m3_type
    output_dtype = base_dtype
    compare_type = torch.float32

    x = torch.randn(16, 16, device="xpu", dtype=base_dtype)
    y = torch.randn(32, 16, device="xpu", dtype=base_dtype).t()

    x_scale = tensor_to_scale(x, input_dtype).float()
    y_scale = tensor_to_scale(y, input_dtype).float()

    x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
    y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

    # Calculate actual F8 mm
    out_scaled_mm = mm_float8(
        x_fp8, y_fp8, a_scale=x_scale, b_scale=y_scale, output_dtype=output_dtype
    )

    # Calculate emulated F8 mm
    out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

    if output_dtype != base_dtype:
        out_scaled_mm = out_scaled_mm.to(compare_type)
        out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

        out_emulated = out_emulated.to(compare_type)
        out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

    if base_dtype in {torch.bfloat16, torch.float16}:
        atol, rtol = 7e-2, 7e-2
    else:
        atol, rtol = 3e-3, 3e-3

    torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)


TestFP8MatmulCuda.test_scaled_mm_vs_emulated = _test_scaled_mm_vs_emulated


@parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
def _test_scaled_mm_change_stride(self, base_dtype):
    torch.manual_seed(42)
    input_dtype = e4m3_type
    output_dtype = base_dtype
    compare_type = torch.float32

    x = torch.empty_strided((16, 16), (16, 1), device="xpu", dtype=base_dtype)
    y = torch.empty_strided((16, 32), (1, 64), device="xpu", dtype=base_dtype)

    x.normal_()
    y.normal_()

    x_scale = tensor_to_scale(x, input_dtype).float()
    y_scale = tensor_to_scale(y, input_dtype).float()

    x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
    y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

    # Calculate actual F8 mm
    out_scaled_mm = mm_float8(
        x_fp8, y_fp8, a_scale=x_scale, b_scale=y_scale, output_dtype=output_dtype
    )

    # Calculate emulated F8 mm
    out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

    if output_dtype != base_dtype:
        out_scaled_mm = out_scaled_mm.to(compare_type)
        out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

        out_emulated = out_emulated.to(compare_type)
        out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

    if base_dtype in {torch.bfloat16, torch.float16}:
        atol, rtol = 7e-2, 7e-2
    else:
        atol, rtol = 3e-3, 3e-3

    torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)


TestFP8MatmulCuda.test_scaled_mm_change_stride = _test_scaled_mm_change_stride


@unittest.skipIf(IS_WINDOWS, f8_msg)
def _test_float8_error_messages(self, device) -> None:
    M, K, N = (1024, 512, 2048)
    fill_value = 0.5
    x = torch.full((M, K), fill_value, device=device)
    y = torch.full((N, K), fill_value, device=device)

    x_fp8 = x.to(e4m3_type)
    y_fp8 = y.to(e4m3_type).t()

    with self.assertRaisesRegex(
        RuntimeError,
        re.escape(
            "For RowWise scaling, scale_a should be (1024, 1) and scale_b "
            "should be (1, 2048). Got scale_a.size()=(1, 1) and scale_b.size()=(1, 2)"
        ),
    ):
        torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=torch.ones((1, 1), device="xpu"),
            scale_b=torch.ones((1, 2), device="xpu"),
            out_dtype=torch.bfloat16,
        )

    with self.assertRaisesRegex(
        RuntimeError,
        re.escape(
            " For RowWise scaling, scale_a should be (1024, 1) and scale_b "
            "should be (1, 2048). Got scale_a.size()=(1024, 1) and scale_b.size()=(1, 2049)"
        ),
    ):
        torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=torch.ones((M, 1), device="xpu"),
            scale_b=torch.ones((1, N + 1), device="xpu"),
            out_dtype=torch.bfloat16,
        )
    with self.assertRaisesRegex(
        RuntimeError,
        re.escape("For non-TensorWise scaling, scale tensors must be 2-dimensional"),
    ):
        torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=torch.ones((M), device="xpu"),
            scale_b=torch.ones((N, N), device="xpu"),
            out_dtype=torch.bfloat16,
        )

    with self.assertRaisesRegex(
        RuntimeError,
        re.escape("Both scale_a and scale_b must be contiguous for RowWise scaling."),
    ):
        torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=torch.ones((M, 1), device="xpu"),
            scale_b=torch.ones((1, N * 2), device="xpu")[:, ::2],
            out_dtype=torch.bfloat16,
        )

    # Note re.compile is used, not re.escape. This is to accomodate fn vs fnuz type message.
    with self.assertRaisesRegex(
        RuntimeError,
        r"Expected b\.dtype\(\) == at::kFloat8_e4m3fnu?z? to be true, but got false\.",
    ):
        torch._scaled_mm(
            x_fp8,
            y_fp8.to(e5m2_type),
            scale_a=torch.ones((M, 1), device="xpu"),
            scale_b=torch.ones((1, N), device="xpu"),
            out_dtype=torch.bfloat16,
        )


TestFP8MatmulCuda.test_float8_error_messages = _test_float8_error_messages


@unittest.skipIf(IS_WINDOWS, f8_msg)
@parametrize("base_dtype", [torch.bfloat16])
def _test_scaled_mm_vs_emulated_row_wise(self, base_dtype):
    torch.manual_seed(42)
    input_dtype = e4m3_type
    output_dtype = base_dtype

    x = torch.randn(16, 16, device="xpu", dtype=base_dtype)
    y = torch.randn(32, 16, device="xpu", dtype=base_dtype).t()

    x_scales = tensor_to_scale(x, input_dtype, dim=1).float()
    y_scales = tensor_to_scale(y, input_dtype, dim=0).float()

    x_fp8 = to_fp8_saturated(x * x_scales, e4m3_type)
    y_fp8 = to_fp8_saturated(y * y_scales, e4m3_type)

    # Calculate actual F8 mm
    out_scaled_mm = mm_float8(
        x_fp8, y_fp8, a_scale=x_scales, b_scale=y_scales, output_dtype=output_dtype
    )

    # Calculate emulated F8 mm
    out_emulated = mm_float8_emulated(x_fp8, x_scales, y_fp8, y_scales, output_dtype)

    if base_dtype in {torch.bfloat16, torch.float16}:
        atol, rtol = 7e-2, 7e-2
    else:
        atol, rtol = 2e-3, 2e-3

    torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)


TestFP8MatmulCuda.test_scaled_mm_vs_emulated_row_wise = (
    _test_scaled_mm_vs_emulated_row_wise
)


def _cublas_and_lt_reduced_precision_fp16_accumulate(self):
    orig_fp16_accumulate = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    x = torch.rand(32, 512, 512, device="xpu", dtype=torch.half)
    w = torch.rand(512, 512, device="xpu", dtype=torch.half)
    b = torch.rand(512, device="xpu", dtype=torch.half)
    out = torch.nn.functional.linear(x, w, b)
    out_cpu = torch.nn.functional.linear(x.cpu(), w.cpu(), b.cpu())
    self.assertEqual(out, out_cpu, atol=5e-3, rtol=8e-3)

    a = torch.rand(16, 128, 128, device="xpu", dtype=torch.half)
    b = torch.rand(16, 128, 128, device="xpu", dtype=torch.half)
    c = torch.rand(16, 128, 128, device="xpu", dtype=torch.half)
    out = torch.baddbmm(a, b, c)
    out_cpu = torch.baddbmm(a.cpu(), b.cpu(), c.cpu())
    self.assertEqual(out, out_cpu, atol=1e-3, rtol=5e-3)
    torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate


TestMatmulCuda.test_cublas_and_lt_reduced_precision_fp16_accumulate = (
    _cublas_and_lt_reduced_precision_fp16_accumulate
)


TestMixedDtypesLinearCuda._default_dtype_check_enabled = True
TestFP8MatmulCuda._default_dtype_check_enabled = True
TestMatmulCuda._default_dtype_check_enabled = True
instantiate_device_type_tests(
    TestMixedDtypesLinearCuda, globals(), only_for=("xpu"), allow_xpu=True
)

instantiate_device_type_tests(
    TestFP8MatmulCuda, globals(), only_for=("xpu"), allow_xpu=True
)

instantiate_device_type_tests(
    TestMatmulCuda, globals(), only_for=("xpu"), allow_xpu=True
)
if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
