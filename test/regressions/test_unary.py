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

floating_types = [torch.float, torch.half, torch.bfloat16, torch.double]
integral_types = [
    torch.int8,
    torch.uint8,
    torch.short,
    torch.int,
    torch.long,
    torch.bool,
]
complex_types = [torch.cfloat, torch.cdouble]
floating_and_complex_types = floating_types + complex_types
all_basic_types = floating_types + integral_types
all_basic_and_complex_types = floating_types + integral_types + complex_types


class Dtypes:  # noqa: UP004
    def __init__(self, include_dtypes, exclude_dtypes=[]):  # noqa: B006
        self.include_dtypes = include_dtypes
        self.exclude_dtypes = exclude_dtypes

    def __call__(self, fn):
        def fn_out(*args, **kwargs):
            for dtype in self.include_dtypes:
                if dtype in self.exclude_dtypes:
                    continue
                kwargs["dtype"] = dtype
                fn(*args, **kwargs)

        return fn_out


class TestSimpleUnary(TestCase):
    def _test_unary_out_ops(self, fn_str, dtype):
        a_cpu = (torch.randn(2049) * 10).to(dtype)
        a_xpu = a_cpu.xpu()
        b_cpu = eval(f"torch.{fn_str}(a_cpu)")
        b_xpu = eval(f"torch.{fn_str}(a_xpu)")
        c_cpu = eval(f"a_cpu.{fn_str}()")
        c_xpu = eval(f"a_xpu.{fn_str}()")
        self.assertEqual(b_cpu, b_xpu.cpu(), atol=1e-4, rtol=1e-4)
        self.assertEqual(c_cpu, c_xpu.cpu(), atol=1e-4, rtol=1e-4)
        d_cpu = eval(f"torch.{fn_str}(a_cpu, out=c_cpu)")
        d_xpu = eval(f"torch.{fn_str}(a_xpu, out=c_xpu)")
        self.assertEqual(c_cpu, c_xpu.cpu(), atol=1e-4, rtol=1e-4)

    @Dtypes(floating_types)
    def test_abs_out(self, dtype):
        self._test_unary_out_ops("abs", dtype)

    @Dtypes(floating_and_complex_types)
    def test_sin_out(self, dtype):
        self._test_unary_out_ops("sin", dtype)

    @Dtypes(floating_and_complex_types)
    def test_cos_out(self, dtype):
        self._test_unary_out_ops("cos", dtype)

    @Dtypes(floating_and_complex_types)
    def test_log_out(self, dtype):
        self._test_unary_out_ops("log", dtype)

    @Dtypes(floating_and_complex_types)
    def test_sqrt_out(self, dtype):
        self._test_unary_out_ops("sqrt", dtype)

    @Dtypes(floating_and_complex_types)
    def test_rsqrt_out(self, dtype):
        self._test_unary_out_ops("rsqrt", dtype)

    @Dtypes(floating_and_complex_types)
    def test_tanh_out(self, dtype):
        self._test_unary_out_ops("tanh", dtype)

    # Regression test for TanhBackwardFunctor precision bug (bf16/half).
    # The XPU kernel must use float32 intermediates (opmath_type) for b*b,
    # otherwise a 1-ULP rounding error in `b*b` propagates to the gradient.
    @Dtypes([torch.bfloat16, torch.float16])
    def test_tanh_backward_reduced_precision(self, dtype):
        z_vals = [0.5, 1.0, 1.5, 2.0, 2.5, -0.5, -1.5, -2.5]
        z_cpu = torch.tensor(z_vals, dtype=dtype, requires_grad=True)
        z_xpu = torch.tensor(z_vals, dtype=dtype, device="xpu", requires_grad=True)
        torch.tanh(z_cpu).backward(torch.ones_like(z_cpu))
        torch.tanh(z_xpu).backward(torch.ones_like(z_xpu))
        self.assertEqual(z_cpu.grad, z_xpu.grad.cpu())

    def test_sigmoid_special_value_complex(self):
        # (xpu_dtype, compare_dtype, numpy_dtype, z)
        cases = [
            (torch.complex32, torch.complex64, np.complex64, complex(501, float("-inf"))),
            (torch.complex64, torch.complex64, np.complex64, complex(501, float("-inf"))),
            (torch.complex128, torch.complex128, np.complex128, complex(3000, float("-inf"))),
        ]
        for xpu_dtype, compare_dtype, numpy_dtype, z in cases:
            with self.subTest(dtype=xpu_dtype):
                x_xpu = torch.tensor(z, dtype=xpu_dtype, device="xpu")
                out_xpu = torch.sigmoid(x_xpu)
                self.assertEqual(out_xpu.dtype, xpu_dtype)

                znp = numpy_dtype(z)
                ref = (1.0 / (1.0 + np.exp(-znp))).astype(numpy_dtype)
                expected = torch.tensor(ref, dtype=compare_dtype)
                self.assertEqual(
                    out_xpu.to(compare_dtype).cpu(), expected, equal_nan=True
                )

    @Dtypes(all_basic_and_complex_types, [torch.bool])
    def test_neg_out(self, dtype):
        self._test_unary_out_ops("neg", dtype)

    @Dtypes(floating_and_complex_types)
    def test_reciprocal_out(self, dtype):
        self._test_unary_out_ops("reciprocal", dtype)

    def test_unary_geometric_bcomplex32(self):
        x_cpu = torch.randn(64, dtype=torch.complex64) * 0.1
        x_xpu = x_cpu.to("xpu").to(torch.bcomplex32)
        unary_geometric_ops = [
            torch.acos,
            torch.acosh,
            torch.asin,
            torch.asinh,
            torch.atan,
            torch.atanh,
            torch.cos,
            torch.cosh,
            torch.sin,
            torch.sinh,
            torch.tan,
            torch.tanh,
        ]

        for op in unary_geometric_ops:
            with self.subTest(op=op.__name__):
                out = op(x_xpu)
                self.assertEqual(out.dtype, torch.bcomplex32)
                self.assertEqual(out.shape, x_xpu.shape)

                ref = op(x_cpu)
                self.assertEqual(
                    out.to(torch.complex64).cpu(), ref, rtol=3e-2, atol=3e-2
                )
