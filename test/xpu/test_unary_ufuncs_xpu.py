# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import math
import unittest

import numpy as np
import torch
from torch import inf, nan
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    floating_types_and,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_SCIPY,
    xfailIfTorchDynamo,
)

if TEST_SCIPY:
    import scipy


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_unary_ufuncs import TestUnaryUfuncs

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
    )

    @dtypes(torch.cdouble)
    def _test_complex_edge_values(self, device, dtype):
        # sqrt Test Reference: https://github.com/pytorch/pytorch/pull/47424
        x = torch.tensor(0.0 - 1.0e20j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)
        # acos test reference: https://github.com/pytorch/pytorch/issues/42952
        if not (dtype == torch.cdouble and device_type in device):
            self.compare_with_numpy(torch.acos, np.arccos, x)

        x = torch.tensor(
            (-1.0e60 if dtype == torch.cdouble else -1.0e20) - 4988429.2j,
            dtype=dtype,
            device=device,
        )
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

    TestUnaryUfuncs._test_complex_edge_values = _test_complex_edge_values

    @dtypes(torch.complex64)
    def _test_tan_complex_cuda_matches_numpy(self, device, dtype):
        # Focused accuracy check for complex tan on CUDA against NumPy reference
        # Includes values near tan singularities on the real axis
        eps = 1e-3
        specials = torch.tensor(
            [
                math.pi / 2 - eps,
                math.pi / 2 + eps,
                -math.pi / 2 - eps,
                -math.pi / 2 + eps,
            ],
            device=device,
            dtype=torch.float32,
        )
        real = torch.randn(1024, device=device, dtype=torch.float32) * (2 * math.pi)
        imag = torch.randn(1024, device=device, dtype=torch.float32) * 5.0
        real = torch.cat([real, specials])
        imag = torch.cat(
            [
                imag,
                torch.linspace(
                    -3,
                    3,
                    steps=specials.numel(),
                    device=device,
                ),
            ]
        )
        z = torch.complex(real, imag).to(dtype)
        self.compare_with_numpy(torch.tan, np.tan, z)

    TestUnaryUfuncs._test_tan_complex_cuda_matches_numpy = (
        _test_tan_complex_cuda_matches_numpy
    )

    @dtypes(torch.complex64)
    def _test_tanh_complex_cuda_matches_numpy(self, device, dtype):
        # Focused accuracy check for complex tanh on CUDA against NumPy reference
        real = torch.randn(2048, device=device, dtype=torch.float32) * (2 * math.pi)
        imag = torch.randn(2048, device=device, dtype=torch.float32) * 5.0
        z = torch.complex(real, imag).to(dtype)
        self.compare_with_numpy(torch.tanh, np.tanh, z)

    TestUnaryUfuncs._test_tanh_complex_cuda_matches_numpy = (
        _test_tanh_complex_cuda_matches_numpy
    )

    # TODO: run on non-native device types
    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @dtypes(torch.double)
    def _test_unary_out_op_mem_overlap(self, device, dtype):
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        positives = torch.randint(1, 100, (2 * sz,), device=device).double()
        ints = torch.randint(-100, 100, (2 * sz,), device=device)
        unary_mem_overlap_cases = [
            ("abs", doubles, True, True, "cpu"),
            ("abs", doubles, True, True, "cuda"),
            ("abs", doubles, True, True, "xpu"),
            ("acos", doubles, True, True, "cpu"),
            ("acos", doubles, True, True, "cuda"),
            ("acos", doubles, True, True, "xpu"),
            ("asin", doubles, True, True, "cpu"),
            ("asin", doubles, True, True, "cuda"),
            ("asin", doubles, True, True, "xpu"),
            ("atan", doubles, True, True, "cpu"),
            ("atan", doubles, True, True, "cuda"),
            ("atan", doubles, True, True, "xpu"),
            ("acosh", doubles, True, True, "cpu"),
            ("acosh", doubles, True, True, "cuda"),
            ("acosh", doubles, True, True, "xpu"),
            ("asinh", doubles, True, True, "cpu"),
            ("asinh", doubles, True, True, "cuda"),
            ("asinh", doubles, True, True, "xpu"),
            ("atanh", doubles, True, True, "cpu"),
            ("atanh", doubles, True, True, "cuda"),
            ("atanh", doubles, True, True, "xpu"),
            ("bitwise_not", ints, True, True, "cpu"),
            ("bitwise_not", ints, True, True, "cuda"),
            ("bitwise_not", ints, True, True, "xpu"),
            ("ceil", doubles, True, True, "cpu"),
            ("ceil", doubles, True, True, "cuda"),
            ("ceil", doubles, True, True, "xpu"),
            ("cos", doubles, True, True, "cpu"),
            ("cos", doubles, True, True, "cuda"),
            ("cos", doubles, True, True, "xpu"),
            ("cosh", doubles, True, True, "cpu"),
            ("cosh", doubles, True, True, "cuda"),
            ("cosh", doubles, True, True, "xpu"),
            ("digamma", doubles, True, True, "cpu"),
            ("digamma", doubles, True, True, "xpu"),
            ("erf", doubles, True, True, "cpu"),
            ("erf", doubles, True, True, "cuda"),
            ("erf", doubles, True, True, "xpu"),
            ("erfc", doubles, True, True, "cpu"),
            ("erfc", doubles, True, True, "cuda"),
            ("erfc", doubles, True, True, "xpu"),
            ("erfinv", doubles, True, True, "cpu"),
            ("erfinv", doubles, True, True, "cuda"),
            ("erfinv", doubles, True, True, "xpu"),
            ("exp", doubles, True, True, "cpu"),
            ("exp", doubles, True, True, "cuda"),
            ("exp", doubles, True, True, "xpu"),
            ("exp2", doubles, True, True, "cpu"),
            ("exp2", doubles, True, True, "cuda"),
            ("exp2", doubles, True, True, "xpu"),
            ("expm1", doubles, True, True, "cpu"),
            ("expm1", doubles, True, True, "cuda"),
            ("expm1", doubles, True, True, "xpu"),
            ("floor", doubles, True, True, "cpu"),
            ("floor", doubles, True, True, "cuda"),
            ("floor", doubles, True, True, "xpu"),
            ("frac", doubles, True, True, "cpu"),
            ("frac", doubles, True, True, "cuda"),
            ("frac", doubles, True, True, "xpu"),
            ("i0", doubles, True, True, "cpu"),
            ("i0", doubles, True, True, "cuda"),
            ("i0", doubles, True, True, "xpu"),
            ("log", positives, True, True, "cpu"),
            ("log", positives, True, True, "cuda"),
            ("log", positives, True, True, "xpu"),
            ("log10", positives, True, True, "cpu"),
            ("log10", positives, True, True, "cuda"),
            ("log10", positives, True, True, "xpu"),
            ("log1p", positives, True, True, "cpu"),
            ("log1p", positives, True, True, "cuda"),
            ("log1p", positives, True, True, "xpu"),
            ("log2", positives, True, True, "cpu"),
            ("log2", positives, True, True, "cuda"),
            ("log2", positives, True, True, "xpu"),
            ("neg", doubles, True, True, "cpu"),
            ("neg", doubles, True, True, "cuda"),
            ("neg", doubles, True, True, "xpu"),
            ("reciprocal", doubles, True, True, "cpu"),
            ("reciprocal", doubles, True, True, "cuda"),
            ("reciprocal", doubles, True, True, "xpu"),
            ("round", doubles, True, True, "cpu"),
            ("round", doubles, True, True, "cuda"),
            ("round", doubles, True, True, "xpu"),
            ("rsqrt", positives, True, True, "cpu"),
            ("rsqrt", positives, True, True, "cuda"),
            ("rsqrt", positives, True, True, "xpu"),
            ("sin", doubles, True, True, "cpu"),
            ("sin", doubles, True, True, "cuda"),
            ("sin", doubles, True, True, "xpu"),
            ("sinh", doubles, True, True, "cpu"),
            ("sinh", doubles, False, True, "cuda"),
            ("sinh", doubles, False, True, "xpu"),
            ("sigmoid", doubles, True, True, "cpu"),
            ("sigmoid", doubles, True, True, "cuda"),
            ("sigmoid", doubles, True, True, "xpu"),
            ("logit", doubles, True, True, "cpu"),
            ("logit", doubles, True, True, "cuda"),
            ("logit", doubles, True, True, "xpu"),
            ("sqrt", doubles, True, True, "cpu"),
            ("sqrt", doubles, False, True, "cuda"),
            ("sqrt", doubles, False, True, "xpu"),
            ("tan", doubles, True, True, "cpu"),
            ("tan", doubles, True, True, "cuda"),
            ("tan", doubles, True, True, "xpu"),
            ("tanh", doubles, True, True, "cpu"),
            ("tanh", doubles, True, True, "cuda"),
            ("tanh", doubles, True, True, "xpu"),
            ("trunc", doubles, True, True, "cpu"),
            ("trunc", doubles, True, True, "cuda"),
            ("trunc", doubles, True, True, "xpu"),
        ]

        for (
            fn,
            inputs,
            has_input_output_mem_overlap_check,
            has_internal_mem_overlap_check,
            dev,
        ) in unary_mem_overlap_cases:
            if dev != device:
                continue
            out_fn = getattr(torch, fn)
            in_fn = getattr(torch.Tensor, fn + "_")

            self.unary_check_input_output_mem_overlap(
                inputs,
                sz,
                out_fn,
                expected_failure=not has_input_output_mem_overlap_check,
            )

            self.check_internal_mem_overlap(
                in_fn,
                1,
                dtype,
                dev,
                expected_failure=not has_internal_mem_overlap_check,
            )

    TestUnaryUfuncs.test_unary_out_op_mem_overlap = _test_unary_out_op_mem_overlap

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypesIfXPU(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def _test_i0_range1(self, device, dtype):
        # This tests the domain for i0 for which float16 does not overflow
        # The domain is (-13.25, 13.25)
        self._i0_range_helper(13.25, device, dtype)

    TestUnaryUfuncs._test_i0_range1 = _test_i0_range1

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypesIfXPU(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def _test_i0_range2(self, device, dtype):
        # This tests the domain for i0 for which float32 and bfloat16 does not overflow
        # The domain is (-88.5, 88.5)
        self._i0_range_helper(88.5, device, dtype)

    TestUnaryUfuncs._test_i0_range2 = _test_i0_range2

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypesIfXPU(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def _test_i0_special(self, device, dtype):
        t = torch.tensor([], device=device, dtype=dtype)
        self._i0_helper(t)

        t = torch.tensor([inf, -inf, nan], device=device, dtype=dtype)
        self.assertTrue(torch.i0(t).isnan().all())

    TestUnaryUfuncs._test_i0_special = _test_i0_special

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypesIfXPU(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def _test_special_i0_i1_vs_scipy(self, device, dtype):
        def check_equal(t, torch_fn, scipy_fn):
            # Test by comparing to scipy
            actual = torch_fn(t)
            if dtype is torch.bfloat16:
                t = t.to(torch.float32)
            expected = scipy_fn(t.cpu().numpy())

            # Casting down for dtype float16 is required since scipy upcasts to float32
            if dtype is torch.bfloat16 or dtype is torch.float16:
                expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

        t = torch.tensor([], device=device, dtype=dtype)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        range = (-1e7, 1e7)
        if dtype == torch.half:
            range = (-65000, 65000)

        t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        # NaN, inf, -inf are tested in reference_numerics tests.
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

    TestUnaryUfuncs._test_special_i0_i1_vs_scipy = _test_special_i0_i1_vs_scipy

    @dtypes(torch.float, torch.double)
    def _test_abs_zero(self, device, dtype):
        # Both abs(0.0) and abs(-0.0) should result in 0.0
        abs_zeros = torch.tensor([0.0, -0.0], device=device, dtype=dtype).abs().tolist()
        for num in abs_zeros:
            self.assertGreater(math.copysign(1.0, num), 0.0)

    TestUnaryUfuncs._test_abs_zero = _test_abs_zero

    @dtypes(torch.bool, torch.int8)
    def _test_narrow_dtypes(self, device, dtype):
        x_int = torch.randint(2, (8 * 1024,), device=device, dtype=torch.int)
        x = x_int.to(dtype)
        # check normal conversion
        self.assertEqual(x_int, x.int())
        x.fill_(0)
        self.assertEqual(x.sum(), 0)
        # test unaligned tensor with non-round number of elements
        x[1:4000].fill_(1)
        self.assertEqual(x.sum(), 3999)

    TestUnaryUfuncs.test_narrow_dtypes = _test_narrow_dtypes

    @dtypes(torch.int8)
    @largeTensorTest("8GB")
    def _test_nonzero_large(self, device, dtype):
        indices = (
            torch.tensor((0, 2, 3, 4, 6, 100, 103, 2**30, 2**31 - 3, 2**31 - 2)),
            torch.tensor((0, 1, 1, 1, 0, 1, 0, 1, 0, 0)),
        )

        x = torch.zeros(2**31 - 1, 2, device=device, dtype=dtype)
        x[indices[0], indices[1]] = 1
        y = torch.nonzero(x, as_tuple=True)
        self.assertEqual(y, indices)
        x = x.view(-1).fill_(0)
        indices = indices[0] * 2
        x[indices] = 1
        y = torch.nonzero(x)
        self.assertEqual(y.view(-1), indices)

    TestUnaryUfuncs.test_nonzero_large = _test_nonzero_large

    def _test_nonzero_static_large(self, device):
        # large enough to have multiple iters per SM even on H100
        # with 132 sms
        size_inp = 1024 * 16 * 132 + 1024 * 16
        x = torch.zeros(size_inp, device=device)
        # unique indices
        indices = torch.randperm(size_inp, device=device)[: size_inp // 2]
        sorted, _ = torch.sort(indices)
        x[sorted] = 1
        res = torch.nonzero_static(x, size=size_inp // 2).view(-1)
        self.assertEqual(res, sorted)
        # no oob writes
        out = torch.full((size_inp,), 10, device=device, dtype=torch.int64)
        res = torch.nonzero_static(x, size=size_inp // 4, out=out[: size_inp // 2])
        self.assertEqual(out[: size_inp // 4], sorted[: size_inp // 4])
        self.assertEqual(
            out[size_inp // 4 :],
            torch.tensor(10, device=device_type).expand_as(out[size_inp // 4 :]),
        )
        # correct fill for 2d
        x = x.view(2, size_inp // 2)
        ref = x.nonzero()
        res = x.nonzero_static(size=size_inp // 2 + 2)
        self.assertEqual(res.shape, [size_inp // 2 + 2, 2])
        self.assertEqual(ref, res[: size_inp // 2])
        self.assertEqual(
            res[size_inp // 2 :],
            torch.tensor(-1, device=device_type).expand_as(res[size_inp // 2 :]),
        )

    TestUnaryUfuncs.test_nonzero_static_large = _test_nonzero_static_large

    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @dtypesIfCUDA(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    @dtypesIfXPU(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def _test_exp(self, device, dtype):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            a = (
                torch.tensor(v, dtype=dtype, device=device)
                * torch.arange(18, device=device)
                / 3
                * math.pi
            )
            a = a.to(dtype)
            # bfloat16 overflows
            if dtype == torch.bfloat16:
                return
            self.compare_with_numpy(torch.exp, np.exp, a)

            if dtype.is_complex:
                inf_real_zero_imag_in = torch.tensor(
                    complex(float("inf"), 0), device=device, dtype=dtype
                )
                inf_real_zero_imag_out = torch.exp(inf_real_zero_imag_in).item()
                self.assertTrue(math.isinf(inf_real_zero_imag_out.real))
                if self.device_type == "cpu":
                    pass
                    # These are commented out because it cannot be consistently reproduced.
                    # This is incorrect. It should be zero. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # self.assertNotEqual(inf_real_zero_imag_out.imag, 0)
                    # This is incorrect. They should equal. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)
                else:
                    self.assertEqual(inf_real_zero_imag_out.imag, 0, atol=0, rtol=0)
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)

                zero_real_inf_imag_in = torch.tensor(
                    complex(0, float("inf")), device=device, dtype=dtype
                )
                zero_real_inf_imag_out = torch.exp(zero_real_inf_imag_in).item()
                self.assertTrue(math.isnan(zero_real_inf_imag_out.real))
                self.assertTrue(math.isnan(zero_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, zero_real_inf_imag_in)

                inf_real_imag_in = torch.tensor(
                    complex(float("inf"), float("inf")), device=device, dtype=dtype
                )
                inf_real_imag_out = torch.exp(inf_real_imag_in).item()
                if self.device_type == "cpu":
                    pass
                    # This is incorrect. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)

                inf_real_nan_imag_in = torch.tensor(
                    complex(float("inf"), float("nan")), device=device, dtype=dtype
                )
                inf_real_nan_imag_out = torch.exp(inf_real_nan_imag_in).item()
                if self.device_type == "cpu":
                    pass
                    # This is incorrect. It should be inf. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_nan_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_nan_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)

                nan_real_inf_imag_in = torch.tensor(
                    complex(float("nan"), float("inf")), device=device, dtype=dtype
                )
                nan_real_inf_imag_out = torch.exp(nan_real_inf_imag_in).item()
                self.assertTrue(math.isnan(nan_real_inf_imag_out.real))
                self.assertTrue(math.isnan(nan_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, nan_real_inf_imag_in)

    TestUnaryUfuncs.test_exp = _test_exp

instantiate_device_type_tests(
    TestUnaryUfuncs, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
