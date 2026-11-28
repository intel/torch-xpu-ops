# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import random

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)

_FOREACH_REDUCE_SIZES = (1, 64, 3072, 1_000_000, 5, 17, 2048, 100, 4097)


class ForeachTest:
    def __init__(self, func):
        self.func = func

    def __call__(
        self, input1, input2, device, is_inplace=False, scalar=None, non_blocking=None
    ):
        input1_for_func = []
        input2_for_func = []
        for i in input1:
            input1_for_func.append(i.clone().to(device))
        for i in input2:
            input2_for_func.append(i.clone().to(device))
        if is_inplace:
            if scalar is not None:
                self.func(input1_for_func, input2_for_func, alpha=scalar)
            elif non_blocking is not None:
                self.func(input1_for_func, input2_for_func, non_blocking=non_blocking)
            else:
                self.func(input1_for_func, input2_for_func)
            return input1_for_func
        else:
            if scalar is not None:
                return self.func(input1_for_func, input2_for_func, alpha=scalar)
            elif non_blocking is not None:
                return self.func(
                    input1_for_func, input2_for_func, non_blocking=non_blocking
                )
            else:
                return self.func(input1_for_func, input2_for_func)


class TestForeachListMethod(TestCase):
    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())

    def create_sample(self, dtype):
        float_types = [
            torch.float,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.cfloat,
            torch.cdouble,
        ]
        int_types = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        if dtype in float_types:
            x1 = [
                torch.randint(1, 100, [5, 8]).to(torch.float).div(100.0).to(dtype)
                for _ in range(250)
            ]
            x2 = [
                torch.randint(1, 100, [5, 8]).to(torch.float).div(100.0).to(dtype)
                for _ in range(250)
            ]
            scalar = random.uniform(0, 1)
        elif dtype in int_types:
            x1 = [torch.randint(1, 100, [5, 8]).to(dtype) for _ in range(250)]
            x2 = [torch.randint(1, 100, [5, 8]).to(dtype) for _ in range(250)]
            scalar = torch.randint(1, 10, [1]).to(dtype).item()
        else:
            AssertionError(
                False, "TestForeachListMethod::create_sample unsupported dtype"
            )
        return x1, x2, scalar

    def test_foreach_add(self, dtype=torch.float):
        claimed_dtypes = get_all_dtypes()
        for dtype_ in claimed_dtypes:
            x1, x2, scalar = self.create_sample(dtype_)

            test = ForeachTest(torch._foreach_add)
            cpu = test(x1, x2, "cpu", is_inplace=False, scalar=scalar)
            xpu = test(x1, x2, "xpu", is_inplace=False, scalar=scalar)
            self.result_compare(cpu, xpu)

            test_ = ForeachTest(torch._foreach_add_)
            cpu_inplace = test_(x1, x2, "cpu", is_inplace=True, scalar=scalar)
            xpu_inplace = test_(x1, x2, "xpu", is_inplace=True, scalar=scalar)
            self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_copy_barebones_unsigned(self):
        # The fused XPU copy kernel doesn't instantiate uint16/uint32/uint64,
        # so these copies must take the slow path instead of failing to dispatch.
        test = ForeachTest(torch._foreach_copy_)
        for dtype in (torch.uint16, torch.uint32, torch.uint64):
            for dst_dtype, src_dtype in (
                (dtype, dtype),
                (dtype, torch.int64),
                (torch.float32, dtype),
            ):
                src = [torch.arange(n).to(src_dtype) for n in (3, 5)]
                dst = [torch.zeros(n, dtype=dst_dtype) for n in (3, 5)]
                cpu = test(dst, src, "cpu", is_inplace=True)
                xpu = test(dst, src, "xpu", is_inplace=True)
                self.result_compare(cpu, xpu)

    @parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    @parametrize("ord", (1.0, 2.0, float("inf")))
    def test_foreach_norm_subgroup_width(self, dtype, ord):
        # intel/torch-xpu-ops#6035: the stage-1 reduction is templated on a
        # compile-time sub-group width, so the launched kernel has to run at
        # that width. On devices where IGC already selects the device maximum
        # (BMG, PVC) this passes with and without the fix; it fails on devices
        # such as Arc A770 where IGC selects a narrower width than the launch.
        torch.manual_seed(6035)
        sizes = _FOREACH_REDUCE_SIZES
        if dtype != torch.float32 and ord == 1.0:
            # The fp16/bf16 output overflows for a 1M-element L1 sum.
            sizes = tuple(n for n in sizes if n != 1_000_000)
        cpu_inputs = [torch.randn(n, dtype=dtype) for n in sizes]
        actual = torch._foreach_norm([x.xpu() for x in cpu_inputs], ord)
        atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
        for out, cpu_input in zip(actual, cpu_inputs):
            self.assertEqual(
                out.cpu().double(),
                cpu_input.double().norm(ord),
                atol=atol,
                rtol=rtol,
            )

    @parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    @parametrize("ord", (1.0, 2.0))
    def test_foreach_powsum_subgroup_width(self, dtype, ord):
        # _foreach_powsum shares the stage-1 reduction with _foreach_norm.
        # It returns the input dtype, so a 1M-element sum of squares overflows
        # half and bfloat16; keep that case for float32 only.
        sizes = _FOREACH_REDUCE_SIZES
        if dtype != torch.float32:
            sizes = tuple(n for n in sizes if n != 1_000_000)
        torch.manual_seed(6035)
        cpu_inputs = [torch.randn(n, dtype=dtype) for n in sizes]
        actual = torch._foreach_powsum([x.xpu() for x in cpu_inputs], ord)
        atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
        for out, cpu_input in zip(actual, cpu_inputs):
            self.assertEqual(
                out.cpu().double(),
                cpu_input.double().abs().pow(ord).sum(),
                atol=atol,
                rtol=rtol,
            )

    @parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    def test_foreach_max_subgroup_width(self, dtype):
        # The unique maximum sits in the last element of a 4096-element input,
        # so only the tail of the single work group handling it can observe it;
        # the second tensor keeps its maximum at the front. A sub-group width
        # mismatch drops the sub-groups past the template SIMD from the
        # reduction and returns a value below the real maximum.
        torch.manual_seed(6035)
        cpu_inputs = []
        for max_index in (4095, 0):
            cpu_input = torch.zeros(4096, dtype=dtype)
            cpu_input[max_index] = 100.0
            cpu_inputs.append(cpu_input)
        actual = torch._foreach_max([x.xpu() for x in cpu_inputs])
        for out, cpu_input in zip(actual, cpu_inputs):
            self.assertEqual(out.cpu().double(), cpu_input.max().double())

    def test_clip_grad_norm_foreach_subgroup_width(self):
        # clip_grad_norm_ uses _foreach_norm by default; this is the
        # user-visible symptom reported in intel/torch-xpu-ops#6035.
        torch.manual_seed(6035)
        cpu_grads = [torch.randn(n) for n in _FOREACH_REDUCE_SIZES]
        params = [
            torch.nn.Parameter(torch.zeros(n, device="xpu"))
            for n in _FOREACH_REDUCE_SIZES
        ]
        for param, cpu_grad in zip(params, cpu_grads):
            param.grad = cpu_grad.xpu()
        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        expected = torch.sqrt(
            sum(cpu_grad.double().pow(2).sum() for cpu_grad in cpu_grads)
        )
        self.assertEqual(total_norm.cpu().double(), expected, atol=1e-5, rtol=1e-5)


instantiate_parametrized_tests(TestForeachListMethod)
