# Owner(s): ["module: intel"]
import random

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import TestCase


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
