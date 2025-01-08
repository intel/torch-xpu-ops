# Owner(s): ["module: intel"]
import random

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import TestCase


class ForeachTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, scalar, device, is_inplace=False, reverse=False):
        input_for_func = []
        for i in input:
            input_for_func.append(i.clone().to(device))
        if reverse:
            x = input_for_func
            input_for_func = scalar
            scalar = x
        if is_inplace:
            self.func(input_for_func, scalar)
            return input_for_func
        else:
            return self.func(input_for_func, scalar)


class TestForeachScalarMethod(TestCase):
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
                torch.randint(1, 100, [5, 8]).to(torch.float).div(1000.0).to(dtype)
                for _ in range(250)
            ]
            scalar = random.uniform(-5, 5)
        elif dtype in int_types:
            x1 = [torch.randint(1, 100, [5, 8]).to(dtype) for _ in range(250)]
            scalar = torch.randint(1, 10, [1]).to(dtype).item()
        else:
            AssertionError(
                False, "TestForeachScalarMethod::create_sample unsupported dtype"
            )
        return x1, scalar

    def test_foreach_add(self, dtype=torch.float):
        claimed_dtypes = get_all_dtypes()
        for dtype_ in claimed_dtypes:
            x1, scalar = self.create_sample(dtype_)

            test = ForeachTest(torch._foreach_add)
            cpu = test(x1, scalar, "cpu")
            xpu = test(x1, scalar, "xpu")
            self.result_compare(cpu, xpu)

            test_ = ForeachTest(torch._foreach_add_)
            cpu_inplace = test_(x1, scalar, "cpu", is_inplace=True)
            xpu_inplace = test_(x1, scalar, "xpu", is_inplace=True)
            self.result_compare(cpu_inplace, xpu_inplace)
