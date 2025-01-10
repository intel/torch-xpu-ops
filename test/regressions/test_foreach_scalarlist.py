# Owner(s): ["module: intel"]
import random

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import TestCase


class ForeachBinaryScalarListTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, scalarlist, device, is_inplace=False):
        input_tensor_for_func = []
        input_scalar_for_func = []

        for i in range(len(input)):
            input_tensor_for_func.append(input[i].clone().to(device))
            input_scalar_for_func.append(scalarlist[i])

        if is_inplace:
            self.func(input_tensor_for_func, input_scalar_for_func)
            return input_tensor_for_func
        else:
            return self.func(input_tensor_for_func, input_scalar_for_func)


class TestForeachScalarListMethod(TestCase):
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
            scalarlist = [random.uniform(0, 1) for _ in range(250)]
        elif dtype in int_types:
            x1 = [torch.randint(1, 100, [5, 8]).to(dtype) for _ in range(250)]
            scalarlist = [
                torch.randint(1, 10, [1]).to(dtype).item() for _ in range(250)
            ]
        else:
            AssertionError(
                False, "TestForeachScalarListMethod::create_sample unsupported dtype"
            )
        return x1, scalarlist

    def test_foreach_add(self, dtype=torch.float):
        claimed_dtypes = get_all_dtypes()
        for dtype_ in claimed_dtypes:
            x1, scalarlist = self.create_sample(dtype_)

            test = ForeachBinaryScalarListTest(torch._foreach_add)
            cpu = test(x1, scalarlist, "cpu")
            xpu = test(x1, scalarlist, "xpu")
            self.result_compare(cpu, xpu)

            test_ = ForeachBinaryScalarListTest(torch._foreach_add_)
            cpu_ = test_(x1, scalarlist, "cpu", True)
            xpu_ = test_(x1, scalarlist, "xpu", True)
            self.result_compare(cpu_, xpu_)
