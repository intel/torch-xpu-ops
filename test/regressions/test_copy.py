# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import float8_types_and
from torch.testing._internal.common_utils import run_tests, TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleCopy(TestCase):
    @dtypes(*float8_types_and(torch.float8_e8m0fnu, torch.float32))
    def test_copy_and_clone(self, dtype):
        a_cpu = torch.randn(16, 64, 28, 28).to(dtype)
        b_cpu = torch.randn(16, 64, 28, 28).to(dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        # naive
        b_cpu.copy_(a_cpu)
        b_xpu.copy_(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
        # clone + permutation
        b_cpu = a_cpu.clone(memory_format=torch.channels_last)
        b_xpu = a_xpu.clone(memory_format=torch.channels_last)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    def test_copy_and_clone_float4(self):
        # Float4_e2m1fn_x2 copy is not implemented by CPU
        a_cpu = torch.randn(16, 64, 28, 28).to(torch.uint8)
        b_cpu = torch.randn(16, 64, 28, 28).to(torch.uint8)
        a_xpu = a_cpu.to(xpu_device).view(torch.float4_e2m1fn_x2)
        b_xpu = b_cpu.to(xpu_device).view(torch.float4_e2m1fn_x2)

        b_cpu.copy_(a_cpu)
        b_xpu.copy_(a_xpu)
        # Float4_e2m1fn_x2 compare is not implemented CPU
        self.assertEqual(b_cpu, b_xpu.view(torch.uint8).to(cpu_device))

        b_cpu = a_cpu.clone(memory_format=torch.channels_last)
        b_xpu = a_xpu.clone(memory_format=torch.channels_last)
        self.assertEqual(b_cpu, b_xpu.view(torch.uint8).to(cpu_device))


instantiate_device_type_tests(TestSimpleCopy, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
