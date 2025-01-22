# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleEmptyDeterministic(TestCase):
    def test_empty_float(self, dtype=torch.float):
        torch.use_deterministic_algorithms(True)
        a_cpu = torch.empty(64, 64, dtype=dtype, device=torch.device("cpu"))
        a_xpu = torch.empty(64, 64, dtype=dtype, device=torch.device("xpu"))
        self.assertEqual(a_cpu, a_xpu.to(cpu_device))
        torch.use_deterministic_algorithms(False)

    def test_empty_int64(self, dtype=torch.int64):
        torch.use_deterministic_algorithms(True)
        a_cpu = torch.empty(64, 64, dtype=dtype, device=torch.device("cpu"))
        a_xpu = torch.empty(64, 64, dtype=dtype, device=torch.device("xpu"))
        self.assertEqual(a_cpu, a_xpu.to(cpu_device))
        torch.use_deterministic_algorithms(False)

    def test_empty_strided_float(self, dtype=torch.float):
        torch.use_deterministic_algorithms(True)
        a_cpu = torch.empty_strided(
            (2, 3), (1, 2), dtype=dtype, device=torch.device("cpu")
        )
        a_xpu = torch.empty_strided(
            (2, 3), (1, 2), dtype=dtype, device=torch.device("xpu")
        )
        self.assertEqual(a_cpu, a_xpu.to(cpu_device))
        torch.use_deterministic_algorithms(False)

    def test_empty_strided_int64(self, dtype=torch.int64):
        torch.use_deterministic_algorithms(True)
        a_cpu = torch.empty_strided(
            (2, 3), (1, 2), dtype=dtype, device=torch.device("cpu")
        )
        a_xpu = torch.empty_strided(
            (2, 3), (1, 2), dtype=dtype, device=torch.device("xpu")
        )
        self.assertEqual(a_cpu, a_xpu.to(cpu_device))
        torch.use_deterministic_algorithms(False)
