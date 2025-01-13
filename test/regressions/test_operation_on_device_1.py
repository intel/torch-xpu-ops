# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestOperationOnDevice1(TestCase):
    def test_sum_on_device1(self, dtype=torch.float):
        if torch.xpu.device_count() >= 2:
            a = torch.randn(2, 3, device=torch.device("xpu:1"))
            torch.xpu.set_device(1)
            res = a.sum()
            ref = a.cpu().sum()
            self.assertEqual(ref, res)
