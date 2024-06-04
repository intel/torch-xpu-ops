import torch
import random
from torch.testing._internal.common_utils import TestCase


class TestNNMethod(TestCase):
    def test_sort_large_slice(self, device=torch.device('xpu')):
        x = torch.randn(4, 1024000, device=device)
        res1val, res1ind = torch.sort(x, stable=True)
        torch.xpu.synchronize()
        # assertIsOrdered is too slow, so just compare to cpu
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())
        res1val, res1ind = torch.sort(x, descending=True, stable=True)
        torch.xpu.synchronize()
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), descending=True, stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())
