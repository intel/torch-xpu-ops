import torch
from torch.testing._internal.common_utils import TestCase

import os

class TestOperationOnDevice1(TestCase):
    def test_sum_on_device1(self, dtype=torch.float):
        mask = os.environ['ZE_AFFINITY_MASK']
        os.environ['ZE_AFFINITY_MASK'] = ''
        if torch.xpu.device_count() >= 2:
            a_xpu0 = torch.randn(2, 3, device=torch.device('xpu:0'))
            a_xpu1 = torch.randn(2, 3, device=torch.device('xpu:1'))
            a_res0 = a_xpu0.sum()
            a_res1 = a_xpu1.sum()
            self.assertEqual(a_xpu0.cpu(), a_xpu1.cpu())
        os.environ['ZE_AFFINITY_MASK'] = mask
        print(os.environ['ZE_AFFINITY_MASK'])
