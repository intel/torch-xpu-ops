import torch
from torch.testing._internal.common_utils import TestCase
import torchvision

class TestTorchVisionMethod(TestCase):
    def test_roi_align(self):
        atol = 1e-1
        rtol = 5e-5
        a_ref = torch.zeros([4, 256, 296, 304]).requires_grad_(True)
        b_ref = torch.zeros([2292, 5]).requires_grad_(True)

        a_xpu = torch.zeros([4, 256, 296, 304], device=torch.device("xpu")).requires_grad_(True)
        b_xpu = torch.zeros([2292, 5], device=torch.device("xpu")).requires_grad_(True)

        ref = torch.ops.torchvision.roi_align(a_ref, b_ref, 0.25, 7, 7, 2, False)
        res = torch.ops.torchvision.roi_align(a_xpu, b_xpu, 0.25, 7, 7, 2, False)
        ref.sum().backward()
        res.sum().backward()
        self.assertEqual(ref, res.cpu())
        self.assertEqual(a_ref.grad, a_xpu.grad.cpu(), rtol=rtol, atol=atol)
