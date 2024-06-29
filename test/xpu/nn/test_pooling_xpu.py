# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests

def max_pool2d_indices(self, device):
    def helper(n, c, h, w, ks):
        if n is None:
            x = torch.randn(
                c, h, w, device="xpu", dtype=torch.float, requires_grad=True
            )
        else:
            x = torch.randn(
                n, c, h, w, device="xpu", dtype=torch.float, requires_grad=True
            )

        ref_x = x.detach().clone().cpu().requires_grad_()

        pool = torch.nn.MaxPool2d(kernel_size=ks, return_indices=True)

        y, idx = pool(x)
        ref_y, ref_idx = pool(ref_x)

        y.sum().backward()
        ref_y.sum().backward()

        self.assertEqual(y, ref_y)
        self.assertEqual(
            idx, ref_idx
        )  # assertEqual implicitly compares shape for tensors
        self.assertEqual(x.grad, ref_x.grad)

    helper(2, 8, 4, 4, ks=2)
    helper(None, 3, 50, 50, ks=5)

def max_pool2d(self, device):
    def helper(n, c, h, w, ks):
        x = torch.randn(
            n, c, h, w, device="xpu", dtype=torch.float, requires_grad=True
        )
        ref_x = x.detach().clone().cpu().requires_grad_()

        pool = torch.nn.MaxPool2d(kernel_size=ks)

        y = pool(x)
        ref_y = pool(ref_x)

        y.sum().backward()
        ref_y.sum().backward()

        self.assertEqual(y, ref_y)
        self.assertEqual(x.grad, ref_x.grad)

    helper(2, 8, 4, 4, ks=2)
    helper(1, 100000, 32, 32, ks=4)
    helper(1, 100000, 1, 4, ks=(1, 4))  # test for max_pool1d

@dtypes(torch.half, torch.float, torch.double)
def max_pool2d_nhwc(self, device, dtype):
    def helper(n, c, h, w, kernel_size, stride=None):
        if stride is None:
            stride = kernel_size
        input = torch.randn(n, c, h, w, dtype=dtype, device=device)
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randn(
            n,
            c,
            (h - kernel_size) // stride + 1,
            (w - kernel_size) // stride + 1,
            dtype=dtype,
            device=device,
        )
        pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
            device
        )

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
            device
        )

        out, ind = pool(input)
        out.backward(grad)
        ref_out, ref_ind = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_ind.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(ind, ref_ind)
        self.assertEqual(input.grad, ref_input.grad)

    helper(4, 8, 8, 8, 7)
    helper(200, 512, 28, 28, 2)
    helper(4, 8, 7, 7, 3, stride=1)
    helper(10, 512, 31, 31, 3, stride=2)
    helper(1, 129, 8, 8, 3, stride=2)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_pooling import TestPoolingNNDeviceType, TestPoolingNN

TestPoolingNNDeviceType.test_max_pool2d_indices = max_pool2d_indices
TestPoolingNNDeviceType.test_max_pool2d = max_pool2d
TestPoolingNNDeviceType.test_max_pool2d_nhwc = max_pool2d_nhwc


instantiate_device_type_tests(TestPoolingNNDeviceType, globals(), only_for="xpu", allow_xpu=True)
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
