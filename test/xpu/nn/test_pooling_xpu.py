# Owner(s): ["module: intel"]

import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import inf, nan
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    subtest,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_pooling import TestAvgPool, TestPoolingNN, TestPoolingNNDeviceType


def _test_avg_pool1d_ceil_mode(self):
    # Regression test for gh-36977
    x = 10 * torch.randn((1, 16, 4))
    y = torch.nn.functional.avg_pool1d(
        x, ceil_mode=True, count_include_pad=True, kernel_size=1, stride=2
    )
    self.assertTrue(not torch.isnan(y).any())

    y = torch.nn.functional.avg_pool1d(
        x.to("xpu"),
        ceil_mode=True,
        count_include_pad=True,
        kernel_size=1,
        stride=2,
    )
    self.assertTrue(not torch.isnan(y).any())


TestAvgPool.test_avg_pool1d_ceil_mode = _test_avg_pool1d_ceil_mode


def _test_avg_pool2d_ceil_mode(self):
    # Regression test for gh-36977
    x = 10 * torch.randn((1, 16, 4, 4))
    y = torch.nn.functional.avg_pool2d(
        x,
        ceil_mode=True,
        count_include_pad=True,
        kernel_size=(1, 2),
        padding=(0, 1),
        stride=2,
    )
    self.assertTrue(not torch.isnan(y).any())

    y = torch.nn.functional.avg_pool2d(
        x.to("xpu"),
        ceil_mode=True,
        count_include_pad=True,
        kernel_size=(1, 2),
        padding=(0, 1),
        stride=2,
    )
    self.assertTrue(not torch.isnan(y).any())


TestAvgPool.test_avg_pool2d_ceil_mode = _test_avg_pool2d_ceil_mode


def _test_avg_pool3d_ceil_mode(self):
    # Regression test for gh-36977
    x = 10 * torch.randn((1, 16, 4, 4, 4))
    y = torch.nn.functional.avg_pool3d(
        x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2, 3), stride=2
    )
    self.assertTrue(not torch.isnan(y).any())

    y = torch.nn.functional.avg_pool3d(
        x.to("xpu"),
        ceil_mode=True,
        count_include_pad=True,
        kernel_size=(1, 2, 3),
        stride=2,
    )
    self.assertTrue(not torch.isnan(y).any())


TestAvgPool.test_avg_pool3d_ceil_mode = _test_avg_pool3d_ceil_mode


def _test_adaptive_pooling_avg_nhwc(self):
    device_list = ["xpu"]

    for device in device_list:
        input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
        pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)


TestPoolingNN.test_adaptive_pooling_avg_nhwc = _test_adaptive_pooling_avg_nhwc


def _test_adaptive_pooling_avg_nhwc_non_contiguous(self):
    device_list = ["xpu"]

    for device in device_list:
        input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
        input = input.contiguous(memory_format=torch.channels_last)
        input = input[:, ::2, :, :].requires_grad_()
        grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
        grad = grad[:, ::2, :, :]
        pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)


TestPoolingNN.test_adaptive_pooling_avg_nhwc_non_contiguous = (
    _test_adaptive_pooling_avg_nhwc_non_contiguous
)


def _test_adaptive_avg_pooling_overflow(self):
    input = torch.randint(-256, 256, (20, 32, 256, 256), dtype=torch.half, device="xpu")
    avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
    out = avg_pool(input)
    self.assertFalse(torch.isinf(out).any())
    self.assertFalse(torch.isnan(out).any())


TestPoolingNN.test_adaptive_avg_pooling_overflow = _test_adaptive_avg_pooling_overflow


def _test_adaptive_avg_pooling_nhwc_overflow(self):
    input = torch.randint(-256, 256, (20, 32, 256, 256), dtype=torch.half, device="xpu")
    input = input.contiguous(memory_format=torch.channels_last)
    avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
    out = avg_pool(input)
    self.assertFalse(torch.isinf(out).any())
    self.assertFalse(torch.isnan(out).any())


TestPoolingNN.test_adaptive_avg_pooling_nhwc_overflow = (
    _test_adaptive_avg_pooling_nhwc_overflow
)


def _test_max_pool2d(self, device):
    def helper(n, c, h, w, ks):
        x = torch.randn(n, c, h, w, device="xpu", dtype=torch.float, requires_grad=True)
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


TestPoolingNNDeviceType.test_max_pool2d = _test_max_pool2d


def _test_max_pool2d_indices(self, device):
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


TestPoolingNNDeviceType.test_max_pool2d_indices = _test_max_pool2d_indices


@parametrize_test(
    "module_name,module_size,output_size,test_index,should_error",
    [
        # Some tests are failing in trunk https://github.com/pytorch/pytorch/issues/103854
        subtest(
            ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), -1, True),
            name="case1",
        ),
        subtest(
            ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), 2 * 2 * 4 * 5, True),
            name="case2",
        ),
        subtest(
            ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), (2 * 2 * 4 * 5) - 1, False),
            name="case3",
        ),
        subtest(
            ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), 2 * 3 * 4 * 2, True),
            name="case4",
        ),
        subtest(
            ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), (2 * 3 * 4 * 2) - 1, False),
            name="case5",
        ),
        subtest(
            ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), -1, True),
            name="case6",
        ),
        subtest(
            ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), 2 * 2 * 2 * 3 * 4 * 5, True),
            name="case7",
        ),
        subtest(
            (
                "MaxUnpool3d",
                (2, 2, 2),
                (1, 3, 4, 5),
                (2 * 2 * 2 * 3 * 4 * 5) - 1,
                False,
            ),
            name="case8",
        ),
        subtest(
            ("MaxUnpool3d", (2, 2, 2), (2, 3, 4, 1), 2 * 2 * 2 * 3 * 4 * 1, True),
            name="case9",
        ),
        subtest(
            (
                "MaxUnpool3d",
                (2, 2, 2),
                (2, 3, 4, 1),
                (2 * 2 * 2 * 3 * 4 * 1) - 1,
                False,
            ),
            name="case10",
        ),
    ],
)
def _test_MaxUnpool_index_errors(
    self, device, module_name, module_size, output_size, test_index, should_error
):
    # NOTE: XPU tests need to be run in a subprocess because they cause device asserts
    if torch.device(device).type == "xpu":
        error_msgs = {
            "MaxUnpool2d": r"Assertion `maxind >= 0 && maxind < outputImageSize` failed",
            "MaxUnpool3d": r"Assertion `index >= 0 && index < outputImageSize` failed",
        }
        script = f"""
import torch
unpool = torch.nn.{module_name}({module_size}).to('{device}')
output = torch.rand({output_size}, dtype=torch.float32, device='{device}')
indices = torch.zeros({output_size}, dtype=torch.int64, device='{device}')
indices.flatten()[0] = {test_index}
unpool(output, indices)
torch.xpu.synchronize()
"""
        p = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            capture_output=True,
            text=True,
        )

        output = p.stdout + "\n" + p.stderr

        error_msg = error_msgs[module_name]

        if should_error:
            self.assertIn(error_msg, output, "The expected error was not found")
        else:
            self.assertNotIn("Error", output, "Should not have produced an error")
    else:
        module_class = getattr(torch.nn, module_name)
        unpool = module_class(module_size).to(device)
        output = torch.rand(output_size, dtype=torch.float32, device=device)
        indices = torch.zeros(output_size, dtype=torch.int64, device=device)
        indices.flatten()[0] = test_index

        if should_error:
            with self.assertRaisesRegex(RuntimeError, r"Found an invalid max index:"):
                unpool(output, indices)
        else:
            unpool(output, indices)


TestPoolingNNDeviceType.test_MaxUnpool_index_errors = _test_MaxUnpool_index_errors


@dtypes(torch.half, torch.float, torch.double)
def _test_max_pool_nan_inf(self, device, dtype):
    for adaptive in ["", "adaptive_"]:
        for num_dim in [1, 2, 3]:
            fn_name = f"{adaptive}max_pool{num_dim}d"
            print("fn_name:", fn_name, flush=True)
            fn = getattr(F, fn_name)

            x = torch.full(
                [1, 1] + num_dim * [3],
                nan,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            res = fn(x, 1 if adaptive else 3)
            res.backward(torch.randn_like(res))
            self.assertTrue(math.isnan(res.item()))
            x.requires_grad_(False)
            res = fn(x, 1 if adaptive else 3)
            self.assertTrue(math.isnan(res.item()))

            x2 = torch.full(
                [1, 1] + num_dim * [3],
                -inf,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            res2 = fn(x2, 1 if adaptive else 3)
            res2.backward(torch.randn_like(res2))
            self.assertTrue(math.isinf(res2.item()))
            x2.requires_grad_(False)
            res2 = fn(x2, 1 if adaptive else 3)
            self.assertTrue(math.isinf(res2.item()))


TestPoolingNNDeviceType.test_max_pool_nan_inf = _test_max_pool_nan_inf

instantiate_device_type_tests(
    TestPoolingNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
