# Owner(s): ["module: intel"]
import itertools
import torch
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import sys
    import os
    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_quantized_op import TestQuantizedOps 

def _test_max_pool2d_pt2e(self):
    kernel_list = [2, 3]
    stride_list = [1, 2]
    padding_list = [0, 2]
    dilation_list = [1, 2]
    ceil_mode_list = [False, True]
    channels_last_input = [False, True]
    options = itertools.product(kernel_list, stride_list, padding_list, dilation_list, ceil_mode_list, channels_last_input)
    for kernel, stride, padding, dilation, ceil_mode, channels_last in options:
        if padding >= (kernel // 2):
            # Continue with invalid input
            continue
        device = torch.device('xpu:0')
        input = torch.randint(0, 8, (1, 3, 8, 8), dtype=torch.uint8, device=device)
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        a_pool = torch.nn.functional.max_pool2d(input.to(torch.float32), kernel_size=kernel,
                                                stride=stride, padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode).to(torch.uint8)
        a_hat = torch.ops.quantized.max_pool2d(input, kernel_size=_pair(kernel),
                                               stride=_pair(stride), padding=_pair(padding),
                                               dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(input.is_contiguous(), a_hat.is_contiguous(),
                         msg="ops.quantized.max_pool2d input output diff memory format")
        self.assertEqual(a_pool, a_hat,
                         msg="ops.quantized.max_pool2d results are off")

TestQuantizedOps.test_max_pool2d_pt2e = _test_max_pool2d_pt2e

instantiate_device_type_tests(TestQuantizedOps, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
