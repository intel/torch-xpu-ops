# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import itertools

import torch
import torch.testing._internal.hypothesis_utils as hu
from hypothesis import assume, given, strategies as st
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import os
    import sys

    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_quantized_op import pool_output_shape, TestQuantizedOps


def _test_max_pool2d_pt2e(self):
    kernel_list = [2, 3]
    stride_list = [1, 2]
    padding_list = [0, 2]
    dilation_list = [1, 2]
    ceil_mode_list = [False, True]
    channels_last_input = [False, True]
    options = itertools.product(
        kernel_list,
        stride_list,
        padding_list,
        dilation_list,
        ceil_mode_list,
        channels_last_input,
    )
    for kernel, stride, padding, dilation, ceil_mode, channels_last in options:
        if padding >= (kernel // 2):
            # Continue with invalid input
            continue
        device = torch.device("xpu:0")
        input = torch.randint(0, 8, (1, 3, 8, 8), dtype=torch.uint8, device=device)
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        a_pool = torch.nn.functional.max_pool2d(
            input.to(torch.float32),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        ).to(torch.uint8)
        a_hat = torch.ops.quantized.max_pool2d(
            input,
            kernel_size=_pair(kernel),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            ceil_mode=ceil_mode,
        )
        self.assertEqual(
            input.is_contiguous(),
            a_hat.is_contiguous(),
            msg="ops.quantized.max_pool2d input output diff memory format",
        )
        self.assertEqual(a_pool, a_hat, msg="ops.quantized.max_pool2d results are off")


@given(
    X=hu.tensor(
        shapes=hu.array_shapes(min_dims=3, max_dims=4, min_side=1, max_side=10),
        # cudnn's support for quantized pooling is limited to
        # int8 currently
        qparams=hu.qparams(dtypes=[torch.qint8]),
    ),
    kernel=st.sampled_from((3, 5, 7)),
    stride=st.sampled_from((None, 1, 2)),
    # currently there is no support for dilation for cudnn
    # pooling
    dilation=st.integers(1, 1),
    padding=st.integers(0, 2),
    ceil_mode=st.booleans(),
)
def _test_max_pool2d_cudnn(self, X, kernel, stride, dilation, padding, ceil_mode):
    X, (scale, zero_point, torch_type) = X
    assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
    iH, iW = X.shape[-2:]
    oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
    assume(oH > 0)
    oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
    assume(oW > 0)

    a = torch.from_numpy(X).to(device="xpu:0")
    a_pool = torch.nn.functional.max_pool2d(
        a,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    a_ref = torch.quantize_per_tensor(
        a_pool, scale=scale, zero_point=zero_point, dtype=torch_type
    )
    a_ref = a_ref.dequantize()
    qa = torch.quantize_per_tensor(
        a, scale=scale, zero_point=zero_point, dtype=torch_type
    )

    # Test the ops.quantized separately, because None is not treated.
    a_hat = torch.ops.quantized.max_pool2d(
        qa,
        kernel_size=_pair(kernel),
        stride=_pair(kernel if stride is None else stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        ceil_mode=ceil_mode,
    )
    self.assertEqual(
        a_ref, a_hat.dequantize(), msg="ops.quantized.max_pool2d results are off"
    )


TestQuantizedOps.test_max_pool2d_pt2e = _test_max_pool2d_pt2e
TestQuantizedOps.test_max_pool2d_cudnn = _test_max_pool2d_cudnn

instantiate_device_type_tests(
    TestQuantizedOps, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
