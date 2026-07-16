# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]
# ruff: noqa: F401

import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import inf, nan
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyOn,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    subtest,
)

try:
    from .xpu_test_utils import XPUImportCtx
except Exception as e:
    from ..xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
    from test_pooling import (
        TestAvgPool,
        TestAvgPoolDeviceType,
        TestPoolingNN,
        TestPoolingNNDeviceType,
    )


# ======================================================================
# 1. dtypesIfXPU decorator additions
# ======================================================================

TestPoolingNNDeviceType.test_max_pool_nan_inf = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_max_pool_nan_inf)

TestPoolingNNDeviceType.test_fractional_max_pool_nan_inf = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_fractional_max_pool_nan_inf)

TestPoolingNNDeviceType.test_AdaptiveMaxPool1d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_AdaptiveMaxPool1d_indices)

TestPoolingNNDeviceType.test_AdaptiveMaxPool2d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_AdaptiveMaxPool2d_indices)

TestPoolingNNDeviceType.test_AdaptiveMaxPool3d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_AdaptiveMaxPool3d_indices)

TestPoolingNNDeviceType.test_MaxPool1d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_MaxPool1d_indices)

TestPoolingNNDeviceType.test_MaxPool2d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_MaxPool2d_indices)

TestPoolingNNDeviceType.test_MaxPool3d_indices = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_MaxPool3d_indices)

TestPoolingNNDeviceType.test_adaptive_pooling_empty_output_size = dtypesIfXPU(
    torch.float32, torch.float64, torch.bfloat16, torch.float16
)(TestPoolingNNDeviceType.test_adaptive_pooling_empty_output_size)

TestPoolingNNDeviceType.test_avg_pool2d_nhwc = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_avg_pool2d_nhwc)

TestPoolingNNDeviceType.test_avg_pool3d_nhwc = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_avg_pool3d_nhwc)

TestPoolingNNDeviceType.test_max_pool2d_nhwc = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_max_pool2d_nhwc)

TestPoolingNNDeviceType.test_max_pool3d_ndhwc = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestPoolingNNDeviceType.test_max_pool3d_ndhwc)

TestPoolingNNDeviceType.test_maxpool_indices_no_batch_dim = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_maxpool_indices_no_batch_dim)

TestPoolingNNDeviceType.test_pool_invalid_size = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_pool_invalid_size)

TestPoolingNNDeviceType.test_pool_large_size = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestPoolingNNDeviceType.test_pool_large_size)


# ======================================================================
# 2. Method overrides (replace hardcoded "cuda" with device)
# ======================================================================


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


@onlyOn(["cuda", "xpu"])
@largeTensorTest("10GB", device="cuda")
@largeTensorTest("10GB", device="xpu")
def _test_adaptive_avg_pool2d_backward_large_index_offsets(self, device):
    height = 32769
    width = 65536
    channels = 2
    output_width = 1024
    input = torch.as_strided(
        torch.empty((1,), dtype=torch.half, device=device),
        (1, channels, height, width),
        (0, 0, 0, 0),
    )
    self.assertGreater(input.numel(), torch.iinfo(torch.int32).max)
    grad_output = torch.ones(
        (1, channels, height, output_width), dtype=torch.half, device=device
    )

    grad_input = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, input)
    sample = grad_input[
        0,
        [0, 0, 1],
        [0, height - 1, height - 1],
        [0, 0, width - 1],
    ]
    expected = torch.full_like(sample, 1 / (width // output_width))
    self.assertEqual(sample, expected)


TestPoolingNNDeviceType.test_adaptive_avg_pool2d_backward_large_index_offsets = (
    _test_adaptive_avg_pool2d_backward_large_index_offsets
)

# ======================================================================
# 3. Instantiate tests
# ======================================================================

instantiate_device_type_tests(
    TestAvgPoolDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestPoolingNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
