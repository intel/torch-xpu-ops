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
    from test_pooling import (
        TestAvgPool,
        TestAvgPoolDeviceType,
        TestPoolingNN,
        TestPoolingNNDeviceType,
    )


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
    TestAvgPoolDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestPoolingNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
