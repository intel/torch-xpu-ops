# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for: torch.native_batch_norm in eval mode returns populated
save_mean/save_invstd on XPU (https://github.com/intel/torch-xpu-ops/issues/3349).

In eval mode (training=False) the CPU backend returns size-0 tensors for
save_mean and save_invstd.  The XPU backend must match this contract.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBatchNormEvalSaveShapes(TestCase):
    def _run(self, dtype):
        torch.manual_seed(0)
        C = 3
        x = torch.randn(2, C, 4, 4, dtype=dtype)
        weight = torch.randn(C, dtype=dtype)
        bias = torch.randn(C, dtype=dtype)
        running_mean = torch.randn(C)
        running_var = torch.abs(torch.randn(C))  # must be non-negative

        cpu_out = torch.native_batch_norm(
            x,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            training=False,
            momentum=0.1,
            eps=1e-5,
        )

        xpu_out = torch.native_batch_norm(
            x.xpu(),
            weight.xpu(),
            bias.xpu(),
            running_mean.clone().xpu(),
            running_var.clone().xpu(),
            training=False,
            momentum=0.1,
            eps=1e-5,
        )

        # save_mean and save_invstd (outputs 1 and 2) must be empty in eval mode
        for i in (1, 2):
            self.assertEqual(
                cpu_out[i].shape,
                xpu_out[i].cpu().shape,
                msg=f"Output {i} shape mismatch: CPU {cpu_out[i].shape} vs XPU {xpu_out[i].shape}",
            )
            self.assertEqual(
                xpu_out[i].numel(),
                0,
                msg=f"Output {i} should be empty in eval mode, got shape {xpu_out[i].shape}",
            )

        # The main output (index 0) must be numerically correct
        atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        self.assertEqual(
            cpu_out[0],
            xpu_out[0].cpu().to(dtype),
            atol=atol,
            rtol=rtol,
        )

    def test_eval_save_shapes_float32(self):
        self._run(torch.float32)

    def test_eval_save_shapes_float16(self):
        self._run(torch.float16)

    def test_eval_save_shapes_bfloat16(self):
        self._run(torch.bfloat16)


if __name__ == "__main__":
    run_tests()
