# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for torch.native_batch_norm returning populated save_mean /
save_invstd in eval mode on XPU.

In eval mode (training=False), save_mean and save_invstd are not used for the
backward pass and should be empty tensors (size 0), matching CPU/CUDA behavior.
XPU previously incorrectly returned tensors of shape [C] for these outputs.

Issue: https://github.com/intel/torch-xpu-ops/issues/3349
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

xpu_device = torch.device("xpu")


class TestNativeBatchNormEvalSaveTensors(TestCase):
    def _run_eval_mode(self, input_shape, dtype=torch.float32):
        C = input_shape[1]
        torch.manual_seed(0)
        x = torch.randn(*input_shape)
        weight = torch.randn(C)
        bias = torch.randn(C)
        running_mean = torch.randn(C)
        running_var = torch.abs(torch.randn(C)) + 0.1  # keep positive

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
            x.to(dtype).to(xpu_device),
            weight.to(dtype).to(xpu_device),
            bias.to(dtype).to(xpu_device),
            running_mean.clone().to(xpu_device),
            running_var.clone().to(xpu_device),
            training=False,
            momentum=0.1,
            eps=1e-5,
        )

        # save_mean and save_invstd must be empty (size 0) in eval mode
        self.assertEqual(
            cpu_out[1].shape,
            xpu_out[1].cpu().shape,
            msg=f"save_mean shape mismatch: CPU {cpu_out[1].shape} vs XPU {xpu_out[1].shape}",
        )
        self.assertEqual(
            cpu_out[2].shape,
            xpu_out[2].cpu().shape,
            msg=f"save_invstd shape mismatch: CPU {cpu_out[2].shape} vs XPU {xpu_out[2].shape}",
        )
        self.assertEqual(
            xpu_out[1].numel(), 0, msg="save_mean should be empty in eval mode"
        )
        self.assertEqual(
            xpu_out[2].numel(), 0, msg="save_invstd should be empty in eval mode"
        )

        # Output tensor values must match CPU reference
        atol, rtol = (
            (1e-2, 1e-2) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-4)
        )
        self.assertEqual(
            cpu_out[0].float(), xpu_out[0].cpu().float(), atol=atol, rtol=rtol
        )

    def test_eval_mode_2d_float32(self):
        self._run_eval_mode((2, 3, 4, 4), torch.float32)

    def test_eval_mode_2d_float16(self):
        self._run_eval_mode((2, 3, 4, 4), torch.float16)

    def test_eval_mode_2d_bfloat16(self):
        self._run_eval_mode((2, 3, 4, 4), torch.bfloat16)

    def test_eval_mode_1d(self):
        self._run_eval_mode((4, 8, 16), torch.float32)

    def test_eval_mode_3d(self):
        self._run_eval_mode((2, 4, 3, 5, 5), torch.float32)

    def test_training_mode_save_tensors_unchanged(self):
        """Training mode must still return size-[C] save_mean / save_invstd."""
        N, C, H, W = 2, 3, 4, 4
        torch.manual_seed(0)
        x = torch.randn(N, C, H, W).to(xpu_device)
        weight = torch.randn(C).to(xpu_device)
        bias = torch.randn(C).to(xpu_device)
        running_mean = torch.zeros(C).to(xpu_device)
        running_var = torch.ones(C).to(xpu_device)

        _, save_mean, save_invstd = torch.native_batch_norm(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            training=True,
            momentum=0.1,
            eps=1e-5,
        )
        self.assertEqual(save_mean.shape, torch.Size([C]))
        self.assertEqual(save_invstd.shape, torch.Size([C]))


if __name__ == "__main__":
    run_tests()
