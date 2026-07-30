# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# Regression test for the XPU replication_pad2d_backward channel-mismatch bug.
#
# replication_pad2d_backward validated the spatial (width/height) dims of
# grad_output but not the channel/plane dim.  A grad_output with a mismatched
# channel count (e.g. 0 channels) passed the checks, a non-empty grad_input was
# allocated from the input shape, and the kernel read grad_output out of bounds
# -> SIGSEGV.  The fix mirrors the 3d backward channel check (and the CPU/CUDA
# fix in pytorch/pytorch#189463) into the 2d XPU backward, raising a clear
# "gradOutput channel unexpected" error instead.  See intel/torch-xpu-ops#4754.
import torch
from torch.testing._internal.common_utils import TestCase

xpu_device = torch.device("xpu")


class TestReplicationPadChannelMismatch(TestCase):
    def test_replication_pad2d_backward_channel_mismatch_xpu(self):
        # channel dim of grad_output (0) does not match input's (2).
        grad_output = torch.ones(2, 0, 6, 8, device=xpu_device)
        inp = torch.ones(2, 2, 4, 4, device=xpu_device)
        with self.assertRaisesRegex(RuntimeError, "gradOutput channel unexpected"):
            torch.ops.aten.replication_pad2d_backward(
                grad_output, inp, [2, 2, 1, 1]
            )

    def test_replication_pad2d_backward_valid_xpu(self):
        # A matching-channel grad_output must still work (no false rejection).
        inp = torch.randn(2, 3, 4, 4, device=xpu_device, requires_grad=True)
        out = torch.nn.functional.pad(inp, [2, 2, 1, 1], mode="replicate")
        out.sum().backward()
        self.assertEqual(inp.grad.shape, inp.shape)

    def test_replication_pad2d_backward_matches_cpu_xpu(self):
        inp = torch.randn(2, 3, 4, 4)
        inp_cpu = inp.clone().requires_grad_()
        out_cpu = torch.nn.functional.pad(inp_cpu, [2, 2, 1, 1], mode="replicate")
        out_cpu.sum().backward()

        inp_xpu = inp.clone().to(xpu_device).requires_grad_()
        out_xpu = torch.nn.functional.pad(inp_xpu, [2, 2, 1, 1], mode="replicate")
        out_xpu.sum().backward()

        self.assertEqual(inp_cpu.grad, inp_xpu.grad.cpu())
