# Owner(s): ["module: inductor"]
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# XPU port of selected tests from test/inductor/test_cuda_repro.py.

import unittest

import torch
import torch._dynamo.testing
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class CudaReproXpuTests(TestCase):
    device = device_type

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        """
        Regression test for https://github.com/intel/torch-xpu-ops/issues/3007.

        Verifies that torch.compile with dynamic=True and flash attention does
        not over-specialise on the seq_len dimension.  The model should produce
        exactly two Dynamo frames: one initial compilation and one recompilation
        that generalises to any seq_len, without creating a new frame for every
        distinct sequence length.

        Root cause: check_flash_attention_head_dim_size in pytorch's
        aten/src/ATen/native/transformers/xpu/sdp_utils.cpp used concrete
        .size() calls instead of .sym_size(), causing static guards on the
        head-dim dimension that prevented dynamic-shape generalisation.
        Fixed upstream: pytorch commit fd1d1b0 (sym_size + XPU platform
        detection in evaluate_platform_supports_flash_attention).
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(1024, 1024)
                self.k = nn.Linear(1024, 1024)
                self.v = nn.Linear(1024, 1024)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()
                queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                attn = F.scaled_dot_product_attention(queries, keys, values)
                return attn

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        model = Model().to(device_type).half()
        model = torch.compile(model, backend=cnts, dynamic=True)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            input1 = torch.rand(5, 512, 1024, device=device_type, dtype=torch.float16)
            input2 = torch.rand(5, 513, 1024, device=device_type, dtype=torch.float16)
            input3 = torch.rand(5, 514, 1024, device=device_type, dtype=torch.float16)

            model(input1)
            model(input2)
            model(input3)

        self.assertEqual(cnts.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
