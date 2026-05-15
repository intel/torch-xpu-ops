# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Owner(s): ["module: inductor"]
"""
Reproducer for https://github.com/intel/torch-xpu-ops/issues/3007.

Verifies that scaled_dot_product_attention with dynamic=True compiles with
exactly 2 Dynamo frames (no over-specialisation on seq_len) when the XPU
flash attention backend is selected.

Run with:
    pytest test/repro/test_flash_attention_dynamic.py -v

Requires XPU hardware with flash attention support (PVC / BMG).
"""

import unittest

import torch
import torch._dynamo.testing
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFlashAttentionDynamic(unittest.TestCase):
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        """
        Regression test: torch.compile with dynamic=True + flash attention must
        not create more than 2 Dynamo frames across calls with different seq_len.

        Before the pytorch fix (commit fd1d1b0), check_flash_attention_head_dim_size
        in aten/src/ATen/native/transformers/xpu/sdp_utils.cpp used concrete
        .size() calls that materialized the head-dim dimension as a static guard.
        This prevented Dynamo from generalising the traced graph after the first
        recompilation, causing a new frame for every distinct seq_len value.
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
            model(torch.rand(5, 512, 1024, device=device_type, dtype=torch.float16))
            model(torch.rand(5, 513, 1024, device=device_type, dtype=torch.float16))
            model(torch.rand(5, 514, 1024, device=device_type, dtype=torch.float16))

        # Exactly 2 frames expected:
        # frame 1 – initial compilation with seq_len=512
        # frame 2 – recompilation that generalises seq_len as a dynamic symbol
        # frame 3+ would indicate over-specialisation (the regression)
        self.assertEqual(
            cnts.frame_count,
            2,
            msg=(
                f"Expected 2 Dynamo frames (initial + 1 generalising recompile), "
                f"got {cnts.frame_count}.  This likely indicates that "
                "check_flash_attention_head_dim_size is using concrete .size() "
                "instead of .sym_size(), preventing dynamic-shape generalisation."
            ),
        )


if __name__ == "__main__":
    unittest.main()
