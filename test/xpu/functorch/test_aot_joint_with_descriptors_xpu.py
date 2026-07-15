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

import unittest

import torch
import torch.fx.traceback as fx_traceback
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TEST_XPU

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from functorch.test_aot_joint_with_descriptors import (
        graph_capture,
        TestAOTJointWithDescriptors,
    )


# ======================================================================
# Change hardcoded "cuda" to device_type.
# Replace upstream requires_cuda decorator to run test both on CUDA and XPU.
# ======================================================================


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


@unittest.skipUnless(TEST_CUDA or TEST_XPU, "Requires CUDA or XPU")
def _test_preserve_annotate_flex_attention(self):
    def score_mod(score, b, h, m, n):
        return score

    def _get_block_causal_mask_mod(seq_idx):
        def block_causal_mask(b, h, q_idx, kv_idx):
            # must use this more complicated mask_mod so autograd seq_nr increases
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    a = 12
    b = 24
    batch_size = 2
    seqlen = a * b
    device = device_type  # CHANGED

    # Create seq_idx tensor - maps each position to a document/sequence ID
    # Example: Split sequence into 2 documents for each batch
    # First half (0:384) belongs to document 0, second half (384:768) to document 1
    seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device=device)
    seq_idx[:, seqlen // 2 :] = 1  # Second half belongs to document 1

    # Get the mask_mod function with seq_idx captured in closure
    mask_mod = _get_block_causal_mask_mod(seq_idx)

    # Create block_mask with the mask_mod function (which only takes 4 args)
    # Note: We don't compile create_block_mask itself, just flex_attention
    block_mask = create_block_mask(mask_mod, None, None, seqlen, seqlen)

    class FlexAttentionModule(torch.nn.Module):
        """Flex attention submodule similar to the sdpa in Llama3 Attention"""

        def forward(self, xq, xk, xv):
            """
            Args:
                xq: Query tensor (bs, n_heads, seqlen, head_dim)
                xk: Key tensor (bs, n_heads, seqlen, head_dim)
                xv: Value tensor (bs, n_heads, seqlen, head_dim)
            Returns:
                Output tensor (bs, n_heads, seqlen, head_dim)
            """
            with fx_traceback.annotate({"compile_with_inductor": "flex_attention"}):
                output = flex_attention(
                    xq, xk, xv, block_mask=block_mask, score_mod=score_mod
                )
            return output

    # Model configuration
    n_heads = 4
    head_dim = 64

    # Create input tensors in the shape expected by FlexAttentionModule
    # Shape: (bs, n_heads, seqlen, head_dim)
    xq = torch.randn(
        batch_size, n_heads, seqlen, head_dim, requires_grad=True, device=device
    )
    xk = torch.randn(
        batch_size, n_heads, seqlen, head_dim, requires_grad=True, device=device
    )
    xv = torch.randn(
        batch_size, n_heads, seqlen, head_dim, requires_grad=True, device=device
    )

    model = FlexAttentionModule().to(device)
    inputs = (xq, xk, xv)

    gm = graph_capture(model, inputs, with_export=True)

    custom_metadata = fx_traceback._get_custom_metadata(gm)

    # not using assertExpectedInline because some CI runs has fewer detach nodes in graph
    # than other CI runs, so we can't use a fixed string to compare against

    self.assertTrue(
        "('get_attr', 'sdpa_score0', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )
    self.assertTrue(
        "('get_attr', 'sdpa_mask0', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )
    self.assertTrue(
        "('call_function', 'flex_attention', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )

    self.assertTrue(
        "('get_attr', 'fw_graph0', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )
    self.assertTrue(
        "('get_attr', 'joint_graph0', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )
    self.assertTrue(
        "('get_attr', 'mask_graph0', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )
    self.assertTrue(
        "('call_function', 'flex_attention_backward', {'compile_with_inductor': 'flex_attention'})"
        in custom_metadata
    )


TestAOTJointWithDescriptors.test_preserve_annotate_flex_attention = (
    _test_preserve_annotate_flex_attention
)


if __name__ == "__main__":
    run_tests()
