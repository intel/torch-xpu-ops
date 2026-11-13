# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""_scaled_dot_product_flash_attention forward and backward; bf16/fp16 only
(no fp32 flash attention). Query, key and value share the layout."""

from ..case import randn, State, table_bench, torch

SMALL = dict(shape=(2, 16, 1024, 64), in_stride=(1048576, 64, 1024, 1), scale=0.125)
LARGE = dict(shape=(4, 64, 512, 64), in_stride=(2097152, 64, 4096, 1), scale=0.125)
OP = "_scaled_dot_product_flash_attention"


def qkv(p, device):
    q, k, v = (
        randn(p["shape"], p["dtype"], device, stride=p["in_stride"]) for _ in range(3)
    )
    return State(q=q, k=k, v=v, scale=p["scale"], is_causal=p["is_causal"])


def flash(s):
    return torch.ops.aten._scaled_dot_product_flash_attention(
        s.q, s.k, s.v, 0.0, s.is_causal, False, scale=s.scale
    )


def setup_bwd(self, p, device):
    """out, logsumexp and the philox state come from a forward run."""
    s = qkv(p, device)
    fwd = flash(s)
    s.grad = randn(p["shape"], p["dtype"], device, stride=p["in_stride"])
    s.out, s.lse, s.seed, s.offset = fwd[0], fwd[1], fwd[6], fwd[7]
    s.seq = p["shape"][2]
    return s


ScaledDotProductFlashAttention = table_bench(
    OP,
    [
        dict(SMALL, is_causal=False),
        dict(SMALL, is_causal=True),
        dict(LARGE, is_causal=False),
    ],
    pr=[0],
    family="attention",
    dtypes=("bf16", "fp16"),
    setup=lambda self, p, device: qkv(p, device),
    fwd=lambda self, s: flash(s),
)
ScaledDotProductFlashAttentionBackward = table_bench(
    OP,
    [dict(SMALL, is_causal=False), dict(LARGE, is_causal=False)],
    pr=[0],
    family="attention",
    passes=("bwd",),
    dtypes=("bf16", "fp16"),
    setup=setup_bwd,
    bwd=lambda self, s: torch.ops.aten._scaled_dot_product_flash_attention_backward(
        s.grad,
        s.q,
        s.k,
        s.v,
        s.out,
        s.lse,
        None,
        None,
        s.seq,
        s.seq,
        0.0,
        s.is_causal,
        s.seed,
        s.offset,
        scale=s.scale,
    ),
)
