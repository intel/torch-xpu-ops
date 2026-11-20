# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Normalization (training mode)."""

from ..case import F, FLOAT_DTYPES, nd_input, OpBench, randn, State, torch


class BatchNorm(OpBench):
    """fp32 weight, bias and running stats (mixed-dtype BN for bf16/fp16)."""

    family = "norm"
    passes = ("fwd", "bwd")
    expected = {
        "fwd": ("_batch_norm_with_update|native_batch_norm",),
        "bwd": ("batch_norm_backward|native_batch_norm_backward",),
    }

    def setup(self, p, device):
        c = p["shape"][1]
        ones, zeros = (torch.ones(c, device=device), torch.zeros(c, device=device))
        x = nd_input(p["shape"], p["dtype"], device, p.get("channels_last", False))
        return State(x=x, w=ones, b=zeros, rm=zeros.clone(), rv=ones.clone())

    def fwd(self, s):
        return F.batch_norm(s.x, s.rm, s.rv, s.w, s.b, training=True)


class BatchNorm1d(BatchNorm):
    op = "batch_norm_1d"
    params = [
        dict(shape=s, dtype=dt)
        for dt in FLOAT_DTYPES
        for s in [(64, 8), (4, 128, 15000), (4, 256, 512)]
    ]
    pr = [dict(shape=(4, 128, 15000), dtype="bf16")]


class BatchNorm2d(BatchNorm):
    op = "batch_norm_2d"
    params = [
        dict(shape=s, dtype=dt, channels_last=cl)
        for dt in FLOAT_DTYPES
        for s in [
            (256, 256, 56, 56),
            (256, 2048, 7, 7),
            (24, 512, 28, 28),
            (24, 1024, 14, 14),
            (4, 8, 640, 1024),
            (4, 48, 20, 32),
        ]
        for cl in (False, True)
    ]
    pr = [
        dict(shape=(24, 512, 28, 28), dtype="bf16", channels_last=cl)
        for cl in (False, True)
    ]


class BatchNorm3d(BatchNorm):
    """(16, 16, 128, 128, 256) is reduced to batch 8 (memory)."""

    op = "batch_norm_3d"
    params = [
        dict(shape=s, dtype=dt, channels_last=cl)
        for dt in FLOAT_DTYPES
        for s in [(2, 5, 6, 3, 5), (2, 8, 64, 64, 64), (8, 16, 128, 128, 256)]
        for cl in (False, True)
    ]
    pr = [dict(shape=(2, 8, 64, 64, 64), dtype="bf16", channels_last=False)]


class GroupNorm(OpBench):
    op = "group_norm"
    family = "norm"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("native_group_norm",), "bwd": ("native_group_norm_backward",)}
    params = [
        dict(shape=s, groups=g, dtype=dt, channels_last=cl, affine=af)
        for g, s in [
            (1, (32, 128, 32, 32)),
            (16, (1024, 128, 32, 32)),
            (32, (32, 32, 64, 64)),
            (32, (16, 512, 256, 256)),  # batch 32 -> 16 (memory)
            (8, (32, 32, 16, 64, 64)),
        ]
        for dt in FLOAT_DTYPES
        for cl in (False, True)
        for af in (False, True)
    ]
    pr = [
        dict(
            shape=(32, 32, 64, 64),
            groups=32,
            dtype="bf16",
            channels_last=False,
            affine=True,
        )
    ]

    def setup(self, p, device):
        c = p["shape"][1]
        x = nd_input(p["shape"], p["dtype"], device, p["channels_last"])
        w = b = None
        if p["affine"]:
            w = torch.ones(c, device=device, dtype=x.dtype)
            b = torch.zeros(c, device=device, dtype=x.dtype)
        return State(x=x, g=p["groups"], w=w, b=b)

    def fwd(self, s):
        return F.group_norm(s.x, s.g, s.w, s.b)


class LayerNorm(OpBench):
    op = "layer_norm"
    family = "norm"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("native_layer_norm",), "bwd": ("native_layer_norm_backward",)}
    params = [
        dict(shape=s, normalized_shape=n, dtype=dt)
        for s, n in [
            ((1, 1024), (1024,)),
            ((2, 4096, 320), (4096, 320)),
            ((512, 3136, 128), (3136, 128)),
            ((48, 49, 196, 1024), (49, 196, 1024)),  # batch 128 -> 48 (memory)
        ]
        for dt in FLOAT_DTYPES
    ]
    pr = [
        dict(shape=(2, 4096, 320), normalized_shape=(4096, 320), dtype="bf16"),
        dict(shape=(512, 3136, 128), normalized_shape=(3136, 128), dtype="bf16"),
    ]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device)
        n = tuple(p["normalized_shape"])
        w = torch.ones(n, device=device, dtype=x.dtype)
        return State(x=x, n=n, w=w, b=torch.zeros(n, device=device, dtype=x.dtype))

    def fwd(self, s):
        return F.layer_norm(s.x, s.n, s.w, s.b)
