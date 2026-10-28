# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""im2col (``F.unfold``) / col2im (``F.fold``); each is the other's backward."""

from ..case import F, FLOAT_DTYPES, OpBench, randn, State

CONV = dict(kernel_size=(7, 7), dilation=(6, 6), padding=1, stride=1)


class Im2Col(OpBench):
    op = "im2col"
    family = "im2col"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("im2col",), "bwd": ("col2im",)}
    params = [
        dict(shape=s, **CONV, dtype=dt)
        for s in [
            (1, 3, 1200, 1200),
            (1, 3, 224, 224),
            (1, 3, 63, 1200),
            (1, 3, 1200, 63),
        ]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(1, 3, 1200, 1200), **CONV, dtype="bf16")]

    def setup(self, p, device):
        return State(
            x=randn(p["shape"], p["dtype"], device), kw={k: p[k] for k in CONV}
        )

    def fwd(self, s):
        return F.unfold(s.x, **s.kw)


class Col2Im(Im2Col):
    op = "col2im"
    expected = {"fwd": ("col2im",), "bwd": ("im2col",)}
    params = [
        dict(shape=s, output_size=o, **CONV, dtype=dt)
        for s, o in [
            ((1, 147, 1359556), (1200, 1200)),
            ((1, 147, 36100), (224, 224)),
            ((1, 147, 33814), (63, 1200)),
            ((1, 147, 33814), (1200, 63)),
        ]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(1, 147, 36100), output_size=(224, 224), **CONV, dtype="bf16")]

    def setup(self, p, device):
        s = super().setup(p, device)
        s.kw["output_size"] = p["output_size"]
        return s

    def fwd(self, s):
        return F.fold(s.x, **s.kw)
