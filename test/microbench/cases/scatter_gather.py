# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""gather / scatter / scatter_add."""

from ..case import FLOAT_DTYPES, full, OpBench, randint, randn, State, torch


def variants(cases):
    """(shape, index_shape, dim) cases in every float dtype."""
    return [
        dict(shape=s, index_shape=i, dim=d, dtype=dt)
        for s, i, d in cases
        for dt in FLOAT_DTYPES
    ]


class Gather(OpBench):
    """The 2.6G-element inputs are reduced 4-8x (memory), keeping the
    input/index aspect ratios."""

    op = "gather"
    family = "scatter_gather"
    expected = {"fwd": ("gather",)}
    params = variants(
        [
            ((2048, 64, 4), (2048, 64, 1), 2),
            ((28, 4096, 9), (28, 4096, 1), 2),
            ((512, 36, 4), (512, 36, 1), 2),
            ((81920000, 4), (81920000, 1), 1),  # 655360000 rows / 8
            ((20480, 25600), (5120, 25600), 0),  # (102400, 25600) / 5
            ((5120, 102400), (1280, 102400), 0),  # (25600, 102400) / 5
            ((10240, 8192), (10240, 2048), 1),
            ((8192, 10240), (2048, 2560), 1),
            ((10240, 8192), (2560, 8192), 0),
            ((8192, 10240), (2048, 10240), 0),
        ]
    )
    pr = [
        dict(shape=(10240, 8192), index_shape=(10240, 2048), dim=1, dtype="bf16"),
        dict(shape=(28, 4096, 9), index_shape=(28, 4096, 1), dim=2, dtype="bf16"),
    ]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device)
        idx = randint(0, p["shape"][p["dim"]], p["index_shape"], device)
        return State(x=x, dim=p["dim"], idx=idx)

    def fwd(self, s):
        return torch.gather(s.x, s.dim, s.idx)


SIZES = [
    (4, 4096, 4096),
    (2048, 4, 4096),
    (2048, 4096, 4),
    (2048, 4096, 4096),
    (4096, 8192, 8192),
    (4097, 8193, 8193),
]
# (dst shape, index/src shape, dim)
SCATTER = (
    [((28, 4096, 1), (28, 4096, 9), 2), ((512, 36, 1), (512, 36, 4), 2)]
    + [((m2, n), (m1, n), 0) for m1, m2, n in SIZES]
    + [((m, n2), (m, n1), 1) for m, n1, n2 in SIZES]
)


class Scatter(OpBench):
    """scatter_ with src; writing the same values every iteration is
    idempotent, so no reset."""

    op = "scatter"
    family = "scatter_gather"
    expected = {"fwd": ("scatter_",)}
    params = variants(SCATTER)
    pr = [dict(shape=(4096, 4096), index_shape=(2048, 4096), dim=0, dtype="bf16")]

    def setup(self, p, device):
        dst = full(p["shape"], 0, p["dtype"], device)
        idx = randint(0, p["shape"][p["dim"]], p["index_shape"], device)
        return State(
            x=dst,
            dim=p["dim"],
            idx=idx,
            src=full(p["index_shape"], 1, p["dtype"], device),
        )

    def fwd(self, s):
        return s.x.scatter_(s.dim, s.idx, s.src)


class ScatterAdd(Scatter):
    """scatter_add_; the destination accumulates, so ``reset()`` zeroes it."""

    op = "scatter_add"
    noise_class = "atomic"
    inplace = True
    expected = {"fwd": ("scatter_add_",)}

    def reset(self, s):
        s.x.zero_()

    def fwd(self, s):
        return s.x.scatter_add_(s.dim, s.idx, s.src)
