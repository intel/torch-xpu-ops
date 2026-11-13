# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Reductions."""

from ..case import FLOAT_DTYPES, OpBench, randn, State, table_bench, torch

# all/any: a mask that is scanned completely (all True / all False) or exits
# early (50% True)
All = table_bench(
    "all",
    [
        dict(shape=(32, 1024, 1024), dtype="bool", values="all_true"),
        dict(shape=(32, 1024, 1024), dtype="bool", values="half"),
        dict(shape=(1000, 320), dim=1, dtype="bool", values="all_true"),
        dict(shape=(1000, 320), dim=1, dtype="bool", values="half"),
    ],
    pr=[0],
    family="reduction",
    values={"x": lambda p: ("mask", {"all_true": 1.0, "half": 0.5}[p["values"]])},
    defaults={"dim": None},
    fwd=lambda self, s: torch.all(s.x) if s.dim is None else torch.all(s.x, s.dim),
)
Any = table_bench(
    "any",
    [
        dict(shape=(32, 1024, 512), dtype="bool", values="all_false"),
        dict(shape=(32, 1024, 512), dtype="bool", values="half"),
        dict(shape=(32, 512, 1024), dtype="bool", values="all_false"),
        dict(shape=(32, 512, 1024), dtype="bool", values="half"),
    ],
    pr=[0],
    family="reduction",
    values={"x": lambda p: ("mask", {"all_false": 0.0, "half": 0.5}[p["values"]])},
    defaults={"dim": None},
    fwd=lambda self, s: torch.any(s.x) if s.dim is None else torch.any(s.x, s.dim),
)
Argmax = table_bench(
    "argmax",
    [
        dict(shape=(64, 12, 1, 4096, 128), dim=-1),
        dict(shape=(16, 2, 640, 959), in_stride=(1227520, 1, 1918, 2), dim=1),
    ],
    pr=[0],
    family="reduction",
    fwd=lambda self, s: torch.argmax(s.x, s.dim),
)
Mean = table_bench(
    "mean",
    [
        dict(
            shape=(128, 2048, 7, 7),
            in_stride=(100352, 1, 14336, 2048),
            dim=(-1, -2),
            keepdim=True,
        ),
        dict(shape=(256, 512, 768), dim=(-1,), keepdim=True),
    ],
    pr=[0],
    family="reduction",
    fwd=lambda self, s: torch.mean(s.x, s.dim, s.keepdim),
)
Std = table_bench(
    "std",
    [
        dict(shape=(512, 128, 768), dim=(-1,), correction=1, keepdim=True),
        dict(shape=(64, 128, 768), dim=(-1,), correction=1, keepdim=True),
    ],
    pr=[0],
    family="reduction",
    fwd=lambda self, s: torch.std(
        s.x, s.dim, correction=s.correction, keepdim=s.keepdim
    ),
)


class Sum(OpBench):
    """Full and along a dim."""

    op = "sum"
    family = "reduction"
    expected = {"fwd": ("sum",)}
    params = [
        dict(shape=(8192, 8192), dtype=dt, dim=d)
        for dt in FLOAT_DTYPES
        for d in (None, 0, 1)
    ]
    pr = [
        dict(shape=(8192, 8192), dtype="bf16", dim=1),
        dict(shape=(8192, 8192), dtype="bf16", dim=0),
    ]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), dim=p["dim"])

    def fwd(self, s):
        return torch.sum(s.x) if s.dim is None else torch.sum(s.x, s.dim)


class MaxDim(Sum):
    """max.dim (values + indices)."""

    op = "max"
    expected = {"fwd": ("max",)}
    params = [
        dict(shape=(8192, 8192), dtype=dt, dim=d) for dt in FLOAT_DTYPES for d in (1, 0)
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", dim=1)]

    def fwd(self, s):
        return torch.max(s.x, s.dim)
