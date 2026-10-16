# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Upsampling (``F.interpolate``)."""

from ..case import F, FLOAT_DTYPES, nd_input, OpBench, State


class Upsample(OpBench):
    family = "upsample"
    passes = ("fwd", "bwd")
    mode = None

    def expected_ops(self, p, pass_):
        name = f"_{self.op}" if self.mode == "nearest-exact" else self.op
        return (name if pass_ == "fwd" else f"{name}_backward",)

    def setup(self, p, device):
        x = nd_input(p["shape"], p["dtype"], device, p.get("channels_last", False))
        sf = p["scale_factor"]
        sf = tuple(sf) if isinstance(sf, (list, tuple)) else sf
        return State(x=x, sf=sf, ac=p.get("align_corners"))

    def fwd(self, s):
        return F.interpolate(s.x, scale_factor=s.sf, mode=self.mode, align_corners=s.ac)


def linear_cases(cases, align_corners):
    return [
        dict(shape=s, scale_factor=sf, align_corners=align_corners, dtype=dt)
        for s, sf in cases
        for dt in FLOAT_DTYPES
    ]


def nearest_cases(cases):
    return [
        dict(shape=s, scale_factor=sf, dtype=dt, channels_last=cl)
        for s, sf in cases
        for dt in FLOAT_DTYPES
        for cl in (False, True)
    ]


class UpsampleBicubic2d(Upsample):
    """(1, 128, 1200, 1200) x3 is reduced to 32 channels (memory)."""

    op = "upsample_bicubic2d"
    mode = "bicubic"
    params = linear_cases(
        [
            ((1, 3, 1200, 1200), (3, 3)),
            ((1, 32, 1200, 1200), (3, 3)),
            ((1, 3, 1200, 1200), (7, 7)),
            ((128, 128, 5, 5), (7, 7)),
        ],
        align_corners=True,
    )
    pr = [
        dict(
            shape=(1, 3, 1200, 1200),
            scale_factor=(3, 3),
            align_corners=True,
            dtype="bf16",
        )
    ]


class UpsampleBilinear2d(Upsample):
    """(1, 128, 1200, 1200) x3 is reduced to 32 channels (memory)."""

    op = "upsample_bilinear2d"
    mode = "bilinear"
    params = linear_cases(
        [
            ((1, 3, 1200, 1200), (3, 3)),
            ((1, 32, 1200, 1200), (3, 3)),
            ((1, 3, 1200, 1200), (7, 7)),
            ((128, 128, 5, 5), (7, 7)),
            ((8, 32, 256, 256), 3),
        ],
        align_corners=False,
    )
    pr = [
        dict(shape=(8, 32, 256, 256), scale_factor=3, align_corners=False, dtype="bf16")
    ]


NEAREST2D = [
    ((8, 32, 256, 256), 3),
    ((8, 512, 16, 16), 1.5),
    ((16, 1024, 23, 23), 2.3),
    ((4, 32, 80, 128), 2),
]


class UpsampleNearest2d(Upsample):
    op = "upsample_nearest2d"
    mode = "nearest"
    params = nearest_cases(NEAREST2D)
    pr = [
        dict(shape=(8, 32, 256, 256), scale_factor=3, dtype="bf16", channels_last=True)
    ]


class UpsampleNearestExact2d(UpsampleNearest2d):
    op = "upsample_nearest_exact2d"
    mode = "nearest-exact"


class UpsampleNearest3d(Upsample):
    """(8, 32, 256, 256, 2) is reduced to batch 4 (memory)."""

    op = "upsample_nearest3d"
    mode = "nearest"
    params = nearest_cases(
        [
            ((4, 32, 256, 256, 2), 3),
            ((8, 512, 16, 16, 4), 1.5),
            ((16, 1024, 23, 23, 7), 2.3),
        ]
    )
    pr = [
        dict(
            shape=(8, 512, 16, 16, 4),
            scale_factor=1.5,
            dtype="bf16",
            channels_last=False,
        )
    ]
