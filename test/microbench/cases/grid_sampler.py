# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""grid_sampler_2d / grid_sampler_3d (``F.grid_sample``); grid uniform in [-1, 1]."""

from ..case import F, FLOAT_DTYPES, OpBench, rand, randn, State


def variants(shapes, modes):
    return [
        dict(shape=s, dtype=dt, mode=m, padding_mode=pm, align_corners=ac)
        for s in shapes
        for dt in FLOAT_DTYPES
        for m in modes
        for pm in ("zeros", "border", "reflection")
        for ac in (True, False)
    ]


class GridSampler2d(OpBench):
    op = "grid_sampler_2d"
    family = "grid_sampler"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("grid_sampler_2d",), "bwd": ("grid_sampler_2d_backward",)}
    params = variants(
        [(6, 5, 7, 3), (4, 32, 128, 128), (16, 128, 512, 512)],
        ("bilinear", "nearest", "bicubic"),
    )
    pr = [
        dict(
            shape=(4, 32, 128, 128),
            dtype="bf16",
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    ]

    def passes_for(self, p):
        # the backward adds into the input gradient atomically: bilinear and
        # bicubic on (16, 128, 512, 512) take 1-10 s per call on BMG
        if p["shape"] == (16, 128, 512, 512) and p["mode"] != "nearest":
            return ("fwd",)
        return self.passes

    def setup(self, p, device):
        n, _, *spatial = p["shape"]
        x = randn(p["shape"], p["dtype"], device)
        grid = rand((n, *spatial, len(spatial)), p["dtype"], device, -1.0, 1.0)
        return State(
            x=x,
            grid=grid,
            mode=p["mode"],
            pm=p["padding_mode"],
            ac=p["align_corners"],
            inputs=[x, grid],
        )

    def fwd(self, s):
        return F.grid_sample(
            s.x, s.grid, mode=s.mode, padding_mode=s.pm, align_corners=s.ac
        )


class GridSampler3d(GridSampler2d):
    op = "grid_sampler_3d"
    expected = {"fwd": ("grid_sampler_3d",), "bwd": ("grid_sampler_3d_backward",)}
    params = variants([(2, 5, 6, 3, 5), (8, 16, 64, 64, 64)], ("bilinear", "nearest"))
    pr = [
        dict(
            shape=(8, 16, 64, 64, 64),
            dtype="bf16",
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    ]
