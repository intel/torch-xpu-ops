# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Softmax family."""

from ..case import FLOAT_DTYPES, grid, OpBench, randn, State, table_bench, torch


class Softmax(OpBench):
    op = "softmax"
    family = "softmax"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("_softmax",), "bwd": ("_softmax_backward_data",)}
    params = grid(
        shape=[(8192, 8192), (64, 8192), (8192, 64)], dtype=FLOAT_DTYPES, dim=[0, 1]
    )
    pr = [
        dict(shape=(8192, 8192), dtype="bf16", dim=1),
        dict(shape=(8192, 8192), dtype="bf16", dim=0),
    ]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), dim=p["dim"])

    def fwd(self, s):
        return torch.softmax(s.x, s.dim)


LogSoftmax = table_bench(
    "_log_softmax",
    [
        dict(
            shape=(4, 2, 640, 959),
            in_stride=(1227520, 1, 1918, 2),
            dim=1,
            half_to_float=False,
        ),
        dict(shape=(128, 1000), dim=1, half_to_float=False),
    ],
    pr=[0],
    family="softmax",
    fwd=lambda self, s: torch._log_softmax(s.x, s.dim, s.half_to_float),
)
# the aten backward op is called directly with s.x as grad_output
LogSoftmaxBackwardData = table_bench(
    "_log_softmax",
    [
        dict(shape=(4096, 50265), output_shape=(4096, 50265), dim=1),
        dict(
            shape=(64, 128, 20005),
            in_stride=(2560640, 1, 128),
            output_shape=(64, 128, 20005),
            dim=-1,
        ),
        dict(shape=(128, 1000), output_shape=(128, 1000), dim=1),
    ],
    pr=[0],
    family="softmax",
    passes=("bwd",),
    expected={"bwd": ("_log_softmax_backward_data",)},
    bwd=lambda self, s: torch._log_softmax_backward_data(
        s.x, s.output, s.dim, s.x.dtype
    ),
)
SafeSoftmax = table_bench(
    "_safe_softmax",
    [
        dict(shape=(16, 12, 512, 512), dim=-1),
        dict(shape=(256, 16, 1, 196, 196), dim=-1),
    ],
    pr=[0],
    family="softmax",
    fwd=lambda self, s: torch.ops.aten._safe_softmax(s.x, s.dim),
)
