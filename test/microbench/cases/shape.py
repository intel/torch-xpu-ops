# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Copy and shape ops."""

from ..case import FLOAT_DTYPES, OpBench, randn, State, table_bench, torch

Cat = table_bench(
    "cat",
    [
        dict(
            shapes=((0,), (4, 8, 1024, 64)),
            strides=(None, (524288, 64, 512, 1)),
            dim=-2,
        ),
        dict(
            shapes=((128, 12, 64, 64, 64), (128, 12, 64, 64, 64)),
            strides=(None, (3145728, 64, 49152, 768, 1)),
            dim=3,
        ),
        dict(shapes=((1000, 4),) * 16, dim=0),
    ],
    pr=[0],
    family="copy",
    fwd=lambda self, s: torch.cat(s.xs, s.dim),
)
UnfoldBackward = table_bench(
    "unfold_backward",
    [
        dict(
            shape=(128, 640, 1, 1, 12, 12),
            input_sizes=(128, 640, 1, 12, 12),
            dim=3,
            size=12,
            step=8,
        ),
        dict(
            shape=(128, 640, 2, 2, 12, 12),
            input_sizes=(128, 640, 2, 20, 12),
            dim=3,
            size=12,
            step=8,
        ),
    ],
    pr=[0],
    family="copy",
    passes=("bwd",),
    expected={"bwd": ("unfold_backward",)},
    bwd=lambda self, s: torch.ops.aten.unfold_backward(
        s.x, s.input_sizes, s.dim, s.size, s.step
    ),
)
# repeats-only call: repeats uniform in [0, 5), no output_size, so the op
# syncs to size its output
RepeatInterleaveRepeats = table_bench(
    "repeat_interleave",
    [dict(shape=(16,), dtype="int64")],
    family="copy",
    values={"x": ("int", 0, 5)},
    fwd=lambda self, s: torch.repeat_interleave(s.x),
)


class RepeatInterleave(OpBench):
    """With a repeats device tensor (a Python int decomposes into expand +
    clone) and ``output_size`` (no host sync): ``repeat_interleave.Tensor``
    builds the gather indices and ``index_select`` gathers."""

    op = "repeat_interleave"
    family = "copy"
    expected = {"fwd": ("repeat_interleave", "index_select")}
    params = [
        dict(shape=s, repeats=8, dim=d, dtype=dt)
        for s in [(16, 8, 23), (4, 2048, 2048)]
        for dt in FLOAT_DTYPES
        for d in (0, 2)
    ]
    pr = [dict(shape=(4, 2048, 2048), repeats=8, dim=2, dtype="bf16")]

    def expect_inputs(self, p, pass_):
        # the first expected op receives the repeats tensor
        return [dict(shape=(p["shape"][p["dim"]],), dtype="int64")]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device)
        n = p["shape"][p["dim"]]
        reps = torch.full((n,), p["repeats"], dtype=torch.long).to(device)
        return State(x=x, reps=reps, dim=p["dim"], out=n * p["repeats"])

    def fwd(self, s):
        return torch.repeat_interleave(s.x, s.reps, s.dim, output_size=s.out)


class Flip(OpBench):
    op = "flip"
    family = "shape"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("flip",), "bwd": ("flip",)}
    params = [
        dict(shape=s, dims=d, dtype=dt)
        for s, d in [
            ((64, 1024, 1024), (0, 1)),
            ((1024, 64, 1024), (0, 2)),
            ((1024, 1024, 64), (1, 2)),
            ((16, 128, 512, 512), (0, 2)),
            ((16, 128, 512, 512), (0, 3)),
            ((16, 128, 512, 512), (1, 3)),
        ]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(16, 128, 512, 512), dims=(0, 3), dtype="bf16")]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), dims=tuple(p["dims"]))

    def fwd(self, s):
        return torch.flip(s.x, s.dims)


class Roll(OpBench):
    """(1024, 1024, 1024) is halved on dim 0 (memory)."""

    op = "roll"
    family = "shape"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("roll",), "bwd": ("roll",)}
    params = [
        dict(shape=s, shifts=sh, dims=d, dtype=dt)
        for s, sh, d in [
            ((512, 1024, 1024), -1, 0),
            ((512, 1024, 1024), (128, 128), (-1, 0)),
            ((512, 1024, 1024), 128, -1),
            ((16, 3, 512, 512), -1, -1),
            ((16, 3, 512, 512), 127, 0),
            ((16, 3, 512, 512), (127, 127), (0, -1)),
        ]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(16, 3, 512, 512), shifts=(127, 127), dims=(0, -1), dtype="bf16")]

    def setup(self, p, device):
        return State(
            x=randn(p["shape"], p["dtype"], device), sh=p["shifts"], d=p["dims"]
        )

    def fwd(self, s):
        return torch.roll(s.x, shifts=s.sh, dims=s.d)


class PadSequence(OpBench):
    """No kernel of its own: allocates the padded output with ``full`` and
    copies each sequence into it."""

    op = "pad_sequence"
    family = "copy"
    composite = True
    expected = {"fwd": ("full|new_full", "copy_")}
    params = [
        dict(shapes=s, dtype=dt, batch_first=bf, padding_value=pv)
        for s in [
            ((25, 300), (22, 300), (15, 300)),
            ((2, 1000), (100, 1000), (8192, 1000)),
        ]
        for dt in FLOAT_DTYPES
        for bf in (False, True)
        for pv in (0.0, 1.0, 2.0)
    ]
    pr = [
        dict(
            shapes=((2, 1000), (100, 1000), (8192, 1000)),
            dtype="bf16",
            batch_first=True,
            padding_value=0.0,
        )
    ]

    def setup(self, p, device):
        seqs = [randn(s, p["dtype"], device) for s in p["shapes"]]
        return State(seqs=seqs, bf=p["batch_first"], pv=p["padding_value"])

    def fwd(self, s):
        return torch.nn.utils.rnn.pad_sequence(
            s.seqs, batch_first=s.bf, padding_value=s.pv
        )
