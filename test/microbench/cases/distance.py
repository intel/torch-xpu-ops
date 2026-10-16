# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""cdist / pdist."""

from ..case import F, OpBench, rand, State, torch


SHAPES = [
    ((8, 16), (2, 16)),
    ((10, 8192), (10, 8192)),
    ((10, 8192), (8192, 8192)),
    ((4, 512, 512), (4, 513, 512)),
    ((1, 512, 8192), (1, 1024, 8192)),
]


MODES = [
    "use_mm_for_euclid_dist_if_necessary",
    "use_mm_for_euclid_dist",
    "donot_use_mm_for_euclid_dist",
]


def _uses_mm(p):
    if p["p"] != 2:
        return False
    if p["compute_mode"] == "use_mm_for_euclid_dist":
        return True
    if p["compute_mode"] == "use_mm_for_euclid_dist_if_necessary":
        return p["x1"][-2] > 25 or p["x2"][-2] > 25
    return False


class Cdist(OpBench):
    """For p=2 with the matmul path the forward is ``_euclidean_dist`` (a
    composite of mm/norm ops, mostly oneDNN), so those cases are forward only."""

    op = "cdist"
    family = "distance"
    passes = ("fwd", "bwd")
    params = [
        dict(x1=a, x2=b, p=pp, compute_mode=m, dtype="fp32")
        for a, b in SHAPES
        for pp in (0.0, 1.0, 2.0)
        for m in (MODES if pp == 2.0 else MODES[:1])
    ]
    pr = [
        dict(
            x1=(4, 512, 512),
            x2=(4, 513, 512),
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
            dtype="fp32",
        )
    ]

    def passes_for(self, p):
        return ("fwd",) if _uses_mm(p) else ("fwd", "bwd")

    def expected_ops(self, p, pass_):
        if pass_ == "bwd":
            return ("_cdist_backward",)
        return ("_euclidean_dist",) if _uses_mm(p) else ("_cdist_forward",)

    def expect_inputs(self, p, pass_):
        if pass_ != "fwd" or _uses_mm(p):
            return None
        return [
            dict(shape=tuple(p["x1"]), dtype=p["dtype"]),
            dict(shape=tuple(p["x2"]), dtype=p["dtype"]),
        ]

    def setup(self, p, device):
        return State(
            x1=rand(p["x1"], p["dtype"], device),
            x2=rand(p["x2"], p["dtype"], device),
            p=p["p"],
            mode=p["compute_mode"],
        )

    def grad_inputs(self, s):
        return [s.x1, s.x2]

    def fwd(self, s):
        return torch.cdist(s.x1, s.x2, p=s.p, compute_mode=s.mode)


FWD_SHAPES = [(2048, 256), (2048, 8192), (16, 8192 * 4)]


BWD_SHAPES = [(256, 256), (256, 8192), (16, 8192 * 4)]


class Pdist(OpBench):
    op = "pdist"
    family = "distance"
    expected = {"fwd": ("_pdist_forward",), "bwd": ("_pdist_backward",)}
    params = [
        dict(shape=s, p=2.0, dtype="fp32")
        for s in FWD_SHAPES + [s for s in BWD_SHAPES if s not in FWD_SHAPES]
    ]
    pr = [dict(shape=(2048, 256), p=2.0, dtype="fp32")]

    def passes_for(self, p):
        shape = tuple(p["shape"])
        return tuple(
            x for x, lst in (("fwd", FWD_SHAPES), ("bwd", BWD_SHAPES)) if shape in lst
        )

    def setup(self, p, device):
        return State(x=rand(p["shape"], p["dtype"], device), p=p["p"])

    def fwd(self, s):
        return F.pdist(s.x, s.p)
