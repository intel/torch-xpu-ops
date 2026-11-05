# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Random number generation ops."""

from ..case import FLOAT_DTYPES, full, grid, OpBench, randn, State, torch

SHAPE = (8192, 8192)


class Rng(OpBench):
    family = "distribution"
    noise_class = "random"
    params = [dict(shape=SHAPE, dtype=dt) for dt in FLOAT_DTYPES]
    pr = [dict(shape=SHAPE, dtype="bf16")]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device))


def inplace(op, *args):
    """``x.<op>_(*args)`` on an (8192, 8192) tensor."""
    ns = {
        "op": op,
        "expected": {"fwd": (f"{op}_",)},
        "fwd": lambda self, s: getattr(s.x, f"{op}_")(*args),
    }
    return type(op, (Rng,), ns)


Cauchy = inplace("cauchy")
Exponential = inplace("exponential", 0.5)
Geometric = inplace("geometric", 0.5)
LogNormal = inplace("log_normal", 1.0, 2.0)
Normal = inplace("normal")
Random = inplace("random", -(2**8), 2**8)
Uniform = inplace("uniform")


class Bernoulli(Rng):
    """bernoulli_ with a scalar p and with a 0-dim device tensor p."""

    op = "bernoulli"
    expected = {"fwd": ("bernoulli_",)}
    params = [
        dict(shape=SHAPE, dtype=dt, p=pk)
        for dt in FLOAT_DTYPES
        for pk in ("float", "tensor")
    ]
    pr = [dict(shape=SHAPE, dtype="bf16", p="float")]

    def setup(self, p, device):
        x = full(p["shape"], 0, p["dtype"], device)
        prob = (
            0.5
            if p["p"] == "float"
            else torch.full((), 0.5, dtype=x.dtype, device=device)
        )
        return State(x=x, prob=prob)

    def fwd(self, s):
        return s.x.bernoulli_(s.prob)


class Multinomial(Rng):
    op = "multinomial"
    expected = {"fwd": ("multinomial",)}
    params = [
        dict(shape=SHAPE, dtype=dt, replacement=r, num_samples=n)
        for dt in FLOAT_DTYPES
        for r in (False, True)
        for n in (2, 128)
    ]
    pr = [dict(shape=SHAPE, dtype="bf16", replacement=False, num_samples=128)]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device).abs_()
        return State(x=x, n=p["num_samples"], r=p["replacement"])

    def fwd(self, s):
        return torch.multinomial(s.x, s.n, s.r)


class Dropout(Rng):
    """Training dropout. ``native_dropout`` is called directly: ``F.dropout``
    only reaches it on GPU devices (it decomposes into bernoulli_/mul on CPU)."""

    op = "dropout"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("native_dropout",), "bwd": ("native_dropout_backward",)}
    params = grid(shape=[SHAPE, (16, 1024)], dtype=FLOAT_DTYPES, p=[0.5])
    pr = [dict(shape=SHAPE, dtype="bf16", p=0.5)]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), p=p["p"])

    def fwd(self, s):
        return torch.native_dropout(s.x, s.p, True)
