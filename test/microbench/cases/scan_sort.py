# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Scan, selection and sort ops."""

from ..case import FLOAT_DTYPES, OpBench, randint, randn, State, to_dtype, torch


class Cumsum(OpBench):
    op = "cumsum"
    family = "scan"
    expected = {"fwd": ("cumsum",)}
    params = [
        dict(shape=s, dtype=dt, dim=d)
        for s in [(8193, 8193), (1234, 8193), (8192, 1234), (1, 4 * 15000)]
        for dt in FLOAT_DTYPES
        for d in (0, 1)
    ]
    pr = [
        dict(shape=(8193, 8193), dtype="bf16", dim=1),
        dict(shape=(8193, 8193), dtype="bf16", dim=0),
    ]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), dim=p["dim"])

    def fwd(self, s):
        return torch.cumsum(s.x, s.dim)


class MaskedSelect(OpBench):
    """Syncs to size the output."""

    op = "masked_select"
    family = "scan"
    noise_class = "host_bound"
    expected = {"fwd": ("masked_select",)}
    params = [dict(shape=(8193, 8193), dtype=dt) for dt in FLOAT_DTYPES]
    pr = [dict(shape=(8193, 8193), dtype="bf16")]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device)
        return State(x=x, mask=x.ge(0.5))

    def fwd(self, s):
        return torch.masked_select(s.x, s.mask)


class Nonzero(OpBench):
    """Syncs to size the output; the 3D input is about 80% nonzero."""

    op = "nonzero"
    family = "scan"
    noise_class = "host_bound"
    expected = {"fwd": ("nonzero",)}
    params = [
        dict(shape=s, dtype=dt)
        for s in [(2047, 2047, 10), (1, 4 * 15000)]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(2047, 2047, 10), dtype="bf16")]

    def setup(self, p, device):
        if len(p["shape"]) == 3:
            # values in {-2..2}: about 80% nonzero
            x = randint(-2, 3, p["shape"], device, dtype=p["dtype"])
        else:
            x = randn(p["shape"], p["dtype"], device)
        return State(x=x)

    def fwd(self, s):
        return torch.nonzero(s.x)


class Topk(OpBench):
    op = "topk"
    family = "sort"
    expected = {"fwd": ("topk",)}
    params = [
        dict(shape=(8193, 8193), k=4096, dtype=dt, dim=d)
        for dt in FLOAT_DTYPES
        for d in (0, 1)
    ]
    pr = [dict(shape=(8193, 8193), k=4096, dtype="bf16", dim=1)]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), k=p["k"], dim=p["dim"])

    def fwd(self, s):
        return torch.topk(s.x, s.k, s.dim, largest=True, sorted=True)


class Unique(OpBench):
    """sorted, return_inverse, return_counts -> _unique2."""

    op = "unique"
    family = "sort"
    noise_class = "host_bound"
    expected = {"fwd": ("_unique2",)}
    params = [dict(shape=(2049, 2049), dtype=dt, values=100) for dt in FLOAT_DTYPES]
    pr = [dict(shape=(2049, 2049), dtype="bf16", values=100)]

    def setup(self, p, device):
        return State(x=randint(0, p["values"], p["shape"], device, dtype=p["dtype"]))

    def fwd(self, s):
        return torch.unique(s.x, sorted=True, return_inverse=True, return_counts=True)


class Sort(OpBench):
    op = "sort"
    family = "sort"
    expected = {"fwd": ("sort",)}
    params = [
        dict(shape=(8193, 8193), dtype=dt, dim=d) for dt in FLOAT_DTYPES for d in (0, 1)
    ]
    pr = [dict(shape=(8193, 8193), dtype="bf16", dim=1)]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), dim=p["dim"])

    def fwd(self, s):
        return torch.sort(s.x, s.dim)


class Randperm(OpBench):
    op = "randperm"
    family = "sort"
    noise_class = "random"
    expected = {"fwd": ("randperm",)}
    params = [dict(n=8193, dtype="fp32")]
    pr = [dict(n=8193, dtype="fp32")]

    def setup(self, p, device):
        return State(n=p["n"], dtype=to_dtype(p["dtype"]), device=device)

    def fwd(self, s):
        return torch.randperm(s.n, dtype=s.dtype, device=s.device)
