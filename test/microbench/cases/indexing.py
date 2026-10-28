# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Indexing ops."""

from ..case import (
    cpu_to,
    FLOAT_DTYPES,
    full,
    OpBench,
    rand,
    randn,
    State,
    to_dtype,
    torch,
)


def spaced(end, steps, device):
    """``steps`` long indices evenly spaced over [0, end] (computed in fp64)."""
    return cpu_to(
        torch.linspace(0, end, steps=steps, dtype=torch.float64).to(torch.long), device
    )


class Indexing(OpBench):
    family = "indexing"

    def expected_ops(self, p, pass_):
        return (self.op,)


class Diag(Indexing):
    """No XPU kernel: a 1D input becomes ``diag_embed`` (fill + diagonal copy),
    a 2D input ``diagonal_copy``."""

    op = "diag"
    composite = True
    params = [
        dict(shape=s, dtype=dt) for s in [(8192,), (8192, 8192)] for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(8192,), dtype="bf16")]

    def expected_ops(self, p, pass_):
        return ("diag_embed",) if len(p["shape"]) == 1 else ("diagonal_copy",)

    def expect_inputs(self, p, pass_):
        return (
            [dict(shape=tuple(p["shape"]), dtype=p["dtype"])]
            if len(p["shape"]) == 1
            else None
        )

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device))

    def fwd(self, s):
        return torch.diag(s.x)


class Index(Indexing):
    """Advanced indexing with a bool mask and with long indices."""

    op = "index"
    params = [
        dict(shape=(4, 15000), dtype=dt, index=m)
        for dt in FLOAT_DTYPES
        for m in ("bool_mask", "long_indices")
    ]
    pr = [dict(shape=(4, 15000), dtype="bf16", index="bool_mask")]

    def setup(self, p, device):
        d, e = (
            rand(p["shape"], p["dtype"], device),
            rand(p["shape"], p["dtype"], device),
        )
        index = (
            (d < e)
            if p["index"] == "bool_mask"
            else torch.tensor([0, 2], device=device)
        )
        return State(e=e, index=index)

    def fwd(self, s):
        return s.e[s.index]


class IndexPut(Index):
    """index_put_; writing the same values every iteration is idempotent, so
    no reset."""

    op = "index_put"
    pr = [dict(shape=(4, 15000), dtype="bf16", index="long_indices")]

    def setup(self, p, device):
        d, e = (
            rand(p["shape"], p["dtype"], device),
            rand(p["shape"], p["dtype"], device),
        )
        mask = d < e
        index = mask if p["index"] == "bool_mask" else tuple(mask.nonzero().unbind(1))
        return State(d=d, index=index, vals=e[mask])

    def fwd(self, s):
        s.d[s.index] = s.vals
        return s.d


class IndexAdd(Indexing):
    """Out of place. 512 unique rows/cols, or 512 indices folded onto 8
    positions (64 accumulations each, the high-contention path)."""

    op = "index_add"
    noise_class = "atomic"
    params = [
        dict(shape=(1024, 1024), dtype=dt, dim=d, distinct=n)
        for dt in FLOAT_DTYPES
        for d in (0, 1)
        for n in (512, 8)
    ]
    pr = [dict(shape=(1024, 1024), dtype="bf16", dim=0, distinct=512)]

    def setup(self, p, device):
        idx = (
            spaced(1022, 512, "cpu")
            if p["distinct"] == 512
            else torch.arange(512) % p["distinct"]
        )
        x = torch.zeros(p["shape"], dtype=to_dtype(p["dtype"]), device=device)
        src_shape = (512, p["shape"][1]) if p["dim"] == 0 else (p["shape"][0], 512)
        src = (
            randn(src_shape, p["dtype"], device)
            if p["dim"]
            else torch.ones(src_shape, dtype=x.dtype, device=device)
        )
        return State(x=x, dim=p["dim"], idx=cpu_to(idx, device), src=src)

    def fwd(self, s):
        return s.x.index_add(s.dim, s.idx, s.src)


class IndexCopy(Indexing):
    op = "index_copy"
    params = [
        dict(shape=(1024, 1024), dtype=dt, dim=d) for dt in FLOAT_DTYPES for d in (0, 1)
    ]
    pr = [dict(shape=(1024, 1024), dtype="bf16", dim=0)]

    def setup(self, p, device):
        dim = p["dim"]
        src_shape = (512, 1024) if dim == 0 else (1024, 512)
        src = (
            full(src_shape, 1, p["dtype"], device)
            if dim == 0
            else randn(src_shape, p["dtype"], device)
        )
        x = full(p["shape"], 0, p["dtype"], device)
        return State(x=x, dim=dim, idx=spaced(1022, 512, device), src=src)

    def fwd(self, s):
        return s.x.index_copy(s.dim, s.idx, s.src)


class IndexFill(Indexing):
    op = "index_fill"
    params = [
        dict(shape=(1024, 1024), dtype=dt, dim=d, value=1 if d == 0 else 2)
        for dt in FLOAT_DTYPES
        for d in (0, 1)
    ]
    pr = [dict(shape=(1024, 1024), dtype="bf16", dim=0, value=1)]

    def setup(self, p, device):
        x = full(p["shape"], 0, p["dtype"], device)
        return State(x=x, dim=p["dim"], idx=spaced(1022, 512, device), v=p["value"])

    def fwd(self, s):
        return s.x.index_fill(s.dim, s.idx, s.v)


class IndexSelect(Indexing):
    """Every other row."""

    op = "index_select"
    params = [
        dict(shape=s, dtype=dt, dim=0)
        for s in [(1024, 1024), (8192, 8192)]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", dim=0)]

    def setup(self, p, device):
        n = p["shape"][0]
        return State(
            x=randn(p["shape"], p["dtype"], device),
            dim=p["dim"],
            idx=spaced(n - 2, n // 2, device),
        )

    def fwd(self, s):
        return torch.index_select(s.x, s.dim, s.idx)


class MaskedFill(Indexing):
    """Out of place, broadcast 1D mask selecting every other column."""

    op = "masked_fill"
    params = [dict(shape=(8192, 8192), dtype=dt, value=1) for dt in FLOAT_DTYPES]
    pr = [dict(shape=(8192, 8192), dtype="bf16", value=1)]

    def setup(self, p, device):
        cols = p["shape"][1]
        mask = torch.zeros(cols, dtype=torch.bool)
        mask[spaced(cols - 2, cols // 2, "cpu")] = True
        return State(
            x=full(p["shape"], 0, p["dtype"], device),
            mask=mask.to(device),
            v=p["value"],
        )

    def fwd(self, s):
        return s.x.masked_fill(s.mask, s.v)


class Take(Indexing):
    op = "take"
    params = [
        dict(shape=(8192, 8192), dtype=dt, n_index=4096 * 4096) for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", n_index=4096 * 4096)]

    def setup(self, p, device):
        return State(
            x=randn(p["shape"], p["dtype"], device),
            idx=spaced(8190 * 8190, p["n_index"], device),
        )

    def fwd(self, s):
        return torch.take(s.x, s.idx)


class Put(Take):
    """put_ (accumulate=False); writing the same values every iteration is
    idempotent, so no reset."""

    op = "put"

    def expected_ops(self, p, pass_):
        return ("put_",)

    def setup(self, p, device):
        x = full(p["shape"], 0, p["dtype"], device)
        idx = spaced(8190 * 8190, p["n_index"], device)
        return State(x=x, idx=idx, src=full((4096, 4096), 1, p["dtype"], device))

    def fwd(self, s):
        return s.x.put_(s.idx, s.src)
