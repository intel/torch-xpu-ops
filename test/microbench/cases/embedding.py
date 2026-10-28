# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""embedding / embedding_bag."""

from ..case import F, FLOAT_DTYPES, OpBench, randint, randn, State, torch


class Embedding(OpBench):
    """Dense weight gradient."""

    op = "embedding"
    family = "embedding"
    noise_class = "atomic"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("embedding",), "bwd": ("embedding_dense_backward",)}
    params = [
        dict(indices=(1024, 8), num_embeddings=2500000, dim=128, dtype=dt)
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(indices=(1024, 8), num_embeddings=2500000, dim=128, dtype="bf16")]

    def setup(self, p, device):
        w = randn((p["num_embeddings"], p["dim"]), p["dtype"], device)
        return State(
            w=w, idx=randint(0, p["num_embeddings"], p["indices"], device), inputs=[w]
        )

    def fwd(self, s):
        return F.embedding(s.idx, s.w)

    def expect_inputs(self, p, pass_):
        if pass_ != "fwd":
            return None
        return [
            dict(shape=(p["num_embeddings"], p["dim"]), dtype=p["dtype"]),
            dict(shape=tuple(p["indices"]), dtype="int64"),
        ]


class EmbeddingBag(OpBench):
    """One index per bag; dense weight gradient."""

    op = "embedding_bag"
    family = "embedding"
    noise_class = "atomic"
    passes = ("fwd", "bwd")
    # fwd without grad tracking dispatches _embedding_bag_forward_only;
    # bwd goes through _embedding_bag_backward -> _embedding_bag_dense_backward
    expected = {
        "fwd": ("_embedding_bag_forward_only|_embedding_bag",),
        "bwd": ("_embedding_bag_backward|_embedding_bag_dense_backward",),
    }
    params = [
        dict(num_embeddings=2500000, dim=128, batch=1024, mode=m, dtype=dt)
        for dt in FLOAT_DTYPES
        for m in ("max", "mean", "sum")
    ]
    pr = [dict(num_embeddings=2500000, dim=128, batch=1024, mode="sum", dtype="bf16")]

    def setup(self, p, device):
        w = randn((p["num_embeddings"], p["dim"]), p["dtype"], device)
        idx = randint(0, p["num_embeddings"], (p["batch"],), device)
        offsets = torch.arange(p["batch"], device=device)
        return State(w=w, idx=idx, offsets=offsets, mode=p["mode"], inputs=[w])

    def fwd(self, s):
        return F.embedding_bag(s.idx, s.w, s.offsets, mode=s.mode)

    def expect_inputs(self, p, pass_):
        if pass_ != "fwd":
            return None
        return [
            dict(shape=(p["num_embeddings"], p["dim"]), dtype=p["dtype"]),
            dict(shape=(p["batch"],), dtype="int64"),
        ]
