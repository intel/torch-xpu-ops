# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""matmul (3D @ 2D -> mm) fwd/bwd.

mm runs on the PyTorch-side oneDNN integration, not torch-xpu-ops code, so
the op never gates PRs (nightly report only). Weights with more than 512M
elements skip fp32 (BMG memory).
"""

from ..case import FLOAT_DTYPES, OpBench, rand, State, torch

# (m, k, n): input (2, m, k) @ weight (k, n)
SHAPES = [
    (4, 4096, 50400),
    (4, 2048, 32000),
    (4, 4096, 128256),
    (4, 5120, 32000),
    (4, 3072, 32064),
    (4, 4096, 50272),
    (4, 4096, 250880),
    (4, 2560, 32000),
    (4, 2048, 50272),
    (4, 1792, 250880),
]


class Matmul(OpBench):
    op = "matmul"
    family = "blas"
    gating = False
    passes = ("fwd", "bwd")
    expected = {"fwd": ("mm",), "bwd": ("mm",)}
    params = [
        dict(m=m, k=k, n=n, batch=2, dtype=dt)
        for m, k, n in SHAPES
        for dt in FLOAT_DTYPES
        if not (dt == "fp32" and k * n > 512 * 1024 * 1024)
    ]

    def setup(self, p, device):
        a = rand((p["batch"], p["m"], p["k"]), p["dtype"], device)
        b = rand((p["k"], p["n"]), p["dtype"], device)
        return State(a=a, b=b, inputs=[a, b])

    def fwd(self, s):
        return torch.matmul(s.a, s.b)

    def expect_inputs(self, p, pass_):
        return None
