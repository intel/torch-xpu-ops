# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Factory and fill ops."""

from ..case import table_bench, torch

Arange = table_bench(
    "arange",
    [dict(start=0, end=3145728, dtype="int64"), dict(end=1024, dtype="int64")],
    pr=[0],
    family="factory",
    defaults={"start": None},
    factory=True,
    fwd=lambda self, s: torch.arange(
        *([] if s.start is None else [s.start]), s.end, dtype=s.dtype, device=s.device
    ),
)
Linspace = table_bench(
    "linspace",
    [dict(start=0.125, end=0.875, steps=7)],
    family="factory",
    factory=True,
    fwd=lambda self, s: torch.linspace(
        s.start, s.end, s.steps, dtype=s.dtype, device=s.device
    ),
)
# fill_ is idempotent, so no reset
Fill = table_bench(
    "fill",
    [
        dict(shape=(2048, 2048), value=-float("inf")),
        dict(shape=(16, 128), value=0, dtype="int64"),
    ],
    family="factory",
    idempotent=True,
    fwd=lambda self, s: s.x.fill_(s.value),
)
