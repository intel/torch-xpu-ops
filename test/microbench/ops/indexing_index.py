# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch
from core.runner import normalize_dtype


def run_op(config, device):
    """config keys:
    - shape: list, e.g., [4, 15000]
    - mode: str ("with_nonzero" or "without_nonzero")
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    mode = config.get("mode", "with_nonzero")
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    d = torch.rand(shape, dtype=dtype, device=device)
    e = torch.rand(shape, dtype=dtype, device=device)
    f = d < e
    g = e[f]

    # Forward
    if mode == "with_nonzero":
        f = d < e
        g = e[f]
    else:
        f = torch.linspace(0, 4 - 2, steps=int(4 / 2), device=device).to(torch.long)
        g = e[f]


def get_default_cases():
    base_shapes = [
        [4, 15000],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    modes = ["with_nonzero", "without_nonzero"]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for mode in modes:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "mode": mode,
                    "backward": False,
                })
    return cases
