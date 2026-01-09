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
    - shape: list, e.g., [8733, 8733]
    - reduce: str ("none", "mean", default: "mean")
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    reduce = config.get("reduce", "mean")
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    B = shape[0]
    S = shape[1]
    input = torch.randn((B, S), dtype=dtype, device=device, requires_grad=True)
    target = torch.randn((B, S), dtype=dtype, device=device)
    loss = torch.nn.SmoothL1Loss(reduction=reduce)

    # Forward
    output = loss(input, target)

    # Backward
    if backward:
        output.backward(torch.ones_like(output, dtype=dtype, device=device))

def get_default_cases():
    base_shapes = [
        [8732, 8732],
        [8192, 8732],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    reduces = ["none", "mean"]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for reduce in reduces:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "reduce": reduce,
                    "backward": True,
                })
    return cases
