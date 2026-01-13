# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
from core.runner import normalize_dtype


def run_op(config, device):
    """config keys:
    - shape: list, e.g., [8733, 8733]
    - reduce: str ("none", "mean", "sum", default: "mean")
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    reduce = config.get("reduce", "mean")
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    M, N = shape[0], shape[1]
    input = torch.randn((M, N), requires_grad=True)
    target = torch.empty((M, N)).random_(2)

    loss = nn.BCELoss(reduce=reduce)
    m = nn.Sigmoid()
    input = m(input).to(dtype=dtype, device=device)
    target = target.to(dtype=dtype, device=device)

    # Forward
    output = loss(input, target)

    # Backward
    if backward:
        grad_output = torch.ones_like(output, dtype=dtype)
        grad_inputs = torch.autograd.grad(output, input, grad_output)


def get_default_cases():
    base_shapes = [
        [8733, 8733],
        [8733, 513],
        [513, 8733],
        [8192, 8192],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    reduces = ["none", "mean", "sum"]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for reduce in reduces:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "reduce": reduce,
                        "backward": True,
                    }
                )
    return cases
