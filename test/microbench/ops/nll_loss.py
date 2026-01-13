# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn.functional as F
from core.runner import normalize_dtype


def run_op(config, device):
    """config keys:
    - shape: list, e.g., [8733, 8733]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    target = torch.empty(shape[0], dtype=torch.long, device=device)
    for i in range(8192):
        target[i] = i
    loss = F.nll_loss
    x = torch.tensor(0.5, dtype=dtype, device=device)

    # Forward
    output = loss(input, target)

    # Backward
    if backward:
        output.backward(x)


def get_default_cases():
    base_shapes = [
        [8192, 8192],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "backward": True,
                }
            )
    return cases
