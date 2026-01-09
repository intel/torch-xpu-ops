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
    - shape: list
    - p: float (dropout probability, 0 ≤ p ≤ 1; default: 0.5)
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    p = config.get("p", 0.5)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    dropout = torch.nn.Dropout(p=p)
    dropout.to(device=device, dtype=dtype)

    output = dropout(input)

    if backward:
        grad = torch.randn_like(output, device=device, dtype=dtype)
        output.backward(grad)


def get_default_cases():
    base_shapes = [
        [8192, 8192],
        [16, 1024],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    p_values = [0.5]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for p in p_values:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "p": p,
                        "backward": True,
                    }
                )
    return cases
