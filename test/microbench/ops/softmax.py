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
    - shape: list, e.g., [8192, 8192]
    - dim: int
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    # Forward: softmax
    softmax = torch.nn.Softmax(dim=dim)
    output = softmax(input)

    # Backward
    if backward:
        grad = torch.randn_like(output, device=device, dtype=dtype)
        output.backward(grad)


def get_default_cases():
    # Original: [(H, W), ...]
    base_shapes = [
        [8192, 8192],
        [64, 8192],
        [8192, 64],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    dims = [0, 1]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for dim in dims:
                cases.append(
                    {
                        "shape": shape,
                        "dim": dim,
                        "datatype": dtype,
                        "backward": True,
                    }
                )
    return cases
