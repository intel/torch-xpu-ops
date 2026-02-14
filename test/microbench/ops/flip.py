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
    - shape: list, e.g., [64, 1024, 1024]
    - dim: tuple or list of int (axes to flip)
    - datatype: torch.dtype
    - backward: bool
    """
    input_shape = config["shape"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(input_shape, device=device, dtype=dtype, requires_grad=True)
    output = torch.flip(input, dim)

    # Backward
    if backward:
        grad = torch.empty_like(output)
        output.backward(grad)


def get_default_cases():
    base_cases = [
        ([64, 1024, 1024], [0, 1]),
        ([1024, 64, 1024], [0, 2]),
        ([1024, 1024, 64], [1, 2]),
        ([16, 128, 512, 512], [0, 2]),
        ([16, 128, 512, 512], [0, 3]),
        ([16, 128, 512, 512], [1, 3]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape, dim in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "dim": dim,
                    "backward": True,
                }
            )
    return cases
