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
    - shape: list, e.g., [1024, 1024, 1024]
    - shifts: int or list of int
    - dim: int or list of int
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    shifts = config["shifts"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    # Forward: roll
    output = torch.roll(input, shifts=shifts, dims=dim)

    # Backward
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def get_default_cases():
    # Original: [(shape, shifts, dim), ...]
    base_cases = [
        ([1024, 1024, 1024], -1, 0),
        ([1024, 1024, 1024], [128, 128], [-1, 0]),
        ([1024, 1024, 1024], 128, -1),
        ([16, 3, 512, 512], -1, -1),
        ([16, 3, 512, 512], 127, 0),
        ([16, 3, 512, 512], [127, 127], [0, -1]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape, shifts, dim in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "dim": dim,
                    "shifts": shifts,
                    "backward": True,
                }
            )
    return cases
