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
    - divisor: float/int or list (tensor shape), e.g., 2.0 or [1, 1024, 1]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    divisor = config["divisor"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    output = torch.remainder(input, divisor)

    # Backward
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def get_default_cases():
    base_shapes = [
        [1024, 1024, 1024],
        [6, 7, 3, 2],
        [8193, 8193, 4, 4],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    divisors = [2.0, -1.5, 3.0]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for div in divisors:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "divisor": div,
                    "backward": True,
                })
    return cases
