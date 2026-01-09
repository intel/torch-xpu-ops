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
    - shape: list, e.g., [1, 1024] or [2, 4096, 320]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    dim = config.get("dim", 1024)
    backward = config.get("backward", True)

    input = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)

    # LayerNorm module
    m = torch.nn.LayerNorm(dim, device=device, dtype=dtype)

    # Forward
    output = m(input)

    # Backward
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def get_default_cases():
    base_shapes = [
        ([1, 1024], [1024]),
        ([2, 4096, 320], [4096, 320]),
        ([512, 3136, 128], [3136, 128]),
        ([128, 49, 196, 1024], [49, 196, 1024]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "dim": shape[1],
                "backward": True,
            })
    return cases
