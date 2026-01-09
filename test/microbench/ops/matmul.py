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
    - shape: (m, k, n) tuple/list → A: [2, m, k], B: [k, n]
    - datatype: torch.dtype
    - backward: bool
    """
    m, k, n = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    m1 = torch.rand(2, m, k, dtype=dtype, device=device, requires_grad=True)
    m2 = torch.rand(k, n, dtype=dtype, device=device, requires_grad=True)

    # Forward: matmul
    output = torch.matmul(m1, m2)

    # Backward
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def get_default_cases():
    base_shapes = [
        [4, 4096, 50400],
        [4, 2048, 32000],
        [4, 4096, 128256],
        [4, 5120, 32000],
        [4, 3072, 32064],
        [4, 4096, 50272],
        [4, 4096, 250880],
        [4, 2560, 32000],
        [4, 2048, 50272],
        [4, 1792, 250880],
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
