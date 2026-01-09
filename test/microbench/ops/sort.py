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
    - dim: float/int or list (tensor shape), e.g., 2.0 or [1, 1024, 1]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype)
    if dim is None:
        torch.sort(input)
    else:
        torch.sort(input, dim)


def get_default_cases():
    base_shapes = [
        [8193, 8193],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    dims = [None, 0, 1]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for dim in dims:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "dim": dim,
                    "backward": False,
                })
    return cases
