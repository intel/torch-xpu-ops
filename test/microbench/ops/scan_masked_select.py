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
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype)
    mask = input.ge(0.5)
    torch.masked_select(input, mask)


def get_default_cases():
    base_shapes = [
        [8193, 8193],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "backward": False,
            })
    return cases
