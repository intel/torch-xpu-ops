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
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    input = torch.zeros(shape, dtype=dtype, device=device)
    masks_ = torch.zeros((8192), dtype=dtype, device=device)
    indices = torch.linspace(0, 8190, steps=4096, device=device).to(torch.long)
    masks_.index_fill_(0, indices, True)
    masks = masks_.to(torch.bool)

    # Forward
    output = input.masked_fill(mask=masks, value=1)


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
                    "backward": False,
                }
            )
    return cases
