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
    - shape: list, e.g., [N, C, H, W]
    - datatype: torch.dtype (output will be cast to this dtype)
    - p: float (success probability, 0 < p ≤ 1; default: 0.5)
    - backward: must be False (geometric_ is inplace and non-differentiable)
    """
    shape = config["shape"]
    p = config.get("p", 0.5)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)

    # geometric_() returns integers, but we respect requested dtype (e.g., for storage)
    # PyTorch geometric_() supports float dtypes (values stored as floats, but integer-valued)
    input = torch.randn(shape, device=device, dtype=dtype)
    input.geometric_(p)


def get_default_cases():
    base_shapes = [
        [8192, 8192],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    p_values = [0.5]
    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for p in p_values:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "p": p,
                    "backward": False,
                })
    return cases
