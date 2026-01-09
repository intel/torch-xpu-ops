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
    - shape: list, e.g., [1, 3, 1200, 1200]
    - scale_factor: float or [float, float]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    scale_factor = config["scale_factor"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    # Forward: bicubic upsample
    output = torch.nn.functional.interpolate(
        input,
        mode="bicubic",
        scale_factor=scale_factor,
        align_corners=True,
    )

    # Backward
    if backward:
        grad = torch.randn_like(output, device=device, dtype=dtype)
        output.backward(grad)


def get_default_cases():
    # Original: zip(shape_list, scale_factor)
    base_cases = [
        ([1, 3, 1200, 1200], [3, 3]),
        ([1, 128, 1200, 1200], [3, 3]),
        ([1, 3, 1200, 1200], [7, 7]),
        ([128, 128, 5, 5], [7, 7]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape, scale in base_cases:
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "scale_factor": scale,
                "mode": "bicubic",
                "backward": True,
            })
    return cases
