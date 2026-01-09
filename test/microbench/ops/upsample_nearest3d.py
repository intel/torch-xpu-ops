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
    - channels_last: bool
    - backward: bool
    """
    shape = config["shape"]
    scale_factor = config["scale_factor"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=torch.channels_last_3d)

    # Forward: bicubic upsample
    output = torch.nn.functional.interpolate(
        input,
        mode="nearest",
        scale_factor=scale_factor,
    )

    # Backward
    if backward:
        output.backward(torch.ones_like(output))


def get_default_cases():
    # Original: zip(shape_list, scale_factor)
    base_cases = [
        ([8, 32, 256, 256, 2], 3),
        ([8, 512, 16, 16, 4], 1.5),
        ([16, 1024, 23, 23, 7], 2.3),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    channels_lasts = [False, True]

    cases = []
    for shape, scale in base_cases:
        for dtype in dtypes:
            for channels_last in channels_lasts:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "scale_factor": scale,
                        "mode": "nearest",
                        "channels_last": channels_last,
                        "backward": True,
                    }
                )
    return cases
