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
    - shape: list, e.g., [N, C, H, W] or [N, C, D, H, W]
    - affine: bool (default: True)
    - datatype: torch.dtype
    - channels_last: bool
    - backward: bool
    """
    shape = config["shape"]
    affine = config.get("affine", True)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    num_groups = shape[0]
    shape_input = (shape[1], shape[2], shape[3], shape[4])
    C = shape[2]
    memory_format = (
        torch.channels_last_3d
        if len(shape_input) == 5
        else torch.channels_last
    )
    input = torch.randn(shape_input, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=memory_format)

        
    gn = torch.nn.GroupNorm(
        num_groups, C, affine=affine, dtype=dtype
    ).to(device)

    # Forward
    output = gn(input)

    # Backward
    if backward:
        grad_out = torch.randn_like(output).to(device)
        (grad_dpcpp,) = torch.autograd.grad(output, input, grad_out)


def get_default_cases():
    base_shapes = [
        [1, 32, 128, 32, 32],  # all channel for 1 group
        [16, 1024, 128, 32, 32],  # normal shape, big memory
        [32, 32, 32, 64, 64],  # normal shape, small memory, 1 channel per group
        [32, 32, 512, 256, 256],  # group_num=32, channel for per group=16,big memory
        [8, 32, 32, 16, 64, 64],  # 3d
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    affine_opts = [False, True]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for channels_last in [False, True]:
                for affine in affine_opts:
                    cases.append({
                        "shape": shape,
                        "datatype": dtype,
                        "channels_last": channels_last,
                        "affine": affine,
                        "backward": True,
                    })
    return cases
