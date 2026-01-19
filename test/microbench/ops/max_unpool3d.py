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
    - shape: [N, C, D, H, W]  # output shape of unpool (i.e., original input shape to maxpool)
    - kernel_size: int or [kH, kW] (default: 2)
    - datatype: torch.dtype
    - channels_last: bool
    - backward: bool
    """
    N, C, D, H, W = config["shape"]
    kernel_size = config.get("kernel_size", 2)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn([N, C, D, H, W], device=device, dtype=dtype)
    if channels_last:
        input = input.to(memory_format=torch.channels_last_3d)

    # Pooled values: [N, C, H_pool, W_pool]
    pool = torch.nn.MaxPool3d(kernel_size, return_indices=True).to(
        device=device, dtype=dtype
    )
    unpool = torch.nn.MaxUnpool3d(kernel_size).to(device=device, dtype=dtype)

    output, indices = pool(input)
    if channels_last:
        x_dpcpp = output.to(memory_format=torch.channels_last_3d).to(
            device=device, dtype=dtype
        )
        indices_dpcpp = indices.to(memory_format=torch.channels_last_3d).to(
            device=device, dtype=torch.int64
        )
    else:
        x_dpcpp = output.to(device=device, dtype=dtype)
        indices_dpcpp = indices.to(device=device, dtype=torch.int64)

    # Backward
    if backward:
        x_dpcpp.requires_grad_(True)
        if channels_last:
            grad_dpcpp = (
                torch.randn([N, C, D, H, W])
                .to(memory_format=torch.channels_last_3d)
                .to(device=device, dtype=dtype)
            )
        else:
            grad_dpcpp = torch.randn([N, C, D, H, W]).to(device=device, dtype=dtype)

    y_dpcpp = unpool(x_dpcpp, indices_dpcpp, output_size=torch.Size([N, C, D, H, W]))

    if backward:
        y_dpcpp.backward(grad_dpcpp)


def get_default_cases():
    base_shapes = [
        [2, 32, 64, 64, 64],
        [4, 33, 64, 64, 64],
        [16, 32, 32, 32, 32],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    kernel_size = 2

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "kernel_size": kernel_size,
                        "channels_last": channels_last,
                        "backward": True,
                    }
                )
    return cases
