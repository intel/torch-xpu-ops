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
    - shape: [N, C, H, W]
    - output_size: int or [H_out, W_out]
    - datatype: torch.dtype
    - channels_last: bool
    - backward: bool
    """
    N, C, H, W = config["shape"]
    output_size = config["output_size"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=True)

    if channels_last:
        input = input.to(memory_format=torch.channels_last)

    # AdaptiveMaxPool2d
    adapt_mp2d = torch.nn.AdaptiveMaxPool2d(
        output_size=output_size, return_indices=True
    )
    # forward
    output = adapt_mp2d(input)

    # Backward
    if backward:
        Wout = output_size[0]
        Hout = output_size[1]
        grad = torch.randn([N, C, Hout, Wout], device=device, dtype=dtype)  
        output[0].backward(grad)


def get_default_cases():
    base_cases = [
        ([8, 512, 7, 7], [1, 1]),
        ([8, 512, 32, 32], [7, 7]),
        ([8, 256, 56, 56], [14, 14]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    
    cases = []
    for shape, out_size in base_cases:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "output_size": out_size,
                    "channels_last": channels_last,
                    "backward": True,
                })
    return cases
