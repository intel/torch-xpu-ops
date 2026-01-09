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
    - datatype: torch.dtype
    - channels_last: bool
    - backward: bool
    """
    N, C, H, W, D, oH, oW, oD = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(N, C, H, W, D, device=device, dtype=dtype, requires_grad=True)

    if channels_last:
        input = input.to(memory_format=torch.channels_last_3d)

    fmp = torch.nn.FractionalMaxPool3d(2, output_size=(oH, oW, oD), return_indices=True)
    # forward
    output = fmp(input)

    # Backward
    if backward:
        grad = torch.randn([N, C, oH, oW, oD], device=device, dtype=dtype)
        output[0].backward(grad)


def get_default_cases():
    base_shapes = [
        [32, 32, 128, 128, 128, 64, 64, 64],
        [1, 3, 144, 144, 144, 72, 72, 72],
        [512, 512, 12, 12, 12, 6, 6, 6],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "channels_last": channels_last,
                    "backward": True,
                })
    return cases
