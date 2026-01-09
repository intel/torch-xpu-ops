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
    - shape: [N, C, H, W]          (input tensor shape)
    - num_features: int             (must equal C; explicit, as in original)
    - datatype: torch.dtype
    - channels_last: bool
    - backward: bool
    """
    N, C, H, W = config["shape"]
    num_features = config["num_features"]  # explicit, matches original intent
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=torch.channels_last)

    BTN = torch.nn.BatchNorm2d(num_features, device=device)
    output = BTN(input)

    if backward:
        grad = torch.randn([C, H, W], device=device, dtype=dtype)
        output[0].backward(grad)


def get_default_cases():
    base_cases = [
        (256, 256, 56, 56, 256),
        (256, 2048, 7, 7, 2048),
        (24, 512, 28, 28, 512),
        (24, 1024, 14, 14, 1024),
        (4, 8, 640, 1024, 8),
        (4, 48, 20, 32, 48),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for N, C, H, W, num_features in base_cases:
        input_shape = [N, C, H, W]
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append({
                    "shape": input_shape,
                    "datatype": dtype,
                    "num_features": num_features,
                    "channels_last": channels_last,
                    "backward": True,
                })
    return cases
