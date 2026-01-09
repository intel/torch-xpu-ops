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
    """config keys: shape[N,C,H,W], out[OH,OW], dtype, channels_last(bool), backward(bool)"""
    N, C, H, W = config["shape"]
    output_size = tuple(config["output_size"])
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    input = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=torch.channels_last)

    output = torch.nn.AdaptiveAvgPool2d(output_size)(input)

    if backward:
        Wout = output_size[0]
        Hout = output_size[1]
        grad = torch.rand([C, Hout, Wout], device=device, dtype=dtype, requires_grad=True)
        output[0].backward(grad)

def get_default_cases():
    base_shapes = [
        ([8, 512, 32, 32], [7, 7]),
        ([8, 256, 56, 56], [14, 14]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for shape, out in base_shapes:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "channels_last": channels_last,
                    "output_size": out,
                    "backward": True,
                })
    return cases
