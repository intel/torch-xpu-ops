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
    - kernel_size: int or [kH, kW]
    - stride: int or [sH, sW]
    - datatype: torch.dtype (optional, default torch.float32)
    - channels_last: bool (optional, default False)
    - backward: bool (optional, default True)
    """
    N, C, H, W = config["shape"]
    kernel_size = config["kernel_size"]
    stride = config.get("stride", kernel_size)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    # Input tensor
    input = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=torch.channels_last)

    if backward:
        input.requires_grad_(True)
        if isinstance(kernel_size, int):
            Wout = (W - kernel_size) / stride + 1
            Hout = (H - kernel_size) / stride + 1
        else:
            Wout = (W - kernel_size[1]) / stride[1] + 1
            Hout = (H - kernel_size[0]) / stride[0] + 1
        grad = torch.rand(
            [C, int(Hout), int(Wout)], device=device, dtype=dtype, requires_grad=True
        )

    AVG2d = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    output = AVG2d(input)

    if backward:
        output[0].backward(grad)


def get_default_cases():
    base_cases = [
        ([16, 24, 112, 112], 3, 2),
        ([16, 1984, 7, 7], (3, 2), (2, 1)),
        ([64, 1024, 112, 112], 6, 4),
        ([16, 2048, 224, 224], 3, 2),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for shape, k, s in base_cases:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "kernel_size": k,
                        "stride": s,
                        "channels_last": channels_last,
                        "backward": True,
                    }
                )
    return cases
