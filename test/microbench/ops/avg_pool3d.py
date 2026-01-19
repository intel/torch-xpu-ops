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
    - shape: [N, C, D, H, W]
    - kernel_size: int or [kD, kH, kW]
    - stride: int or [sD, sH, sW] (default: same as kernel_size)
    - datatype: torch.dtype (optional, default torch.float32)
    - channels_last: bool (optional, default False) — uses channels_last_3d if True
    - backward: bool (optional, default True)
    """
    N, C, D, H, W = config["shape"]
    kernel_size = config["kernel_size"]
    stride = config.get("stride", kernel_size)  # PyTorch default: stride = kernel_size
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    # Input tensor
    input = torch.randn(N, C, D, H, W, device=device, dtype=dtype, requires_grad=True)
    if channels_last:
        input = input.to(memory_format=torch.channels_last_3d)

    if backward:
        input.requires_grad_(True)
        if isinstance(kernel_size, int):
            Dout = (D - kernel_size) / stride + 1
            Hout = (H - kernel_size) / stride + 1
            Wout = (W - kernel_size) / stride + 1
        else:
            Dout = (D - kernel_size[0]) / stride[0] + 1
            Hout = (H - kernel_size[1]) / stride[1] + 1
            Wout = (W - kernel_size[2]) / stride[2] + 1
        grad = torch.randn(
            [C, int(Dout), int(Hout), int(Wout)], device=device, dtype=dtype
        )

    # Forward
    AVG3d = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride)
    output = AVG3d(input)

    # Backward
    if backward:
        output[0].backward(grad)


def get_default_cases():
    # Format: ([N, C, D, H, W], kernel_size, stride)
    base_cases = [
        ([16, 24, 28, 19, 19], 3, 2),
        ([16, 1984, 7, 7, 7], (3, 2, 2), (2, 1, 2)),
        ([64, 1024, 14, 14, 14], 6, 4),
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
