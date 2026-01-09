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
    - input_shape: [N, Ck, L]          (Ck = C * kH * kW)
    - output_size: [H_out, W_out]     or int → (H_out, H_out)
    - kernel_size: int or [kH, kW]
    - dilation: int or [dH, dW]       (default: 1)
    - datatype: torch.dtype
    - backward: bool
    - channels_last: ignored (no effect for fold)
    """
    input_shape = config["input_shape"]
    output_size = config["output_size"]
    kernel_size = (7, 7)
    dilation = (6, 6)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(input_shape, device=device, dtype=dtype, requires_grad=True)

    output = torch.nn.functional.fold(
        input,
        output_size,
        kernel_size,
        dilation,
        1,
        1,
    )

    if backward:
        torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))


def get_default_cases():
    base_cases = [
        ([1, 147, 1359556], [1200, 1200]),
        ([1, 147, 36100], [224, 224]),
        ([1, 147, 33814], [63, 1200]),
        ([1, 147, 33814], [1200, 63]),
    ]

    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for input_shape, output_size in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "input_shape": input_shape,
                    "output_size": output_size,
                    "datatype": dtype,
                    "kernel_size": [7, 7],
                    "backward": True,
                }
            )
    return cases
