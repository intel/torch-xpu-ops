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
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    kernel_size = (7, 7)
    dilation = (6, 6)
    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    # Forward: im2col = unfold
    output = torch.nn.functional.unfold(
        input, kernel_size, dilation=dilation, padding=1, stride=1
    )

    # Backward
    if backward:
        torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))


def get_default_cases():
    base_shapes = [
        [1, 3, 1200, 1200],
        [1, 3, 224, 224],
        [1, 3, 63, 1200],
        [1, 3, 1200, 63],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "backward": True,
                }
            )
    return cases
