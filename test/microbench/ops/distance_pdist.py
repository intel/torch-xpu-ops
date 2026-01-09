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
    - shape: [N, D] or [B, N, D]  (last dim = feature dim)
    - p: float or int (default: 2)
    - datatype: torch.dtype
    - backward: bool
    - channels_last: ignored (no effect for pdist)
    """
    input_shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(input_shape, device=device, dtype=dtype, requires_grad=True)

    # Forward: pairwise distance (condensed form)
    output = torch.nn.functional.pdist(input, 2)

    # Backward
    if backward:
        grad = torch.randn_like(output)
        output.backward(grad)


def get_default_cases():
    forward_shapes = [
        [2048, 256],
        [2048, 8192],
        [16, 8192 * 4],
    ]
    backward_shapes = [
        [256, 256],
        [256, 8192],
        [16, 8192 * 4],
    ]

    dtypes = [torch.float32]

    cases = []
    # Forward-only cases (no grad needed)
    for shape in forward_shapes:
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "backward": False,
            })

    # Backward cases (require 2D input: [N, D])
    for shape in backward_shapes:
        if len(shape) != 2:
            continue
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "backward": True,
            })
    return cases
