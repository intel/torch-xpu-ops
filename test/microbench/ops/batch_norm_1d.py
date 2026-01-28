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
    - shape: [N, C] or [N, C, L]  (1D: C = num_features)
    - datatype: torch.dtype (optional, default torch.float32)
    - backward: bool (optional, default True)
    """
    shape = config["shape"]
    num_features = config["num_features"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    BTN1d = torch.nn.BatchNorm1d(num_features, device=device)
    output = BTN1d(input)

    if backward:
        grad = torch.empty_like(output)
        output.backward(grad)


def get_default_cases():
    base_cases = [
        ([64, 8], 8),
        ([4, 128, 15000], 128),
        ([4, 256, 512], 256),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for shape, num_features in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "num_features": num_features,
                    "backward": True,
                }
            )
    return cases
