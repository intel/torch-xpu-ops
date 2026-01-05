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
    - input1_shape: list, e.g., [N, P] or [B, N, P]
    - input2_shape: list, e.g., [M, P] or [B, M, P] (last dim must match input1)
    - p: int (default: 2)
    - compute_mode: str (default: "use_mm_for_euclid_dist_if_necessary")
    - datatype: torch.dtype
    - backward: bool
    - channels_last: ignored (no effect for cdist)
    """
    shape_pair = config["shape"]
    input1_shape, input2_shape = shape_pair
    p = config.get("p", 2)
    compute_mode = config.get("compute_mode", "use_mm_for_euclid_dist_if_necessary")
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    input1 = torch.randn(input1_shape, device=device, dtype=dtype, requires_grad=True)
    input2 = torch.randn(input2_shape, device=device, dtype=dtype, requires_grad=True)

    # Forward
    output = torch.cdist(input1, input2, p=p, compute_mode=compute_mode)

    # Backward
    if backward:
        grad = torch.empty_like(output)
        output.backward(grad)


def get_default_cases():
    base_cases = [
        ([8, 16], [2, 16]),
        ([10, 8192], [10, 8192]),
        ([10, 8192], [8192, 8192]),
        ([4, 512, 512], [4, 513, 512]),
        ([1, 512, 8192], [1, 1024, 8192]),
    ]
    p_values = [0, 1, 2]
    compute_modes = [
        "use_mm_for_euclid_dist_if_necessary",
        "use_mm_for_euclid_dist",
        "donot_use_mm_for_euclid_dist",
    ]
    dtypes = [torch.float32]

    cases = []
    for shape1, shape2 in base_cases:
        for p in p_values:
            for mode in compute_modes:
                for dtype in dtypes:
                    cases.append({
                        "shape": [shape1, shape2],
                        "datatype": dtype,
                        "p": p,
                        "mode": mode,
                        "backward": True,
                    })
    return cases
