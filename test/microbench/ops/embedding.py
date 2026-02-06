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
    - input_shape: list
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)
    dict_len = 2500000
    vect_len = 128
    num_iter = 20

    # Embedding module
    emb = torch.nn.Embedding(dict_len, vect_len, dtype=dtype, device=device)
    input = torch.randint(0, dict_len, shape, device=device)

    # Forward
    output = emb(input)

    # Backward
    if backward:
        grad = torch.randn(1024, 8, vect_len, dtype=dtype, device=device)
        output.backward(grad)


def get_default_cases():
    base_shapes = [
        [1024, 8],
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
