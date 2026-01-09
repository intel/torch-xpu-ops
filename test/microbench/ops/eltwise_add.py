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
    - shape: list
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    # Create input 'a'
    a = torch.randn(shape[0], dtype=dtype, device=device)
    if shape[1] == 0.5:
        b = int(shape[1])
    else:
        b = torch.randn(shape[1], dtype=dtype, device=device)
    if shape[0] == 100000:
        a = torch.as_strided(a, (8192, 8192), (20000, 2))
        b = torch.as_strided(b, (8192, 8192), (20000, 2))

    output = a + b


def get_default_cases():
    base_shapes = [
        ([8192, 8192], [8192, 8192]),  # contiguous input
        ([100000, 10000], [100000, 10000]),  # non-contiguous input
        ([8190, 8190], [8190, 8190]),  # non-vectorized input
        ([8192, 8192], 0.5),  # scalar input
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            cases.append({
                "shape": shape,
                "datatype": dtype,
                "backward": False,
            })
    return cases
