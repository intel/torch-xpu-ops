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
    - shape: list, e.g., [1024, 1024]
    - dim: int (axis to index along)
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    input = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.linspace(0, 1022, steps=512, device=device).to(torch.long)
    y_0 = torch.ones((512, 1024), dtype=dtype, device=device)
    y_1 = torch.randn((1024, 512), dtype=dtype, device=device)

    # Forward
    if dim == 0:
        output = input.index_copy(dim, indices, y_0)
    else:
        output = input.index_copy(dim, indices, y_1)


def get_default_cases():
    base_shapes = [
        [1024, 1024],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    dims = [0, 1]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for dim in dims:
                cases.append(
                    {
                        "shape": shape,
                        "datatype": dtype,
                        "dim": dim,
                        "backward": False,
                    }
                )
    return cases
