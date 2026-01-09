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
    - shape: list, e.g., [2048, 64, 4]
    - kernel_size: list, e.g., [2048, 64, 1]
    - dim: int
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    kernel_size = config["kernel_size"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    g_xpu = torch.Generator(device=device)
    g_xpu.manual_seed(25)
    torch.manual_seed(25)
    # Input tensor
    input = torch.randn(shape, device=device, dtype=dtype)

    # Index tensor: values in [0, shape[dim])
    index = torch.randint(1, shape[dim], kernel_size, device=device, generator=g_xpu)

    # Forward
    output = torch.gather(input, dim=dim, index=index)


def get_default_cases():
    # Original: [(shape, kernel_size, dim), ...]
    base_cases = [
        ([2048, 64, 4], [2048, 64, 1], 2),
        ([28, 4096, 9], [28, 4096, 1], 2),
        ([512, 36, 4], [512, 36, 1], 2),
        ([102400 * 6400, 4], [102400 * 6400, 1], 1),
        ([102400, 4 * 6400], [25600, 4 * 6400], 0),
        ([4 * 6400, 102400], [1 * 6400, 102400], 0),
        ([10240, 8192], [10240, 2048], 1),
        ([8192, 10240], [2048, 2560], 1),
        ([10240, 8192], [2560, 8192], 0),
        ([8192, 10240], [2048, 10240], 0),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape, kernel_size, dim in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "kernel_size": kernel_size,
                    "dim": dim,
                    "backward": False,
                }
            )
    return cases
