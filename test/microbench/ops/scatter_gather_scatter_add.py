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
    - shape: list, src tensor shape (e.g., [28, 4096, 9, 1])
    - dim: int
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dim = config["dim"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    g_xpu = torch.Generator(device=device)
    g_xpu.manual_seed(25)
    torch.manual_seed(25)

    if dim == 2:
        m, n, k1, k2 = shape[0], shape[1], shape[2], shape[3]
        src = torch.ones((m, n, k1), dtype=dtype, device=device)
        index = torch.randint(0, k2, (m, n, k1), generator=g_xpu, device=device)
        zeros = torch.zeros(m, n, k2, dtype=dtype, device=device)
    elif dim == 0:
        m1, m2, n = shape[0], shape[1], shape[2]
        src = torch.ones((m1, n), dtype=dtype, device=device)
        index = torch.randint(0, m2, (m1, n), generator=g_xpu, device=device)
        zeros = torch.zeros(m2, n, dtype=src.dtype, device=device)
    else:
        m, n1, n2 = shape[0], shape[1], shape[2]
        src = torch.ones((m, n1), dtype=dtype, device=device)
        index = torch.randint(0, n2, (m, n1), generator=g_xpu, device=device)
        zeros = torch.zeros(m, n2, dtype=src.dtype, device=device)

    dst = zeros.scatter_add_(dim, index, src)


def get_default_cases():
    # Original: [(shape, dim), ...]
    base_cases = [
        ([28, 4096, 9, 1], 2),
        ([512, 36, 4, 1], 2),
        ([4, 4096, 4096], 0),
        ([2048, 4, 4096], 0),
        ([2048, 4096, 4], 0),
        ([2048, 4096, 4096], 0),
        ([4096, 8192, 8192], 0),
        ([4097, 8193, 8193], 0),
        ([4, 4096, 4096], 1),
        ([2048, 4, 4096], 1),
        ([2048, 4096, 4], 1),
        ([2048, 4096, 4096], 1),
        ([4096, 8192, 8192], 1),
        ([4097, 8193, 8193], 1),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    cases = []
    for shape, dim in base_cases:
        for dtype in dtypes:
            cases.append(
                {
                    "shape": shape,
                    "datatype": dtype,
                    "dim": dim,
                    "backward": False,
                }
            )
    return cases
