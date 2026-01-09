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
    - shape: [N, C, D, H, W]
    - mode: str ("bilinear", "nearest")
    - padding_mode: str ("zeros", "border", "reflection")
    - align_corners: bool
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    mode = config.get("mode", "bilinear")
    padding_mode = config.get("padding_mode", "zeros")
    align_corners = config.get("align_corners", False)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    N, C, D, H, W = shape
    input = torch.randn(
        N, C, D, H, W, dtype=dtype, device=device, requires_grad=True
    )
    grid = torch.randn(
        N, D, H, W, 3, dtype=dtype, device=device, requires_grad=True
    )

    # Forward
    output = torch.nn.functional.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # Backward
    if backward:
        output.sum().backward()


def get_default_cases():
    base_shapes = [
        [2, 5, 6, 3, 5],
        [8, 16, 64, 64, 64],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    modes = ["bilinear", "nearest"]
    padding_modes = ["zeros", "border", "reflection"]
    align_corners_opts = [True, False]

    cases = []
    for shape in base_shapes:
        for dtype in dtypes:
            for mode in modes:
                for padding_mode in padding_modes:
                    for align_corners in align_corners_opts:
                        cases.append({
                            "shape": shape,
                            "datatype": dtype,
                            "mode": mode,
                            "padding_mode": padding_mode,
                            "align_corners": align_corners,
                            "backward": True,
                        })
    return cases
