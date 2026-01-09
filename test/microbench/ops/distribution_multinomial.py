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
    - input_shape: list, e.g., [C] or [B, C]  (C = num_classes)
    - num_samples: int (>0)
    - replacement: bool
    - datatype: torch.dtype (for input; output is always Long)
    - backward: must be False (multinomial is non-differentiable)
    """
    input_shape = config["input_shape"]
    num_samples = config["num_samples"]
    replacement = config["replacement"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))

    # Multinomial requires non-negative input; use abs(randn) as in original
    input = torch.randn(input_shape, device=device, dtype=dtype).abs()
    input.multinomial(num_samples, replacement=replacement)


def get_default_cases():
    base_input_shapes = [
        [8192, 8192],
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    replacement_vals = [False, True]
    num_samples_vals = [2, 128]

    cases = []
    for shape in base_input_shapes:
        for dtype in dtypes:
            for replacement in replacement_vals:
                for num_samples in num_samples_vals:
                    cases.append({
                        "input_shape": shape,
                        "datatype": dtype,
                        "replacement": replacement,
                        "num_samples": num_samples,
                        "backward": False,
                    })
    return cases
