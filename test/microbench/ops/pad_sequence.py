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
    - shape: list of sequences' shapes, e.g., [[25,300], [22,300], [15,300]]
    - batch_first: bool (default: False)
    - padding_value: float (default: 0.0)
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    batch_first = config.get("batch_first", False)
    padding_value = config.get("padding_value", 0.0)
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", False)

    a = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True)
    c = torch.randn(shape[2], device=device, dtype=dtype, requires_grad=True)

    # Forward: pad_sequence
    output = torch.nn.utils.rnn.pad_sequence(([a, b, c]), batch_first, padding_value)

    # Backward
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def get_default_cases():
    base_cases = [
        ([25, 300], [22, 300], [15, 300]),
        ([2, 1000], [100, 1000], [8192, 1000]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    batch_first_opts = [False, True]
    padding_vals = [0.0, 1.0, 2.0]

    cases = []
    for shape in base_cases:
        for dtype in dtypes:
            for batch_first in batch_first_opts:
                for pad_val in padding_vals:
                    cases.append(
                        {
                            "shape": shape,
                            "datatype": dtype,
                            "batch_first": batch_first,
                            "padding_value": pad_val,
                            "backward": False,
                        }
                    )
    return cases
