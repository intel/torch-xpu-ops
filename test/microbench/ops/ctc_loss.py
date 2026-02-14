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
    - shape: list, e.g., [8733, 8733]
    - datatype: torch.dtype
    - backward: bool
    """
    shape = config["shape"]
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)

    T, N, C, S = shape[0], shape[1], shape[2], shape[3]
    g_cpu = torch.Generator()
    g_cpu.manual_seed(15)
    torch.manual_seed(15)
    log_probs = (
        torch.randn(T, N, C, dtype=dtype, device=device)
        .log_softmax(2)
        .detach()
        .requires_grad_()
    )
    targets = torch.randint(1, N, (N, S), dtype=torch.long, device=device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
    target_lengths = torch.randint(1, S, (N,), dtype=torch.long, device=device)

    # Forward
    loss_dpcpp = torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths
    )

    # Backward
    if backward:
        loss_dpcpp.backward()


def get_default_cases():
    base_shapes = [
        [32, 32, 32, 16],
        [128, 128, 128, 128],
        [8, 8, 4, 8],
    ]
    dtypes = [torch.float32]

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
