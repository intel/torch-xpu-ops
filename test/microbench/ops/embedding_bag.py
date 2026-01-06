# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import random
from core.runner import normalize_dtype


def run_op(config, device):
    """config keys:
    - batch_size: int
    - mode: str ("sum", "mean", "max")
    - datatype: torch.dtype
    - backward: bool
    """
    reduce = config.get("reduce", "sum")
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    backward = config.get("backward", True)
    dict_len = 2500000
    vect_len = 128
    batch = config.get("batch", 1024)

    input = torch.empty([batch], dtype=torch.long, device=device)
    emb = torch.nn.EmbeddingBag(
        dict_len, vect_len, mode=reduce, dtype=dtype, device=device
    )
    for i in range(batch):
        input[i] = random.randint(0, dict_len - 1)

    bag = torch.empty([batch], dtype=torch.long, device=device)

    for i in range(batch):
        bag[i] = i
 
    # Forward
    output = emb(input, bag)

    # Backward
    if backward:
        grad = torch.randn(batch, vect_len, dtype=dtype, device=device)
        output.backward(grad)


def get_default_cases():
    modes = ["sum", "mean", "max"]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    batch = [1024]

    cases = []
    for dtype in dtypes:
        for reduce in modes:
            cases.append({
                "shape": batch,
                "datatype": dtype,
                "reduce": reduce,
                "backward": True,
            })
    return cases
