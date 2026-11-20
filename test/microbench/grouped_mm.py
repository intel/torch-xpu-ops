# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
# (groups, rows per group, K, N): one expert layer of a MoE model per entry,
# called as torch._grouped_mm(x, weight.transpose(-2, -1), offs) with weight
# [groups, N, K]. Forward and backward cover the 2D x 3D (forward, grad_input)
# and 2D x 2D (grad_weight) cases. Shapes are DeepSeek-V3 16B (w13: K 2048,
# N 2816; w2: K 1408, N 2048) at 16 x ~190, 8 x ~190, 8 x ~6k and 64 x ~770
# rows per expert, and DeepSeek-V3 671B (w13: K 7168, N 4096; w2: K 2048,
# N 7168) at 8 x ~512.
shape_list = [
    (16, 192, 2048, 2816),
    (16, 192, 1408, 2048),
    (8, 192, 2048, 2816),
    (8, 192, 1408, 2048),
    (8, 6144, 2048, 2816),
    (8, 6144, 1408, 2048),
    (64, 768, 2048, 2816),
    (64, 768, 1408, 2048),
    (8, 512, 7168, 4096),
    (8, 512, 2048, 7168),
]


def grouped_mm(groups, rows, k, n, dtype, backward):
    x = torch.randn(groups * rows, k, dtype=dtype, device=device)
    weight = torch.randn(groups, n, k, dtype=dtype, device=device)
    offs = torch.arange(1, groups + 1, dtype=torch.int32, device=device) * rows
    if backward:
        x.requires_grad_(True)
        weight.requires_grad_(True)
    output = torch._grouped_mm(x, weight.transpose(-2, -1), offs=offs)

    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16]:
            # warm up
            grouped_mm(*shape, dtype, backward)

            # go
            print(
                "groups:",
                shape[0],
                "; rows per group:",
                shape[1],
                "; K:",
                shape[2],
                "; N:",
                shape[3],
                "; datatype:",
                dtype,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    grouped_mm(*shape, dtype, backward)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                grouped_mm(*shape, dtype, backward)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
