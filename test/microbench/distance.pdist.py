# Copyright 2020-2025 Intel Corporation
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
forward_shape_list = [(2048, 256), (2048, 8192), (16, 8192 * 4)]
backward_shape_list = [(256, 256), (256, 8192), (16, 8192 * 4)]

for backward in [False, True]:
    shape_list = backward_shape_list if backward else forward_shape_list
    for shape in shape_list:
        for dtype in [torch.float32]:
            input = torch.rand(shape, device=device, dtype=dtype)
            if backward:
                input.requires_grad_(True)

            # warm up
            b = torch.nn.functional.pdist(input, 2)

            # go
            print("shape:", shape, "; datatype:", dtype, "; backward:", backward)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(20):
                    b = torch.nn.functional.pdist(input, 2)
                    if backward:
                        gy = torch.empty_like(b)
                        b.backward(gy)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                b = torch.nn.functional.pdist(input, 2)
                if backward:
                    gy = torch.empty_like(b)
                    b.backward(gy)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
