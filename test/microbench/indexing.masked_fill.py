import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
device = "xpu"
backward = False
num_iter = 20
cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.zeros(shape, dtype=dtype, device=device)
        masks_ = torch.zeros((8192), dtype=dtype, device=device)
        indices = torch.linspace(0, 8190, steps=4096, device=device).to(torch.long)
        masks_.index_fill_(0, indices, True)
        masks = masks_.to(torch.bool)

        # warm up
        for i in range(10):
            y_1 = input.masked_fill(mask=masks, value=1)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(num_iter):
                cache_r = cache_w * i
                y_1 = input.masked_fill(mask=masks, value=1)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            cache_r = cache_w * i
            y_1 = input.masked_fill(mask=masks, value=1)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
