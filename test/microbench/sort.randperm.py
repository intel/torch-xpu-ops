import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False
num_iter = 20

shape_list = [(8193)]

for shape in shape_list:
    for dtype in [torch.float32]:
        # warm up
        torch.randperm(shape, dtype=dtype, device=device)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                torch.randperm(shape, dtype=dtype, device=device)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            torch.randperm(shape, dtype=dtype, device=device)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
