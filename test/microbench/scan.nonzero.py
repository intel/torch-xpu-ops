import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(2047, 2047, 10), (1, 4 * 15000)]
device = "xpu"
backward = False
num_iter = 20

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        if shape == (2047, 2047, 10):
            input = torch.randint(-2, 3, shape, dtype=dtype, device=device)
        else:
            input = torch.randn(shape, dtype=dtype, device=device)

        # warm up
        torch.nonzero(input)

        # go
        print("shape:", shape, "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(num_iter):
                torch.nonzero(input)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            torch.nonzero(input)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
