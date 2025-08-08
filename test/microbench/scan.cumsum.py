import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False
num_iter = 20

shape_list = [(8193, 8193), (1234, 8193), (8192, 1234), (1, 4 * 15000)]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [0, 1]:
            input = torch.randn(shape, dtype=dtype, device=device)

            # warm up
            torch.cumsum(input, 0)
            torch.cumsum(input, 1)

            # go
            print(
                "shape:",
                (shape),
                "; datatype:",
                dtype,
                "; dim:",
                dim,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    torch.cumsum(input, 0)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                torch.cumsum(input, 0)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
