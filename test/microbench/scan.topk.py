import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False
num_iter = 20

shape_list = [(8193, 8193)]
k = 4096
largest = True
sorted = True

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [None, 0, 1]:
            input = torch.randn(shape, dtype=dtype, device=device)
            # warm up
            torch.topk(input, k)
            torch.topk(input, k, 0, largest, sorted)
            torch.topk(input, k, 1, largest, sorted)

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
                    if dim is None:
                        torch.topk(input, k)
                    else:
                        torch.topk(input, k, dim, largest, sorted)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                if dim is None:
                    torch.topk(input, k)
                else:
                    torch.topk(input, k, dim, largest, sorted)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
