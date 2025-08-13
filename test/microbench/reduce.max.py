import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [(8192, 8192)]
backward = False
num_iter = 20

# dim = 1: reduce along contiguous dim
# dim = 0: reduce along strided dim
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [1, 0]:
            input = torch.randn(8192, 8192, dtype=dtype, device=device)

            # warm up
            output = torch.max(input, 1)
            output = torch.max(input, 0)

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
                    output = torch.max(input, dim)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                output = torch.max(input, dim)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
