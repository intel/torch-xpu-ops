import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [(8192, 8192)]
backward = False
num_iter = 20

# dim = None: reduce all
# dim = 0: reduce along strided dim
# dim = 1: reduce along contiguous dim
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [None, 0, 1]:
            input = torch.randn(shape, dtype=dtype, device=device)

            # warm up
            output = torch.sum(input)
            output = torch.sum(input, 0)
            output = torch.sum(input, 1)

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
                        output = torch.sum(input)
                    else:
                        output = torch.sum(input, dim)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                if dim is None:
                    output = torch.sum(input)
                else:
                    output = torch.sum(input, dim)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
