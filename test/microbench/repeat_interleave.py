import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (16, 8, 23),
    (4, 2048, 2048),
]
device = "xpu"
backward = False
num_iter = 20

for shape in shape_list:
    for repeats in [8]:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for dim in [0, 2]:
                input = torch.randn(shape, device=device, dtype=dtype)

                if backward:
                    input.requires_grad_(True)

                # warm up
                for i in range(5):
                    output = torch.repeat_interleave(input, repeats, dim)

                    if backward:
                        gy = torch.empty_like(output)
                        output.backward(gy)
                # go
                print(
                    "shape:",
                    shape,
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
                        output = torch.repeat_interleave(input, repeats, dim)

                        if backward:
                            gy = torch.empty_like(output)
                            output.backward(gy)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    output = torch.repeat_interleave(input, repeats, dim)
                    if backward:
                        gy = torch.empty_like(output)
                        output.backward(gy)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
