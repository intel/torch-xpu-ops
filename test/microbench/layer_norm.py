import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20

shape_list = [
    ((1, 1024), (1024)),
    ((2, 4096, 320), (4096, 320)),
    ((512, 3136, 128), (3136, 128)),
    ((128, 49, 196, 1024), (49, 196, 1024)),
]


for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape[0], device=device, dtype=dtype)

        if backward:
            input.requires_grad_(True)

        # warm up
        m = torch.nn.LayerNorm(shape[1], device=device, dtype=dtype)
        output = m(input)
        if backward:
            gy = torch.empty_like(output)
            output.backward(gy)

        # go
        print(
            "shape:",
            shape[0],
            "; datatype:",
            dtype,
            "; dim:",
            shape[1],
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                m = torch.nn.LayerNorm(shape[1], device=device, dtype=dtype)
                output = m(input)
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            m = torch.nn.LayerNorm(shape[1], device=device, dtype=dtype)
            output = m(input)
            if backward:
                gy = torch.empty_like(output)
                output.backward(gy)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
