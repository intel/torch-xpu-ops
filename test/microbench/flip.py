import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
shape_list = [
    ((64, 1024, 1024), (0, 1)),
    ((1024, 64, 1024), (0, 2)),
    ((1024, 1024, 64), (1, 2)),
    ((16, 128, 512, 512), (0, 2)),
    ((16, 128, 512, 512), (0, 3)),
    ((16, 128, 512, 512), (1, 3)),
]

backward = True
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape[0], device=device, dtype=dtype)

        if backward:
            input.requires_grad_(True)

        # warm up
        output = torch.flip(input, shape[1])
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
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(num_iter):
                output = torch.flip(input, shape[1])
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            output = torch.flip(input, shape[1])
            if backward:
                gy = torch.empty_like(output)
                output.backward(gy)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
