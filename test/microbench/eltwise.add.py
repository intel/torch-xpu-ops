import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False
num_iter = 20

shape_list = [
    ((8192, 8192), (8192, 8192)),  # contiguous input
    ((100000, 10000), (100000, 10000)),  # non-contiguous input
    ((8190, 8190), (8190, 8190)),  # non-vectorized input
    ((8192, 8192), (0.5)),  # scalar input
]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        a = torch.randn(shape[0], dtype=dtype, device=device)
        if shape[1] == 0.5:
            b = int(shape[1])
        else:
            b = torch.randn(shape[1], dtype=dtype, device=device)
        if shape[0] == 100000:
            a = torch.as_strided(a, (8192, 8192), (20000, 2))
            b = torch.as_strided(b, (8192, 8192), (20000, 2))

        # warm up
        for i in range(10):
            output = a + b

        # go
        print(
            "shape:",
            (shape[0], shape[1]),
            "; datatype:",
            dtype,
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                output = a + b
        print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            output = a + b
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
