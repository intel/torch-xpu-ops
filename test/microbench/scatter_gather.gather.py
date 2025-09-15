import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    ((2048, 64, 4), (2048, 64, 1), 2),  # LQCD shape
    ((28, 4096, 9), (28, 4096, 1), 2),
    ((512, 36, 4), (512, 36, 1), 2),
    ((102400 * 6400, 4), (102400 * 6400, 1), 1),  # big shape thin
    ((102400, 4 * 6400), (25600, 4 * 6400), 0),  # big shape fat
    ((4 * 6400, 102400), (1 * 6400, 102400), 0),
    ((10240, 8192), (10240, 2048), 1),  # medium shape
    ((8192, 10240), (2048, 2560), 1),
    ((10240, 8192), (2560, 8192), 0),
    ((8192, 10240), (2048, 10240), 0),
]

device = "xpu"
backward = False
num_iter = 20

g_xpu = torch.Generator(device=device)
g_xpu.manual_seed(25)
torch.manual_seed(25)
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        shapes = shape[0]
        ishapes = shape[1]
        dim = shape[2]
        a = torch.randn(shapes, dtype=dtype, device=device)
        index = torch.randint(1, shapes[dim], ishapes, device=device, generator=g_xpu)
        print(
            "shape:",
            shapes,
            "; kernel_size:",
            ishapes,
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
                torch.gather(a, dim, index)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            torch.gather(a, dim, index)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
