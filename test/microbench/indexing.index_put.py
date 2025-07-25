import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(4, 15000)]
device = "xpu"
backward = False
num_iter = 20

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for mode in ["with_nonzero", "without_nonzero"]:
            d = torch.rand(4, 15000, dtype=dtype, device=device)
            e = torch.rand(4, 15000, dtype=dtype, device=device)
            f = d < e
            g = e[f]

            if mode == "with_nonzero":
                # warm up
                for i in range(100):
                    d[f] = g

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; mode:",
                    mode,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        d[f] = g
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    d[f] = g
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
            else:
                f = f.nonzero()
                index = []
                for i in range(f.dim()):
                    index.append(f.select(1, i))
                # warm up
                for i in range(100):
                    d[index] = g

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; mode:",
                    mode,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        d[index] = g
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    d[index] = g
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
