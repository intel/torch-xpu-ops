import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20
shape_list = [
    ((8, 16), (2, 16)),
    ((10, 8192), (10, 8192)),
    ((10, 8192), (8192, 8192)),
    ((4, 512, 512), (4, 513, 512)),
    ((1, 512, 8192), (1, 1024, 8192)),
]

for shape in shape_list:
    for p in [0, 1, 2]:
        for compute_mode in [
            "use_mm_for_euclid_dist_if_necessary",
            "use_mm_for_euclid_dist",
            "donot_use_mm_for_euclid_dist",
        ]:
            for dtype in [torch.float32]:
                input1 = torch.rand(shape[0], device=device, dtype=dtype)
                input2 = torch.rand(shape[1], device=device, dtype=dtype)
                if backward:
                    input1.requires_grad_(True)
                    input2.requires_grad_(True)

                # warm up
                output = torch.cdist(input1, input2, p, compute_mode)
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; P:",
                    p,
                    "; mode:",
                    compute_mode,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        output = torch.cdist(input1, input2, p, compute_mode)
                        if backward:
                            gy = torch.empty_like(output)
                            output.backward(gy)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    output = torch.cdist(input1, input2, p, compute_mode)
                    if backward:
                        gy = torch.empty_like(output)
                        output.backward(gy)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
