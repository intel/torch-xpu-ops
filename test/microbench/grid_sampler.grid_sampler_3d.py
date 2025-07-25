import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (2, 5, 6, 3, 5),
    (8, 16, 64, 64, 64),
]

device = "xpu"
backward = True
num_iter = 20

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for mode in ["bilinear", "nearest"]:
            for padding_mode in ["zeros", "border", "reflection"]:
                for align_corners in [True, False]:
                    N, C, D, H, W = shape
                    input = torch.randn(N, C, D, H, W, dtype=dtype, device=device)
                    grid = torch.randn(N, D, H, W, 3, dtype=dtype, device=device)

                    if backward:
                        input.requires_grad_(True)
                        grid.requires_grad_(True)

                    # warm up
                    output = torch.nn.functional.grid_sample(
                        input,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    if backward:
                        output.sum().backward()

                    # go
                    print(
                        "shape:",
                        (shape),
                        "; datatype:",
                        dtype,
                        "; mode:",
                        mode,
                        "; padding_mode:",
                        padding_mode,
                        "; align_corners:",
                        align_corners,
                        "; backward:",
                        backward,
                    )
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                        record_shapes=True,
                    ) as prof:
                        for i in range(num_iter):
                            output = torch.nn.functional.grid_sample(
                                input,
                                grid,
                                mode=mode,
                                padding_mode=padding_mode,
                                align_corners=align_corners,
                            )
                            if backward:
                                output.sum().backward()
                    print(prof.key_averages().table(sort_by="xpu_time_total"))

                    # E2E time
                    torch.xpu.synchronize()
                    t1 = time.time()
                    for i in range(num_iter):
                        output = torch.nn.functional.grid_sample(
                            input,
                            grid,
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                        if backward:
                            output.sum().backward()
                    torch.xpu.synchronize()
                    t2 = time.time()
                    e2e_time = (t2 - t1) / num_iter
                    print("E2E total time:", f"{float(e2e_time):.20f}")
