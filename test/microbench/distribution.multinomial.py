import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [(8192, 8192)]
backward = False
num_iter = 20

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for replacement in [False, True]:
            for num_samples in [2, 128]:
                input = torch.randn(shape, dtype=dtype, device=device).abs()
                # warm up
                input.multinomial(num_samples, replacement)

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; replacement:",
                    replacement,
                    "; num_samples:",
                    num_samples,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.XPU,
                    ],
                    record_shapes=True,
                ) as prof:
                    for _ in range(num_iter):
                        input.multinomial(num_samples, replacement)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    input.multinomial(num_samples, replacement)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
