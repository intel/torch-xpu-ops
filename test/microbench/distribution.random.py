import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
backward = False
num_iter = 20

if __name__ == "__main__":
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=torch.bfloat16, device=torch.device("xpu"))

            # warm up
            input.random_(-(2**8), 2**8)

            # go
            print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    input.random_(-(2**8), 2**8)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                input.random_(-(2**8), 2**8)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
