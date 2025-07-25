import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20

shape_list = [(8192, 8192), (64, 8192), (8192, 64)]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [0, 1]:
            H, W = (int(shape[0]), int(shape[1]))
            input = torch.randn((H, W)).to(dtype=dtype, device=device)

            softmax = torch.nn.Softmax(dim=dim)
            softmax.to(device=device, dtype=dtype)
            grad_dpcpp = torch.randn((H, W)).to(device=device, dtype=dtype)
            input.requires_grad_(True)

            # warm up
            output = softmax(input)
            output.backward(grad_dpcpp)

            # go
            print(
                "shape:",
                (shape),
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
                    output = softmax(input)
                    output.backward(grad_dpcpp)
            print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                output = softmax(input)
                output.backward(grad_dpcpp)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
