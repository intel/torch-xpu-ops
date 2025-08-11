import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
shape_list = [
    ((1, 147, 1359556), (1200, 1200)),
    ((1, 147, 36100), (224, 224)),
    ((1, 147, 33814), (63, 1200)),
    ((1, 147, 33814), (1200, 63)),
]
kernel_size = (7, 7)
dilation = (6, 6)

backward = True
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape[0], dtype=dtype, device=device, requires_grad=True)
        if backward:
            input.requires_grad_(True)
        output_size = shape[1]

        # warm up
        output = torch.nn.functional.fold(
            input, output_size, kernel_size, dilation, 1, 1
        )
        if backward:
            torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))

        # go
        print(
            "shape:",
            shape[0],
            "; datatype:",
            dtype,
            "; output_size:",
            shape[1],
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                output = torch.nn.functional.fold(
                    input, output_size, kernel_size, dilation, 1, 1
                )
                if backward:
                    torch.autograd.grad(
                        output, input, grad_outputs=torch.ones_like(output)
                    )
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            output = torch.nn.functional.fold(
                input, output_size, kernel_size, dilation, 1, 1
            )
            if backward:
                torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
