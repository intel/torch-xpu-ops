import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20
shape_list = [
    (1, 32, 128, 32, 32),  # all channel for 1 group
    (16, 1024, 128, 32, 32),  # normal shape, big memory
    (32, 32, 32, 64, 64),  # normal shape, small memory, 1 channel per group
    (32, 32, 512, 256, 256),  # group_num=32, channel for per group=16,big memory
    (8, 32, 32, 16, 64, 64),  # 3d
]


for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for channels_last in [False, True]:
            for affine in [False, True]:
                num_groups = shape[0]
                shape_input = (shape[1], shape[2], shape[3], shape[4])
                C = shape[2]
                memory_format = (
                    torch.channels_last_3d
                    if len(shape_input) == 5
                    else torch.channels_last
                )

                if channels_last:
                    input = (
                        torch.randn(shape_input)
                        .to(memory_format=memory_format)
                        .to(device=device, dtype=dtype)
                    )
                else:
                    input = torch.randn(shape_input).to(device=device, dtype=dtype)

                if backward:
                    input.requires_grad_(True)

                m = torch.nn.GroupNorm(num_groups, C, affine=affine, dtype=dtype).to(
                    device
                )

                # warm up
                for i in range(5):
                    output = m(input)

                    if backward:
                        grad_out = torch.randn_like(output).to(device)
                        (grad_dpcpp,) = torch.autograd.grad(output, input, grad_out)

                # go
                print(
                    "shape:",
                    (shape_input),
                    "; datatype:",
                    dtype,
                    "; channels_last:",
                    channels_last,
                    "; affine:",
                    affine,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        output = m(input)

                        if backward:
                            grad_out = torch.randn_like(output).to(device)
                            (grad_dpcpp,) = torch.autograd.grad(output, input, grad_out)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    output = m(input)
                    if backward:
                        grad_out = torch.randn_like(output).to(device)
                        (grad_dpcpp,) = torch.autograd.grad(output, input, grad_out)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
