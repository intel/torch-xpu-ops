import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True

shape_list = [(1024, 1024, 1024), (6, 7, 3, 2), (8193, 8193, 4, 4)]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for divisor in [2, -1.5, 3]:
            input = torch.randn(shape, device=device, dtype=dtype)
            if backward:
                input.requires_grad_(True)

            # warm
            output = torch.remainder(input, divisor)
            if backward:
                gy = torch.empty_like(output)
                output.backward(gy)

            # go
            print(
                "shape:",
                shape[0],
                "; datatype:",
                dtype,
                "; divisor:",
                divisor,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(20):
                    output = torch.remainder(input, divisor)
                    if backward:
                        gy = torch.empty_like(output)
                        output.backward(gy)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
