import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False

shape_list = [(8193, 8193)]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [None, 0, 1]:
            input = torch.randn(shape, dtype=dtype, device=device)

            # warm up
            torch.sort(input)
            torch.sort(input, 0)
            torch.sort(input, 1)

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
                for i in range(20):
                    if dim is None:
                        torch.sort(input)
                    else:
                        torch.sort(input, dim)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
