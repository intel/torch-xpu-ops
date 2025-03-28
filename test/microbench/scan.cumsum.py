import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False

shape_list = [(8193, 8193), (1234, 8193), (8192, 1234), (1, 4 * 15000)]

for dim in [0, 1]:
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=dtype, device=device)

            # warm up
            torch.cumsum(input, 0)
            torch.cumsum(input, 1)

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
                    torch.cumsum(input, 0)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
