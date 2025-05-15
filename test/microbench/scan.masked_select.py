import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8193, 8193)]
device = "xpu"
backward = False

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape, dtype=dtype, device=device)
        mask = input.ge(0.5)
        # warm up
        torch.masked_select(input, mask)

        # go
        print("shape:", shape, "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(20):
                torch.masked_select(input, mask)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
