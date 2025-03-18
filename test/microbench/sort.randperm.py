import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False

shape_list = [(8193)]

for shape in shape_list:
    for dtype in [torch.float32]:
        # warm up
        torch.randperm(shape, dtype=dtype, device=device)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                torch.randperm(shape, dtype=dtype, device=device)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
