import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False

shape_list = [
    (2049, 2049)
]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randint(100, shape, dtype=dtype, device=device)

        # warm up
        torch.unique(input, sorted=True, return_inverse=True, return_counts=True)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
            for i in range(20):
                output = torch.unique(input, sorted=True, return_inverse=True, return_counts=True)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
