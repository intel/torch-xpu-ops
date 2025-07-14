import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192), (8192, 8192)]
device = "xpu"
backward = False

cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape, dtype=dtype, device=device)

        # warm up
        output = torch.diag(input)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(10):
                cache_r = cache_w * i
                output = torch.diag(input)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
