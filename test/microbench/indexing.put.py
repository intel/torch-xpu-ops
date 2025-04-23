import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
device = "xpu"
backward = False
cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.linspace(0, 8190 * 8190, steps=4096 * 4096, device=device).to(
            torch.long
        )
        sources = torch.ones((4096, 4096), dtype=dtype, device=device)

        # warm up
        for i in range(10):
            input.put_(index=indices, source=sources)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(20):
                cache_r = cache_w * i
                input.put_(index=indices, source=sources)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
