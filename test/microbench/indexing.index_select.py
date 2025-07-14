import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 1024), (8192, 8192)]
device = "xpu"
backward = False
cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        main_size = shape[0]
        step = int(main_size / 2)
        input = torch.randn(shape, dtype=dtype, device=device)
        indices = torch.linspace(0, shape[0] - 2, steps=step, device=device).to(
            torch.long
        )

        # warm up
        for i in range(10):
            y_0 = torch.index_select(input, 0, indices)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(20):
                cache_r = cache_w * i
                y_0 = torch.index_select(input, 0, indices)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
