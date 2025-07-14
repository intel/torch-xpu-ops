import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 1024)]
device = "xpu"
backward = False

cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [0, 1]:
            input = torch.zeros(shape, dtype=dtype, device=device)
            indices = torch.linspace(0, 1022, steps=512, device=device).to(torch.long)
            y_0 = torch.ones((512, 1024), dtype=dtype, device=device)
            y_1 = torch.randn((1024, 512), dtype=dtype, device=device)

            # warm up
            for i in range(10):
                output = input.index_copy(0, indices, y_0)

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
                for i in range(10):
                    cache_r = cache_w * i
                    if dim == 0:
                        output = input.index_copy(dim, indices, y_0)
                    else:
                        output = input.index_copy(dim, indices, y_1)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
