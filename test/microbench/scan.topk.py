import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False

shape_list = [(8193, 8193)]
k = 4096
largest = True
sorted = True

for dim in [None, 0, 1]:
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=dtype, device=device)
            # warm up
            torch.topk(input, k)
            torch.topk(input, k, 0, largest, sorted)
            torch.topk(input, k, 1, largest, sorted)

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
                        torch.topk(input, k)
                    else:
                        torch.topk(input, k, dim, largest, sorted)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
