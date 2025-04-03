import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [(8192, 8192)]
backward = False

# dim = 1: reduce along contiguous dim
# dim = 0: reduce along strided dim
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for dim in [1, 0]:
            input = torch.randn(8192, 8192, dtype=dtype, device=device)

            # warm up
            output = torch.max(input, 1)
            output = torch.max(input, 0)

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
                    output = torch.max(input, dim)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
