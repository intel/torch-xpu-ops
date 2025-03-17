import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

shape_list = [
            ((64, 1024, 1024), (0, 1)),
            ((1024, 64, 1024), (0, 2)),
            ((1024, 1024, 64), (1, 2)),
            ((16, 128, 512, 512), (0, 2)),
            ((16, 128, 512, 512), (0, 3)),
            ((16, 128, 512, 512), (1, 3))
]

backward = True
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape[0], device=device, dtype=dtype)

        if backward:
            input.requires_grad_(True)
        
        # warm up
        output = torch.flip(input, shape[1])
        if backward:
            gy = torch.empty_like(output)
            output.backward(gy) 
        
        # go
        print("shape:", shape[0], "; datatype:", dtype, "; backward:", backward)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
            for i in range(20):
                output = torch.flip(input, shape[1])
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
        