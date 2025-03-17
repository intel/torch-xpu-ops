import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [
    (8192, 8192)
]

backward = False

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape, dtype=dtype, device=device)
        # warm up
        input.exponential_(0.5)
        
        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
            for i in range(20):
                input.exponential_(0.5)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
