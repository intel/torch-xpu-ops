import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

forward_shape_list = [(2048, 256), (2048, 8192), (16, 8192 * 4)]
backward_shape_list = [(256, 256), (256, 8192), (16, 8192 * 4)]

for backward in [False, True]:
    shape_list = backward_shape_list if backward else forward_shape_list
    for shape in shape_list:
        for dtype in [torch.float32]:
            input = torch.rand(shape, device=device, dtype=dtype)
            if backward:
                input.requires_grad_(True)
            
            # warm up
            b = torch.nn.functional.pdist(input, 2)
            
            # go
            print("shape:", shape, "; datatype:", dtype, "; backward:", backward)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
                for i in range(20):
                    b = torch.nn.functional.pdist(input, 2)
                    if backward:
                        gy = torch.empty_like(b)
                        b.backward(gy)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
