import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
shape_list = [(8192, 8192)]

cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape).to(device).to(dtype)
        target = torch.empty(shape[0], dtype=torch.long).to(device)
        for i in range(8192):
            target[i] = i
        x = torch.tensor(0.5).to(device).to(dtype)
        input.requires_grad = True

        # warm up
        output = F.nll_loss(input, target)
        output.backward(x)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                cache_r = cache_w
                output = F.nll_loss(input, target)
                cache_r = cache_w
                output.backward(x)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
