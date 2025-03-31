import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
shape_list = [(8732, 8732), (8192, 8732)]
cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for reduce in ["none", "mean"]:
            B = shape[0]
            S = shape[1]
            input = torch.randn((B, S), requires_grad=True).to(
                dtype=dtype, device=device
            )
            target = torch.randn((B, S)).to(dtype=dtype, device=device)
            loss = torch.nn.SmoothL1Loss(reduction=reduce)

            # warm up
            output = loss(input, target)
            output.backward(torch.ones_like(output, dtype=dtype, device=device))

            # go
            print(
                "shape:",
                (B, S),
                "; datatype:",
                dtype,
                "; backward:",
                backward,
                "; reduce: 0" if (reduce == "none") else "; reduce: 1",
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
            ) as prof:
                for i in range(20):
                    cache_r = cache_w * i
                    output = loss(input, target)
                    cache_r = cache_w * i
                    output.backward(
                        torch.ones_like(output, dtype=torch.float, device=device)
                    )
            print(prof.key_averages().table(sort_by="xpu_time_total"))
