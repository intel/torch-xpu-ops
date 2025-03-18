import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True

shape_list = [(8192, 8192), (64, 8192), (8192, 64)]

for dim in [0, 1]:
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            H, W = (int(shape[0]), int(shape[1]))
            input = torch.randn((H, W)).to(dtype=dtype, device=device)

            softmax = torch.nn.Softmax(dim=dim)         
            softmax.to(device=device, dtype=dtype)
            grad_dpcpp = torch.randn((H, W)).to(device=device, dtype=dtype)
            input.requires_grad_(True)

            # warm up
            output = softmax(input)
            output.backward(grad_dpcpp)

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
                    output = softmax(input)
                    output.backward(grad_dpcpp)
            print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))
