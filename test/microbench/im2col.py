import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

shape_list = [(1, 3, 1200, 1200), (1, 3, 224, 224), (1, 3, 63, 1200), (1, 3, 1200, 63)]
kernel_size = (7, 7)
dilation = (6, 6)

backward = True
for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape, dtype=dtype, device=device, requires_grad=backward)

        # warmup
        output = torch.nn.functional.unfold(
            input, kernel_size, dilation=dilation, padding=1, stride=1
        )
        if backward:
            torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                output = torch.nn.functional.unfold(
                    input, kernel_size, dilation=dilation, padding=1, stride=1
                )
                if backward:
                    torch.autograd.grad(
                        output, input, grad_outputs=torch.ones_like(output)
                    )
        print(prof.key_averages().table(sort_by="xpu_time_total"))
