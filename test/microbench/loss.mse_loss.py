import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
shape_list = [(8192, 8192)]

cache_r = torch.randn((1024 * 1024 * 1024), device=device)
cache_w = torch.randn((1024 * 1024 * 1024), device=device)

def _do_test(loss, input, target, dtype, device):
    input = input.to(dtype=dtype, device=device)
    target = target.to(dtype=dtype, device=device)

    output = loss(input, target)
    grad_output = torch.ones_like(output, dtype=dtype, device=device)
    grad_inputs = torch.autograd.grad(output, input, grad_output)

    # warm up
    output = loss(input, target)
    output.backward(grad_output)

    # go
    print(
        "shape:",
        (shape),
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
            cache_r = cache_w
            output = loss(input, target)
            cache_r = cache_w
            output.backward(grad_output)
    print(prof.key_averages().table(sort_by="xpu_time_total"))

for shape in shape_list:
    for reduce in ["none", "mean"]:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, requires_grad=True)
            target = torch.randn(shape)
            loss = nn.MSELoss(reduction=reduce)
            _do_test(loss, input, target, dtype, device)
