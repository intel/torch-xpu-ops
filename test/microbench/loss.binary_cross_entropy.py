import time

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20
shape_list = [(8733, 8733), (8733, 513), (513, 8733), (8192, 8192)]

cache_r = torch.randn(1024 * 1024 * 1024, device=device)
cache_w = torch.randn(1024 * 1024 * 1024, device=device)


def _do_test(loss, input, target, dtype, device):
    output = loss(input, target)
    grad_output = torch.ones_like(output, dtype=dtype)
    grad_inputs = torch.autograd.grad(output, input, grad_output)

    return output, grad_inputs


for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        M, N = shape[0], shape[1]
        input = torch.randn((M, N), requires_grad=True)
        target = torch.empty((M, N)).random_(2)
        for reduce in ["none", "mean", "sum"]:
            loss = nn.BCELoss(reduce=reduce)
            m = nn.Sigmoid()
            input = m(input).to(dtype=dtype, device=device)
            target = target.to(dtype=dtype, device=device)
            # warm up
            _do_test(loss, input, target, dtype, device)

            # go
            print(
                "shape:",
                (M, N),
                "; datatype:",
                dtype,
                "; reduce:",
                reduce,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    cache_r = cache_w + 1
                    output_xpu, grad_input_xpu = _do_test(
                        loss, input, target, dtype, device
                    )
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                cache_r = cache_w + 1
                output_xpu, grad_input_xpu = _do_test(
                    loss, input, target, dtype, device
                )
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
